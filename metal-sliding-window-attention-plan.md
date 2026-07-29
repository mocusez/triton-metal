# Metal 后端：`hard-sliding_window_self_attention.py` 完整修复计划

> 状态：**全部完成（Phase A/B/C/D），未提交。** verbatim
> `hard-sliding_window_self_attention.py` 在 Metal 上跑通，maxerr 2.4e-07；
> 504/504 配置扫描通过。日期 2026-07-29。分支 `metal-develop`。
> metal pytest 581 / lit 117。
> 相关：`metal-flash-attention-plan.md`（`metal.flash_attention` 的设计出处）。

---

## 0. 现状 — 实测证据

跑 verbatim 的 `leet-triton/hard-sliding_window_self_attention.py`（走 `solve()`，
默认 `BLOCK_M=BLOCK_N=16`，`BLOCK_D=max(16, next_pow2(d))`）：

**编译通过、launch 成功、output 缓冲区一个字节都没被写。** 把 output 预填成
`7.0` 检测「是否被写过」：

| M, d, window | 结果 | Q[0][0] |
|---|---|---|
| 32,16,4 | 全是 7.0，写入 0 个元素 | 0.5544 |
| 64,16,8 | 全是 7.0，写入 0 个元素 | -1.0653 |
| 48,32,16 | 只写了 **1** 个元素 | 1.2129 |
| 33,16,7 | 全是 7.0，写入 0 个元素 | -1.3376 |

kernel 本身语义正确：`TRITON_INTERPRET=1` 下 maxerr = 2.4e-7。所以是后端问题。

**这是静默错误（silent miscompile），比编译失败严重。** 没有任何 warning、
没有 diagnostic、没有非零退出码。

### 0a. 生成的 MSL（`TRITON_KERNEL_DUMP=1`，节选）

```metal
kernel void attention(
  device float *v0 …,   // Q
  device float *v1 …,   // K
  device float *v2 …,   // V
  device float *v3 …,   // output
  device uint32_t *v4,  // M
  device uint32_t *v5,  // N
  device uint32_t *v6,  // d
  device uint32_t *v7,  // window_size
  …)
{
  v4[0]; v6[0]; v7[0];              // M / d / window_size 读一下就扔了
  int v8 = (v5[0] + 15);

  // ---- metal.flash_attention (online softmax, simdgroup dots) ----
  uint _fa_N     = v0[0];           // ← v0 是 Q！
  uint _fa_dm    = v0[0];
  uint _fa_dhead = v0[0] / v0[0];
  uint _fa_coloff = tgid.y * _fa_dhead;
  float _fa_scale = 1.0f / sqrt((float)_fa_dhead);
  …                                 // 全窗口 attention，无 window mask
}
```

`v0[0]` 就是 `Q[0][0]`，一个随机 f32 被当 `uint` 读。Q00 为负 →
`_fa_N = 0` → 主循环不执行、`row < _fa_N` 恒假 → 一个元素都不写。
Q00 = 1.21 → `_fa_N = 1`、`_fa_dhead = 1/1 = 1` → 恰好写 1 个元素。
与上表逐行吻合，机理确认。

### 0b. 绕开 FA 路径后是硬失败

把 `BLOCK_M` 提到 64（语义不变，只是触发 matcher 的 `BM > 32` 闸门拒绝）：

```
error: 'ttg.convert_layout' op ttg.convert_layout: broader staged-transpose
       deferred to L1d3 (rank≠2 or shape/elem-type change or non-blocked
       encoding or sizePerThread > 1)
  at vals_exp (line 39), vals_v (line 42)
RuntimeError: Metal backend: convert-tritongpu-to-metal failed
```

即：通用路径撞的正是当初 `tryFlashAttentionLoop` 被造出来绕开的那堵
L1d3 墙（dot B 的 A 操作数是计算出来的 `exp`，matmul track 吸收不了）。
**所以真要支持滑窗，只能走 `metal.flash_attention` 这条专用路，没有捷径。**

---

## 1. 根因 — 三个独立的 bug（全部实测确认）

### 1a. FA matcher 只做结构计数，不做语义验证 → 过度匹配

`tryFlashAttentionLoop`（`third_party/metal/lib/Conversion/TritonGPUToMetal/TritonGPUToMetal.cpp:11402`）
的全部准入条件：

| # | 条件 | SWA 是否满足 |
|---|---|---|
| 1 | 恰好 3 个 iter_args（1 rank-2 f32 + 2 rank-1 f32） | ✅ `out_vals` / `ma` / `sum` |
| 2 | body 里恰好 2 个 `tt.dot` | ✅ |
| 3 | body 里恰好 2 个 `tt.reduce` | ✅ `tl.max` + `tl.sum` |
| 4 | 其中一个 dot 的 B 侧 cone 有 `tt.trans` | ✅ `tl.permute(vals_k,(1,0))` |
| 5 | 三个 dot 操作数能 trace 到 `tt.load` | ✅ |
| 6 | func 里恰好 1 个 `tt.store` | ✅ |
| 7 | func 里恰好 1 个 `arith.divsi` | ✅ —— 但那是 `tl.cdiv(N,16)`！ |
| 8 | `BM<=32`，`BM/BN/BD % 8 == 0`，threadgroup ≤ 32 KiB | ✅ 16/16/16 |

**没有一条检查实际算的是什么。** 后果：

* **滑窗 mask 被整个丢掉。** `|offset_m - offset_n| <= window_size` 在
  TTGIR 里是 `subi → math.absi → cmpi sle → select`（见 `attention.ttgir`
  `%mask_61`…`%vals_qk_ma`），matcher 完全无视它，emitter 渲染的是无窗口
  的完整 attention。`window_size`（`v7`）在 MSL 里只被读一次就扔了。

* **标量参数绑错。** 条件 7 假设「函数里唯一的 divsi 就是 `d_head = d_model/h`」。
  SWA 里唯一的 divsi 是 `%1 = arith.divsi %0, %c16`（`%0 = N + 15`，即
  `tl.cdiv(N, BLOCK_N)`）。于是：
  - `dModelVal ← %0`（= N+15，一个计算值，不是 kernel arg）
  - `hVal ← %c16_i32`（一个常量！）
  - `nVal ← forOp.getUpperBound()` = `%1` = `cdiv(N,16)`（不是 N）

### 1b. `bridgePtrToMemref` / `bufName` 静默回落到 `v0`

`bridgePtrToMemref`（`TritonGPUToMetal.cpp:7786`）只是造一个
`UnrealizedConversionCastOp`，**不校验来源**。真正的 buffer 解析在
emitter 的 `bufName`（`ModuleTranslation.cpp:1808`）：

```cpp
auto it = _buffers.find(m.getAsOpaquePointer());
return "v" + std::to_string(it != _buffers.end() ? it->second : 0);
//                                                  ^^^^^^^^^^^^^ 静默回落
```

kernel 标量参数在转换后是 `metal.get_element(v<i>, 0)`，`bufName` 会走过去；
但 1a 塞进来的是**计算值**（`divsi` 结果、`addi` 结果、常量），走不到任何
`GetElementOp`，`_buffers` 查不到 → 返回 `v0`。

**这是通用隐患，不止影响这一个 kernel**：任何把非 kernel-arg 值交给
`bridgePtrToMemref` 的 matcher 都会静默读到 buffer 0。

### 1c. `BM < 32` 时 lane guard 缺失 → threadgroup 越界（已存在于已提交代码）

emitter 的 online-softmax 段（`ModuleTranslation.cpp:1885` 起）：

```metal
uint q = _fa_lane;  uint row = _fa_rowoff + q;     // _fa_lane ∈ [0,32)
if (row < _fa_N) {
  … _fa_rmax[q] … _fa_rsum[q] …                    // 数组只有 BM 个元素
  _fa_pbuf[q*BN + kk] = p;                         // 缓冲区只有 BM*BN 个
  _fa_obuf[q*BD + d] *= scaler;                    // 缓冲区只有 BM*BD 个
} else {
  for (kk) _fa_pbuf[q*BN + kk] = 0.0f;             // 同样越界
}
```

`q` 取到 31，但缓冲区按 `BM` 分配。`BM == 32` 时恰好不越界，**`BM < 32`
时全部越界写**。epilogue 段（`:1948`）同理。

现有测试 `test_metal_backend_flash_attention.py` 9 个 case 全部
`BLOCKSIZE_N == 32`，所以从未暴露。实测（同一个 MHA kernel，只改 BLOCK）：

```
BLOCKSIZE_N=32: maxerr=3.6e-07   ✅
BLOCKSIZE_N=16: maxerr=1.7e+04   ❌  ← 已提交路径的静默错误
```

SWA 的驱动用 `BLOCK_M=16`，所以即使 1a/1b 修好，不修 1c 也照样错。

---

## 2. IR 差异对照 — MHA（已支持）vs SWA（目标）

两份 TTGIR 已 dump 比对。**同一个数学算法，但几乎每一处的 IR 拼写都不同**，
这是 matcher 必须泛化而不是「多加一个 if」的原因。

| 维度 | MHA `hard-mult_head_attention.py` | SWA `hard-sliding_window_self_attention.py` |
|---|---|---|
| grid | `(cdiv(N,BM), h)` 2D | `(cdiv(M,BM),)` 1D |
| 列偏移 | `pid1 * d_head` | 无（= 0） |
| 行 stride | `d_model` | `d` |
| 特征宽度 | `d_head = d_model / h`（`arith.divsi`） | `d`（直接是 kernel arg） |
| 行数 / 键数 | 同一个 `N` | 分开的 `M` / `N` 两个 arg |
| 主循环 | `scf.for %iv = 0 to %N step 32` | `scf.for %s = 0 to cdiv(N,16) step 1`，`offset_n = %s*16` |
| scale | `mulf(S, splat(divf(1.0, sqrt(sitofp d_head))))` | `divf(S, splat(sqrt(sitofp d)))` |
| max 输入 | `select(bounds_mask, S·scale, -inf)` | `select(window_mask, S/scale, -100.0)`（**不含** bounds mask） |
| exp 输入 | `exp(subf(已 mask 的 S, m))` | `exp(subf(**原始** S, m))` 之后再 `select(bounds,·,0)` → `select(window,·,0)` |
| 窗口 mask | 无 | `cmpi sle, math.absi(subi(bcast m, bcast n)), splat(window_size)` |
| acc 累积 | `math.fma(acc, scaler, dot)` | `mulf(acc, scaler)` 喂给 `tt.dot` 的 C 操作数 |
| sum 累积 | `math.fma(sum, scaler, denom)` | `addf(mulf(sum, scaler), denom)` |
| BM / BN / BD | 32 / 32 / 16 | 16 / 16 / max(16, pow2(d)) |

数值等价性核对（两处看起来不同、实际等价，需在计划里记账）：

1. **`-100` vs `-inf` 填充**：SWA 用 `-100` 填 masked-max。若某 block 整块
   在窗口外，SWA 的 `ma` 变成 `-100` 而不是 `-inf`；此时 `out_vals` 与
   `sum` 都恰好是 0，`0 * exp(-100 - m)` = `0 * exp(-inf - m)` = 0，结果一致。
   emitter 用 `-INFINITY` 是**更**正确的写法。
2. **SWA 的 masked-max 不含 bounds mask**：ragged tail 里越界的 K 行按
   `other=0` 载入 → `S = 0` → 可能把 `ma` 抬到 ≥ 0。但 softmax 有平移不变性，
   且那些位置的 `vals_exp` 被 bounds mask 归零，不进 `sum`。只差舍入。
3. **`S / sqrt(w)` vs `S * (1/sqrt(w))`**：差 1 ulp 量级。现有 FA 测试用
   `atol=rtol=1e-3`，够。

**新增数值风险（窗口特有）**：整块在窗口外时 `m_cur = -INFINITY`，若此时
`m_old` 也是 `-INFINITY`（行 `m` 的前若干个 block 全在窗口外，例如
`m=50, w=4, kb=0`），则 `scaler = exp(-inf - (-inf)) = exp(NaN) = NaN`，
`_fa_rsum` 和 `_fa_obuf` 整行被污染。**必须显式 guard。**
（无窗口时不可达：`row < N && kb < N` ⇒ 至少 `kk=0` 在界内 ⇒ `m_cur` 有限。
所以这不是现有 MHA 路径的潜在 bug。）

---

## 3. 方案分两部分

修复必须**分两次落地**，因为止血和加功能是两件事，风险与紧迫度都不同：

* **Part 1 — 止血**：让 FA matcher 只认它真正能编译的东西。落地后 SWA 从
  「静默算错」变成「干净的编译错误」，与后端其它所有不支持的 kernel 一致。
  这一部分**独立成立、必须先合**，即使 Part 2 最终不做。
* **Part 2 — 支持**：泛化 `metal.flash_attention`（分离 M/N、可选 head 划分、
  可选 window）+ 泛化 matcher（归一化两种代数拼写），让 SWA 真正跑对。

---

## 4. Part 1 — 止血（bug 1a / 1b / 1c）

### Phase A — 最小闸门 ✅ 已完成 2026-07-29（未提交）

实测结果：

| 项 | 修复前 | 修复后 |
|---|---|---|
| SWA（4 个 case） | 静默写 0 个元素 | `RuntimeError: convert-tritongpu-to-metal failed` |
| MHA `BLOCKSIZE_N=32` | maxerr 3.6e-07 | maxerr 3.6e-07（无回归） |
| MHA `BLOCKSIZE_N=16` | **maxerr 1.7e+04** | maxerr 2.7e-07 |

改动 4 个文件、+118 −11 行：

* `TritonGPUToMetal.cpp` `tryFlashAttentionLoop` — 新增 (5a) `isKernelArg` 闸门
  （7 个操作数全查）、(5b) 角色化的 `divsi` 查找（替换 `nDiv != 1`）。
* `ModuleTranslation.cpp` `bufName` — `: 0` 回落改成 `op.emitError()` + `_emitFailed`。
* `ModuleTranslation.h` — 新增 `_emitFailed` 成员。
* `ModuleTranslation.cpp` `translateModule` — 检查 `_emitFailed` 返回 `failure()`。
* emitter online-softmax 段 + epilogue 段 — 三处 lane guard 加 `q < BM`。

测试：`test_metal_backend_flash_attention.py` +4 个 `BLOCKSIZE_N=16` case
（`test_flash_attention_block16_lane_guard`）；新增
`test_metal_backend_sliding_window_attention.py`（9 个 `xfail(strict=True)`
+ 2 个跨阶段有效的锁）；新增 lit `flash_attention_reject.mlir`。

零回归：metal pytest 566 passed / 12 xfailed（同一 binary + 原测试基线：
561 / 2；差值 = 新增的 4 + 2 passed、9 xfailed，±1 来自 `l1d2d_probe`
既有的 flaky 命令式 xfail）。metal lit 115/115（基线 114 + 新增 1）。

> ⚠️ 与本次改动无关的既有失败，勿误记为回归：
> `test/TritonGPU/coalesce-propagate-reduce.mlir`（本次 ninja 根本没有重链
> `triton-opt`），以及 `python/test/unit/{language,runtime,tools}` 下若干
> CUDA-only 的 collection error。

原始设计如下。

---

### Phase A 设计（~0.5 天，立即消除静默错误）

三道互相独立的闸门，任何一道都能挡住当前的 SWA 误匹配；三道都加，因为
它们防的是不同的东西。

**A1. 标量操作数必须是 kernel entry-block 参数**
`tryFlashAttentionLoop` 步骤 (6) 之后加：

```cpp
auto isKernelArg = [&](mlir::Value v) {
  auto ba = mlir::dyn_cast<mlir::BlockArgument>(v);
  return ba && ba.getOwner() == &funcOp.getBody().front();
};
if (!isKernelArg(nVal) || !isKernelArg(dModelVal) || !isKernelArg(hVal))
  return mlir::failure();
```

对 4 个 buffer 操作数（`qPtr/kPtr/vPtr/oPtr`）同样校验——`unwrapPtrToKernelArg`
在失败时会**原样返回输入**而不是空值，目前没人检查这一点。

> 单这一条就挡住 SWA：`nVal` 是 divsi 结果、`hVal` 是常量。

**A2. `divsi` 必须真的是 `d_head = d_model / h`**
不再「取函数里唯一的 divsi」，而是：从列偏移表达式
（`store` 指针链里 `pid1 * X` 的那个 `X`）反查 `d_head`，并要求它是一个
`arith.divsi`，其两个操作数都是 kernel arg。找不到 → `failure()`。
无 head 划分（列偏移为 0）的情形在 Part 1 阶段直接 `failure()`，Part 2 放开。

**A3. emitter 禁止静默回落到 `v0`**
`ModuleTranslation.cpp:1808` 的 `bufName`：

```cpp
auto it = _buffers.find(m.getAsOpaquePointer());
if (it == _buffers.end()) {
  op.emitError() << "metal.flash_attention: operand does not resolve to a "
                    "kernel buffer (matcher bug — refusing to emit)";
  return std::string();   // 或走 translateModule 的失败路径
}
```

`ModuleTranslation::translateModule` 已返回 `LogicalResult`，且
`metal.sdpa` / `metal.matmul` 已有 `op.emitError()` 先例
（`ModuleTranslation.cpp:2123` / `:3964`），照抄那个模式即可。

> 这条是**通用加固**：以后任何 matcher 犯同类错误都会当场炸而不是读 buffer 0。

**A4. 修 lane guard（bug 1c）**
emitter 的 online-softmax 段和 epilogue 段，把

```metal
uint q = _fa_lane; uint row = _fa_rowoff + q;
if (row < _fa_N) { … }
```

改成

```metal
uint q = _fa_lane; uint row = _fa_rowoff + q;
if (q < <BM>u && row < _fa_N) { … }
else if (q < <BM>u) { /* pbuf 清零 */ }
```

（`_fa_rmax`/`_fa_rsum` 初始化处已有 `_fa_lane < BM` 的守卫，对齐它。）

**Phase A 验收**
- `pixi run pytest -s --tb=short python/test/unit/test_metal_backend_flash_attention.py`
  9 个 case 全绿（无回归）。
- 新增参数化 case `BLOCKSIZE_N=16`（`(64,64,4)` / `(48,64,4)`），修 1c 前红、修后绿。
  **这是 1c 的独立回归证据，必须先看到它红。**
- 新增 `python/test/unit/test_metal_backend_sliding_window_attention.py`，
  此阶段整体 `pytest.mark.xfail(raises=RuntimeError)` —— 断言 SWA 现在
  **干净地报错**而不是静默算错。Part 2 完成时把 xfail 摘掉。
- 新增 lit `test/Dialect/Metal/convert-tritongpu-to-metal/flash_attention_reject.mlir`：
  用 SWA 的 TTGIR 做诱饵，`CHECK-NOT: metal.flash_attention`。
- 全量零回归：`pytest python/test/unit/`（当前基线 558）+ `lit test/`（当前 114）。

### Phase B — 语义验证器 ✅ 已完成 2026-07-29（未提交）

实现为 `FaTemplate`（`TritonGPUToMetal.cpp`，紧接 `faConeHasTrans` 之后），
在 matcher 的 envelope 检查之后作为步骤 (7a) 调用。

**做法**：从 `scf.yield` 反向做**角色游走**，把每个值绑定到 FA 模板的一个槽位并
把 defining op 标记为 claimed；layout/shape 管道（`convert_layout` /
`broadcast` / `expand_dims`）沿途 peel 并 claim。最后一道闸门是**覆盖检查**：
loop body 里每一个 op 都必须被 claim，否则拒绝。

除了算法本身，还验证了原 matcher 完全没查的东西：
* **iter_arg 初值**（acc/sum = 0，max = −inf）—— emitter 是硬编码这三个的；
* **loop 形状**（lb = 0、step = BN、ub = N kernel arg）；
* **地址**（`base + major*d_model + col`）与**掩码**（`major < N` 且 `d < d_head`），
  三个 load 加 store 各一份；
* **scale** = `1/sqrt(d_head)`（接受 Python `d_head + 0.0` 提升产生的 `addf`）；
* mask 填充值必须是 `-inf`，两个 dot 的 C 操作数必须是 0；
* 两个 `tt.reduce` 的 combine 必须恰好是单个 `maxnumf`/`maximumf` 与 `addf`，axis = 1。

**故意严格**：等价但不同的拼写（`mulf`+dot-C 而非 `fma`、`S / sqrt(w)` 而非
`S * (1/sqrt(w))`、cdiv 形式的循环）一律拒绝，因为每一处都是 emitter 可能
悄悄不一致的地方。Phase D 会带着测试有意放开这些。

**诱饵实测**（三个与 MHA 结构完全相同、只差一个 op 的 kernel）：

| 诱饵 | Phase B 之前 | Phase B 之后（拒绝原因） |
|---|---|---|
| causal mask | 编译通过，maxerr **2.4** | `softmax mask is not exactly (row < N) & (key < N)` |
| ALiBi 线性 bias | 编译通过，maxerr **1.4** | `logits are not <dot> * scale` |
| scale 漏了 sqrt | 编译通过，maxerr **0.95** | `logit scale is not 1/sqrt(d_head)` |

（「之前」这一列是把 `TritonGPUToMetal.cpp` stash 掉重编译实测的，不是推断。）

`TRITON_METAL_FA_DEBUG=1` 打印拒绝原因和未被 claim 的 op —— 只在通过了全部
结构闸门之后才打印，所以普通的「这个循环不是 FA」不会刷屏。

测试：`test_flash_attention_rejects_near_miss_kernels`（3 个诱饵，断言
「报错 **或** 数值正确」，这条不变式在以后支持了某个变体时依然成立）；
lit `flash_attention_reject_causal.mlir`。

零回归：metal pytest 570 passed / 11 xfailed，metal lit 116/116。

> Phase B 计划里预留的「先只记录不拒绝、跑一遍 MHA 收集白名单再翻成硬拒绝」
> 这一步**没有用上**：模板一次就通过了 MHA 全部 13 个 case（9 个原有 + 4 个
> BLOCKSIZE_N=16），没有出现意料之外的 op。用 `TRITON_METAL_FA_DEBUG` 拿到
> 拒绝原因即可，不需要额外的观察模式编译轮次。

原始设计如下。

---

### Phase B 设计（3–5 天，把「结构计数」换成「模板验证」）

Phase A 是点防御。真正的通用防御是：**要求 loop body 里的每一个 op 都被
FA 模板认领**，未认领的 op 一律 `failure()`。这样以后任何语义不同的
attention 变体（causal mask、ALiBi bias、dropout、softcap…）都会被自动拒绝，
而不是等着谁去实测发现算错。

设计：从 `scf.yield` 的三个操作数反向遍历，把 SSA 值绑定到「角色」上：

```
yield[acc] ← fma(accIter, bcast(expand(scaler)), dotB)
           | dot(P, V, mulf(accIter, bcast(expand(scaler))))
yield[max] ← maxnumf(reduce_max(Smasked, axis=1), maxIter)
yield[sum] ← fma(sumIter, scaler, reduce_add(P))
           | addf(mulf(sumIter, scaler), reduce_add(P))
scaler     ← exp(subf(maxIter, maxNew))
P          ← select(mask*, exp(subf(Sscaled, bcast(expand(maxNew)))), 0.0)*
Sscaled    ← mulf(dotA, splat(divf(1.0, sqrt(sitofp W))))
           | divf(dotA, splat(sqrt(sitofp W)))
dotA       ← dot(Qload, trans(Kload), 0)
dotB       ← dot(P, Vload, acc?)
mask*      ← 每个 cmpi 必须归类为 bounds mask（offset < 界）
              或 window mask（cmpi sle, absi(subi(rowIdx, colIdx)), splat(arg)）
```

收尾断言：`claimedOps ∪ {地址算术 / layout 搬运 / 常量}` 必须覆盖 body 的
**全部** op。「地址算术 / layout 搬运」是一个显式白名单
（`tt.addptr`、`tt.broadcast`、`tt.expand_dims`、`tt.splat`、
`ttg.convert_layout`、`arith.addi/muli`、`arith.cmpi`、`arith.andi`、
`tt.make_range`、`arith.constant`），且只允许出现在 3 个 load 的地址/掩码 cone
与已识别的 mask cone 里。

**Phase B 验收**
- MHA 9 case 仍全绿（模板必须认得已支持的拼写）。
- 3 个诱饵 lit：causal mask（`cmpi sge` 而非 window 形状）、
  logits 上多一个 `mulf` bias、dot 前多一个 `tanh` softcap ——
  全部 `CHECK-NOT: metal.flash_attention`。
- 从 `-DDEBUG` 下打一条 `LLVM_DEBUG` 说明拒绝原因（哪个 op 没被认领），
  便于后续扩展时定位。

---

## 5. Part 2 — 支持滑窗（Phase C / D）

### Phase C + D ✅ 已完成 2026-07-29（未提交）

**verbatim `hard-sliding_window_self_attention.py` 现在在 Metal 上跑通，
maxerr 2.4e-07。**

扫描 M ∈ {1,7,16,17,31,32,63,64,65,100,128,255,256} × d ∈ {3,8,16,17,32,64} ×
window ∈ {0,1,3,7,15,64,1000}：**504/504 编译并数值正确，0 错**。其余 42 个
（全部是 M=1）是干净的编译错误，见下面的 envelope。

**Phase C 实际改动**
* op 签名（`MetalOps.td`）：新增 `$m`（query 行数，与 `$n` 分开）、
  `Optional<...>:$h`、`Optional<...>:$window`、`OptionalAttr<I64Attr>:$window_const`，
  加 `AttrSizedOperandSegments`。汇编格式用关键字前缀
  （`… heads %h window %w`）而不是两个相邻的 `(`,` …^)?`，否则 parser 有歧义。
* emitter：`_fa_M`/`_fa_N` 分开；无 head 划分时 `_fa_dhead = _fa_dm`、
  `_fa_coloff = 0u`；带窗口时发 `int _fa_win` 与
  `abs((int)row - (int)(kb+kk)) <= _fa_win`，插在 masked-max 与 p 两处；
  NaN guard `scaler = (m_old == m_new) ? 1.0f : exp(m_old - m_new)`。

**Phase D 实际改动**（`FaTemplate`，从 509 行长到 795 行）
计划里列的六条全部实现，另外多了两条计划里没写的：

| # | 放开的拼写 | 说明 |
|---|---|---|
| 1 | 循环形式 B | `range(0, cdiv(N,BN))` step 1，key = `iv*BN + arange`；`nVal` 从 `divsi(addi(N, BN-1), BN)` 反解 |
| 2 | 无 head 划分 | `h` 缺省；`d_head` 从 Q load 掩码的 `arange(BD) < d` 项反解；并强制 kernel 不得读 grid dim y/z |
| 3 | M / N 分离 | 行界从 Q load 掩码读，键界从循环边界读 |
| 4 | scale 两种拼写 | `dot * splat(1/sqrt(w))` 与 `dot / splat(sqrt(w))` |
| 5 | 窗口 mask | `cmpi sle, absi(subi(Row, Key)), <band>`，新增 `Idx::Window` 项 |
| 6 | select 链 | P 允许 0/1/2 层 `select(mask, ·, 0)`，内层允许一个 `-inf` 填充的 select |
| 7 | 累积两种拼写 | `fma` 与 `mulf`+`addf` / dot-with-rescaled-C |
| 8 | 地址两种拼写 | 单层 `addptr(base, row*dm + col)` 与两层 `addptr(broadcast(addptr(base, row*dm)), col)` |

两个数值等价性判据写进了代码注释，因为它们不是显然的：
* **masked-max 的掩码只要求是有效掩码的子集**（不是相等）。原 kernel 的 max
  只过窗口不过边界，emitter 两者都过 → 两个 `m` 不同。但 online softmax 除的是
  它自己的 running sum，结果对 `m` 的任意平移不变，所以只差舍入（实测 2.4e-07）。
  emitter 的选择数值上更稳健。
* **masked-max 的填充值接受 `-inf` 或任意负的有限常量**（原 kernel 用 `-100`）。
  同样由平移不变性保证。正数填充被拒绝——那意味着「优先选被 mask 掉的项」，
  只可能是 bug。

**计划外发现：`window_size = 1` 需要单独处理。**
Triton 会把任何等于 1 的 kernel 参数**从签名里整个删掉**并折成
`arith.constant dense<1>`。所以 w=1 根本没有 buffer 可指——和 `h` 缺省是同一类
问题。加了 `window_const` 属性走字面量。不加的话 w=1 会是一个干净的编译错误，
但 w=0 和 w=2 都能跑、唯独 w=1 不能，太别扭了。

**已知 envelope**（都是干净的编译错误，不是错误答案）
* `M == 1`：M 同时也是 N，被 Triton 折成常量后循环边界退化成常数、行/键界失去
  buffer。要支持得给每个标量都加属性回退，一行的 attention 不值得。已用
  `test_sliding_window_attention_single_row_rejects` 钉住「必须报错」。
* `d > 64`：`BLOCK_D = max(16, next_pow2(d))` ≥ 128 → threadgroup 10784 floats
  > 8192 预算 → 拒绝。已用 `test_sliding_window_attention_over_budget_rejects` 钉住。
* 运行时传入**负**的 window：emitter 给 0，原 kernel 给 `0/0 = NaN`。常量负
  window 会被拒绝；运行时的检查不到，属于 garbage-in。

零回归：metal pytest **581 passed / 3 xfailed**，metal lit **117/117**
（全量 lit 386/389，唯一失败是既有的 `TritonGPU/coalesce-propagate-reduce.mlir`，
本次 ninja 没有重链 `triton-opt`）。三个 Phase B 诱饵仍然被拒绝。
MHA 13 个 case 仍然 maxerr ~3e-07。

lit：`flash_attention_reject.mlir` 从「拒绝」测试转正成
`sliding_window_attention.mlir`（正向，`CHECK-SAME: window` + `CHECK-NOT: heads`）；
`flash_attention_emit.mlir` 更新签名；新增 `sliding_window_attention_emit.mlir`
（两个 kernel，覆盖 window-as-operand 与 window-as-attribute）。
`flash_attention_reject_causal.mlir` 保留为拒绝测试。

原始设计如下。

---

### Phase C 设计 — 泛化 `metal.flash_attention` op + emitter（2–3 天）

**op 签名改动**（`third_party/metal/include/Dialect/Metal/IR/MetalOps.td:646`）：

```tablegen
let arguments = (ins Metal_MemRefType:$q, Metal_MemRefType:$k,
                     Metal_MemRefType:$v, Metal_MemRefType:$out,
                     Metal_MemRefType:$m,          // 新增：query 行数
                     Metal_MemRefType:$n,          // key 数
                     Metal_MemRefType:$d_model,    // 行 stride
                     Optional<Metal_MemRefType>:$h,       // 缺省 = 1（无 head 划分）
                     Optional<Metal_MemRefType>:$window,  // 缺省 = 无窗口
                     I64Attr:$bm, I64Attr:$bn, I64Attr:$bd);
```

`$h` 做成 optional 而不是「传一个常量 1 的 buffer」，是因为**没有**常量的
buffer 可传——这正是 bug 1b 的成因，不能再走一遍。MHA 侧 `$m` 与 `$n` 传
同一个 `%N`。`assemblyFormat` 用 `(`,` $h^)? (`,` $window^)?` + 自定义
directive；两个 lit 测试相应更新。

**emitter 改动**（`ModuleTranslation.cpp:1786`）：

```metal
uint _fa_M   = <M>[0];                       // 行界（原来复用 _fa_N）
uint _fa_N   = <N>[0];
uint _fa_dm  = <DM>[0];
uint _fa_dhead = <H 存在 ? DM[0] / H[0] : DM[0]>;
uint _fa_coloff = <H 存在 ? tgid.y * _fa_dhead : 0u>;
int  _fa_win = <WINDOW 存在 ? (int)WINDOW[0] : 0>;
```

窗口谓词（只在 `$window` 存在时生成）：

```metal
#define _FA_INWIN(row, key)  (abs((int)(row) - (int)(key)) <= _fa_win)
```

插入两处：masked-max 循环、exp/pbuf 循环。**pbuf 必须归零**（它直接喂
dot B），所以是 `p = (in_bounds && in_win) ? exp(...) : 0.0f`。

**NaN guard（§2 的窗口特有风险）**：

```metal
float scaler = (m_old == m_new) ? 1.0f : exp(m_old - m_new);
```

（`m_old == m_new == -INFINITY` 时 `exp(NaN)` → NaN，会污染整行。
写成 `m_old == m_new` 而不是 `isinf` 判断，顺带省掉常见的 no-op block 的
`exp` 调用。）

行界从 `_fa_N` 换成 `_fa_M`：Q 载入、softmax 段、epilogue 三处的
`row < _fa_N` 全改 `row < _fa_M`；键界 `kb + kk < _fa_N` 保持不变。

### Phase D — matcher 泛化（3–4 天）

在 Phase B 的模板验证器上放开：

1. **两种循环形式**。从 body 里 `offset_n` 的构造反查，而不是信 `getUpperBound()`：
   - 形式 A：`ub = Nval`、`step = BN`、`offset_n = iv + arange(BN)`
   - 形式 B：`ub = cdiv(Nval, BN)`、`step = 1`、`offset_n = iv*BN + arange(BN)`

   `Nval` 一律从 body 里的 bounds mask `cmpi slt, offset_n, splat(Nval)` 取
   （那才是权威的键界），再**独立验证**循环恰好覆盖 `ceil(Nval/BN)` 块。
   形式 B 需识别 `cdiv` 的 canonical 展开 `divsi(addi(N, BN-1), BN)`。

2. **无 head 划分**。列偏移为常量 0（`pid1` 不参与地址）时，`$h` 置空，
   `d_head = d_model = ` 行 stride。同时要求 grid 是 1D（func 里没有
   `tt.get_program_id y` 的非平凡使用）。

3. **分离 M / N**。行 mask 的 `cmpi slt, offset_m, splat(Mval)` 取 `Mval`，
   键 mask 取 `Nval`；两者可以是同一个 SSA 值。

4. **scale 两种拼写**（见 §2 表），并**验证** `W` 确实等于特征宽度
   （`d_head` 或 `d_model`）—— 现在这一条完全没验。

5. **窗口 mask 识别**：`cmpi sle, math.absi(subi(bcastRow, bcastCol)), splat(warg)`，
   要求 `warg` 是 kernel arg、`bcastRow` 溯源到 `offset_m`、`bcastCol` 溯源到
   `offset_n`。识别到就绑定 `$window`，否则 `$window` 置空。
   `math.absi` 目前不在任何白名单里，要加。

6. **`select` 链**：SWA 的 P 是 `select(window, select(bounds, exp, 0), 0)` 两层，
   MHA 是「masked-S 直接 exp」零层。模板要接受 0/1/2 层任意顺序的
   `select(mask, ·, 0.0)`，且要求这些 mask 是已识别集合的子集。

7. **acc/sum 两种累积拼写**（`math.fma` vs `mulf`+dot-C / `mulf`+`addf`）。

**envelope 检查更新**：`BM <= 32` 保留（每 lane 一行），`BD` 的
threadgroup 预算公式不变。`BLOCK_D = max(16, next_pow2(d))`，`d=128` →
`BD=128` → 10784 floats > 8192 → 拒绝，落到 §0b 的干净报错。**这是已知
envelope 边界，要写进测试注释和 op 的 description。**

---

## 6. 测试计划

### pytest — `python/test/unit/test_metal_backend_sliding_window_attention.py`（新增）

verbatim 拷贝 `leet-triton/hard-sliding_window_self_attention.py` 的 kernel，
对拍手写的 masked-softmax 参考（不能用 `F.scaled_dot_product_attention`，
它的 `attn_mask` 语义要另外构造，直接写 `softmax(masked_fill(QK^T/√d))@V` 更清楚）。

| 场景 | (M, d, window) | 目的 |
|---|---|---|
| block 对齐 | (64, 16, 8) | 基线 |
| ragged M | (33, 16, 7) | 行 tail mask |
| ragged + 大 window | (48, 32, 16) | `d == BD` 无 padding |
| `d < BD` | (32, 12, 4) | padded-column masking（`BD=16`） |
| window = 0 | (64, 16, 0) | 只有对角线；`denom == 1` |
| window ≥ N | (32, 16, 64) | 退化成全注意力，应等于 SDPA |
| window < BLOCK_N | (128, 16, 3) | **整块在窗口外 → NaN guard 的专项 case** |
| 多 program | (256, 16, 8) | `tgid.x` 分块 |
| d = 64 | (64, 64, 16) | `BD=64`，threadgroup 5664 floats |
| 超预算 | (64, 128, 16) | `BD=128` → 期望干净报错（`pytest.raises`） |

Phase A 落地时整个文件先 `xfail(raises=RuntimeError)`；Phase D 落地后转正。

### pytest — `test_metal_backend_flash_attention.py`（扩充）

- 新增 `BLOCKSIZE_N=16` 参数化（bug 1c 的回归锁）。
- **注意：这个 case 必须在修 1c 之前先跑一遍看到它红**（当前 maxerr 1.7e4），
  否则它锁不住任何东西。

### lit

| 文件 | 内容 |
|---|---|
| `convert-tritongpu-to-metal/flash_attention.mlir` | 更新（op 签名变了） |
| `convert-tritongpu-to-metal/sliding_window_attention.mlir` | 新增：`CHECK: metal.flash_attention`、`CHECK-SAME: bm = 16`、window 操作数在位 |
| `convert-tritongpu-to-metal/flash_attention_reject.mlir` | 新增：causal-mask / bias / softcap 三个诱饵，`CHECK-NOT` |
| `metal-translate/flash_attention_emit.mlir` | 更新（`_fa_M`、lane guard） |
| `metal-translate/sliding_window_attention_emit.mlir` | 新增：`CHECK` 窗口谓词 + NaN guard 出现在 MSL 里 |

### 零回归门槛

`pixi run pytest -s --tb=short python/test/unit/`（基线 558）
+ `cd build/cmake.macosx-11.0-arm64-cpython-3.12 && ninja triton-opt && lit -v test/`（基线 114）。

---

## 7. 风险与开放问题

1. **Phase B 的「全覆盖断言」可能误伤 MHA。** Triton 会生成一些死的
   `arith.extsi`/`trunci` 溢出检查链（`findStrideSplatSource` 的注释里已记录
   过这个现象）。白名单要含这些，否则已支持的 kernel 会被自己的验证器拒掉。
   *缓解：Phase B 先做成「记录未认领 op 但不拒绝 + `LLVM_DEBUG` 打印」跑一遍
   MHA 9 case，把实际出现的 op 收进白名单，再翻成硬拒绝。*

2. **optional 操作数的 `assemblyFormat`。** MLIR 的 optional operand 需要
   `AttrSizedOperandSegments` 或自定义 parser/printer。选前者最省事，代价是
   IR 里多一个 `operandSegmentSizes` 属性，两个已有 lit 测试要改 CHECK 行。

3. **`BLOCK_D = max(16, next_pow2(d))` 与 threadgroup 预算的交互。**
   leet 驱动对 `d > 64` 会生成 `BD >= 128`，必然超预算。这不是 bug，但
   要确保它落在 §0b 的**干净报错**上而不是别的地方——需要一个专项 case。

4. **`_fa_win` 的符号。** `window_size` 是 i32（无 `tt.divisibility`），
   buffer 里按 `uint32_t` 读，必须显式 `(int)` 再比较，否则负 window
   会变成巨大的正数（虽然负 window 是无意义输入，但不该静默变成全窗口）。

5. **Part 2 的收益判断。** 如果只想止损，Part 1 单独合入就够了——SWA 会得到
   和其它不支持 kernel 一致的干净报错。Part 2 是 6–8 天的功能开发。
   **建议 Part 1 先合，再决定 Part 2 是否排期。**

---

## 8. 工期与顺序

| 阶段 | 内容 | 估时 | 可独立合入 |
|---|---|---|---|
| A | 最小闸门 A1–A4 + 1c 回归锁 | 0.5 天 | ✅ **优先** |
| B | 语义验证器（模板 + 全覆盖断言） | 3–5 天 | ✅ |
| C | op 签名 + emitter 泛化（M/N、head、window、NaN guard） | 2–3 天 | ⚠️ 需与 D 一起才有用 |
| D | matcher 泛化（两种循环 / 两种 scale / 窗口 / select 链 / 累积拼写） | 3–4 天 | ⚠️ 需与 C 一起 |
| — | 测试 + 全量零回归 + commit 拆分 | 1 天 | — |

合计 **9.5–13.5 天**。Part 1（A+B）= 3.5–5.5 天；Part 2（C+D）= 6–8 天。

commit 拆分建议（对齐仓库既有风格，一个 blocker 一个 commit）：

1. `[Metal] Flash attention: fix threadgroup overflow when BM < 32`
2. `[Metal] Flash attention: reject non-kernel-arg scalars; no silent buffer-0 fallback`
3. `[Metal] Flash attention: template-verified matcher (reject unclaimed body ops)`
4. `[Metal] flash_attention: split M/N, optional head split, optional window`
5. `[Metal] Sliding-window attention: recognize cdiv loop, /sqrt scale, abs-diff mask`
