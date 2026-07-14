# Metal 后端：`hard-mult_head_attention.py` 修复计划

**日期:** 2026-07-14 ｜ **分支:** metal-develop ｜ **状态:** 计划 —— **架构已定:选项 A(硬件级 simdgroup flash attention)**

目标：让 `leet-triton/hard-mult_head_attention.py`（flash-attention / online-softmax
风格的多头注意力）在 Triton Metal 后端上 **bit-exact** 编译并运行。

---

## 0. 现状 — 编译在哪一步失败（实测）

实测编译在 `convert-tritongpu-to-metal` 阶段失败：

```
RuntimeError: Metal backend: convert-tritongpu-to-metal failed
```

4 个 `ttg.convert_layout` 被 reject（`TritonGPUToMetal.cpp:8100`）：

| 行 | 值 | 根因 |
|---|---|---|
| 44 | `data_V` → dotB.opnd1 | dot B 的 operand cvt（`blocked→dot_op`），未被 matmul track 吸收 |
| 58 | `softmax_nom` → dotB.opnd0 | 同上；且 A 操作数是 **计算值** `exp(...)` |
| 62 | `accumulator` (fma 广播) | dotB 引入 `#blocked2`(spt=[2,2]) 与其它布局不匹配产生的 blocked↔blocked cvt |
| 65 | 最终 store | 同上 |

### TTGIR 结构（BLOCKSIZE_N=32, d_head=16, N-tile=32, num_warps=4）

```
scf.for %ci = 0 to N step 32
    iter_args(acc<32x16>, run_sum<32>, run_max<32>)   # 3 个 iter_arg，run_max 初值 -inf
  data_K, data_V = tt.load(..., mask, other=0.0)       # masked，带 other
  S = tt.dot(Q<32x16>, trans(K)<16x32>) * scale        # Dot A → <32x32,#blocked>
  S = select(mask, S, -inf)                            # 越界置 -inf
  m  = tt.reduce(max, axis=1)(S)                        # 行 max
  m' = maxnumf(m, run_max)                              # online 更新
  P  = exp(S - m'[:,None])                              # softmax 分子（计算值！）
  l  = tt.reduce(add, axis=1)(P)                        # 行 sum
  run_sum' = fma(run_sum, exp(run_max-m'), l)
  O  = tt.dot(P<32x32>, V<32x16>)                       # Dot B → <32x16,#blocked2>
  acc' = fma(acc, exp(run_max-m')[:,None], O)           # 累加 + rescale
  yield acc', run_sum', m'
acc = acc / run_sum[:,None]                             # epilogue
store(out, acc, mask)
```

**Dot A 的 operand cvt 已被 matmul track 吸收**（reject 不在第 46 行）；**Dot B 没有**，
因为它的 A 操作数 `P = exp(...)` 是计算值。

---

## 1. 根因分析（基于代码，两个调查 agent 已核实）

### 1a. matmul track 是一条“寄存器→内存”的封闭管线
- `tt.dot` 被降级成 `metal.simdgroup_*`，结果是 **`simdgroup_matrix` 寄存器值**，只能
  ① 存回 device memory（`SimdgroupStoreOp`），或 ② 直接喂给下一个
  `simdgroup_multiply_accumulate`。（`MetalOps.td:976/989`，`TritonGPUToMetal.cpp:7637/7649`）
- **没有任何 op 能把 `simdgroup_matrix` 逐元素消费**（无 exp/sub/max/reduce），也没有
  matrix→per-thread-tensor 的转换。
- `preprocessDotCvtChains.peel`（`:7706-7730`）只接受 operand cvt 的 source 是
  **load / trans(load) / scf.for**，**不接受计算 cone**。且两个操作数“一起 peel 或都不 peel”
  （`:7732-7734`）→ Dot B 因 `softmax_nom=exp(...)` 使 `newA` 为 null，整条 dot 被跳过，
  连本可单独处理的 `data_V` 的 cvt 也一并留下 → 撞 reject。
- 唯一的“链式 dot”路径 `tryFusedLoRAEpilogue`（`:7003`）要求第一个 dot 的输出
  **原封不动**做第二个 dot 的 A 操作数（`d.getA() != forOp.getResult(i)` → fail），
  **中间不能有任何计算** → 无法表达 softmax。
- ⇒ `dot → {reduce, exp, sub, div} → dot` 在当前 matmul track 里 **根本无法表示**。

### 1b. scalarizing 路径的现状（有几处此前的判断已过时，需修正）
- ✅ **多标量 iter_arg 已支持**：`scf.for` guard 只查 iter_arg 的**类型**不查个数
  （`:8430-8449`），emitter 的 `_scfForIterArgsMulti`（`MT:1138-1161`）能发射 N 个
  f32/i32 累加器并整体 yield 回写。**“只支持单标量 iter_arg” 的旧结论已失效。**
- ✅ **`-inf` / `NaN` 常量已能发射**：`emitFloatLiteral`（`MT:35-46`）渲染
  `-INFINITY`/`INFINITY`/`NAN`。`tl.full(-inf)` / `tl.where(...,-inf)` **不再是 blocker**。
  （reduce identity 里仍硬编 `-FLT_MAX`，注释“can't render -inf”已过时但数值无碍。）
- ✅ **computed-cone 的 rank-2 axis=1 reduce 已支持**（Wall 17 Inc1/2）：`evalRank2ConeAt`
  能走 load / f32 {add,sub,mul,div,max} / 一元 math / broadcast / expand_dims / where /
  cmp。
- ❌ **reduce cone 不能含 `tt.dot`**：`DotOp` 不在任何分支，落到 `:3371 return nullptr` /
  `:3501 return false`，报 `"rank-2 reduce: computed-cone src has an unsupported producer"`。
- ❌ **reduce cone 的叶子不能是 loop-carried / scf.if 结果**：BlockArgument 的
  `getDefiningOp()` 为 null → `:3218 return nullptr`（这正是 adder 卡在 `q0_rope←d←scf.if`
  的原因）。
- ❌ **2D tile 的 iter_arg 无法表示**：TypeConverter 把每个 tensor 映射成**单个标量**
  （`:87-89`，一个线程一个元素）。`tensor<32x16>`（512 元素）当不了单标量 iter_arg；
  `TileInfo`/`elemPerThread>1` 的 tile-loop 只包裹**直线 FuncOp body**，**不覆盖 iter_arg**。
- ❌ **Increment 2.5（loop-carried 标量的 threadgroup staging）**：progress.txt 仅有设计
  （`qbuf[M]` + barrier + 多趟填充），**无代码**。
- ❌ **Increment 3（链式 reduce：`max[:,None]`/`sum[:,None]` 回灌）**：**无代码**，每个
  `ReduceLowering` 只分配自己的 `rowBuf`，无跨 reduce 交接。
- ❌ **一个循环内 simdgroup dot 与 per-thread scalar reduce 互斥**：guard 不允许
  `simdgroup_matrix` 与 f32 iter_arg 混在同一个 loop。

### 1c. 其它潜在问题（被 reject 掩盖，修复时会浮现）
- `data_K`/`data_V` 是带 `other=0.0` 的 masked load 且喂给 dot；现在 `other-into-dot`
  guard（`:8007`）只因中间隔了 cvt 才没触发。真正修复必须正确处理 mask/other。
- 存在一个 **未被使用的 Stage-5 SDPA 库 op**（`Metal_...Attention`，`MetalOps.td:528-640`），
  接受 memref 操作数，但 **conversion 从不发射它**（是另一条手写 track）。

---

## 2. 架构决策（三选一，需确认）

flash-attention 的 `dot → softmax → dot` 结构，matmul track 和 scalarizing 路径**都无法
直接延伸**。三条路线：

### 选项 A — 扩展 simdgroup matmul track 做“真·flash attention”
需新增：`simdgroup_matrix ↔ threadgroup-tensor` 桥（把 S 存 threadgroup、按 per-thread
tensor 读回做 softmax、把 P 重新 stage 成 simdgroup 再做 dot B），且要掌握 Apple
simdgroup_matrix 的 lane 布局才能对行做 reduce。
- 👍 性能最好（保留硬件 matmul）。
- 👎 工作量最大、最脆；等于重造 NVIDIA MMA→ldmatrix 那套。**~4–6 周**。

### 选项 B — 全 scalarize（不用 simdgroup_matrix）+ 补齐通用原语
把整个 kernel 走 scalarizing TypeConverter：dot 变成 threadgroup-staged 的标量内积循环，
reduce 用现成 butterfly，循环携带 scalarized 状态。需要建：
(1) 2D-tile / register-array 的 iter_arg 表示；(2) 对 loop-carried / dot 结果做 in-loop
reduce；(3) Increment 2.5 staging；(4) Increment 3 链式 reduce。
- 👍 是**通用能力**，(2)(3)(4) 顺带解锁 adder_transformer；不脆（非形状匹配）。
- 👎 量大且 dot 标量化很慢；4 项原语都要从零建。**~4–6 周**。

### 选项 C —（推荐）专用 `tryFlashAttentionLoop` matcher + 手写 MSL FA 模板
仿照已落地的 `tryFusedLoRAEpilogue`：在 cvt 分类器**之前**的 matmul pre-pass 里，结构匹配
整个 online-softmax 循环，抽取 Q/K/V 指针、scale、mask、block sizes，**整体替换**为一段
自包含的 MSL flash-attention body（K/V tile 进 threadgroup，行内做 online softmax，
Dot 用 simdgroup 或小规模标量内积）。
- 👍 工作量有界且局部（一个 matcher + 一个 emit path + 模板）；与项目已验证的 fused-matcher
  模式一致；不必解决“2D iter_arg / loop-carried reduce”这些通用难题；能保留 simdgroup dot。
- 👎 形状脆弱（只匹配这一类 kernel）；kernel 稍变则需扩匹配。**~1.5–3 周**。

**★ 已选定：选项 A**（硬件级 simdgroup flash attention，性能优先）。选项 B/C 见 §5 备选。

### 选项 A 复核：可行性比初判乐观——桥的原语大多已存在
二次侦查（读 `MetalOps.td` + `ModuleTranslation.cpp`）发现，A 需要的
`simdgroup_matrix ↔ threadgroup` 桥**大部分已经实现**：
- ✅ **正确的 Apple simdgroup 模式已落地**：matmul emitter（`ModuleTranslation.cpp:469-810`）
  已经把 tile 经 threadgroup `_stage_shared` 暂存、再 `simdgroup_load` **从 threadgroup** 读、
  链式 `simdgroup_multiply_accumulate`。op 文档明确：直接对 `device T*` 做链式 sgmma 会在
  Apple GPU family 9 / Metal 17.5 上**丢输出列**，所以“threadgroup 暂存”既是必须、也已实现。
- ✅ `ThreadgroupAllocaOp` / `metal.barrier` / threadgroup 标量 load/store（`MetalOps.td:126/154/176/196`）
  齐备，可分配 `sbuf[M][N]` / `obuf[M][d]` 等 tile。
- ✅ `SimdgroupStoreOp`（`:989`）已能把 matrix 暂存到 per-threadgroup scratch（其“→scratch”半段
  正是我们要的 store-to-threadgroup）；`SimdgroupLoadOp`（`:965`）/ staged-load 可从 threadgroup 读。
- ✅ masked staged load（`SimdgroupLoadDeviceStagedMaskedOp`）处理 K/V 的
  `mask + other=0.0`；`-inf`、多标量 iter_arg 均已支持（见 §1b）。
- ⚠️ 存在 Stage-4 `SoftmaxOp`/`LogsumexpOp`（`:571/592`）与 Stage-5 SDPA op（`:614`），但
  **SDPA op 限定 `D∈{64,128}, N∈{1,8}`**，本 kernel 是 `d_head=16, N=64`，**不匹配**；且这些
  库 op **不由 conversion 发射**（另一条手写 track）。→ 不能直接复用，但 online-softmax 可借鉴
  其 MLX 数值写法。

**关键设计决定**：O 累加器与 online-softmax 状态（run_max/run_sum）与整个 softmax **常驻
threadgroup（标量域）**，`simdgroup_matrix` **只在两个 matmul 处瞬态使用**（dot 出结果即
`simdgroup_store` 回 threadgroup）。这样 per-row rescale O、行 max/sum 都在标量域完成，
**绕开 simdgroup_matrix 不透明 lane 布局做行 reduce 的难题**，同时两个 FLOP 大头仍走硬件。

---

## 3. 实施计划（选项 A；每阶段独立可测）

> ### ✅ Phase 0 已完成（2026-07-14）—— Option A 全线去风险成功
> 手写 simdgroup flash-attention 已通过验证，产出物：`metal-flash-attention-phase0-spike.py`（repo 根）。
> - **数值**：对拍 `torch.scaled_dot_product_attention`，max_abs_err ≈ **2–5e-7**（f32 舍入级）。
>   覆盖 clean(N=64)、masked-tail(N=48/96)、ragged(N=33)。
> - **✅ 证伪 family-9 丢列**：两个 dot 都从 **threadgroup** `simdgroup_load`（Q/Kᵀ/V/P 全部 tile 先入
>   threadgroup），链式 `simdgroup_multiply_accumulate` 结果正确。
> - **✅ 计算出的 P 走 threadgroup load 正确**：P=exp(shift) 由标量写入 `pbuf`，再 `simdgroup_load`
>   喂 Dot B，无精度问题。
> - **✅ 关键设计成立**：S/P/O + run_max/run_sum 常驻 threadgroup 标量域，simdgroup_matrix 仅瞬态；
>   per-row rescale、行 max/sum 全在标量域，绕开不透明 lane 布局。
> - **✅ num_warps=4（128 线程，real solve() 配置）通过**：FA body 用单 warp（`active = ltid.x<32`），
>   多余 warp idle 但仍执行所有 `threadgroup_barrier`（barrier 在 guard 外）。
> - **⚠️ 关键 emit 教训**：active-warp guard 必须用 **`thread_position_in_threadgroup`（局部）**，
>   **不是** `thread_position_in_grid`（全局）——否则第 2 个 query block 的 warp 全局 id.x∈[32,64) 被
>   误判 inactive，整块输出为 0（已在 spike 中复现并修正）。Phase 3 emit 必须照此。
> - **⚠️ 当前 spike 假设 `d_head==BD==16`**（tile 循环界硬编 4/4/2、4/2/4）。d_head≠16 / 非 8 整数倍 /
>   BLOCKSIZE_d>d_head（需 mask_d）是 Phase 4 泛化项。
> - MSL 约定（从验证过的 8×8 matmul 提取）：`device float* [[buffer(N)]]` / `device uint32_t*
>   [[buffer(k)]]` 读 `v[0]`；`ltid [[thread_position_in_threadgroup]]`、`tgid
>   [[threadgroup_position_in_grid]]`；`simdgroup_float8x8`、`simdgroup_load(m,&buf[off],stride)`、
>   `simdgroup_multiply_accumulate(acc,a,b,acc)`、`simdgroup_store`。dispatch：
>   `lib.k(*tensors, *int_scalars, threads=grid*tg, group_size=(num_warps*32,1,1))`。


设计要点（复述 §2）：**两个 matmul 走 simdgroup 硬件；S/P/O 及 online-softmax 状态全部常驻
threadgroup 标量域**。`simdgroup_matrix` 只是 dot 的瞬态输入/输出，dot 出结果立即
`simdgroup_store` 回 threadgroup。发射的 MSL body 由新 matcher 整体驱动。

### Phase 0 — 去风险 spike（2–3 天）
- **手写目标 MSL flash-attention**（MHA 确切形状 M=32, d_head=16, N=64, h=4），走 simdgroup
  路线：K/V staged→threadgroup、`simdgroup_load` from threadgroup、`simdgroup_multiply_accumulate`
  得 S→`simdgroup_store` 到 `sbuf`；标量做 online softmax(run_max/run_sum、rescale)得 `pbuf`；
  `simdgroup_load(pbuf)`×`simdgroup_load(V)`→O_tile→标量累加进 `obuf`；epilogue 除 run_sum。
- 用 `torch.mps.compile_shader` 跑，核对 vs `torch.scaled_dot_product_attention`。
- **验证三件事**：(a) 对**计算出来的 P** 做 `simdgroup_load(threadgroup)` 正确（op 文档称这是
  Apple 唯一可靠路径）；(b) online-softmax 数值；(c) 得到 matcher 要发射的确切 MSL。
- 产出：**emit 模板** + 确认硬件路径无 family-9 丢列问题。

> **架构精化（Phase 0 发现）**：无需建“可组合的 threadgroup 桥 op”。spike 证明
> `simdgroup_store`/`simdgroup_load` 直接吃 threadgroup 指针即可。整个 FA body 应作为**单个
> 高层 op `metal.flash_attention`** + 手写 emitter（照搬 Phase-0 已验证的 MSL 模板），与现有
> `SoftmaxOp`/`QmmOp`/Stage-5 `SDPAOp` 完全同构。Phase 1/2/3 据此重排如下。

> ### ✅ Phase 1 已完成（2026-07-14）—— op + emitter 落地并验证
> - **op**：`MetalOps.td` 新增 `metal.flash_attention`（operands: q/k/v/out/n/d_model/h 七个
>   `Metal_MemRefType`；attrs: bm/bn/bd）。
> - **emitter**：`ModuleTranslation.cpp` `translate(FlashAttentionOp)` 发射 Phase-0 模板（参数化
>   BM/BN/BD + buffer 名解析）；注册进两个 dispatch TypeSwitch；`translateKernel` 在有 FA op 时
>   补 `tgid` + `ltid [[thread_position_in_threadgroup]]` 到签名。
> - **构建**：`ninja` 通过（TableGen 自动生成 op 类）。
> - **验证**：① lit `test/Dialect/Metal/metal-translate/flash_attention_emit.mlir` 通过（emitter
>   输出结构对拍）；② 数值——把 emitter 的 MSL 经 `torch.mps.compile_shader` 跑，对拍 torch SDPA
>   **err≈2–5e-7**（N∈{64,48,96,33}，num_warps∈{1,4}）；③ 全 Metal lit 105 项无回归。
> - emitter 里的地址算：`buf[row*d_model + head*d_head + d]`（d_model 即行 stride），scale 运行时
>   `1/sqrt(d_model/h)`。scalar 参数走 `!metal.memref<? x ui32>`，读 `v[0]`。

### Phase 1 — 定义 `metal.flash_attention` op + emitter（4–6 天）
- 在 `MetalOps.td` 新增 `FlashAttentionOp`（**非 Pure**）：operands = Q/K/V/O memref + N/d_model/h
  （UI32）+ 属性 block sizes（BM/BN/BD）、grid(head) 信息；无 result（写 O memref）。
- 在 `ModuleTranslation.cpp` 写 `translate(FlashAttentionOp)`：直接发射 Phase-0 模板（含单-warp
  guard、barrier 在 guard 外、threadgroup `sbuf/pbuf/obuf/ktbuf/vbuf`、online softmax、两个
  simdgroup dot）。**局部线程索引用 `thread_position_in_threadgroup`**（Phase 0 教训）。
- **验收**：手工构造一个 `metal.flash_attention` 的 lit/IR，`ttgir_to_msl` 输出与 Phase-0 模板一致；
  pytest 直接 dispatch 对拍 torch SDPA（≈复用 spike）。

> ### ✅ Phase 2 + 3 已完成（2026-07-14）—— verbatim MHA 端到端 bit-exact 跑通
> - **matcher**：`TritonGPUToMetal.cpp` `tryFlashAttentionLoop` + `runFlashAttentionMatcher`
>   在 cvt 分类器**前**运行。匹配 3-iter_arg（1 个 rank-2 acc + 2 个 rank-1 状态）的 online-
>   softmax `scf.for`（2 dot + 2 reduce），按 “dotA.B 的 cone 含 tt.trans” 区分 Q@Kᵀ / P@V，
>   trace 到 Q/K/V load + 唯一 store + 唯一 divsi(=d_head)，抽 N/d_model/h + BM/BN/BD，建
>   `metal.flash_attention`，然后自底向上 DCE 掉 loop+epilogue+loads+offset。
> - **关键 bug 修复**：scalar 参数(N/d_model/h)转换后是 `get_element(buf[0])`，emitter 的
>   `bufName` 走 cast 会停在 get_element(非 buffer)→`_buffers[]` miss→operator[] 返回 0→全变
>   `v0`(数值错 1.3)。修复：`bufName` 跟随 `GetElementOp` 到其 memref buffer。指针参数无此问题。
> - **验证**：verbatim `hard-mult_head_attention.py` 编译+运行 **err≈3.5e-7**；6 种形状(N∈
>   {64,48,33,128,96,256}, d_head=16, masked-tail/ragged)全 PASS；lit
>   `convert-tritongpu-to-metal/flash_attention.mlir`(TTGIR→metal.flash_attention, CHECK-NOT
>   tt.dot/reduce/cvt/scf.for) PASS；pytest `test_metal_backend_flash_attention.py`(6 例) PASS；
>   **回归**：Metal lit 106、pytest 295 passed/1 xfailed，无回归(matcher guard 够窄不误触)。
> - **剩余 = Phase 4 泛化**：目前 envelope 硬约束 d_head==BD==16、BM≤32、BM/BN/BD 是 8 的倍数。
>   d_head≠16 / 非 8 倍数 / BLOCKSIZE_d>d_head(需 mask_d 列掩码) / BM>32 仍未支持(matcher 会
>   干净 fall through → 回到原 reject，不会误编译)。

### Phase 2 — FA matcher（3–5 天）
- 在 `runOnOperation` 的 matmul pre-pass（**cvt 分类器 `:8048` 之前**）新增
  `tryFlashAttentionLoop(forOp)`。以 dump 的真实 TTGIR 为锚点，结构匹配：
  - `scf.for` 带 3 iter_arg（2D acc + 两个 1D reduce 状态，run_max 初值 -inf）；
  - body 内 2 个 `tt.dot`，之间 `select(mask) → reduce(maxnumf,axis=1) → sub → exp →
    reduce(addf,axis=1)`；有 `fma`/`mul` by `exp(Δmax)` 的 rescale 与 3 元 yield；
  - 抽取：Q(hoisted load)、K/V(addptr+mask+other)、scale、两类 mask、block sizes、head grid。
- 命中即 **erase 整个 loop + epilogue**，替换为 Phase 1 的 `metal.flash_attention` op。
- **验收**：MHA 编译**越过** cvt 分类器（4 个 reject 消失）并跑通；非匹配形状**干净 fall through**。

### Phase 3 — 端到端接通（2–4 天）
- verbatim 跑 `leet-triton/hard-mult_head_attention.py`（N=64,d_model=64,h=4）编译+运行成功。
- 正确处理两类 mask（`offset_N<N`、`d<d_head`）、K/V masked load 的 `other=0.0`。
- epilogue：`obuf / run_sum`，masked store。
- **验收**：MHA 确切形状端到端跑通（数值 Phase 4 收）。

> ### ✅ Phase 4（列掩码泛化）已完成（2026-07-14）
> - **column masking**：`BLOCKSIZE_d = max(16, d_head)`，d_head<16 的头会把 tile pad 到 16 列。
>   emitter 现在按 `d < d_head`(运行时)掩码 Q/K/V load 与输出 store：pad 列载 0（Dot A 收缩、
>   Dot B 输出仍正确），且不写回。d_head==BD 时为无操作(原路径不变)。
> - **threadgroup 预算 guard**：matcher 加 32KiB 预算检查(`3·BM·BD+2·BD·BN+2·BM·BN+2·BM ≤
>   8192` floats)，BD=64(~48KiB)干净 fall through 到原 reject,不发超预算 kernel。
> - **验证**：d_head=8 现在 bit-exact(之前数值错 1.37)；d_head∈{8,16,32} pytest 全 PASS；
>   Metal lit 106、pytest 299,无回归。3 个 commit 已落 metal-develop。
> - **仍不支持**(kernel 自身或超预算,非 backend 缺陷)：d_head 使 `max(16,d_head)` 非 2 的幂
>   (如 24 → Triton 前端 `tl.arange` 报错)；d_head=64(超 threadgroup 预算)；BM>32。

### Phase 4 — 数值对拍 + 泛化（4–6 天）
- 对拍 `torch.scaled_dot_product_attention`（online vs 一次性 softmax 只差 f32 舍入，容差
  ~1e-3；专测 `-inf` 行、全 masked 行、`exp(run_max-m')` 边界）。
- 参数化：`BLOCKSIZE_N`、`BLOCKSIZE_d`、`N`、`d_model`、`h`；masked-tail（N 非 BN 整数倍、
  d_head 非 8 整数倍）；多头 grid `(cdiv(N,BN), h)`；多 warp（dot 用 `factorWarps`）。
- **注意 d_head=16 的 8×8 tile 划分**：Dot A 的 K 维=16=2×8，Dot B 的 N 维=16=2×8，都需 K/N
  方向多 tile；d_head 非 8 整数倍时走 masked staged load 的 col_extent。
- **验收**：多形状对拍通过。

### Phase 5 — 测试 + 加固（2–3 天）
- lit：`test/Dialect/Metal/convert-tritongpu-to-metal/flash_attention.mlir`（matcher 收敛
  FA loop；近似形状仍 reject）。
- pytest：`python/test/unit/test_metal_backend_flash_attention.py`（多形状 vs torch SDPA），
  命名 `test_<feature>_<condition>`。
- “像 FA 但不完全匹配” → 明确 diagnostic，不静默误降级。
- **验收**：全 Metal 套件不回归（当前 223 passed / lit 86）；新增用例通过。

**选项 A 合计：~3–5 周**（桥原语已大半存在，主要成本在 matcher 鲁棒性 + online-softmax +
d_head/tile 泛化）。

---

## 4. 备选（未选中，存档）
- **选项 C**：同样的 `tryFlashAttentionLoop` matcher，但 dot 用小规模标量内积而非 simdgroup。
  更简单(~1.5–3 周)，但放弃硬件 matmul 性能。—— A 与 C 共用 matcher 骨架(Phase 2)，只是 emit
  body 内部不同；**若 A 的 simdgroup 路径遇阻，可快速回退到 C 的标量 dot 先跑通**。
- **选项 B**：全 scalarize + 建通用原语（2D-tile iter_arg / Increment 2.5 staging / Increment 3
  链式 reduce / in-loop 标量 dot）。~4–6 周，通用且**顺带解锁 adder_transformer**，但 dot 慢。

---

## 5. 风险与开放问题（选项 A）
- **数值一致性**：online softmax（分块 rescale）与一次性 softmax 在 f32 下应等价；`-inf` 行、
  全 masked 行、`exp(run_max-m')` 边界专测。对拍用容差而非 bit-exact-to-online。
- **matcher 脆弱性**：Triton 前端 pass（canonicalize / layout 选择 / hoist / CSE）可能改变 IR
  形状。用 dump 真实 TTGIR 锚定；加“像 FA 但不匹配→明确报错”兜底。这是 A/C 共同的主风险。
- **Apple family-9 丢列陷阱**：链式 sgmma 必须 `simdgroup_load` from **threadgroup**（Phase 0
  优先证伪此风险）；对**计算出的 P** 做 threadgroup load 是新用法，需 spike 确认位精度。
- **`other`-into-dot / mask**：K/V 越界元素必须按 `other=0.0` 读，否则污染 S。
- **per-row rescale O**：靠“O 常驻 threadgroup、标量 rescale”规避 simdgroup_matrix 行操作——
  已在设计中排除，但要确认 obuf 的 threadgroup 容量（M×d f32）在预算内。
- **不解决 adder**：选项 A 与 adder_transformer 的 `q0_rope←scf.if` 阻塞正交；如需一并解，
  另起选项 B 的 Increment 2.5/3。

---

## 6. 下一步
1. 立即做 **Phase 0 spike**（手写 simdgroup MSL FA 对拍 torch SDPA）—— 证伪 family-9 丢列 +
   计算 P 的 threadgroup load 精度，产出 emit 模板。**这是 A 全线的地基。**
2. Phase 0 通过 → 进 Phase 1（桥 op）→ Phase 2（matcher）。
3. 若 Phase 0 暴露 simdgroup 路径硬伤 → 按 §4 回退选项 C（标量 dot），复用 Phase 2 matcher。
