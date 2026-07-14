# Metal 后端：adder_transformer 修复计划

**日期:** 2026-07-14 ｜ **分支:** metal-develop ｜ **状态:** ✅ **完成 —— adder verbatim 编译+运行+通过 smoke（bs=128 单程序 & bs=256 双程序）；leet-triton 全部通过**

> ## ✅ 完成小结（2026-07-14）
> `medium-adder_transformer_inference.py` 现在在 Metal 后端**编译 + 运行 + 通过自带 smoke test**。
> 三处改动（全部未提交，零回归：Metal pytest 299+2、lit 106/106）：
> 1. **Inc 2.5/3 —— staged-leaf reduce**（`TritonGPUToMetal.cpp`）：M≤tpb 时每线程 reduce 自己那行，
>    loop-carried / 链式 reduce 叶子经 `getRemappedValue` 就地解析(inline-fill),Inc 2.5 与 Inc 3 同一机制。
> 2. **emitter-CSE**（`ModuleTranslation.cpp`）：多用途白名单值物化成 `vN` 临时,adder MSL 37MB→10KB。
> 3. **多程序 global-row 修复**（`TritonGPUToMetal.cpp`）：reduce 的 device 行索引加 `tgid.x*tpb`
>    （原来用 program-local `r`,导致程序 1 读程序 0 的 KV 行 → 分数错乱溢出 → NaN；单程序无影响）。
> **"深体 NaN"的真相**：不是 RoPE/MLP/argmax,而是上面第 3 条——多程序 reduce 寻址漏了 per-program 偏移。
> 新增 durable pytest `test_metal_backend_adder_transformer.py`(bs∈{128,256})。

**（下方为原方案与调查过程,已全部落地。）**


目标：让 `leet-triton/medium-adder_transformer_inference.py`（自回归 adder-transformer 推理）
在 Metal 后端编译并运行，对拍参考。**这是 leet-triton 唯一剩余的 blocker**（MHA 已完成，其余
kernel 全过）。

---

## 0. 现状（实测）

编译在 line 101 `max_score = tl.max(score, axis=1)` 处失败：
```
error: failed to legalize operation 'tt.reduce' ... tensor<128x64xf32> -> tensor<128xf32>
RuntimeError: Metal backend: convert-tritongpu-to-metal failed
```
编译已经**走到** line 101，说明前面全部(embedding / RMSNorm / RoPE / 注意力投影 / KV 读写)
都已 lower。卡点是 softmax 的 reduce。

---

## 1. 全部的墙（静态分析 + fork 代码核实）

adder 的注意力是 **GEMV 式**（`score = q_rope[:,None]*k_seq + ...` 广播乘，**不是 `tl.dot`**），
所以 MHA 的 matcher 不适用。真正缺的能力：

- **Wall A（Increment 2.5）**：reduce cone 含 **loop-carried per-row 标量叶子**
  `q0_rope[:,None]` / `q1_rope[:,None]`。它们经 RoPE→RMSNorm 追到 `d = scf.if(pos<31){load}
  else{next_token}`，`next_token` 是外层 `for pos` 的 iter_arg（控制流/循环携带）。
  `evalRank1ValueAt` 在 BlockArgument(`:3007-3009`) / scf.if 结果(`:3163`) 处 `return nullptr`
  → reduce 匹配失败 → `failed to legalize`。
- **Wall B（Increment 3）**：**链式 reduce** `max_score → p=exp(score-max[:,None]) → sum_p →
  p/=sum_p[:,None] → attn0=sum(p·v0_seq)`。后面的 reduce cone 含前一个 `tt.reduce` 结果叶子
  （`max_score[:,None]`），同样在 `:3163` `return nullptr`。三个链式 reduce：max→sum_p→attn0。
- **已 OK**：标量 loop IV `pos`（经 `splat(pos)` → `:3203` 返回 `splat.getSrc()`，uniform 每线程）。
- **已支持（编译已到 :101）**：自回归 `scf.for` 带 i32-tensor iter_arg（`next_token[128]`）、
  `scf.if` yield tensor（`d`）、`arith.sitofp`、`tl.sigmoid`(→exp)、RMSNorm/RoPE(sin/cos)、
  KV cache 循环内 masked store 然后 masked load（每线程只读写自己 batch 行，程序序保证）。
- **⚠️ 未验证的尾巴**：`if pos>=30` 的 `tl.static_range(10)` argmax + `tl.store`（编译从没到过
  这里）。可能是**额外的墙**，需在 Inc 2.5/3 落地后 probe。

---

## 2. 设计

**关键发现（fork）**：rank-2 axis=1 reduce **已经是一个 staging 引擎**——
`ReduceLowering`(`:3560-3893`) 已有 `rowBuf = ThreadgroupAlloca<storeTy,M>`（hoist 到 tile loop
上方 `:3691`）、`localTid = tidGlobal - tgid*tpb`(`:3701`)、grid-stride per-row fill
(`:3752-3852`)、`BarrierOp`(`:3852`)、read-back `GetElement(rowBuf, outIdx)`(`:3855-3891`)。
**唯一缺口**是 cone evaluator 不能求值 loop-carried / reduce-result 叶子。

**统一原语 —— “staged rank-1 leaf”**：一个不能 re-emit 的 rank-1 per-row 叶子（loop-carried
标量 **或** 前一个 reduce 的结果），stage 进 threadgroup buffer，在正确点填充，reduce 从 buffer 读。
Inc 2.5 与 Inc 3 共用这一个机制，只是 buffer 来源不同。

> **为何不用“stage 整个 reduce 输入 tile”**（另一种思路）：那需要 reduce 与主 tile loop 跨 op
> 协调（在 tile loop 里捕获 score 每元素写 sbuf）。leaf-staging 让 **reduce 保持自包含**（在自己的
> fill loop 里 re-derive cone，只把无法 re-derive 的叶子换成 staged 读），改动更小、更契合现有
> scaffold。**采用 leaf-staging。**

### Increment 2.5 — stage loop-carried per-row 标量
1. 谓词 `isStagedRank1Leaf(v)` = 当前返回 nullptr 的 loop-carried / scf.if 情况。
2. 在 rowBuf 旁 alloc `qbuf = ThreadgroupAlloca<f32, M>`；fill：
   `for ri: r = localTid + ri*tpb; if (r<M) Store(getRemappedValue(v), qbuf, r);` + `Barrier`。
3. 给 `evalRank1ValueAt` / `evalRank2ConeAt` 加 `DenseMap<Value,Value> staged` 参数；命中 staged
   叶子返回 `GetElement(qbuf, idxVal)`。两个 `*ConeSupported` 谓词同步放行 staged 叶子。
4. 复用现有 rowBuf/fill/barrier/localTid scaffold；净新增 ≈ 谓词 + qbuf fill + map 穿线。

### Increment 3 — 链式 reduce 的 rowBuf 交接
同机制，但 staged buffer 来源是**前一个 reduce 的 rowBuf**。需要 `DenseMap<Operation*, Value>
reduceRowBuf`，在 `replaceOp`(`:3891`) **之前** populate——因为 replaceOp 会 erase `tt.reduce`，
把 `max_score` 的 use 改写成 `GetElement(rowBuf_max, 消费线程的行)`，而不是 sum_p fill loop 扫描
的任意 `r`。所以 reduce 必须在替换前被发现/关联（小 pre-pass tag，或一批一起 lower）。

---

## 3. 两个 ordering 风险（fork 标注 —— 任一都可能迫使 pre-pass 重构而非 in-pattern 编辑）

1. **getRemappedValue 对 deep-cone `q0_rope`**：现有用法(`:4043/:4298/:4414`)都是 operand-adjacent
   值（如 `splat.getSrc()`）。`q0_rope` 深埋在 cone 里、不是 reduce 的操作数；若它的 producers 在
   reduce pattern 跑时尚未被 driver 转换，`getRemappedValue` 返回 null。**未验证——最大未知。**
2. **reduce→rowBuf 关联跨 replaceOp**（Inc 3）：见上。需要在替换前建立映射。

---

## 4. 分阶段计划

> ### ⚠️ Phase 0 spike 结果（2026-07-14）—— 发现比预估更深的架构问题
> spike 复现了 blocker 并**暴露了一个 fork 估计里没抓到的架构不匹配**，把 Inc 2.5 从“加个 staged
> 叶子、复用 scaffold”升级成“重构 reduce 的 hoist/fill/read”。
> - ✅ **最小 repro 复现**：loop-carried per-row 标量进 rank-2 reduce cone → 同样的
>   `failed to legalize 'tt.reduce'`（`tensor<128x64>`，同 layout）。
> - ✅ **外层循环携带 per-row iter_arg 本身可用**：去掉 reduce、只留 `for` + per-row 逐元素更新，
>   编译+运行 bit-exact（该 kernel MSL 里 `acc` 是 1 elem/thread、无 tile loop）。
> - ⚠️ **架构不匹配（核心发现）**：reduce 把 rowBuf fill **hoist 到最外层 scf.for 之上**
>   （`findOutermostScfFor` + `:3694 setInsertionPoint(tileLoop)`），这只在 reduce **循环无关**
>   （device-rooted）时正确。adder 的 reduce cone **依赖 loop-carried `q0_rope`**，必须**每次
>   `for pos` 迭代重算**；hoist 一次 = 结果 stale。**这不是“stage 叶子”能解决的，需要把
>   hoist/fill/read 拆开、对 loop-dependent cone 走 inline（fill 进循环体、alloc 仍 hoist）。**
> - ⚠️ **多 elem/thread + tile loop**：在 reduce 的 2D layout 下，loop-carried 叶子是 **16
>   elem/thread**、且身处 tile loop 内 → `getRemappedValue` 的 scope/dominance 更复杂（叶子活在
>   tile loop + 外层循环内，而 fill 当前在 tile loop 之上）。
> - **修正评估**：Inc 2.5 = reduce hoisting 重构（条件 hoist：loop-dependent → 不 hoist fill）
>   + staged 叶子 + 多 elem/thread 处理。**> 原估 3–5 天**，是 fork 标注的“可能迫使 pre-pass
>   重构”真的发生了。adder 总体上修正为 **~2.5–4 周 + 未知尾巴**，不确定性偏高。
> - **spike 的价值**：以 1 天成本证明了 adder 是一次**真正的 reduce 基础设施重写**，不是增量补丁。

> ### ✅ Phase 0b — inline-fill 可行性已验证（2026-07-14）
> 在 reduce lowering 里加了一个一次性 probe（已 revert），在 reduce 位置（循环内）对每个
> `expand_dims` 叶子调用 `getRemappedValue`。**结论：inline-fill 可行。**
> - **repro**：loop-carried 叶子 `getRemappedValue = NON-NULL → f32`（干净的 per-thread 标量）。
> - **真 adder**（第一个 reduce）：5 个 expand_dims 叶子——`q0_rope`/`q1_rope`（`tensor<128xf32>`,
>   **`re-emittable=0`**，正是 blocker）两者 **`getRemappedValue = NON-NULL → f32`**；其余
>   (seq_idx `<64xi32>`、per-row i32/i1 mask) `re-emittable=1` 走原 re-emit 路径。
> - **意外收获**：现有 `rank1ConeSupported` 谓词**自动区分**哪些叶子要 stage（=0）哪些不用（=1）。
> - **风险下调**：fork 标注的最大未知（deep-cone getRemappedValue）**已证可行**。剩下是可控工程：
>   (a) alloc 仍 hoist；(b) loop-dependent cone 时 fill 走 inline（循环内）；(c) 把 `re-emittable=0`
>   的叶子经 getRemappedValue 写进 qbuf（每线程写自己拥有的行——16 elem/thread 的 fill 映射用
>   现有 MakeRange/tile 机制）；(d) Inc 3 链式（buffer 来源换成前一个 reduce 的 rowBuf）。

> ### 🔴 Phase 0c — 真正的核心要求浮出：**tile-loop 切分**（2026-07-14）
> 一开始动手就发现：完整 Inc 2.5 不是“给 reduce 加 staged 叶子”，而是**要把 FuncOp 的 tile loop
> 绕着 reduce 切开**——一个**全函数变换**，比 plan/fork 假设的深一层。
> - **根因**：adder 的 `max_score` 是**跨列 reduce**（要先知道一行的全部列才能出结果），却又在**同一个
>   tile loop 里被逐元素消费**（`p = exp(score - max_score[:,None])`）。现有 reduce 之所以能用
>   **hoist**（fill 提到 tile loop 之上、算一次），是因为 device cone **与 tile loop 独立**（从 device
>   重导）。而 staged 叶子 `q0_rope` 是**在 tile loop 内**逐元素算出来的：
>   - qbuf 必须在 reduce fill **之前**被**全部行**填满；
>   - 但 `max_score` 又必须在 tile loop **内部**可用（给 p）；
>   - ⇒ 只能把 tile loop **切成**：`[tile-loop-1: 算+stage q0_rope] → [barrier] → [reduce fill] →
>     [barrier] → [tile-loop-2: 消费 max_score]`。
> - **为何无法绕过**：qbuf fill 必须挂进**现有** tile loop（q0_rope 在那儿逐 iv 算出，`getRemappedValue`
>   只给当前 iv 的标量）；无法在独立 loop 里重导 q0_rope（loop-carried 不可 re-emit）。所以 fill 挂 tile
>   loop、reduce 在其后、结果又在 tile loop 内被消费 → tile loop 必须切分。
> - **repro vs adder**：repro 的 `m` 作为 rank-1 消费（`acc+m*0.01`），adder 的 `max_score` 被
>   `[:,None]` 重新展开成 rank-2（`p`）→ adder 从**第一个 reduce** 就需要切分。
> - **性质升级**：这是**核心 tile-loop lowering 的重构**（新的全函数 pass：定位 tile loop、在 reduce 处
>   切分、跨切分 stage 所有 live per-row 值、插 barrier），**不是**局部 reduce-pattern 编辑。风险/工作量
>   显著高于修正后的 2–4 周框架；且是高风险改动，不适合在会话里盲目一次性 hack 出来。
> - **建议**：这是一个**独立的多周编译器项目**的自然起点，宜专门、审慎地做——而不是在一次长会话尾部堆代码。
>   前面 4 层 de-risk（blocker→hoisting→getRemappedValue 可行→tile-loop 切分）已全部记录在案。

> ### ✅✅ Phase 0d — 正确设计已验证（Phase 0c 的“tile-loop 切分”判断是错的）
> 开始建原型后发现：**M ≤ tpb 时每个 fill 线程只 reduce 自己那一行（r == localTid）**，所以
> loop-carried per-row 叶子 = 该线程的 `getRemappedValue` 标量，**无需跨线程 staging、无需 tile-loop
> 切分**（Phase 0c 想复杂了）。而且 `score`/`p` 只被 reduce 消费（逐列 re-derive），**没有 [128,64] 的
> body tile loop**。这条路已用**可运行原型**验证。
>
> **原型实现（未提交，`TritonGPUToMetal.cpp`）**：
> - `g_stagedLeaves` map + `evalRank1ValueAt`/`rank1ConeSupported` 认它；`collectStagingLeaves` 收
>   `expand_dims` 里 `!rank1ConeSupported` 的 per-row 叶子；`M<=tpb` 时 `getRemappedValue` 建 staged
>   map、放行 gate；reduce fill **inline**（alloc 仍 hoist，fill 不 hoist）；`findFirstLoadInCone`
>   遇 staged 叶子当 opaque（关键修复：否则它钻进 q0_rope 的 cone 找 load、addptr 检查失败）。
> - **Inc 2.5 与 Inc 3 是同一个机制**：sum_p/attn0 的 cone 里 `max_score[:,None]`/`sum_p[:,None]` 也是
>   staged 叶子，`getRemappedValue`(reduce 结果) = `rowBuf[localTid]`（该线程自己的行），链式 reduce 自动
>   成立——不需要单独的 Inc 3 代码。
>
> **验证**：① repro（loop-carried 标量进 reduce cone）**bit-exact**（err 2.98e-8）；② 真 adder 的
> **3 个 reduce（max_score/sum_p/attn0）全部 legalize**（都 reached replaceOp）；③ **零回归**（Metal
> pytest 299 passed、reduce+FA 32 passed）。
>
> **🔴 最后一堵墙 = emitter 无 CSE（与 reduce 无关）**：adder 编译出的 **MSL 达 37 MB**（只有 3 个
> barrier，不是死锁），xcrun 编译卡死 → 运行超时。根因是 **MSL emitter 把表达式内联、无 CSE/临时变量**，
> adder 的深链（embedding→RMSNorm→RoPE→attention→MLP→static_range(10) argmax，且依赖 loop-carried
> `next_token` / `scf.if`）被指数级展开。试过把 staged 叶子物化进 qbuf（想减小体积），**没用**（体积不变，
> 且在 repro 上 segfault，已 revert）——证明 37MB 是**通用 emitter-CSE 问题**，不是 staged-leaf 内联。
>
> **结论**：Inc 2.5/3（reduce 侧）**已解决并验证**；adder 剩下的是一个**独立的 emitter-level CSE/临时变量
> 物化**工作（让深表达式发射一次、按名引用）。这是下一个 wall，量级另算。

> ### ✅ emitter-CSE 已实现 —— adder 现在**编译 + 运行端到端**（只剩输出 NaN）
> - **根因**：37MB MSL = **单条 36.6MB 行**,`e0=w0-w1·d·d` 内联 ~1900 次。MSL emitter 逐用途内联、
>   无 CSE,adder 深链每个值复用 2-3× → 2^depth 爆炸。
> - **实现**（`ModuleTranslation.cpp` `translate(Region)`,把现有 `_letBound`(tg_load)机制推广):
>   任何**多用途**(`!hasOneUse`)的**白名单纯值 op**(arith 加减乘除/cmp/select/sitofp、math
>   exp/sqrt/sin/cos/…、metal binary_exp/unary_exp/get_element)在其 IR 位置发射一次
>   `T v<N> = <expr>;`,后续用途渲染成 `v<N>`。白名单是关键——一开始用 `!isStatementPrintable`
>   兜底会在 two-load reduce 上 **crash**(translateValue 对某些 op 不能独立发射)。
> - **效果**:adder MSL **37MB → 10KB**(3600×),xcrun 秒编,不再卡死。
> - **回归**:全 Metal **pytest 299 passed 零回归**;reduce_rank2_computed(曾 crash)现通过;repro
>   仍 bit-exact。**8 个 lit 挂**——纯 CSE 文本 churn(多用途值现在是 `vN` 临时变量),数值不变,
>   CHECK 行机械更新即可。
> - **⚠️ 剩下:adder 输出 NaN**。零填 KV cache 时 attention 段算出有限值(0),说明 NaN 在 adder 的
>   **非-reduce 深体**(RoPE/MLP/argmax/autoregressive)——一条**从未跑过、现在首次被执行**的
>   emitter 路径里的潜在 bug,与我的 reduce/CSE 代码无关。是最后一个待查 bug(面大)。
>
> **总账**:Inc 2.5/3(reduce staging)+ emitter-CSE **都已解决并零数值回归**;adder 从"fails to
> legalize"推进到"**编译+运行端到端**"。剩:① 8 个 lit CHECK 机械更新;② adder 深体 NaN(独立 bug)。

### Phase 0 — de-risk spike（1–2 天）★ 先做
写一个**最小 kernel**：外层 `for` 循环携带一个 per-row i32 iter_arg（模拟 next_token），循环内
`x = carried[:,None] * device_load[128,64]`，`m = tl.max(x, axis=1)`，存 m。手动/试验性地实现
staged-leaf，验证：
- `getRemappedValue(carried_per_row)` 在 reduce lower 时**返回非 null**；
- staged qbuf fill + reduce 数值正确。

**这一步一票否决/确认 Inc 2.5 可行性。** 若 getRemappedValue 不行 → 换方案（用一个 pre-pass 先把
q0_rope materialize 成显式 per-thread 值，或退回 stage-whole-input）。

### Phase 1 — Increment 2.5（3–5 天）
落地 staged loop-carried 叶子，让 `max_score` reduce 合法化。
**验收**：adder 编译**越过 line 101**，停在下一墙（sum_p / 链式）。新增最小 lit + pytest。

### Phase 2 — Increment 3（4–7 天）
`reduceRowBuf` map + 链式交接（max→sum_p→attn0，三个 reduce）。
**验收**：adder 越过**整个 softmax + attn0**，停在（若有）尾巴墙。

### Phase 3 — 尾巴（未知，先 probe）
probe `if pos>=30` 的 static_range argmax + store 块；修任何新墙（预计 argmax 的 where/maximum
链已支持，store masked 已支持，但未证）。

### Phase 4 — 端到端 + 测试
verbatim `medium-adder_transformer_inference.py` 编译+运行，对拍其 `__main__` 参考；lit（TTGIR→
staged reduce）+ pytest；全 Metal 套件回归。

---

## 5. 工作量与风险总结
- Inc 2.5 ≈ **3–5 天**；Inc 3 ≈ **4–7 天**；共享一个 staged-leaf 机制。
- 尾巴（argmax 块）**未验证**，可能追加。
- 现实端到端：**~2 周 + 未知尾巴**。
- **主风险**：§3 的两个 ordering 问题，任一都可能把“in-pattern 小改”升级成“pre-pass 重构”。
- 与 MHA 的差别：MHA 是**有界**的 bespoke matcher；adder 是**核心 reduce 基础设施改造**，范围更大、
  尾巴不确定。

## 6. 建议
1. **先做 Phase 0 spike**（1–2 天）——直接打掉最大未知（getRemappedValue for deep-cone）。
2. spike 通过 → Inc 2.5 → Inc 3 → 尾巴 → 端到端。
3. spike 若暴露 getRemappedValue 不可行 → 暂停，重估方案（pre-pass materialize / stage-whole-input），
   再决定是否投入。
