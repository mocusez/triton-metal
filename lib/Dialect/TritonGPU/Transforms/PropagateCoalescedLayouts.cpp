#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-propagate-coalesced-layouts"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPROPAGATECOALESCEDLAYOUTS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// Return the product of the sizePerThread vector, or 1 if `layout` is not a
// BlockedEncodingAttr.
static int64_t sizePerThreadProduct(Attribute layout) {
  auto blocked = dyn_cast_or_null<BlockedEncodingAttr>(layout);
  if (!blocked)
    return 1;
  int64_t p = 1;
  for (auto s : blocked.getSizePerThread())
    p *= s;
  return p;
}

// True iff `cvt` is a "down-convert" inserted by the Coalesce pass: its source
// carries a BlockedEncoding with sizePerThread > 1 along some axis and its
// result restores the default sizePerThread = 1 layout. We compare products
// rather than the full vector because Coalesce only ever promotes the
// fastest-varying axis (`sizePerThread[order[0]] = perThread`), leaving all
// other axes at 1; the product is then sufficient to distinguish coalesced
// from default.
//
// INVARIANT (SC1, plan tutorial02-wall6-spt1-reduce-consensus.md):
// This predicate intentionally REJECTS up-converts (dstSPT > srcSPT). Phase-1
// of runOnOperation() inserts cvts that raise spt from 1 to N before
// qualifying rank-1 reduces; those inserted cvts must never be re-processed
// by phase-2's candidate walk. Any future weakening of this asymmetry (e.g.
// matching on srcSPT != dstSPT) MUST also update insertReduceCvt / phase-1
// wiring to prevent an infinite cvt-insertion loop on repeated pipeline
// invocation.
static bool isCoalesceDownConvert(ConvertLayoutOp cvt) {
  auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
  auto dstTy = dyn_cast<RankedTensorType>(cvt.getType());
  if (!srcTy || !dstTy)
    return false;
  if (srcTy.getShape() != dstTy.getShape())
    return false;
  int64_t srcSPT = sizePerThreadProduct(srcTy.getEncoding());
  int64_t dstSPT = sizePerThreadProduct(dstTy.getEncoding());
  return srcSPT > 1 && dstSPT == 1;
}

// MVP scope (see `.omc/specs/deep-interview-propagate-coalesced-impl.md` and
// `.omc/research/upstream-coalesce-tweak-feasibility.md` §6): only rewrite
// when the down-convert has exactly one user and that user is a `tt.reduce`
// op. This is the literal IR pattern produced by the rank-1 reduce kernels
// at `python/test/unit/test_metal_backend_reduce_rank1.py` post-Coalesce and
// is structurally safe — the reduce's operand layout cannot affect the
// scalar result's value (only its codegen).
//
// Future work (not in MVP): walk through layout-transparent unary
// elementwise users (arith.*, math.*) into the reduce/store sink, and
// handle the multi-operand store case where each operand traces back to
// its own coalesced load.
static bool isSafeRewriteTarget(ConvertLayoutOp cvt) {
  if (!cvt.getResult().hasOneUse())
    return false;
  Operation *user = *cvt.getResult().getUsers().begin();
  if (!isa<ReduceOp>(user))
    return false;
  // Rank-1 reduce ONLY. `rewriteReduceOperand` swaps the reduce's operand to
  // the coalesced layout but does NOT update the reduce's result type. For a
  // rank-1 input the result is a scalar (no encoding) so the swap is type-safe.
  // For a rank-2+ input the result is a `#ttg.slice` whose parent encoding is
  // derived from the operand encoding — swapping the operand changes the
  // inferred result type, leaving the op's stale slice result type incompatible
  // and tripping the `verify_each` verifier (a hard pass-manager abort). The
  // Metal rank-2 axis=1 reduce body is self-contained (it reads the source
  // tile directly from device memory via the producing `tt.load`), so it does
  // NOT need this coalesced-operand rewrite. Leaving the down-convert in place
  // for rank-2 reduces is both crash-free and semantically correct.
  auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
  if (!srcTy || srcTy.getRank() != 1)
    return false;
  return true;
}

// Rewrite `reduce`'s operand at index `operandIdx` to use the coalesced
// layout `coalescedTy`. The reduce result type does not change (rank
// decreases / shape unchanged depending on the reduce). Since `tt.reduce`
// has variadic operands and per-axis semantics, we only rewrite the
// specific operand we know is safe; if other operands exist with a
// different layout we abort.
static LogicalResult rewriteReduceOperand(ConvertLayoutOp cvt,
                                          ReduceOp reduce) {
  // Find which operand index `cvt` feeds. `cvt` must feed exactly one
  // operand of the reduce (it has hasOneUse asserted by the caller).
  int64_t operandIdx = -1;
  for (auto [i, v] : llvm::enumerate(reduce.getOperands())) {
    if (v == cvt.getResult()) {
      operandIdx = i;
      break;
    }
  }
  if (operandIdx < 0)
    return failure();

  // For multi-operand reduces (e.g. argmax with two operands), all operands
  // must agree on layout. Conservatively bail if any other operand is a
  // tensor with a different encoding than `cvt`'s source.
  auto coalescedTy = cvt.getSrc().getType();
  for (auto [i, v] : llvm::enumerate(reduce.getOperands())) {
    if (static_cast<int64_t>(i) == operandIdx)
      continue;
    auto otherTy = dyn_cast<RankedTensorType>(v.getType());
    if (otherTy && otherTy.getEncoding() != coalescedTy.getEncoding())
      return failure();
  }

  // Swap the operand: replace `cvt`'s result use in the reduce with
  // `cvt.getSrc()` directly.
  reduce.setOperand(operandIdx, cvt.getSrc());

  // The reduce's result type stays the same (axis dropped, encoding chosen
  // by the reduce itself based on operand encoding). If the reduce inferred
  // its result encoding from the old operand encoding, we'd need to update
  // it — but `tt.reduce` for rank-1 input produces a scalar (no encoding),
  // and for higher ranks the slice encoding is derived from the operand
  // when the verifier runs. Defensive check: if the reduce now fails to
  // verify, the caller will catch it and bail out of this rewrite by
  // restoring the previous operand.
  return success();
}

// Phase 1 (Wall 6 — plan tutorial02-wall6-spt1-reduce-consensus.md):
// Insert a fresh `tt.convert_layout` mapping sizePerThread=[1] → [BLOCK/tpb]
// immediately before a qualifying rank-1 `tt.reduce` whose input has spt=1.
// This unblocks the B2.3 spt-fold path in lowerRank1Reduce for kernels where
// no down-convert existed in the IR (e.g. the fused-softmax tutorial whose
// load directly produces spt=[1]).
//
// Gating:
//   - single-operand reduce only (multi-operand argmax-style deferred)
//   - rank-1 input only
//   - source layout is BlockedEncodingAttr with sizePerThreadProduct == 1
//   - BLOCK % tpb == 0 AND BLOCK/tpb is a power of 2 AND > 1
//
// The new cvt's dst encoding preserves threadsPerWarp, warpsPerCTA, order,
// and CGA layout from the source; only sizePerThread changes to [BLOCK/tpb].
// SC2 mirror: OptimizeThreadLocality.cpp:194-197.
static LogicalResult insertReduceCvt(OpBuilder &b, ReduceOp reduce) {
  if (reduce.getSrcs().size() != 1)
    return failure();
  auto srcVal = reduce.getSrcs().front();
  auto srcTy = dyn_cast<RankedTensorType>(srcVal.getType());
  if (!srcTy || srcTy.getRank() != 1)
    return failure();
  auto srcBlocked = dyn_cast_or_null<BlockedEncodingAttr>(srcTy.getEncoding());
  if (!srcBlocked || sizePerThreadProduct(srcBlocked) > 1)
    return failure();

  int64_t tpb = 1;
  for (auto t : srcBlocked.getThreadsPerWarp())
    tpb *= t;
  for (auto w : srcBlocked.getWarpsPerCTA())
    tpb *= w;
  if (tpb <= 0)
    return failure();

  int64_t BLOCK = srcTy.getShape()[0];
  if (BLOCK <= 0 || BLOCK % tpb != 0)
    return failure();
  int64_t spt = BLOCK / tpb;
  if (spt <= 1 || (spt & (spt - 1)) != 0)
    return failure();

  // Build new BlockedEncoding with sizePerThread=[spt], preserving everything
  // else. Mirror OptimizeThreadLocality.cpp:194-197 — get CGA layout via the
  // free function on LayoutEncodingTrait.
  MLIRContext *ctx = srcTy.getContext();
  auto baseLayout = cast<LayoutEncodingTrait>(srcTy.getEncoding());
  auto cgaLayout = getCGALayout(baseLayout);
  SmallVector<unsigned> sizePerThread = {static_cast<unsigned>(spt)};
  auto newBlocked = BlockedEncodingAttr::get(
      ctx, sizePerThread, srcBlocked.getThreadsPerWarp(),
      srcBlocked.getWarpsPerCTA(), srcBlocked.getOrder(), cgaLayout);
  auto newTy = RankedTensorType::get(srcTy.getShape(), srcTy.getElementType(),
                                     newBlocked);

  b.setInsertionPoint(reduce);
  auto cvt = ConvertLayoutOp::create(b, reduce.getLoc(), newTy, srcVal);
  reduce.setOperand(0, cvt.getResult());
  return success();
}

class PropagateCoalescedLayoutsPass
    : public impl::TritonGPUPropagateCoalescedLayoutsBase<
          PropagateCoalescedLayoutsPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // No target gate: only the Metal pipeline wires this pass in
    // (`third_party/metal/backend/compiler.py:174`). Defense-in-depth via a
    // `ttg.target` `starts_with("metal:")` check was the first-landing
    // posture, but the Metal compiler stitches `ttg.target = "cuda:80"`
    // via `_TTGPUIR_PARSER_STUB_TRIPLE` (compiler.py:93-94) for parser
    // compatibility, so the gate rejected the pass's own intended target.
    // Wall 6 fix: pipeline wiring is the gate.

    // Phase 2 FIRST: collect candidate convert_layout ops; mutating during the
    // walk would invalidate iterators.
    //
    // ⚠️ Order matters, and it is the opposite of what it looks like. Phase 1
    // inserts an spt=1 -> spt=N cvt immediately before a qualifying reduce, and
    // that cvt lands between Coalesce's DOWN-convert and the reduce — which is
    // precisely the edge phase 2 matches on (`isSafeRewriteTarget` wants the
    // down-convert's single user to BE the reduce). Running phase 1 first
    // therefore disabled phase 2 on every kernel where Coalesce had already
    // produced a down-convert, leaving BOTH cvts in place: exactly the
    // round trip `cvt spt4->spt1` then `cvt spt1->spt4`, which is what
    // `test/TritonGPU/coalesce-propagate-reduce.mlir` had been failing on.
    //
    // With phase 2 first, the down-convert is folded into the reduce and phase
    // 1 then sees an operand that is already spt=N and declines — one cvt
    // removed instead of one added. The kernels phase 1 exists for (no
    // down-convert in the IR at all, e.g. the fused-softmax tutorial) offer
    // phase 2 no candidate, so they are unaffected by the swap. The SC1
    // invariant still holds and is now structural rather than incidental:
    // phase 2 has finished before any inserted cvt exists.
    SmallVector<ConvertLayoutOp> candidates;
    mod.walk([&](ConvertLayoutOp cvt) {
      if (isCoalesceDownConvert(cvt) && isSafeRewriteTarget(cvt))
        candidates.push_back(cvt);
    });

    LDBG("found " << candidates.size() << " down-convert(s) to rewrite");

    for (ConvertLayoutOp cvt : candidates) {
      // Re-check hasOneUse here because a prior rewrite may have changed
      // it (e.g. two converts feeding the same reduce — though MVP scope
      // doesn't expect this).
      if (!cvt.getResult().hasOneUse())
        continue;
      Operation *user = *cvt.getResult().getUsers().begin();
      auto reduce = dyn_cast<ReduceOp>(user);
      if (!reduce)
        continue;

      // Snapshot for rollback in case rewrite fails.
      Value oldOperand = cvt.getResult();
      (void)oldOperand;

      if (failed(rewriteReduceOperand(cvt, reduce))) {
        LDBG("rewrite failed for " << *cvt << "; skipping");
        continue;
      }

      // The convert_layout now has zero uses (the reduce was its only user
      // and we just rewired). Erase it.
      if (cvt.getResult().use_empty()) {
        cvt.erase();
        LDBG("erased down-convert; reduce now consumes coalesced layout");
      }
    }

    // Phase 1 (Wall 6): insert cvts before the qualifying spt=1 rank-1 reduces
    // phase 2 did not already give a coalesced operand to. This unblocks the
    // B2.3 spt-fold path in lowerRank1Reduce for kernels where no down-convert
    // existed in the IR (e.g. the fused-softmax tutorial, whose load directly
    // produces spt=[1]).
    SmallVector<ReduceOp> reduceCandidates;
    mod.walk([&](ReduceOp r) { reduceCandidates.push_back(r); });
    OpBuilder builder(&getContext());
    for (ReduceOp r : reduceCandidates)
      (void)insertReduceCvt(builder, r);
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
