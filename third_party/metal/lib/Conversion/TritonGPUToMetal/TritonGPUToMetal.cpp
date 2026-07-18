//===--- TritonGPUToMetal.cpp - TritonGPU → Metal conversion pass ---------===//
//
// AC3 half-slice (finish) per
// `.omc/specs/deep-interview-ac3-half-slice-finish.md`. Lowers unmasked
// vector_add-shaped TTGIR into the metal dialect under the
// 1-element-per-thread assumption. Multi-element-per-thread layouts are
// rejected by the pre-pass layout guard.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonGPUToMetal/Passes.h"

#include "Dialect/Metal/IR/MetalDialect.h"
#include "Dialect/Metal/IR/MetalOps.h"
#include "Dialect/Metal/IR/MetalTypes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace triton {
namespace metal {

#define GEN_PASS_DEF_CONVERTTRITONGPUTOMETAL
#include "Conversion/TritonGPUToMetal/Passes.h.inc"

namespace {

// L2b: project a Triton element type to the type used for Metal storage
// (memref element + threadgroup scratch). `Metal_Type` (MetalOps.td:17) admits
// signless I1 and I8 but NOT signless I16/I32/I64, so `metal.get_element` /
// `metal.store` / `metal.tg_{load,store}_indexed` — all of which require the
// value type to equal the memref element type AND be in `Metal_Type` — cannot
// operate on a signless-i32 buffer directly. We therefore route i32 storage
// through ui32 (bit-preserving; downstream signless arith bridges back with
// `builtin.unrealized_conversion_cast`). This mirrors the scalar-wrap path's
// `wrapperElementType` (:303) and `ReduceLowering`'s ui32 staging.
//
// Scope L2b: i32 ONLY. i8 stays signless (it is already in `Metal_Type`);
// i16/i64 are out of scope per the spec's non-goals.
static mlir::Type metalStorageElementType(mlir::Type t) {
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(t))
    if (intTy.isSignless() && intTy.getWidth() == 32)
      return mlir::IntegerType::get(t.getContext(), 32,
                                    mlir::IntegerType::Unsigned);
  return t;
}

// Inverse of `metalStorageElementType`: bit-preserving cast of a value carrying
// a SIGNED/UNSIGNED integer type back to the signless integer of the same width.
//
// Device buffers are typed with the ui32 storage element type (see
// `wrapperElementType`), so a `metal.get_element` read inside a reduce/scan cone
// yields a ui32 value. Every `arith.*` integer op — addi, cmpi, select, andi —
// requires SIGNLESS operands ("operand #0 must be signless-non-zero-bitwidth-
// integer-like"), so rebuilding an arith node directly on a cone leaf produced a
// module that failed its own verifier. That is why `tl.sum(tl.where(v > 0, 1, 0))`
// over an i32 buffer used to abort the pass while the leaf-only `tl.sum(v)` was
// fine: only the former rebuilds arith ops on top of the leaf.
//
// The emitter forwards `unrealized_conversion_cast` as a no-op (MSL treats int
// and uint interchangeably in expression context), so this is free at runtime
// and purely a type-system reconciliation.
static mlir::Value toSignlessInt(mlir::Value value, mlir::OpBuilder &rewriter,
                                 mlir::Location loc) {
  auto intTy = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  if (!intTy || intTy.isSignless())
    return value;
  auto signless = mlir::IntegerType::get(value.getContext(), intTy.getWidth());
  return mlir::UnrealizedConversionCastOp::create(
             rewriter, loc, mlir::TypeRange{signless}, mlir::ValueRange{value})
      .getResult(0);
}

// L2b: bit-preserving cast of `value` to the storage element type of `memref`
// (a MetalMemRefType) when they differ, via `builtin.unrealized_conversion_cast`.
// Lets a signless-i32 arith value feed a ui32-typed `metal.store` / scratch op.
static mlir::Value
castToMemrefStorage(mlir::Value value, mlir::Value memref,
                    mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc) {
  auto storageTy = mlir::cast<MetalMemRefType>(memref.getType()).getType();
  if (value.getType() == storageTy)
    return value;
  return mlir::UnrealizedConversionCastOp::create(
             rewriter, loc, mlir::TypeRange{storageTy}, mlir::ValueRange{value})
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// TypeConverter
//===----------------------------------------------------------------------===//

class TritonGPUToMetalTypeConverter : public mlir::TypeConverter {
public:
  TritonGPUToMetalTypeConverter(mlir::MLIRContext *ctx) {
    addConversion([](mlir::Type t) -> std::optional<mlir::Type> { return t; });

    addConversion([](mlir::RankedTensorType t) -> std::optional<mlir::Type> {
      return t.getElementType();
    });

    addConversion([ctx](mlir::triton::PointerType ptr) -> std::optional<mlir::Type> {
      // L2b: route signless-i32 pointees through ui32 storage so the memref
      // element type satisfies `Metal_Type` (see metalStorageElementType).
      return MetalMemRefType::get(
          ctx, metalStorageElementType(ptr.getPointeeType()), 0);
    });

    // Source materializer: when the framework needs a value of the
    // un-converted (original) type but only the converted value is
    // available, emit an unrealized_conversion_cast that the
    // applyFullConversion pass folds away once all uses convert.
    addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1)
            return {};
          return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
              .getResult(0);
        });

    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          if (inputs.size() != 1)
            return {};
          return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
              .getResult(0);
        });
  }
};

//===----------------------------------------------------------------------===//
// Tile-loop helpers (BLOCK_SIZE > threads_per_block).
//
// When the kernel processes more tensor elements than the launch has
// threads (e.g. BLOCK_SIZE=1024 on num_warps*warp_size=128 threads, so
// elem_per_thread=8), the conversion wraps the per-thread body in an
// outer `scf.for(0, E, 1)` and computes a per-iteration index. Both
// canonical Triton blocked layouts are supported:
//   * contiguous (sizePerThread[0] > 1): idx = tid * E + iv
//   * strided   (sizePerThread[0] == 1, layout repeats E times): idx = tid + iv * T
// See `.omc/specs/deep-interview-metal-block-size-loop.md`.
//===----------------------------------------------------------------------===//

struct TileInfo {
  int64_t elemPerThread;
  int64_t threadsPerBlock;
  bool contiguous; // true if sizePerThread[0] > 1
  // 2D-aware additions (`.omc/specs/deep-interview-metal-2d-maskedaccess-emitperiterindex.md`):
  int64_t rank;
  llvm::SmallVector<int64_t, 2> shape; // logical tile dim sizes
};

// Per-op callers (LoadOp / StoreOp lowerings at :2036, :2394, :2494, :2571) pass
// tensor<...x!tt.ptr<...>> and need a valid TileInfo. The kernel-level walker
// findTileInfo (:155) and the duplicate "largest blocked tensor" walker inside
// the rank-2 pattern dispatcher (:507) each do their own ptr-element skip on
// the bestSize tiebreaker (see AC1-bis insertions). Dead pointer-arithmetic
// chains from multi-tile matmul rewrites are removed by the fixed-point DCE
// pass ("AC4 v6: dead-code eliminate" block at ~:4404) before any caller of
// findTileInfo (:278 in FuncOpLowering, :2755 in preprocessMaskedStoreSentinels)
// runs.
static std::optional<TileInfo> tileFromTensor(mlir::Type t) {
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t);
  if (!rt)
    return std::nullopt;
  auto blocked = mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
      rt.getEncoding());
  if (!blocked)
    return std::nullopt;
  if (rt.getRank() < 1)
    return std::nullopt;
  // Rank-generic: take products across all axes. For 1D this collapses
  // to the original single-axis math. For 2D/3D this gives the flat
  // elem_per_thread / threadsPerBlock counts. See
  // `.omc/specs/deep-interview-metal-2d-layouts-foundation.md`.
  auto sizePerThread = blocked.getSizePerThread();
  auto threadsPerWarp = blocked.getThreadsPerWarp();
  auto warpsPerCTA = blocked.getWarpsPerCTA();
  int64_t tpb = 1;
  for (auto t : threadsPerWarp) tpb *= t;
  for (auto w : warpsPerCTA) tpb *= w;
  int64_t total = 1;
  for (auto s : rt.getShape()) total *= s;
  if (tpb == 0)
    return std::nullopt;
  int64_t E = total / tpb;
  // Contiguous if ANY axis has sizePerThread > 1 (the per-thread tile
  // is multi-element along at least one dim).
  bool contiguous = llvm::any_of(sizePerThread,
                                  [](auto s) { return s > 1; });
  TileInfo info{E, tpb, contiguous, rt.getRank(), {}};
  for (auto s : rt.getShape()) info.shape.push_back(s);
  return info;
}

// Same, but also accepts a rank-(R-1) `#ttg.slice<{dim, parent = #blocked}>`.
//
// A 1D vector that gets `tt.expand_dims`'d / `tt.broadcast`'d into a 2D tile
// (`skip[:, None] * A`) carries a slice encoding, not a blocked one, so plain
// `tileFromTensor` rejects it and the load fails to legalize. Relaxing
// `tileFromTensor` itself is NOT an option: its nullopt on slice encodings is
// load-bearing at the other call sites — `findTileInfo` (:243) would let the
// parent's M*N size win the tile-loop sizing tiebreaker, and `ReduceLowering`'s
// outTile walk (:4515) steps OVER the slice-encoded reduce result precisely
// because of it, landing on the rank-1 #blocked store value; make slice resolve
// there and the reduce reads rowBuf[M] with an index in [0, M*N).
//
// Downstream only `rank` is actually consumed (`emitLoadStoreIndex` branches on
// it, then passes the scalarized tt.addptr offset straight through — and
// `MakeRangeLowering` already emits per-thread-correct values for slice
// encodings via the parent's order-driven div/rem). So `rank`/`shape` stay the
// op's own rank-1 view, and only the thread geometry comes from the parent.
//
// Read the geometry off `slice.getParent()`, never off the slice: a
// SliceEncodingAttr's own getThreadsPerWarp/getWarpsPerCTA return the
// dim-REMOVED vectors, so their product is parent_tpb / threadsPerWarp[dim] —
// not the thread count the launch actually has.
//
// LOADS ONLY. Under a slice encoding the value is REPLICATED across the sliced
// axis (N-to-1 for dim=1), so every replica computes the same address and reads
// the same value — redundant but correct. A slice-encoded STORE would have a
// whole replica set writing one address from values that need not agree, so
// those keep failing to legalize rather than silently racing.
static std::optional<TileInfo> tileFromLoadPtrTensor(mlir::Type t) {
  if (auto info = tileFromTensor(t))
    return info;
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t);
  if (!rt || rt.getRank() < 1)
    return std::nullopt;
  auto slice = mlir::dyn_cast_or_null<mlir::triton::gpu::SliceEncodingAttr>(
      rt.getEncoding());
  if (!slice)
    return std::nullopt;
  auto parent = mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
      slice.getParent());
  if (!parent)
    return std::nullopt;
  int64_t tpb = 1;
  for (auto t : parent.getThreadsPerWarp()) tpb *= t;
  for (auto w : parent.getWarpsPerCTA()) tpb *= w;
  if (tpb == 0)
    return std::nullopt;
  int64_t total = 1;
  for (auto s : rt.getShape()) total *= s;
  bool contiguous =
      llvm::any_of(parent.getSizePerThread(), [](auto s) { return s > 1; });
  TileInfo info{total / tpb, tpb, contiguous, rt.getRank(), {}};
  for (auto s : rt.getShape()) info.shape.push_back(s);
  return info;
}

// True iff `v` has a forward use-chain that reaches a `tt.store` WITHOUT first
// being consumed by a `tt.reduce`. The reduce is treated as a terminal sink:
// its result value is a NEW (post-reduce) value and is not traversed.
//
// Used by `findTileInfo` to size the tile loop from the LIVE post-reduce
// tensors only. The Metal rank-1/rank-2 reduce bodies are self-contained —
// they read their source tile straight from device memory via the producing
// `tt.load` — so the pre-reduce input tensors are dead after lowering and must
// NOT drive the tile-loop trip count. If they did, the loop would iterate over
// the (larger) pre-reduce element count while the reduce-output store still
// indexes with its own (smaller) E_out, writing out of bounds.
static bool reachesStoreNotThroughReduce(mlir::Value v) {
  llvm::SmallVector<mlir::Operation *, 8> wl;
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  for (auto *u : v.getUsers())
    wl.push_back(u);
  while (!wl.empty()) {
    auto *u = wl.pop_back_val();
    if (!seen.insert(u).second)
      continue;
    if (mlir::isa<mlir::triton::StoreOp>(u))
      return true;
    if (mlir::isa<mlir::triton::ReduceOp>(u))
      continue; // sink: do not traverse the reduce result
    for (auto res : u->getResults())
      for (auto *uu : res.getUsers())
        wl.push_back(uu);
  }
  return false;
}

// Walk the original tt.func body for the canonical ttg.blocked tensor and
// derive the tile info from its layout. Among 2D blocked tensors, prefer
// the one with the LARGEST element count — this skips the intermediate
// 16x1 / 1x16 expand_dims tensors and returns the full (BLOCK_M, BLOCK_N)
// tile. For 1D kernels the first 1D blocked tensor wins (preserves prior
// behavior). Returns nullopt if no blocked tensor is found.
//
// Reduce-aware sizing: when the function contains a `tt.reduce`, only tensors
// that reach a `tt.store` without passing through the reduce are eligible to
// size the loop (see `reachesStoreNotThroughReduce`). This makes the loop
// OUTPUT-driven for reduce kernels (E_out) instead of input-driven, which the
// self-contained rank-1/rank-2 reduce bodies require. For functions with no
// reduce the eligibility filter is skipped entirely, preserving the prior
// largest-tensor behavior byte-for-byte.
static std::optional<TileInfo>
findTileInfo(mlir::triton::FuncOp funcOp) {
  bool hasReduce = false;
  funcOp.walk([&](mlir::triton::ReduceOp) { hasReduce = true; });

  std::optional<TileInfo> result;
  int64_t bestSize = 0;
  funcOp.walk([&](mlir::Operation *op) {
    for (auto v : op->getResults()) {
      auto info = tileFromTensor(v.getType());
      if (!info) continue;
      // AC4-v6 intent: don't let stale tensor<...x!tt.ptr<...>> win the
      // bestSize tiebreaker. Per-op callers of tileFromTensor need ptr-element
      // acceptance; this defense lives at the heuristic site, independent of
      // pass order (see DCE block at ~:4404).
      if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
          rt && mlir::isa<mlir::triton::PointerType>(rt.getElementType()))
        continue;
      // Reduce-aware: skip tensors consumed solely by a self-contained reduce
      // (they would over-size the loop and OOB the reduce-output store).
      if (hasReduce && !reachesStoreNotThroughReduce(v))
        continue;
      int64_t sz = 1;
      for (auto s : info->shape) sz *= s;
      if (sz > bestSize) {
        bestSize = sz;
        result = info;
      }
    }
  });
  return result;
}

// Emit a per-iteration ui32 index for a load/store inside the tile loop.
// When the surrounding tile loop is absent (E == 1), falls back to the
// existing `metal.thread_id "x"` direct-use semantics.
// Wall 13 fix (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC8):
// when the kernel contains user-level scf.for loops (e.g. tl.range for
// per-row scans like softmax), `op->getParentOfType<scf::ForOp>()` returns
// the IMMEDIATE parent — the user loop, not the FuncOpLowering-inserted
// tile loop. emitPerIterIndex needs the TILE loop iv. The tile loop is
// always the OUTERMOST scf.for. This helper walks the parent chain and
// returns it.
static mlir::scf::ForOp findOutermostScfFor(mlir::Operation *op) {
  mlir::scf::ForOp outermost;
  mlir::Operation *p = op->getParentOp();
  while (p) {
    if (auto f = mlir::dyn_cast<mlir::scf::ForOp>(p))
      outermost = f;
    p = p->getParentOp();
  }
  return outermost;
}

// localTid = thread_id.x - threadgroup_id.x * tpb. Multi-program safe: each
// threadgroup must address its own slice by a LOCAL offset (Wall 13).
static mlir::Value emitLocalTid(mlir::OpBuilder &rewriter, mlir::Location loc,
                                int64_t tpb) {
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
  auto i32 = rewriter.getI32Type();
  mlir::Value tid =
      ThreadIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"))
          .getResult();
  mlir::Value tidI32 = mlir::UnrealizedConversionCastOp::create(
                           rewriter, loc, mlir::TypeRange{i32},
                           mlir::ValueRange{tid})
                           .getResult(0);
  mlir::Value tg =
      ThreadgroupIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"))
          .getResult();
  mlir::Value tgI32 = mlir::UnrealizedConversionCastOp::create(
                          rewriter, loc, mlir::TypeRange{i32},
                          mlir::ValueRange{tg})
                          .getResult(0);
  auto cTpb = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
  auto tgOff =
      mlir::arith::MulIOp::create(rewriter, loc, tgI32, cTpb.getResult());
  return mlir::arith::SubIOp::create(rewriter, loc, tidI32, tgOff.getResult())
      .getResult();
}

static mlir::Value emitLocalTidUI32(mlir::OpBuilder &rewriter,
                                    mlir::Location loc, int64_t tpb) {
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
  return mlir::UnrealizedConversionCastOp::create(
             rewriter, loc, mlir::TypeRange{ui32},
             mlir::ValueRange{emitLocalTid(rewriter, loc, tpb)})
      .getResult(0);
}

static mlir::Value emitPerIterIndex(const TileInfo &tile,
                                    mlir::scf::ForOp parentFor,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Location loc) {
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
  auto i32 = rewriter.getI32Type();
  auto tid =
      ThreadIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"));
  if (tile.elemPerThread <= 1 || !parentFor) {
    // Degenerate (E == 1): the existing path already uses `tid` directly
    // as the ui32 index. Caller can use this directly.
    return tid.getResult();
  }
  // Wall 13 fix: use LOCAL thread id (= global_tid - tgid*tpb). For
  // multi-program launches each threadgroup must address its row's columns
  // by local thread offset; using global_tid shifts every threadgroup's
  // window by tgid*tpb and skips columns [0, tgid*tpb) of the row.
  auto tidGlobalI32 =
      mlir::UnrealizedConversionCastOp::create(rewriter,
              loc, mlir::TypeRange{i32}, mlir::ValueRange{tid.getResult()})
          .getResult(0);
  auto tgUI32 =
      ThreadgroupIdOp::create(rewriter, loc, ui32,
                                rewriter.getStringAttr("x"));
  auto tgI32 =
      mlir::UnrealizedConversionCastOp::create(rewriter,
              loc, mlir::TypeRange{i32},
              mlir::ValueRange{tgUI32.getResult()})
          .getResult(0);
  auto cTpb = mlir::arith::ConstantOp::create(
      rewriter, loc,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(tile.threadsPerBlock)));
  auto tgOffsetI32 = mlir::arith::MulIOp::create(rewriter, loc, tgI32,
                                                  cTpb.getResult());
  auto tidI32 = mlir::arith::SubIOp::create(rewriter, loc, tidGlobalI32,
                                             tgOffsetI32.getResult())
                    .getResult();
  auto iv = parentFor.getInductionVar();
  mlir::Value idxI32;
  if (tile.contiguous) {
    // idx = tid * E + iv
    auto cE = mlir::arith::ConstantOp::create(rewriter,
        loc, rewriter.getI32IntegerAttr(tile.elemPerThread));
    auto mul =
        mlir::arith::MulIOp::create(rewriter, loc, tidI32, cE.getResult());
    idxI32 =
        mlir::arith::AddIOp::create(rewriter, loc, mul.getResult(), iv).getResult();
  } else {
    // strided: idx = tid + iv * T
    auto cT = mlir::arith::ConstantOp::create(rewriter,
        loc, rewriter.getI32IntegerAttr(tile.threadsPerBlock));
    auto mul =
        mlir::arith::MulIOp::create(rewriter, loc, iv, cT.getResult());
    idxI32 =
        mlir::arith::AddIOp::create(rewriter, loc, tidI32, mul.getResult())
            .getResult();
  }
  return mlir::UnrealizedConversionCastOp::create(rewriter,
          loc, mlir::TypeRange{ui32}, mlir::ValueRange{idxI32})
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// FuncOp → metal.module { metal.kernel ... }
//
// Scalar (non-pointer, non-tensor) kernel args are wrapped as a 1-element
// `MetalMemRefType<T, 1>` because `metal.kernel`'s verifier rejects any
// kernel arg that isn't a memref. At kernel entry we emit a
// `metal.get_element %arg[0]` prologue that materializes the scalar, and
// RAUW the original arg's uses to that materialized value. Supported scalar
// types match `KernelOp::verify`'s allowed memref-element set:
//   F16 | F32 | BF16 | Index | i1 | i8 | i16 | i32 | i64.
// When the body's blocked layout implies elem_per_thread > 1, the spliced
// body is wrapped in an outer `scf.for(0, E, 1)` (the tile loop) so each
// thread processes E elements per kernel invocation. See
// `.omc/specs/deep-interview-metal-block-size-loop.md`.
// See also `.omc/specs/deep-interview-metal-dynamic-scalar-args.md`.
//===----------------------------------------------------------------------===//

static bool isWrappableScalar(mlir::Type t) {
  if (t.isF16() || t.isF32() || t.isBF16() || t.isIndex())
    return true;
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(t)) {
    switch (intTy.getWidth()) {
    case 1:
    case 8:
    case 16:
    case 32:
    case 64:
      return true;
    }
  }
  return false;
}

// Project a Triton signless integer to its Metal_Type-allowed unsigned
// counterpart (Metal_Type covers UI8..UI64/SI8..SI64 but not signless I8..I64).
// Floats, index, and i1 already fit Metal_Type and pass through unchanged.
// The returned type is what the wrapper's memref element type uses; the
// prologue bridges back to the original signless type with
// `builtin.unrealized_conversion_cast` so downstream signless-arith stays
// correct (same pattern the prior session uses for `metal.thread_id`).
static mlir::Type wrapperElementType(mlir::Type t) {
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(t)) {
    if (intTy.getWidth() > 1 && intTy.isSignless())
      return mlir::IntegerType::get(t.getContext(), intTy.getWidth(),
                                     mlir::IntegerType::Unsigned);
  }
  return t;
}

struct FuncOpLowering : public mlir::OpConversionPattern<mlir::triton::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    // Compute the tile shape from the ORIGINAL body (before any region
    // inlining empties it).
    auto tileInfo = findTileInfo(op);
    llvm::SmallVector<mlir::Type> kernelArgTypes;
    llvm::SmallVector<bool> isAddressSpaceDevice;
    // For arg i, holds the original scalar type if it was wrapped, else null.
    llvm::SmallVector<mlir::Type> wrappedScalarType;
    for (auto argType : op.getFunctionType().getInputs()) {
      mlir::Type converted = getTypeConverter()->convertType(argType);
      if (!converted)
        return rewriter.notifyMatchFailure(op, "argument type not convertible");
      if (!mlir::isa<MetalMemRefType>(converted) &&
          isWrappableScalar(converted)) {
        kernelArgTypes.push_back(
            MetalMemRefType::get(ctx, wrapperElementType(converted), 1));
        isAddressSpaceDevice.push_back(true);
        wrappedScalarType.push_back(converted);
      } else {
        kernelArgTypes.push_back(converted);
        isAddressSpaceDevice.push_back(mlir::isa<MetalMemRefType>(converted));
        wrappedScalarType.push_back(nullptr);
      }
    }
    auto metalModule = ModuleOp::create(rewriter, loc);
    rewriter.setInsertionPointToStart(&metalModule.getBody().front());
    mlir::OperationState kernelState(loc, KernelOp::getOperationName());
    kernelState.addAttribute("name", rewriter.getStringAttr(op.getName()));
    kernelState.addAttribute("address_space_device",
                             rewriter.getBoolArrayAttr(isAddressSpaceDevice));
    mlir::Region *kernelRegion = kernelState.addRegion();
    auto *kernelBlock = new mlir::Block();
    kernelRegion->push_back(kernelBlock);
    for (auto t : kernelArgTypes)
      kernelBlock->addArgument(t, loc);
    auto kernel = llvm::cast<KernelOp>(rewriter.create(kernelState));

    rewriter.inlineRegionBefore(op.getBody(), kernel.getBodyRegion(),
                                kernel.getBodyRegion().end());
    auto &kernelRegion2 = kernel.getBodyRegion();
    auto &emptyBlock = kernelRegion2.front();
    auto &origBlock = *(std::next(kernelRegion2.begin()));

    // Emit the get_element prologue for wrapped args at the start of
    // emptyBlock, then RAUW original arg uses against either the new
    // memref arg (passthrough) or the materialized scalar (wrapped).
    rewriter.setInsertionPointToStart(&emptyBlock);
    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto idxAttr = rewriter.getIntegerAttr(ui32, 0);
    mlir::Value idxZero;
    llvm::SmallVector<mlir::Value> rauwSources;
    rauwSources.reserve(emptyBlock.getNumArguments());
    for (auto [i, newArg] : llvm::enumerate(emptyBlock.getArguments())) {
      if (auto scalarTy = wrappedScalarType[i]) {
        if (!idxZero)
          idxZero = ConstantOp::create(rewriter, loc, idxAttr).getResult();
        auto wrapperTy = wrapperElementType(scalarTy);
        auto getEl =
            GetElementOp::create(rewriter, loc, wrapperTy, newArg, idxZero);
        mlir::Value materialized = getEl.getResult();
        // Bridge UI<W>->I<W> (signless) when the wrapper transport type
        // differs from the original arg type. Downstream signless arith
        // (addi/cmpi/muli) requires this.
        if (wrapperTy != scalarTy) {
          materialized =
              mlir::UnrealizedConversionCastOp::create(rewriter, 
                      loc, mlir::TypeRange{scalarTy},
                      mlir::ValueRange{materialized})
                  .getResult(0);
        }
        rauwSources.push_back(materialized);
      } else {
        rauwSources.push_back(newArg);
      }
    }

    for (auto [origArg, src] :
         llvm::zip(origBlock.getArguments(), rauwSources)) {
      origArg.replaceAllUsesWith(src);
    }
    while (origBlock.getNumArguments())
      origBlock.eraseArgument(0);

    // Record the end of the prologue BEFORE splicing in the original body.
    // After splice we'll move the original body ops (excluding tt.return)
    // into an outer `scf.for` when elem_per_thread > 1.
    mlir::Operation *lastProloguePtr =
        emptyBlock.empty() ? nullptr : &emptyBlock.back();

    emptyBlock.getOperations().splice(emptyBlock.end(),
                                      origBlock.getOperations());
    origBlock.erase();

    // BLOCK_SIZE > threads_per_block: wrap the spliced body in an outer
    // `scf.for(0, E, 1)`. Load/store lowerings detect the parent for and
    // compute a per-iteration index via `emitPerIterIndex`. `tileInfo` was
    // captured at the very start of matchAndRewrite while the original
    // body was still intact.
    if (tileInfo && tileInfo->elemPerThread > 1 && !emptyBlock.empty()) {
      mlir::Operation *returnOp = &emptyBlock.back();
      mlir::Block::iterator firstOrigBodyIt =
          lastProloguePtr ? std::next(lastProloguePtr->getIterator())
                          : emptyBlock.begin();
      mlir::Block::iterator returnIt = returnOp->getIterator();
      if (lastProloguePtr)
        rewriter.setInsertionPointAfter(lastProloguePtr);
      else
        rewriter.setInsertionPointToStart(&emptyBlock);
      auto i32 = rewriter.getI32Type();
      auto cZero = mlir::arith::ConstantOp::create(rewriter, 
          loc, rewriter.getI32IntegerAttr(0));
      auto cE = mlir::arith::ConstantOp::create(rewriter, 
          loc, rewriter.getI32IntegerAttr(tileInfo->elemPerThread));
      auto cStep = mlir::arith::ConstantOp::create(rewriter, 
          loc, rewriter.getI32IntegerAttr(1));
      auto forOp = mlir::scf::ForOp::create(rewriter, 
          loc, cZero.getResult(), cE.getResult(), cStep.getResult());
      auto &forBlock = forOp.getRegion().front();
      auto forYieldIt = std::prev(forBlock.end());
      forBlock.getOperations().splice(forYieldIt, emptyBlock.getOperations(),
                                       firstOrigBodyIt, returnIt);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.return → metal.return
//===----------------------------------------------------------------------===//

struct ReturnOpLowering
    : public mlir::OpConversionPattern<mlir::triton::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::ReturnOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.get_program_id <axis> -> metal.threadgroup_id <axis> -> signless i32
//
// Each MTLComputeCommandEncoder dispatchThreadgroups call receives a
// (gridX, gridY, gridZ) triple; inside the kernel,
// `[[threadgroup_position_in_grid]]` gives the current threadgroup's
// index along each axis. Triton's tt.get_program_id axis enum (x=0,
// y=1, z=2) maps directly to the MSL `tgid.x/.y/.z` components.
// See `.omc/specs/deep-interview-metal-pid-lowering.md`.
//===----------------------------------------------------------------------===//

struct GetProgramIdLowering
    : public mlir::OpConversionPattern<mlir::triton::GetProgramIdOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::GetProgramIdOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int axis = op.getAxisAsInt();
    const char *dim = axis == 0 ? "x" : axis == 1 ? "y" : "z";
    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto i32 = rewriter.getI32Type();
    auto tg = ThreadgroupIdOp::create(rewriter, loc, ui32,
                                        rewriter.getStringAttr(dim));
    // Bridge ui32 -> signless i32 so downstream arith.* ops (which
    // require signless operands) can consume it. Mirrors the
    // metal.thread_id bridge used elsewhere in this pass.
    auto castI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tg.getResult()})
            .getResult(0);
    rewriter.replaceOp(op, castI32);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.get_num_programs <axis> -> metal.threadgroups_per_grid <axis> -> i32
//
// Mirrors GetProgramIdLowering: emits the MSL [[threadgroups_per_grid]]
// builtin via the new metal.threadgroups_per_grid op, then bridges
// ui32 -> signless i32 for downstream arith consumers. Tutorial driver:
// `leet-triton/tutorials_python/02-fused-softmax.py:89` (tl.num_programs(0)).
// See `.omc/plans/tutorial02-fused-softmax-fix-consensus.md`.
//===----------------------------------------------------------------------===//

struct GetNumProgramsLowering
    : public mlir::OpConversionPattern<mlir::triton::GetNumProgramsOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::GetNumProgramsOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int axis = static_cast<int>(op.getAxis());
    const char *dim = axis == 0 ? "x" : axis == 1 ? "y" : "z";
    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto i32 = rewriter.getI32Type();
    auto tgpg = ThreadgroupsPerGridOp::create(
        rewriter, loc, ui32, rewriter.getStringAttr(dim));
    auto castI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tgpg.getResult()})
            .getResult(0);
    rewriter.replaceOp(op, castI32);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.make_range[start..end] → metal.cast metal.thread_id "x" : ui32 -> i32
//
// Under 1-elt-per-thread + start=0 + end==BLOCK_SIZE, each thread holds
// exactly the value `thread_id_in_block`, which under single-block launch
// equals `thread_position_in_grid` (= metal.thread_id "x").
//===----------------------------------------------------------------------===//

// 1D path: metal-dialect's `metal.cast` rejects signless i32, and the
// downstream arith chain requires signless. We dodge the type-system tangle
// by lowering make_range to a placeholder `arith.constant 0 : i32`. The
// downstream arith.muli/addi fold to 0 — those values are NEVER read by
// the 1D `LoadLowering`/`StoreLowering`, which use `metal.thread_id "x"`
// directly as the index.
//
// 2D path: when the result type carries `#ttg.slice<{dim = X, parent =
// #blocked2D}>`, emit a per-thread axis value (row if axis==0, col if
// axis==1) computed from the flat per-iter idx:
//     row = idx / BLOCK_N
//     col = idx % BLOCK_N
// This is a valid bijection for elementwise ops (which don't constrain
// physical thread placement) and lets the downstream arith chain
// (`muli`, `addi`) naturally produce
// `(pid_m*BLOCK_M + row) * stride_m + (pid_n*BLOCK_N + col)` as the
// per-thread offset. See
// `.omc/specs/deep-interview-metal-2d-maskedaccess-session2.md`.
struct MakeRangeLowering
    : public mlir::OpConversionPattern<mlir::triton::MakeRangeOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::MakeRangeOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resTy = mlir::cast<mlir::RankedTensorType>(op.getType());
    if (auto slice = mlir::dyn_cast_or_null<
            mlir::triton::gpu::SliceEncodingAttr>(resTy.getEncoding())) {
      if (auto parent = mlir::dyn_cast_or_null<
              mlir::triton::gpu::BlockedEncodingAttr>(slice.getParent())) {
        if (parent.getOrder().size() == 2) {
          int axis = 1 - static_cast<int>(slice.getDim());
          // Walk the entire enclosing module so we see all rank-2 ttg.blocked
          // tensors regardless of where the conversion driver is in
          // converting individual ops (the triton.func may already be a
          // metal.kernel by now; mlir-level types of replaced values are
          // post-conversion scalars). The largest 2D blocked tensor we can
          // still see in the module gives BLOCK_M, BLOCK_N.
          mlir::Operation *modOp = op->getParentOfType<mlir::ModuleOp>();
          std::optional<TileInfo> tile;
          if (modOp) {
            int64_t bestSize = 0;
            modOp->walk([&](mlir::Operation *inner) {
              for (auto v : inner->getResults()) {
                auto info = tileFromTensor(v.getType());
                if (!info) continue;
                // AC4-v6 intent (mirrored from findTileInfo): skip
                // tensor<...x!tt.ptr<...>> in the bestSize tiebreaker.
                if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
                    rt && mlir::isa<mlir::triton::PointerType>(rt.getElementType()))
                  continue;
                int64_t sz = 1;
                for (auto s : info->shape) sz *= s;
                if (sz > bestSize) {
                  bestSize = sz;
                  tile = info;
                }
              }
            });
          }
          if (tile && tile->rank == 2 && tile->shape.size() == 2) {
            int64_t blockN = tile->shape[1];
            auto parentFor = findOutermostScfFor(op); // Wall 13 fix: tile loop, not user loop
            auto i32 = rewriter.getI32Type();
            auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
            // Metal `[[thread_position_in_grid]]` is GLOBAL, not per-CTA.
            // For multi-CTA 2D dispatch we need the threadgroup-local
            // tid: `lid.x = id.x - tgid.x * threadsPerCTA`.
            auto tidGlobal = ThreadIdOp::create(
                rewriter, loc, ui32, rewriter.getStringAttr("x"));
            mlir::Value tidI32 =
                mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, mlir::TypeRange{i32},
                    mlir::ValueRange{tidGlobal.getResult()})
                    .getResult(0);
            auto tgX = ThreadgroupIdOp::create(
                rewriter, loc, ui32, rewriter.getStringAttr("x"));
            mlir::Value tgI32 =
                mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, mlir::TypeRange{i32},
                    mlir::ValueRange{tgX.getResult()})
                    .getResult(0);
            auto tpbAttr =
                rewriter.getI32IntegerAttr(tile->threadsPerBlock);
            auto tpbConst =
                mlir::arith::ConstantOp::create(rewriter, loc, tpbAttr);
            auto tgOffset = mlir::arith::MulIOp::create(
                rewriter, loc, tgI32, tpbConst.getResult());
            mlir::Value localTidI32 =
                mlir::arith::SubIOp::create(rewriter, loc, tidI32,
                                             tgOffset.getResult())
                    .getResult();
            // Per-iter local idx in tile: strided vs contiguous (same
            // shape as emitPerIterIndex but using localTid instead of
            // global tid).
            mlir::Value idxI32 = localTidI32;
            if (parentFor && tile->elemPerThread > 1) {
              auto iv = parentFor.getInductionVar();
              if (tile->contiguous) {
                auto cE = mlir::arith::ConstantOp::create(
                    rewriter, loc,
                    rewriter.getI32IntegerAttr(tile->elemPerThread));
                auto mul = mlir::arith::MulIOp::create(
                    rewriter, loc, localTidI32, cE.getResult());
                idxI32 = mlir::arith::AddIOp::create(rewriter, loc,
                                                      mul.getResult(), iv)
                             .getResult();
              } else {
                auto cT = mlir::arith::ConstantOp::create(rewriter, loc,
                                                          tpbAttr);
                auto mul = mlir::arith::MulIOp::create(rewriter, loc, iv,
                                                        cT.getResult());
                idxI32 = mlir::arith::AddIOp::create(rewriter, loc,
                                                      localTidI32,
                                                      mul.getResult())
                             .getResult();
              }
            }
            // Pick div/rem and the divisor based on the PARENT
            // BlockedEncoding's `order` permutation. With order=[1,0] (dim 1
            // contiguous), the canonical linearization is `tid = row*N + col`
            // (divisor = N), so axis=0 (row) = tid/N (div) and axis=1 (col) =
            // tid%N (rem). With order=[0,1] (dim 0 contiguous), linearization
            // is `tid = col*M + row` (divisor = M), so axis=0 (row) = tid%M
            // (rem) and axis=1 (col) = tid/M (div). Pre-L1d2 the codebase only
            // saw order=[1,0] kernels; the L1d2 staged-transpose body's dst
            // encoding is order=[0,1] (`#blocked1` in matrix_transpose TTGIR),
            // so this branch is now load-bearing.
            int64_t blockM = tile->shape[0];
            auto parentOrder = parent.getOrder();
            bool rowMajor =
                parentOrder.size() == 2 && parentOrder[0] == 1 &&
                parentOrder[1] == 0;
            int64_t divisor = rowMajor ? blockN : blockM;
            auto bn = mlir::arith::ConstantOp::create(
                rewriter, loc, rewriter.getI32IntegerAttr(divisor));
            mlir::Value result;
            if ((axis == 0 && rowMajor) || (axis == 1 && !rowMajor))
              result = mlir::arith::DivSIOp::create(rewriter, loc, idxI32,
                                                     bn.getResult())
                           .getResult();
            else
              result = mlir::arith::RemSIOp::create(rewriter, loc, idxI32,
                                                     bn.getResult())
                           .getResult();
            rewriter.replaceOp(op, result);
            return mlir::success();
          }
        }
      }
    }
    // 3D path: a rank-1 make_range whose result is a doubly-nested `#ttg.slice`
    // over a rank-3 `#ttg.blocked` parent — the `offset[:,None,None]` /
    // `offset[None,:,None]` / `offset[None,None,:]` broadcast cones of 3D
    // kernels (e.g. 3D convolution). Emit the per-(thread, tile-iv) coordinate
    // along the surviving original axis. As in the 2D path, ANY (thread,iv)->
    // element bijection is valid for elementwise / reduction ops (they do not
    // constrain physical thread placement) provided every make_range and the
    // load/store share it. We decompose the flat local index (strided:
    // localTid + iv*T; contiguous: localTid*E + iv) by the parent `order`.
    if (auto outerSlice = mlir::dyn_cast_or_null<
            mlir::triton::gpu::SliceEncodingAttr>(resTy.getEncoding())) {
      if (auto innerSlice = mlir::dyn_cast_or_null<
              mlir::triton::gpu::SliceEncodingAttr>(outerSlice.getParent())) {
        if (auto parent3d = mlir::dyn_cast_or_null<
                mlir::triton::gpu::BlockedEncodingAttr>(
                innerSlice.getParent())) {
          if (parent3d.getOrder().size() == 3) {
            // Surviving original axis = the one removed by neither slice. The
            // inner slice indexes the rank-3 dim list; the outer slice then
            // indexes the rank-2 remainder.
            llvm::SmallVector<int64_t, 3> remaining{0, 1, 2};
            int innerDim = static_cast<int>(innerSlice.getDim());
            int outerDim = static_cast<int>(outerSlice.getDim());
            if (innerDim >= 0 && innerDim < (int)remaining.size())
              remaining.erase(remaining.begin() + innerDim);
            if (outerDim >= 0 && outerDim < (int)remaining.size())
              remaining.erase(remaining.begin() + outerDim);
            // Largest non-pointer rank-3 blocked tensor visible in the module
            // gives the tile shape / thread count (mirrors the 2D walk).
            mlir::Operation *modOp = op->getParentOfType<mlir::ModuleOp>();
            std::optional<TileInfo> tile;
            if (remaining.size() == 1 && modOp) {
              int64_t bestSize = 0;
              modOp->walk([&](mlir::Operation *inner) {
                for (auto v : inner->getResults()) {
                  auto info = tileFromTensor(v.getType());
                  if (!info || info->rank != 3)
                    continue;
                  if (auto rt =
                          mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
                      rt && mlir::isa<mlir::triton::PointerType>(
                                rt.getElementType()))
                    continue;
                  int64_t sz = 1;
                  for (auto s : info->shape)
                    sz *= s;
                  if (sz > bestSize) {
                    bestSize = sz;
                    tile = info;
                  }
                }
              });
            }
            if (tile && tile->rank == 3) {
              int64_t axis = remaining[0];
              auto i32 = rewriter.getI32Type();
              auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
              // localTid = id.x - tgid.x * threadsPerBlock.
              auto tidGlobal = ThreadIdOp::create(rewriter, loc, ui32,
                                                  rewriter.getStringAttr("x"));
              mlir::Value tidI32 =
                  mlir::UnrealizedConversionCastOp::create(
                      rewriter, loc, mlir::TypeRange{i32},
                      mlir::ValueRange{tidGlobal.getResult()})
                      .getResult(0);
              auto tgX = ThreadgroupIdOp::create(rewriter, loc, ui32,
                                                 rewriter.getStringAttr("x"));
              mlir::Value tgI32 =
                  mlir::UnrealizedConversionCastOp::create(
                      rewriter, loc, mlir::TypeRange{i32},
                      mlir::ValueRange{tgX.getResult()})
                      .getResult(0);
              auto tpbConst = mlir::arith::ConstantOp::create(
                  rewriter, loc,
                  rewriter.getI32IntegerAttr(tile->threadsPerBlock));
              auto tgOffset = mlir::arith::MulIOp::create(
                  rewriter, loc, tgI32, tpbConst.getResult());
              mlir::Value idxI32 =
                  mlir::arith::SubIOp::create(rewriter, loc, tidI32,
                                              tgOffset.getResult())
                      .getResult();
              // Fold the tile-loop iv when E>1 (same shape as emitPerIterIndex).
              auto parentFor = findOutermostScfFor(op);
              if (parentFor && tile->elemPerThread > 1) {
                auto iv = parentFor.getInductionVar();
                if (tile->contiguous) {
                  auto cE = mlir::arith::ConstantOp::create(
                      rewriter, loc,
                      rewriter.getI32IntegerAttr(tile->elemPerThread));
                  auto mul = mlir::arith::MulIOp::create(rewriter, loc, idxI32,
                                                         cE.getResult());
                  idxI32 = mlir::arith::AddIOp::create(rewriter, loc,
                                                       mul.getResult(), iv)
                               .getResult();
                } else {
                  auto cT = mlir::arith::ConstantOp::create(
                      rewriter, loc,
                      rewriter.getI32IntegerAttr(tile->threadsPerBlock));
                  auto mul = mlir::arith::MulIOp::create(rewriter, loc, iv,
                                                         cT.getResult());
                  idxI32 = mlir::arith::AddIOp::create(rewriter, loc, idxI32,
                                                       mul.getResult())
                               .getResult();
                }
              }
              // Row-major-by-`order` tile strides; coord = (idx/stride)%shape.
              auto order = parent3d.getOrder();
              llvm::SmallVector<int64_t, 3> strides(3, 1);
              int64_t accStride = 1;
              for (size_t i = 0; i < order.size(); ++i) {
                int64_t d = order[i];
                strides[d] = accStride;
                accStride *= tile->shape[d];
              }
              mlir::Value coord = idxI32;
              if (strides[axis] != 1) {
                auto cS = mlir::arith::ConstantOp::create(
                    rewriter, loc, rewriter.getI32IntegerAttr(strides[axis]));
                coord = mlir::arith::DivSIOp::create(rewriter, loc, coord,
                                                     cS.getResult())
                            .getResult();
              }
              auto cM = mlir::arith::ConstantOp::create(
                  rewriter, loc, rewriter.getI32IntegerAttr(tile->shape[axis]));
              coord = mlir::arith::RemSIOp::create(rewriter, loc, coord,
                                                   cM.getResult())
                          .getResult();
              rewriter.replaceOp(op, coord);
              return mlir::success();
            }
          }
        }
      }
    }
    // 1D path: emit a real per-thread index value so that downstream
    // arithmetic on arange (`>> 1`, `* k`, `% n`, `& mask`) survives the
    // conversion. Lmultiload Phase C (see `.omc/specs/deep-interview-
    // lmultiload-phase-c-makerange.md`): replaces the prior `arith.constant
    // 0` placeholder. `MakeRangeLowering` is the single source of truth
    // for the per-thread term; `emitLoadStoreIndex` rank-1 stops adding
    // localTid itself.
    if (auto info = tileFromTensor(resTy)) {
      if (info->rank == 1) {
        auto i32 = rewriter.getI32Type();
        auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
        // localTid = id.x - tgid.x * threadsPerBlock (per-CTA-local index).
        auto tidGlobal = ThreadIdOp::create(
            rewriter, loc, ui32, rewriter.getStringAttr("x"));
        mlir::Value tidI32 =
            mlir::UnrealizedConversionCastOp::create(
                rewriter, loc, mlir::TypeRange{i32},
                mlir::ValueRange{tidGlobal.getResult()})
                .getResult(0);
        auto tgX = ThreadgroupIdOp::create(
            rewriter, loc, ui32, rewriter.getStringAttr("x"));
        mlir::Value tgI32 =
            mlir::UnrealizedConversionCastOp::create(
                rewriter, loc, mlir::TypeRange{i32},
                mlir::ValueRange{tgX.getResult()})
                .getResult(0);
        auto tpbConst = mlir::arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getI32IntegerAttr(info->threadsPerBlock));
        auto tgOffset = mlir::arith::MulIOp::create(
            rewriter, loc, tgI32, tpbConst.getResult());
        mlir::Value localTidI32 =
            mlir::arith::SubIOp::create(rewriter, loc, tidI32,
                                         tgOffset.getResult())
                .getResult();
        // Tile-loop wrap: if E>1 and we're inside the outer scf.for, fold
        // the iv in the same shape `emitPerIterIndex` would (contiguous:
        // localTid*E + iv; strided: localTid + iv*T).
        auto parentFor = findOutermostScfFor(op); // Wall 13 fix: tile loop, not user loop
        mlir::Value idxI32 = localTidI32;
        if (parentFor && info->elemPerThread > 1) {
          auto iv = parentFor.getInductionVar();
          if (info->contiguous) {
            auto cE = mlir::arith::ConstantOp::create(
                rewriter, loc,
                rewriter.getI32IntegerAttr(info->elemPerThread));
            auto mul = mlir::arith::MulIOp::create(
                rewriter, loc, localTidI32, cE.getResult());
            idxI32 = mlir::arith::AddIOp::create(rewriter, loc,
                                                  mul.getResult(), iv)
                         .getResult();
          } else {
            auto cT = mlir::arith::ConstantOp::create(
                rewriter, loc,
                rewriter.getI32IntegerAttr(info->threadsPerBlock));
            auto mul = mlir::arith::MulIOp::create(rewriter, loc, iv,
                                                    cT.getResult());
            idxI32 = mlir::arith::AddIOp::create(rewriter, loc,
                                                  localTidI32,
                                                  mul.getResult())
                         .getResult();
          }
        }
        rewriter.replaceOp(op, idxI32);
        return mlir::success();
      }
    }
    // Unsupported encoding (no blocked layout): keep the constant-0
    // placeholder so we don't break legalization of edge cases.
    auto zero = rewriter.getI32IntegerAttr(0);
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, zero);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.splat → identity (under tensor→scalar TypeConverter, splat is a no-op).
//===----------------------------------------------------------------------===//

struct SplatLowering
    : public mlir::OpConversionPattern<mlir::triton::SplatOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::SplatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ttg.convert_layout lowering:
//   - L1c identity passthrough (srcTy == dstTy or post-converter scalarized
//     identity, e.g. L3a slice→blocked rank-1).
//   - L1d2 staged-transpose body for in-envelope rank-2 blocked↔blocked cvts
//     with sizePerThread=[1,1] on both sides: emit
//       threadgroup_alloca → tg_store_indexed[srcIdx] → barrier →
//       tg_load_indexed[dstIdx] → barrier → replaceOp
//     where srcIdx / dstIdx are derived algebraically from each side's
//     `BlockedEncodingAttr.order`. The pre-pass classifier guarantees only
//     in-envelope cvts (sizePerThread=[1,1]) reach this body; any other
//     non-identity cvt is rejected by the pre-pass as out-of-envelope (L1d3).
// See `.omc/specs/deep-interview-leet-triton-l1d2-staged-transpose-body.md`.
//===----------------------------------------------------------------------===//

// A ttg.convert_layout is a pure relabel under the Metal scalarizing model —
// safe to lower as a scalar identity (no cross-thread data movement) — when
// EITHER:
//   * at least one side is a SliceEncodingAttr. Slice encodings only arise from
//     `tt.expand_dims` (broadcast-prep `x[None,:]` / `x[:,None]`) and `tt.reduce`
//     outputs — index/broadcast relabels, never data transposes. (Covers the
//     subarray-sum / adder broadcast cones.) OR
//   * both sides are blocked, sizePerThread is all-1 on both, AND the shape has
//     at most ONE non-unit dimension — a degenerate convert that can only
//     permute size-1 axes, so no element actually moves. (Covers the rank>=2
//     index cones stacked expand_dims produces, e.g. `tensor<1x1x1024>` in
//     3D subarray-sum.)
// A GENUINE transpose (both blocked, >=2 non-unit dims, or sizePerThread>1) is
// NOT identity-safe: the rank-2 sizePerThread=[1,1] case goes to the staged-
// transpose body below; everything else is rejected (L1d3) by the pre-pass.
// True if a slice-encoded cone bottoms out in a value whose per-thread content
// depends on the slice layout — a `tt.load` or `tt.make_range`, both of which
// MakeRangeLowering places by projecting the PARENT tile's index onto the
// surviving axis (row = idx / N for dim=1). Converting such a value to a plain
// rank-1 #blocked layout, where thread t holds element t, genuinely moves data;
// treating that cvt as a scalar identity hands every lane element `t / N`.
//
// `tt.reduce` / `tt.scan` results are NOT such leaves and terminate the walk:
// ReduceLowering reads its result out of rowBuf under the DESTINATION layout's
// indexing already, so bridging it with an identity cvt is correct — that is
// the case the identity rule was written for.
static bool sliceConeHasLayoutDependentLeaf(mlir::Value root) {
  llvm::SmallVector<mlir::Value, 16> wl{root};
  llvm::SmallPtrSet<mlir::Value, 16> seen;
  while (!wl.empty()) {
    mlir::Value v = wl.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!rt || !mlir::isa_and_nonnull<mlir::triton::gpu::SliceEncodingAttr>(
                   rt.getEncoding()))
      continue;
    auto *def = v.getDefiningOp();
    if (!def)
      continue;
    if (mlir::isa<mlir::triton::ReduceOp, mlir::triton::ScanOp>(def))
      continue;
    if (mlir::isa<mlir::triton::LoadOp, mlir::triton::MakeRangeOp>(def))
      return true;
    for (auto operand : def->getOperands())
      wl.push_back(operand);
  }
  return false;
}

static bool isScalarIdentityConvert(mlir::triton::gpu::ConvertLayoutOp op) {
  auto srcRtt = mlir::dyn_cast<mlir::RankedTensorType>(op.getSrc().getType());
  auto dstRtt =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
  if (!srcRtt || !dstRtt)
    return false;
  if (mlir::isa_and_nonnull<mlir::triton::gpu::SliceEncodingAttr>(
          srcRtt.getEncoding()) ||
      mlir::isa_and_nonnull<mlir::triton::gpu::SliceEncodingAttr>(
          dstRtt.getEncoding())) {
    // The comment above assumes slice encodings only ever arise from
    // tt.expand_dims and tt.reduce outputs — pure index/broadcast relabels. A
    // slice-encoded tt.load breaks that assumption. `normalizeBlockedDivergentCvt`
    // re-encodes those cones so the cvt really does become an identity; if it
    // bailed (a shared non-cloneable value, an unhandled boundary), refuse the
    // identity shortcut so the kernel gets a clean rejection instead of
    // silently reading element `t / N` in every lane.
    if (mlir::isa_and_nonnull<mlir::triton::gpu::SliceEncodingAttr>(
            srcRtt.getEncoding()) &&
        mlir::isa_and_nonnull<mlir::triton::gpu::BlockedEncodingAttr>(
            dstRtt.getEncoding()) &&
        sliceConeHasLayoutDependentLeaf(op.getSrc()))
      return false;
    return true;
  }
  auto srcB = mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
      srcRtt.getEncoding());
  auto dstB = mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
      dstRtt.getEncoding());
  if (!srcB || !dstB)
    return false;
  for (auto s : srcB.getSizePerThread())
    if (s != 1)
      return false;
  for (auto s : dstB.getSizePerThread())
    if (s != 1)
      return false;
  int nonUnit = 0;
  for (auto s : srcRtt.getShape())
    if (s != 1)
      ++nonUnit;
  return nonUnit <= 1;
}

struct ConvertLayoutLowering
    : public mlir::OpConversionPattern<mlir::triton::gpu::ConvertLayoutOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Path 1: structural identity (srcTy == dstTy).
    if (op.getSrc().getType() == op.getResult().getType()) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return mlir::success();
    }
    // Path 2: scalar-identity relabel (slice broadcast/reduce cones, or a
    // degenerate blocked↔blocked permuting only size-1 axes). See
    // `isScalarIdentityConvert`. Genuine transposes fall through to Path 3.
    if (isScalarIdentityConvert(op)) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return mlir::success();
    }
    auto srcRtt =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getSrc().getType());
    auto dstRtt =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    // Path 2b: rank-1 relabel whose ONLY consumers are `tt.store`s.
    //
    // A scatter (`tl.store(dst + idx_tensor, vals)`) arrives as a pair of these
    // cvts feeding one store. They are a genuine permutation — src has
    // sizePerThread>1 (`idx = tid*E + iv`), dst is strided (`idx = tid + iv*T`)
    // — so passing one through in general would hand a consumer another
    // element's value. StoreLowering does not consume it: it peels back to the
    // pre-cvt operands and performs the whole store in the SOURCE layout, which
    // writes the same (address, value) set (see its scatter-peel comment). So
    // for a store-only consumer set this cvt is genuinely dead, and forwarding
    // the source is how a dead op gets legalized here — the forwarded value has
    // no remaining reader.
    //
    // The user check is what keeps this sound: any OTHER consumer (an
    // elementwise op, a reduce cone, a second store shape that does not peel)
    // means the permutation is observable, and the cvt falls through to Path 3
    // and is rejected instead.
    if (srcRtt && dstRtt && srcRtt.getRank() == 1 && dstRtt.getRank() == 1 &&
        srcRtt.getShape() == dstRtt.getShape() &&
        srcRtt.getElementType() == dstRtt.getElementType() &&
        // Vacuously true once the store has already been legalized and erased —
        // which is the usual ordering, and why this must NOT require a
        // non-empty use list. A dead relabel has no reader to mislead.
        llvm::all_of(op.getResult().getUsers(), [](mlir::Operation *u) {
          return mlir::isa<mlir::triton::StoreOp>(u);
        })) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return mlir::success();
    }
    // Path 3: L1d2 staged-transpose body for in-envelope rank-2 blocked↔
    // blocked cvts with sizePerThread=[1,1] on both sides.
    if (!srcRtt || !dstRtt || srcRtt.getRank() != 2 || dstRtt.getRank() != 2 ||
        srcRtt.getShape() != dstRtt.getShape() ||
        srcRtt.getElementType() != dstRtt.getElementType())
      return rewriter.notifyMatchFailure(
          op, "ttg.convert_layout: out-of-envelope");
    auto srcBlocked =
        mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
            srcRtt.getEncoding());
    auto dstBlocked =
        mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
            dstRtt.getEncoding());
    if (!srcBlocked || !dstBlocked)
      return rewriter.notifyMatchFailure(
          op, "ttg.convert_layout: non-blocked encoding");
    auto srcSpt = srcBlocked.getSizePerThread();
    auto dstSpt = dstBlocked.getSizePerThread();
    if (srcSpt.size() != 2 || dstSpt.size() != 2 || srcSpt[0] != 1 ||
        srcSpt[1] != 1 || dstSpt[0] != 1 || dstSpt[1] != 1)
      return rewriter.notifyMatchFailure(
          op, "ttg.convert_layout: sizePerThread > 1 deferred to L1d3");
    auto srcOrder = srcBlocked.getOrder();
    auto dstOrder = dstBlocked.getOrder();
    if (srcOrder.size() != 2 || dstOrder.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "ttg.convert_layout: order rank mismatch");

    auto loc = op.getLoc();
    int64_t M = srcRtt.getDimSize(0);
    int64_t N = srcRtt.getDimSize(1);
    int64_t bufSize = M * N;
    mlir::Type elemTy = srcRtt.getElementType();
    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto i32 = rewriter.getI32Type();

    // 1) Fresh threadgroup buffer of shape M*N with the scalarized element
    // type. The staging buffer holds the logical tile in row-major order
    // (linear index = row*N + col regardless of side encoding); the
    // transpose effect comes from each side's `order` mapping localTid →
    // (row, col) differently.
    auto bufTy =
        MetalMemRefType::get(rewriter.getContext(), elemTy, bufSize);
    mlir::Value buf =
        ThreadgroupAllocaOp::create(rewriter, loc, bufTy).getResult();

    // 2) localTid = id.x - tgid.x * threadsPerBlock (matches Lmultiload
    // Phase C's MakeRangeLowering canonical pattern). srcBlocked and
    // dstBlocked share the same threadsPerBlock product by the in-envelope
    // predicate (same total element count, both are 1 elem/thread).
    int64_t tpb = 1;
    for (auto t : srcBlocked.getThreadsPerWarp()) tpb *= t;
    for (auto w : srcBlocked.getWarpsPerCTA()) tpb *= w;
    auto tidGlobal = ThreadIdOp::create(rewriter, loc, ui32,
                                        rewriter.getStringAttr("x"));
    mlir::Value tidI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tidGlobal.getResult()})
            .getResult(0);
    auto tgX = ThreadgroupIdOp::create(rewriter, loc, ui32,
                                       rewriter.getStringAttr("x"));
    mlir::Value tgI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tgX.getResult()})
            .getResult(0);
    auto tpbConst = mlir::arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
    auto tgOffset = mlir::arith::MulIOp::create(rewriter, loc, tgI32,
                                                tpbConst.getResult());
    mlir::Value localTidI32 =
        mlir::arith::SubIOp::create(rewriter, loc, tidI32,
                                    tgOffset.getResult())
            .getResult();

    // 3) Index math: given a blocked encoding with sizePerThread=[1,1],
    // each thread holds one logical (row, col) on that side. Linearization
    // along the encoding's `order`:
    //   * order == [1, 0] (dim 1 contiguous): localTid → (row, col) =
    //     (localTid / N, localTid % N), buf[row*N + col] = localTid.
    //   * order == [0, 1] (dim 0 contiguous): localTid → (row, col) =
    //     (localTid % M, localTid / M), buf[row*N + col] =
    //     (localTid % M) * N + (localTid / M).
    // The transpose effect arises from the src/dst `order` divergence.
    auto cM = mlir::arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(M)));
    auto cN = mlir::arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(N)));
    auto sideIdxI32 = [&](llvm::ArrayRef<unsigned> order) -> mlir::Value {
      if (order[0] == 1 && order[1] == 0) {
        // Row-major: buf index == localTid.
        return localTidI32;
      }
      if (order[0] == 0 && order[1] == 1) {
        // Column-major: buf index = (localTid % M) * N + (localTid / M).
        auto rowI32 = mlir::arith::RemSIOp::create(
                          rewriter, loc, localTidI32, cM.getResult())
                          .getResult();
        auto colI32 = mlir::arith::DivSIOp::create(
                          rewriter, loc, localTidI32, cM.getResult())
                          .getResult();
        auto rowMul = mlir::arith::MulIOp::create(rewriter, loc, rowI32,
                                                   cN.getResult())
                          .getResult();
        return mlir::arith::AddIOp::create(rewriter, loc, rowMul, colI32)
            .getResult();
      }
      return {};
    };
    mlir::Value srcIdxI32 = sideIdxI32(srcOrder);
    mlir::Value dstIdxI32 = sideIdxI32(dstOrder);
    if (!srcIdxI32 || !dstIdxI32)
      return rewriter.notifyMatchFailure(
          op,
          "ttg.convert_layout: unsupported order permutation (only "
          "[1,0]/[0,1] for rank-2)");
    mlir::Value srcIdxUI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{srcIdxI32})
            .getResult(0);
    mlir::Value dstIdxUI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{dstIdxI32})
            .getResult(0);

    // 4) Stage: store this thread's source-side scalar into the buffer at
    // srcIdx, barrier, load this thread's dest-side scalar from the buffer
    // at dstIdx, barrier (paranoid-safe for chained cvts), replace.
    TgStoreIndexedOp::create(rewriter, loc, buf, srcIdxUI32, adaptor.getSrc());
    BarrierOp::create(rewriter, loc);
    auto loaded =
        TgLoadIndexedOp::create(rewriter, loc, elemTy, buf, dstIdxUI32);
    BarrierOp::create(rewriter, loc);
    rewriter.replaceOp(op, loaded.getResult());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// arith.constant dense<C> on tensor -> scalar arith.constant.
//
// Under our tensor->element-type TypeConverter, a tensor-valued
// arith.constant with a SplatElementsAttr collapses to a single scalar
// (all elements equal). Non-splat dense attributes would require
// per-element scalarization and are rejected with a clean diagnostic.
// See `.omc/specs/deep-interview-metal-constdense-expanddims-broadcast.md`.
//===----------------------------------------------------------------------===//

struct ArithConstantDenseLowering
    : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = op.getType();
    auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(resTy);
    if (!rtt)
      return mlir::failure();  // scalar arith.constant: leave alone
    auto denseAttr =
        mlir::dyn_cast<mlir::SplatElementsAttr>(op.getValue());
    if (!denseAttr)
      return rewriter.notifyMatchFailure(
          op, "arith.constant: non-splat dense not supported");
    auto elemTy = rtt.getElementType();
    mlir::TypedAttr scalarAttr;
    if (mlir::isa<mlir::IntegerType>(elemTy)) {
      scalarAttr = rewriter.getIntegerAttr(
          elemTy, denseAttr.getSplatValue<llvm::APInt>());
    } else if (mlir::isa<mlir::FloatType>(elemTy)) {
      scalarAttr = rewriter.getFloatAttr(
          elemTy, denseAttr.getSplatValue<llvm::APFloat>());
    } else {
      return rewriter.notifyMatchFailure(
          op, "arith.constant: unsupported element type");
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, scalarAttr);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.expand_dims / tt.broadcast -> identity (under typeconverter).
//
// Both ops add or expand size-1 axes on tensor types. Under our
// tensor->scalar conversion, each thread holds one element, so the
// shape change adds no scalar information. Pass the source value
// through, matching the SplatLowering pattern.
//===----------------------------------------------------------------------===//

struct ExpandDimsLowering
    : public mlir::OpConversionPattern<mlir::triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::ExpandDimsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return mlir::success();
  }
};

struct BroadcastLowering
    : public mlir::OpConversionPattern<mlir::triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::BroadcastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.reshape / tt.trans -> identity (under TypeConverter).
//
// Both ops only change the logical tensor shape (reshape) or axis order
// (trans). Under our tensor->scalar TypeConverter each thread holds one
// scalar element, so neither op changes the per-thread value — the
// downstream IR's address arithmetic re-derives indices from the new
// shape and consumes the same scalar. Same identity-passthrough pattern
// as expand_dims / broadcast. See
// `.omc/specs/deep-interview-metal-matmul-session1-reshape-trans-simd-scaffold.md`.
//===----------------------------------------------------------------------===//

struct ReshapeLowering
    : public mlir::OpConversionPattern<mlir::triton::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::ReshapeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return mlir::success();
  }
};

struct TransLowering
    : public mlir::OpConversionPattern<mlir::triton::TransOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::TransOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// arith.muli on tensor → arith.muli on scalar (identity under typeconverter).
// arith.addi on tensor → arith.addi on scalar (identity under typeconverter).
// arith.addf on tensor → metal.binary_exp addOp.
//===----------------------------------------------------------------------===//

struct ArithMuliLowering
    : public mlir::OpConversionPattern<mlir::arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::MulIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithAddILowering
    : public mlir::OpConversionPattern<mlir::arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AddIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// L2: Elementwise integer arith op patterns. Each follows the MulI/AddI
// template — `replaceOpWithNewOp` rebuilds the op on the (now scalar)
// converted operands. Width-agnostic; semantics handled at MSL emit time.
// See `.omc/specs/deep-interview-leet-triton-l2-int-arith-broad.md`.
//===----------------------------------------------------------------------===//

struct ArithSubILowering
    : public mlir::OpConversionPattern<mlir::arith::SubIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SubIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithDivSILowering
    : public mlir::OpConversionPattern<mlir::arith::DivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::DivSIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithRemSILowering
    : public mlir::OpConversionPattern<mlir::arith::RemSIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::RemSIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::RemSIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithShRSILowering
    : public mlir::OpConversionPattern<mlir::arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ShRSIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ShRSIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithShLILowering
    : public mlir::OpConversionPattern<mlir::arith::ShLIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ShLIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithOrILowering
    : public mlir::OpConversionPattern<mlir::arith::OrIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::OrIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(op, adaptor.getLhs(),
                                                    adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithXOrILowering
    : public mlir::OpConversionPattern<mlir::arith::XOrIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::XOrIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithDivUILowering
    : public mlir::OpConversionPattern<mlir::arith::DivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::DivUIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::DivUIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithRemUILowering
    : public mlir::OpConversionPattern<mlir::arith::RemUIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::RemUIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::RemUIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithShRUILowering
    : public mlir::OpConversionPattern<mlir::arith::ShRUIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ShRUIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ShRUIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
    return mlir::success();
  }
};

struct ArithSelectLowering
    : public mlir::OpConversionPattern<mlir::arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return mlir::success();
  }
};

struct ArithAddFLowering
    : public mlir::OpConversionPattern<mlir::arith::AddFOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AddFOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto addEnum = BinaryExpOperatorAttr::get(rewriter.getContext(),
                                              BinaryExpOperator::addOp);
    rewriter.replaceOpWithNewOp<BinaryExpOp>(op, adaptor.getLhs().getType(),
                                             addEnum, adaptor.getLhs(),
                                             adaptor.getRhs());
    return mlir::success();
  }
};

// Session L4 honest-divergence: GLU (`x2 * (1 + erf(...)) * 0.5`) generates
// `arith.mulf` on tensor<Nxf32> which had no existing lowering. The L4 spec
// stated `arith.mulf` was already shipped but it wasn't, so we add a mirror
// of `ArithAddFLowering` here to unblock AC.E1 without expanding op scope.
struct ArithMulFLowering
    : public mlir::OpConversionPattern<mlir::arith::MulFOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::MulFOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mulEnum = BinaryExpOperatorAttr::get(rewriter.getContext(),
                                              BinaryExpOperator::mulOp);
    rewriter.replaceOpWithNewOp<BinaryExpOp>(op, adaptor.getLhs().getType(),
                                             mulEnum, adaptor.getLhs(),
                                             adaptor.getRhs());
    return mlir::success();
  }
};

// Elementwise `arith.maxnumf` / `arith.maximumf` (`tl.maximum`) → per-thread
// metal.binary_exp maxOp. Mirrors ArithAddFLowering; needed by a loop-carried
// column-max accumulator (`acc = tl.maximum(acc, tile)`) that
// reassociateLoopCarriedAxis0Reduce turns into a scalar `max(s, reduce(...))`.
template <typename OpTy>
struct ArithMaxFLowering : public mlir::OpConversionPattern<OpTy> {
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto maxEnum = BinaryExpOperatorAttr::get(rewriter.getContext(),
                                              BinaryExpOperator::maxOp);
    rewriter.template replaceOpWithNewOp<BinaryExpOp>(
        op, adaptor.getLhs().getType(), maxEnum, adaptor.getLhs(),
        adaptor.getRhs());
    return mlir::success();
  }
};

// Elementwise i32 `arith.maxsi` / `arith.minsi` (`tl.maximum` / `tl.minimum`
// on int32) → the same SCALAR op on the per-thread operands. metal.binary_exp
// rejects signless i32, but ModuleTranslation emits scalar maxsi/minsi as MSL
// `max((int)a,(int)b)` (signed). Enables loop-carried i32 column-min/max after
// reassociateLoopCarriedAxis0Reduce.
template <typename OpTy>
struct ArithIntMinMaxLowering : public mlir::OpConversionPattern<OpTy> {
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<OpTy>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return mlir::success();
  }
};

// Scalarize a tensor `arith.sitofp` (`idx.to(tl.float32)` etc.). The scalar
// form is emitted by ModuleTranslation as the MSL constructor cast `T(x)`; here
// we just rebuild it on the converted scalar operand/result so the tensor op
// legalizes. Mirrors how the binary arith lowerings scalarize.
struct ArithSIToFPLowering
    : public mlir::OpConversionPattern<mlir::arith::SIToFPOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SIToFPOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy || mlir::isa<mlir::RankedTensorType>(resTy))
      return rewriter.notifyMatchFailure(op, "sitofp: result not scalarizable");
    rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(op, resTy,
                                                       adaptor.getIn());
    return mlir::success();
  }
};

// Scalarize a tensor `arith.extf` (float widen, e.g. fp16 load `.to(tl.float32)`)
// and `arith.truncf` (float narrow, e.g. store an fp32 result to an fp16 output).
// The MSL emitter spells both as a constructor cast `T(x)` (see the ExtFOp /
// TruncFOp case in translateValue). fp16-in / fp32-compute is the common shape
// for layer-norm / softmax / attention kernels.
struct ArithExtFLowering
    : public mlir::OpConversionPattern<mlir::arith::ExtFOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ExtFOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy || mlir::isa<mlir::RankedTensorType>(resTy))
      return rewriter.notifyMatchFailure(op, "extf: result not scalarizable");
    rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(op, resTy,
                                                     adaptor.getIn());
    return mlir::success();
  }
};

struct ArithTruncFLowering
    : public mlir::OpConversionPattern<mlir::arith::TruncFOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::TruncFOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy || mlir::isa<mlir::RankedTensorType>(resTy))
      return rewriter.notifyMatchFailure(op, "truncf: result not scalarizable");
    rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(op, resTy,
                                                       adaptor.getIn());
    return mlir::success();
  }
};

// Scalarize a tensor integer width cast: `arith.extui` (zero-extend, the shape
// Triton emits for `tl.cast(<i1 predicate>, tl.int32)` — the per-digit histogram
// idiom `tl.sum(tl.cast(digits == b, tl.int32))` in a radix sort), `arith.extsi`
// (sign-extend) and `arith.trunci` (narrow). The MSL emitter already spells all
// three as the C-style cast `(T)(x)` (see the ExtSIOp/ExtUIOp/TruncIOp case in
// translateValue), so — exactly like ArithExtFLowering / ArithTruncFLowering —
// this only has to rebuild the op on the converted scalar operand.
//
// i1 is emitted as MSL `bool`, so `(int)(pred)` yields 0/1 == zero-extension,
// which is correct for extui and for trunci-to-bool (C's bool conversion is
// `!= 0`, matching arith.trunci's low-bit semantics for the 0/1 values Triton
// produces). It is NOT correct for a SIGN-extending i1 -> iN: arith.extsi
// defines that as all-ones (-1), while `(int)(bool)` gives +1. Rather than emit
// a silently-wrong sign, reject that one case and let it stay a hard
// legalization error.
template <typename OpTy>
struct ArithIntCastLowering : public mlir::OpConversionPattern<OpTy> {
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = this->getTypeConverter()->convertType(op.getType());
    if (!resTy || mlir::isa<mlir::RankedTensorType>(resTy))
      return rewriter.notifyMatchFailure(op, "int cast: result not scalarizable");
    // Guard the i1 sign-extension mismatch described above.
    if (std::is_same<OpTy, mlir::arith::ExtSIOp>::value) {
      auto inTy = adaptor.getIn().getType();
      if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(inTy))
        if (intTy.getWidth() == 1)
          return rewriter.notifyMatchFailure(
              op, "extsi from i1: MSL (T)(bool) yields +1, not all-ones");
    }
    rewriter.template replaceOpWithNewOp<OpTy>(op, resTy, adaptor.getIn());
    return mlir::success();
  }
};

// Scalarize a tensor `arith.negf` (`-x`). ModuleTranslation emits scalar negf as
// `(-x)`; rebuild it on the converted scalar operand so the tensor op legalizes.
struct ArithNegFLowering
    : public mlir::OpConversionPattern<mlir::arith::NegFOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::NegFOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy || mlir::isa<mlir::RankedTensorType>(resTy))
      return rewriter.notifyMatchFailure(op, "negf: result not scalarizable");
    rewriter.replaceOpWithNewOp<mlir::arith::NegFOp>(op, resTy,
                                                     adaptor.getOperand());
    return mlir::success();
  }
};

// Wall 9 (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC1-AC2):
// `arith.subf` on rank-1 (and ranked) f32 tensors. Mirror of ArithAddFLowering
// with BinaryExpOperator::subOp.
struct ArithSubFLowering
    : public mlir::OpConversionPattern<mlir::arith::SubFOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SubFOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto subEnum = BinaryExpOperatorAttr::get(rewriter.getContext(),
                                              BinaryExpOperator::subOp);
    rewriter.replaceOpWithNewOp<BinaryExpOp>(op, adaptor.getLhs().getType(),
                                             subEnum, adaptor.getLhs(),
                                             adaptor.getRhs());
    return mlir::success();
  }
};

// Wall 12 (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC6-AC7):
// `arith.divf` on rank-1 (and ranked) f32 tensors. Mirror of ArithAddFLowering
// with BinaryExpOperator::divOp.
struct ArithDivFLowering
    : public mlir::OpConversionPattern<mlir::arith::DivFOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::DivFOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto divEnum = BinaryExpOperatorAttr::get(rewriter.getContext(),
                                              BinaryExpOperator::divOp);
    rewriter.replaceOpWithNewOp<BinaryExpOp>(op, adaptor.getLhs().getType(),
                                             divEnum, adaptor.getLhs(),
                                             adaptor.getRhs());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// math.sqrt / math.erf → metal.unary_exp (Session L4 transcendentals).
//
// Triton frontend `tl.sqrt` / `tl.erf` produce `math::SqrtOp` / `math::ErfOp`
// on fp32 tensors. We lower them to the existing `metal.unary_exp` with the
// `sqrtOp` / `erfOp` enum cases; MSL emission is `metal::precise::sqrt(...)`
// and `metal::erf(...)` respectively (no `metal::precise::erf` exists in MSL
// stdlib). fp32 only — non-f32 element types are rejected via `failure()`.
//===----------------------------------------------------------------------===//

struct MathSqrtLowering
    : public mlir::OpConversionPattern<mlir::math::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::math::SqrtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy || !resultTy.isF32())
      return mlir::failure();
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(),
                                          UnaryExpOperator::sqrtOp);
    rewriter.replaceOpWithNewOp<UnaryExpOp>(op, resultTy, attr,
                                            adaptor.getOperand());
    return mlir::success();
  }
};

struct MathErfLowering
    : public mlir::OpConversionPattern<mlir::math::ErfOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::math::ErfOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy || !resultTy.isF32())
      return mlir::failure();
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(),
                                          UnaryExpOperator::erfOp);
    rewriter.replaceOpWithNewOp<UnaryExpOp>(op, resultTy, attr,
                                            adaptor.getOperand());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// math.exp / math.log / math.rsqrt → metal.unary_exp (Session L4b transcendentals).
//
// Triton frontend `tl.exp` / `tl.log` / `tl.rsqrt` produce `math::ExpOp` /
// `math::LogOp` / `math::RsqrtOp` on fp32 tensors. We lower them to the
// existing `metal.unary_exp` with the `expOp` / `logOp` / `rsqrtOp` enum
// cases; MSL emission is `metal::precise::exp/log/rsqrt(...)`. fp32 only —
// non-f32 element types are rejected via `failure()`.
//===----------------------------------------------------------------------===//

struct MathExpLowering
    : public mlir::OpConversionPattern<mlir::math::ExpOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::math::ExpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy || !resultTy.isF32())
      return mlir::failure();
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(),
                                          UnaryExpOperator::expOp);
    rewriter.replaceOpWithNewOp<UnaryExpOp>(op, resultTy, attr,
                                            adaptor.getOperand());
    return mlir::success();
  }
};

// `tl.sin` / `tl.cos` (RoPE in the adder transformer) → math.{sin,cos} on f32
// tensors → metal.unary_exp {sin,cos}Op → MSL metal::precise::{sin,cos}.
struct MathSinLowering
    : public mlir::OpConversionPattern<mlir::math::SinOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::math::SinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy || !resultTy.isF32())
      return mlir::failure();
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(),
                                          UnaryExpOperator::sinOp);
    rewriter.replaceOpWithNewOp<UnaryExpOp>(op, resultTy, attr,
                                            adaptor.getOperand());
    return mlir::success();
  }
};

struct MathCosLowering
    : public mlir::OpConversionPattern<mlir::math::CosOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::math::CosOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy || !resultTy.isF32())
      return mlir::failure();
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(),
                                          UnaryExpOperator::cosOp);
    rewriter.replaceOpWithNewOp<UnaryExpOp>(op, resultTy, attr,
                                            adaptor.getOperand());
    return mlir::success();
  }
};

struct MathLogLowering
    : public mlir::OpConversionPattern<mlir::math::LogOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::math::LogOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy || !resultTy.isF32())
      return mlir::failure();
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(),
                                          UnaryExpOperator::logOp);
    rewriter.replaceOpWithNewOp<UnaryExpOp>(op, resultTy, attr,
                                            adaptor.getOperand());
    return mlir::success();
  }
};

struct MathRsqrtLowering
    : public mlir::OpConversionPattern<mlir::math::RsqrtOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::math::RsqrtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy || !resultTy.isF32())
      return mlir::failure();
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(),
                                          UnaryExpOperator::rsqrtOp);
    rewriter.replaceOpWithNewOp<UnaryExpOp>(op, resultTy, attr,
                                            adaptor.getOperand());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.reduce → sequential row-scan body (Session L3a, axis=1, f32/i32).
//
// Lowers an in-envelope `tt.reduce` on `tensor<MxNxT>` (T ∈ {f32, i32},
// axis=1, combine ∈ {arith.addf, arith.addi}) into the threadgroup-shared
// row-scan algorithm:
//
//   threadgroup T buf[M*N];
//   buf[tid] = input;
//   threadgroup_barrier(mem_flags::mem_threadgroup);
//   row = tid / N; base = row * N;
//   acc = buf[base+0] + buf[base+1] + ... + buf[base+N-1];
//
// The L3 pre-pass has already validated the envelope (rank=2, axis=1,
// dtype, combine op, static shape, ≤32 KiB tile). The f32 scan is rerolled
// into `scf.for` + a single f32 `iter_arg` via Wall 15's translator (cf.
// `ModuleTranslation::translate(scf::ForOp)`); the i32 (ui32-storage) path
// keeps the fully-unrolled emission.
// See `.omc/specs/deep-interview-leet-triton-l3a-reduce-body-axis1.md`.
//===----------------------------------------------------------------------===//

// Forward declaration for the Synthesis-path spt-fold sub-branch (B2.3).
// `findBaseMemref` is defined later in the file alongside `LoadLowering`.
static mlir::Value findBaseMemref(mlir::Value origPtrVal,
                                  mlir::ConversionPatternRewriter &rewriter);

// Wall 13 fix (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC8):
// Walk the original ptr chain (tt.addptr → tt.splat → tt.addptr → block arg)
// and accumulate the i32 SCALAR offsets of any tt.addptr nodes traversed.
// For the softmax tutorial's shape `tt.addptr(tt.splat(tt.addptr(funcArg,
// row*stride)), col_offsets)`, the OUTER tensor-offset is consumed by the
// per-k iteration; this helper returns the INNER scalar offset
// (row*stride) which must be added to the per-k index for correctness.
// Returns null if no scalar offsets are present (the simple direct-load
// shape).
static mlir::Value accumulateScalarAddPtrOffsets(
    mlir::Value origPtrVal, mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc);

// Same, but also walks tt.broadcast / tt.expand_dims so the per-program scalar
// term of a 2D tile address is reachable. See the definition for why the rank-2
// axis=1 reduce needs it.
static mlir::Value accumulateScalarAddPtrOffsetsThroughShape(
    mlir::Value origPtrVal, mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc);

// Rank-1 tt.reduce lowering. Threadgroup-buffer parallel-tree-reduce body.
// BLOCK <= tpb regimes use the adaptor scalar (BLOCK < tpb uses identity-fill
// tail; BLOCK == tpb has no tail). BLOCK > tpb uses the spt-fold path: walks
// back to the producing tt.load, emits `spt` inline `metal.get_element` calls,
// folds via BinaryExpOp, then enters the same butterfly.
//
// Combine dispatch (Phase B):
//   - arith.addf  → arith.AddFOp (scalar f32 add)
//   - arith.addi  → arith.AddIOp (scalar i32 add, on ui32 storage type)
//   - arith.maximumf → metal.BinaryExpOp maxOp (lowers to MSL max(a,b))
//
// Identity constants (for BLOCK < tpb tail fill):
//   - sum:  0.0 (f32) / 0 (i32)
//   - maxf: -inf via bit pattern 0xff800000

// Forward declaration; definition at TritonGPUToMetal.cpp:2706.
// Wall 7 (.omc/plans/tutorial02-wall7-masked-spt-reduce-consensus.md Step 0):
// `lowerRank1Reduce` (line 1399) is lexically above the definition so the
// masked B2.3 path needs a forward decl to extract `other` splat constants.
static std::optional<mlir::TypedAttr>
extractSplatConstantAttr(mlir::Value other);

// Forward declarations for the W-B rich-cone rank-1 reduce path: the general
// per-element cone evaluator + its dry-run support predicate are defined below
// (near the rank-2 reduce), but `lowerRank1Reduce` (above them) invokes both
// when the single-load Wall-11 walker fails on a computed cone.
// Pool of shared (inbuf, scanbuf) threadgroup buffers, keyed by the tt.scan
// that should use them; populated by `preprocessScanBuffers` (defined below,
// where the rationale lives) and consumed by `ScanLowering`.
using ScanBufPool =
    llvm::DenseMap<mlir::Operation *, std::pair<mlir::Value, mlir::Value>>;

static mlir::Value evalRank1ValueAt(mlir::Value v, mlir::Value idxVal,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Location loc, int depth);
static bool rank1ConeSupported(mlir::Value v, int depth);

//===----------------------------------------------------------------------===//
// Wall 11 (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC4-AC5):
// recursive walker for rank-1 reduce B2.3 source chains. Permits the chain
// `tt.load → arith.{addf,subf,mulf,divf} → math.exp → tt.reduce` with
// `tt.splat`-broadcast scalar operands on binary ops. Anything else is
// rejected via notifyMatchFailure.
//===----------------------------------------------------------------------===//

namespace {

struct Wall11ChainStep {
  enum class Kind { Add, Sub, Mul, Div, Exp };
  Kind kind;
  bool unary = false;           // true for Exp; false for Add/Sub/Mul/Div
  bool splatOnRhs = false;      // binary: running-value LHS, splat RHS
  mlir::Value splatScalar;      // binary: pre-converted splat scalar src
};

// Architect F2 (stale splat), F5 (splat dominance), F7 (rank-1 guard at
// every depth), F9 (failure semantics: walker calls notifyMatchFailure and
// the caller propagates `failure()` verbatim). Critic D2 (two-splat rejected),
// D6 (cvt-mid-chain rejected before unknown-op fallback).
static mlir::LogicalResult walkBackThroughElementwiseChain(
    mlir::Value v, int depth,
    llvm::SmallVectorImpl<Wall11ChainStep> &chain,
    mlir::triton::LoadOp &outLoad, mlir::Operation *reduceOp,
    mlir::ConversionPatternRewriter &rewriter,
    mlir::DominanceInfo &dominance) {
  // Depth guard (Critic D2 spec; mitigates R3).
  if (depth > 8)
    return rewriter.notifyMatchFailure(
        reduceOp, "rank-1 reduce B2.3: chain depth > 8");

  // Architect F7: every chain producer must be rank-1.
  auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!rtt || rtt.getRank() != 1)
    return rewriter.notifyMatchFailure(
        reduceOp, "rank-1 reduce B2.3: non-rank-1 producer in chain");

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return rewriter.notifyMatchFailure(
        reduceOp,
        "rank-1 reduce B2.3: chain value has no defining op (block arg)");

  // Terminator: tt.load.
  if (auto load = mlir::dyn_cast<mlir::triton::LoadOp>(def)) {
    outLoad = load;
    return mlir::success();
  }

  // Critic D6: cvt mid-chain emits a cvt-specific diagnostic before the
  // generic unknown-op fallback.
  if (mlir::isa<mlir::triton::gpu::ConvertLayoutOp>(def))
    return rewriter.notifyMatchFailure(
        reduceOp, "rank-1 reduce B2.3: cvt mid-chain not supported "
                  "(see spec follow-up F1)");

  auto doBinary = [&](mlir::Value lhs, mlir::Value rhs,
                      Wall11ChainStep::Kind kind) -> mlir::LogicalResult {
    auto lhsSplat = lhs.getDefiningOp<mlir::triton::SplatOp>();
    auto rhsSplat = rhs.getDefiningOp<mlir::triton::SplatOp>();
    // Critic D2: two-splat binary has no per-iter value.
    if (lhsSplat && rhsSplat)
      return rewriter.notifyMatchFailure(
          reduceOp,
          "rank-1 reduce B2.3: binary op with two splat operands has no "
          "per-iter value");
    if (!lhsSplat && !rhsSplat)
      return rewriter.notifyMatchFailure(
          reduceOp,
          "rank-1 reduce B2.3: binary op with two tensor operands (no splat) "
          "not supported by this walker");

    Wall11ChainStep step;
    step.unary = false;
    step.kind = kind;
    mlir::Value runningSide;
    mlir::triton::SplatOp splat;
    if (rhsSplat) {
      splat = rhsSplat;
      step.splatOnRhs = true;
      runningSide = lhs;
    } else {
      splat = lhsSplat;
      step.splatOnRhs = false;
      runningSide = rhs;
    }

    // Architect F2: stale splat check (splat may have been rewritten by
    // SplatLowering before this walker runs).
    if (!splat.getOperation()->getBlock())
      return rewriter.notifyMatchFailure(
          reduceOp, "rank-1 reduce B2.3: splat already converted");

    step.splatScalar = splat.getSrc();

    // Architect F5: dominance — splat's scalar source must dominate the
    // reduce's insertion point.
    if (mlir::Operation *scalarDef = step.splatScalar.getDefiningOp()) {
      if (!dominance.dominates(scalarDef, reduceOp))
        return rewriter.notifyMatchFailure(
            reduceOp, "rank-1 reduce B2.3: splat operand dominance");
    }

    // Recurse on the running side; on success, push the step so `chain` is
    // in load-to-reduce order.
    if (mlir::failed(walkBackThroughElementwiseChain(runningSide, depth + 1,
                                                      chain, outLoad, reduceOp,
                                                      rewriter, dominance)))
      return mlir::failure();
    chain.push_back(step);
    return mlir::success();
  };

  auto doUnary = [&](mlir::Value operand,
                     Wall11ChainStep::Kind kind) -> mlir::LogicalResult {
    if (mlir::failed(walkBackThroughElementwiseChain(operand, depth + 1, chain,
                                                      outLoad, reduceOp,
                                                      rewriter, dominance)))
      return mlir::failure();
    Wall11ChainStep step;
    step.unary = true;
    step.kind = kind;
    chain.push_back(step);
    return mlir::success();
  };

  if (auto addOp = mlir::dyn_cast<mlir::arith::AddFOp>(def))
    return doBinary(addOp.getLhs(), addOp.getRhs(),
                    Wall11ChainStep::Kind::Add);
  if (auto subOp = mlir::dyn_cast<mlir::arith::SubFOp>(def))
    return doBinary(subOp.getLhs(), subOp.getRhs(),
                    Wall11ChainStep::Kind::Sub);
  if (auto mulOp = mlir::dyn_cast<mlir::arith::MulFOp>(def))
    return doBinary(mulOp.getLhs(), mulOp.getRhs(),
                    Wall11ChainStep::Kind::Mul);
  if (auto divOp = mlir::dyn_cast<mlir::arith::DivFOp>(def))
    return doBinary(divOp.getLhs(), divOp.getRhs(),
                    Wall11ChainStep::Kind::Div);
  if (auto expOp = mlir::dyn_cast<mlir::math::ExpOp>(def))
    return doUnary(expOp.getOperand(), Wall11ChainStep::Kind::Exp);

  // Architect F9: unknown producer — AC4-exact diagnostic substring.
  return rewriter.notifyMatchFailure(
      reduceOp,
      llvm::formatv("rank-1 reduce B2.3: unsupported producer in elementwise "
                    "chain: {0}",
                    def->getName().getStringRef())
          .str());
}

// Per-spt-idx scalar re-emission. Walks `chain` in load-to-reduce order
// (as produced by the walker above), starting from `loadedScalar`, emitting
// one BinaryExpOp (Add/Sub/Mul/Div) or UnaryExpOp (Exp) per step. Architect
// F4 / Critic D4: (lhs, rhs) order is preserved so non-commutative subf/divf
// emit operands in the original-IR order.
static mlir::Value emitScalarChain(
    llvm::ArrayRef<Wall11ChainStep> chain, mlir::Value loadedScalar,
    int /*spt_idx*/, mlir::Location loc,
    mlir::ConversionPatternRewriter &rewriter) {
  mlir::Value running = loadedScalar;
  for (const auto &step : chain) {
    if (step.unary) {
      auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(),
                                            UnaryExpOperator::expOp);
      running = UnaryExpOp::create(rewriter, loc, running.getType(), attr,
                                   running)
                    .getResult();
      continue;
    }
    mlir::Value lhs = step.splatOnRhs ? running : step.splatScalar;
    mlir::Value rhs = step.splatOnRhs ? step.splatScalar : running;
    BinaryExpOperator opKind = BinaryExpOperator::addOp;
    switch (step.kind) {
      case Wall11ChainStep::Kind::Add: opKind = BinaryExpOperator::addOp; break;
      case Wall11ChainStep::Kind::Sub: opKind = BinaryExpOperator::subOp; break;
      case Wall11ChainStep::Kind::Mul: opKind = BinaryExpOperator::mulOp; break;
      case Wall11ChainStep::Kind::Div: opKind = BinaryExpOperator::divOp; break;
      case Wall11ChainStep::Kind::Exp:
        // Unreachable: Exp is unary; handled above.
        break;
    }
    auto attr = BinaryExpOperatorAttr::get(rewriter.getContext(), opKind);
    running = BinaryExpOp::create(rewriter, loc, lhs.getType(), attr, lhs, rhs)
                  .getResult();
  }
  return running;
}

} // namespace

static mlir::LogicalResult
lowerRank1Reduce(mlir::triton::ReduceOp op,
                 typename mlir::OpConversionPattern<
                     mlir::triton::ReduceOp>::OpAdaptor adaptor,
                 mlir::ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(
      op.getSrcs().front().getType());
  if (!rtt || rtt.getRank() != 1)
    return mlir::failure();
  if (rtt.isDynamicDim(0))
    return mlir::failure();
  if (op.getAxis() != 0)
    return mlir::failure();

  mlir::Type elemTy = rtt.getElementType();
  bool isF32 = elemTy.isF32();
  bool isI32 = elemTy.isInteger(32);
  if (!isF32 && !isI32)
    return mlir::failure();

  // Identify the combine op (the pre-pass already validated, but be defensive).
  mlir::Operation *combine = nullptr;
  if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
    for (auto &nested : op->getRegion(0).front()) {
      if (mlir::isa<mlir::triton::ReduceReturnOp>(nested))
        continue;
      combine = &nested;
      break;
    }
  }
  if (!combine)
    return mlir::failure();
  bool isAddF = mlir::isa<mlir::arith::AddFOp>(combine);
  bool isAddI = mlir::isa<mlir::arith::AddIOp>(combine);
  // Accept both arith.maximumf (IEEE maximum) and arith.maxnumf (IEEE maxNum).
  // Triton's tl.max emits arith.maxnumf; both map to MSL max(a,b).
  bool isMaxF = mlir::isa<mlir::arith::MaximumFOp>(combine) ||
                mlir::isa<mlir::arith::MaxNumFOp>(combine);
  // Triton's tl.max on signed i32 emits arith.maxsi (signed max). Lowered via
  // a si32-typed butterfly/accumulator so MSL's `max` performs a SIGNED
  // comparison (see storeTy and the BinaryExpOp maxOp translator).
  bool isMaxI = mlir::isa<mlir::arith::MaxSIOp>(combine);
  // Triton's tl.min on signed i32 emits arith.minsi (signed min). Mirrors
  // isMaxI: si32-typed accumulator + BinaryExpOp minOp, identity INT32_MAX.
  bool isMinI = mlir::isa<mlir::arith::MinSIOp>(combine);
  if (isF32 && !(isAddF || isMaxF))
    return mlir::failure();
  if (isI32 && !(isAddI || isMaxI || isMinI))
    return mlir::failure();

  // Derive tpb from the blocked encoding directly (do NOT use
  // tileFromTensor.elemPerThread — it's 0 in the BLOCK < tpb regime).
  auto srcBlocked = mlir::dyn_cast_or_null<
      mlir::triton::gpu::BlockedEncodingAttr>(rtt.getEncoding());
  if (!srcBlocked)
    return mlir::failure();
  int64_t tpb = 1;
  for (auto t : srcBlocked.getThreadsPerWarp()) tpb *= t;
  for (auto w : srcBlocked.getWarpsPerCTA()) tpb *= w;
  if (tpb <= 0 || (tpb & (tpb - 1)) != 0)
    return mlir::failure(); // require power-of-two tpb for the butterfly

  int64_t BLOCK = rtt.getDimSize(0);
  if (BLOCK <= 0)
    return mlir::failure();

  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
  auto si32 = rewriter.getIntegerType(32, /*isSigned=*/true);
  auto i32 = rewriter.getI32Type();
  // storeTy: the metal buffer/element ops reject SIGNLESS i32, so i32 is
  // routed through a signed Metal_Type. i32 add uses ui32 (bit-preserving
  // two's-complement accumulation); i32 signed-max uses si32 so MSL emits
  // `int32_t` and `max(...)` compares as signed. f32 passes through.
  mlir::Type storeTy = elemTy;
  if (isI32)
    storeTy = (isMaxI || isMinI) ? mlir::Type(si32) : mlir::Type(ui32);

  // Combine dispatch helper: emits metal.BinaryExpOp for storeTy values.
  // The MSL translator handles metal.BinaryExpOp for all scalar types;
  // raw arith.addf / arith.addi are NOT handled by ModuleTranslation (only
  // tensor-typed arith is lowered to BinaryExpOp by ArithAddFLowering /
  // ArithAddILowering). We must use metal.BinaryExpOp directly here, mirroring
  // the rank-2 reduce body's approach (see :1530, :1729, :1739).
  auto emitCombine = [&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
    if (isAddF || isAddI) {
      auto addEnum = BinaryExpOperatorAttr::get(rewriter.getContext(),
                                                BinaryExpOperator::addOp);
      return BinaryExpOp::create(rewriter, loc, lhs.getType(), addEnum, lhs,
                                 rhs)
          .getResult();
    }
    // arith.maximumf / arith.maxsi → metal.BinaryExpOp maxOp (MSL max(a, b));
    // arith.minsi → minOp (MSL min(a, b)).
    auto op = isMinI ? BinaryExpOperator::minOp : BinaryExpOperator::maxOp;
    auto opEnum = BinaryExpOperatorAttr::get(rewriter.getContext(), op);
    return BinaryExpOp::create(rewriter, loc, lhs.getType(), opEnum, lhs, rhs)
        .getResult();
  };

  // Combine identity, materialised as a storeTy-typed Value. Integer
  // identities are built as a SIGNLESS arith.constant (the verifier rejects
  // signed/unsigned-typed integer constants — 'arith.constant op integer
  // return type must be signless') and then bridged to storeTy (ui32 for add,
  // si32 for signed-max) via unrealized_conversion_cast.
  auto buildIdentityVal = [&]() -> mlir::Value {
    if (isF32 && isAddF)
      return mlir::arith::ConstantOp::create(
                 rewriter, loc, rewriter.getF32FloatAttr(0.0f))
          .getResult();
    if (isF32 && isMaxF) {
      // -FLT_MAX (most-negative finite f32): the MSL float-constant emitter
      // can't render -inf, and max(x, -FLT_MAX) == x for every finite x.
      return mlir::arith::ConstantOp::create(
                 rewriter, loc,
                 rewriter.getF32FloatAttr(-std::numeric_limits<float>::max()))
          .getResult();
    }
    // i32 add → 0; i32 signed-max → INT32_MIN (max(x, INT_MIN) == x);
    // i32 signed-min → INT32_MAX (min(x, INT_MAX) == x).
    int32_t identVal = 0;
    if (isMaxI)
      identVal = std::numeric_limits<int32_t>::min();
    else if (isMinI)
      identVal = std::numeric_limits<int32_t>::max();
    auto z = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(identVal));
    return mlir::UnrealizedConversionCastOp::create(
               rewriter, loc, mlir::TypeRange{storeTy},
               mlir::ValueRange{z.getResult()})
        .getResult(0);
  };

  // B2.3 (spt-fold path): BLOCK > tpb, with the
  // tritongpu-propagate-coalesced-layouts pre-pass having promoted the rank-1
  // source layout to sizePerThread=[spt] where BLOCK == tpb*spt.
  //
  // Strategy: walk back to the producing `tt.load` and re-emit `spt` direct
  // `metal.get_element` calls inline (idx = tid*spt + spt_idx, the contiguous
  // formula for spt>1 in the BLOCK == tpb*spt case where the tile-loop trip
  // count E = BLOCK/(tpb*spt) = 1). Fold the `spt` scalars in place via the
  // combine BinaryExpOp, then fall through to the same B2.4 butterfly that
  // the BLOCK <= tpb regimes use. No TileInfo.spt field, no LoadLowering
  // 1:N change, no tileFromTensor change — see
  // .omc/plans/option-beta-spt-load-lowering.md.
  mlir::Value inputScalarPT;
  if (BLOCK > tpb) {
    int64_t spt = srcBlocked.getSizePerThread()[0];
    // Wall 8 (.omc/plans/tutorial02-wall8-spt1-direct.md): when spt == 1 and
    // BLOCK > tpb, each thread directly loads E = BLOCK/tpb elements at
    // stride tpb (idx = tid + k*tpb). Independent of Wall 6's cvt survival;
    // robust to remove-layout-conversions canonicalizing the cvt away in
    // scf.for-wrapped multi-use kernels (the softmax tutorial path).
    // When spt > 1, the existing path emits stride-1 within thread
    // (idx = tid*spt + k); E equals spt.
    // Wall 14 (.omc/plans/tutorial02-wall14-block-gt-tpb-consensus.md AC1):
    // bounded-unroll envelope guard, generalized to cover spt > 1 cyclic
    // tile addressing. Each thread always owns E = BLOCK/tpb elements; the
    // address formula now decomposes k into (outer_tile, inner_idx) when
    // BLOCK > tpb*spt:
    //   outer = k / spt;  inner = k % spt
    //   addr = outer * (tpb*spt) + tid*spt + inner
    // The existing accept cases stay byte-identical because:
    //   - spt1Direct (spt=1): outer = k, inner = 0 ⇒ addr = k*tpb + tid
    //   - non-spt1Direct, BLOCK == tpb*spt: E = spt, k < spt ⇒ outer = 0,
    //     inner = k ⇒ addr = tid*spt + k (matches the existing formula at
    //     TritonGPUToMetal.cpp:1959 verbatim).
    // Tutorial02 hits the new cyclic case at BLOCK > tpb*spt (typically
    // sizePerThread=[4] capped, BLOCK ∈ {2048, 4096, ..., 16384}, tpb=256).
    bool spt1Direct = (spt == 1);
    int64_t E = BLOCK / tpb;
    if (BLOCK != tpb * E)
      return rewriter.notifyMatchFailure(
          op, "rank-1 reduce: BLOCK not divisible by tpb (cyclic tile "
              "addressing requires BLOCK = N * tpb)");
    if (E < 1 || E > 64 || (E & (E - 1)) != 0)
      return rewriter.notifyMatchFailure(
          op, "rank-1 reduce E = BLOCK/tpb outside supported envelope "
              "[1, 64] or not a power of 2 (tutorial02 autotune caps "
              "BLOCK <= 16384 at tpb = num_warps*warp_size = 8*32 = 256 "
              "=> E <= 64; see "
              ".omc/plans/tutorial02-wall14-block-gt-tpb-consensus.md AC1)");
    if (!spt1Direct && (spt & (spt - 1)) != 0)
      return rewriter.notifyMatchFailure(
          op, "rank-1 reduce spt-fold: spt > 1 must be a power of 2 for "
              "cyclic tile decomposition");

    // Walk back to the producing tt.load, optionally through one cvt hop
    // inserted by tritongpu-propagate-coalesced-layouts (Wall 6 —
    // .omc/plans/tutorial02-wall6-spt1-reduce-consensus.md, SC3 / spec AC5).
    mlir::Value reduceSrc = op.getSrcs().front();
    if (auto cvt = reduceSrc.getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>()) {
      auto cvtSrcTy =
          mlir::dyn_cast<mlir::RankedTensorType>(cvt.getSrc().getType());
      auto cvtDstTy =
          mlir::dyn_cast<mlir::RankedTensorType>(cvt.getType());
      if (!cvtSrcTy || !cvtDstTy ||
          cvtSrcTy.getShape() != cvtDstTy.getShape())
        return rewriter.notifyMatchFailure(
            op, "rank-1 reduce B2.3 cvt-walkthrough: cvt source shape "
                "mismatch (see "
                ".omc/specs/deep-interview-tutorial02-wall6-spt1-reduce.md "
                "AC5)");
      reduceSrc = cvt.getSrc();
    }
    // Wall 11 (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC4):
    // walk back through whitelisted elementwise ops to the producing tt.load.
    // Empty chain = trivial reduce<-tt.load case → Wall 7/8 behavior preserved
    // bit-for-bit. Architect F9: walker issues notifyMatchFailure on its own
    // rejection paths; the caller propagates failure() verbatim.
    mlir::DominanceInfo dominance(op->getParentOp());
    llvm::SmallVector<Wall11ChainStep, 4> chain;
    mlir::triton::LoadOp loadOp;
    // W-B: computed cones (multi-load / select / cmp / andi / make_range — e.g.
    // speculative decoding's `tl.sum(where(mask, where(q>p, q-p, 0), 0))`) are
    // NOT single-load chains. When the Wall-11 walker fails, fall back to the
    // general per-element cone evaluator `evalRank1ValueAt`, which re-derives
    // each logical element from persistent leaves (each load addressed + masked
    // independently). Validate the whole cone up front so an unsupported one is
    // a clean notifyMatchFailure rather than a mid-emission abort.
    bool richCone = false;
    if (mlir::failed(walkBackThroughElementwiseChain(
            reduceSrc, /*depth=*/0, chain, loadOp, op, rewriter, dominance))) {
      if (!rank1ConeSupported(reduceSrc, 0))
        return mlir::failure(); // walker already emitted a diagnostic
      richCone = true;
    }
    // Wall 7 (.omc/plans/tutorial02-wall7-masked-spt-reduce-consensus.md):
    // masked tt.load extraction. Canonical mask shape only: cmpi slt
    // (make_range start=0) (tt.splat scalar). Other shapes degrade to
    // notifyMatchFailure preserving the original wall-6 reject semantics.
    // TODO(Wall 8): tl.sum on arith.divf chain — input is non-load, current
    // 'src not produced by tt.load' reject at line ~1525 fires. See
    // .omc/specs/deep-interview-tutorial02-wall7-masked-spt-reduce.md §Risks.
    mlir::Value maskBoundN;       // i32 scalar; null = unmasked
    mlir::TypedAttr otherAttrPre; // populated only if user provided `other`
    if (!richCone && loadOp.getMask()) {
      mlir::Value maskVal = loadOp.getMask();
      // Mirror Wall 6 cvt-walkthrough for the mask path: propagate-coalesced
      // may have inserted an spt=1 -> spt=N cvt on the mask alongside the
      // value path.
      if (auto cvt =
              maskVal.getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>()) {
        auto sTy =
            mlir::dyn_cast<mlir::RankedTensorType>(cvt.getSrc().getType());
        auto dTy = mlir::dyn_cast<mlir::RankedTensorType>(cvt.getType());
        if (!sTy || !dTy || sTy.getShape() != dTy.getShape())
          return rewriter.notifyMatchFailure(
              op, "rank-1 reduce B2.3 masked: mask cvt shape mismatch");
        maskVal = cvt.getSrc();
      }
      auto cmp = maskVal.getDefiningOp<mlir::arith::CmpIOp>();
      if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::slt)
        return rewriter.notifyMatchFailure(
            op,
            "rank-1 reduce B2.3 masked: mask not cmpi slt "
            "(see .omc/specs/deep-interview-tutorial02-wall7-masked-spt-reduce"
            ".md AC2)");
      auto range =
          cmp.getLhs().getDefiningOp<mlir::triton::MakeRangeOp>();
      if (!range || range.getStart() != 0)
        return rewriter.notifyMatchFailure(
            op,
            "rank-1 reduce B2.3 masked: mask lhs not make_range(start=0)");
      auto splat = cmp.getRhs().getDefiningOp<mlir::triton::SplatOp>();
      if (!splat)
        return rewriter.notifyMatchFailure(
            op, "rank-1 reduce B2.3 masked: mask rhs not tt.splat");
      maskBoundN = splat.getSrc();
      if (loadOp.getOther()) {
        auto opt = extractSplatConstantAttr(loadOp.getOther());
        if (!opt)
          return rewriter.notifyMatchFailure(
              op, "rank-1 reduce B2.3 masked: `other` not splat-constant");
        otherAttrPre = *opt;
      }
    }
    mlir::Value memref;
    if (!richCone) {
      if (!loadOp.getPtr().getDefiningOp<mlir::triton::AddPtrOp>())
        return rewriter.notifyMatchFailure(
            op, "rank-1 reduce src tt.load missing tt.addptr");
      memref = findBaseMemref(loadOp.getPtr(), rewriter);
      if (!memref)
        return rewriter.notifyMatchFailure(
            op, "rank-1 reduce src tt.load: base memref not found");
    }

    // Emit `spt` scalar element loads at idx = tid*spt + spt_idx.
    // Wall 13 fix (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC8):
    // compute LOCAL thread id = global_tid - tgid * tpb. metal.thread_id "x"
    // is [[thread_position_in_grid]] (GLOBAL across all threadgroups). For
    // multi-program launches (tutorial02 sets num_programs = NUM_SM * occupancy),
    // each threadgroup must reduce its own row's columns independently;
    // using global tid here would make each threadgroup's threads read a
    // tgid-shifted window of the row, producing wrong reduces.
    mlir::Value tidGlobalUI32 =
        ThreadIdOp::create(rewriter, loc, ui32,
                            rewriter.getStringAttr("x"))
            .getResult();
    mlir::Value tidGlobalI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tidGlobalUI32})
            .getResult(0);
    mlir::Value tgGlobalUI32 =
        ThreadgroupIdOp::create(rewriter, loc, ui32,
                                  rewriter.getStringAttr("x"))
            .getResult();
    mlir::Value tgGlobalI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tgGlobalUI32})
            .getResult(0);
    auto cTpb = mlir::arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
    mlir::Value tgOffsetI32 = mlir::arith::MulIOp::create(
                                  rewriter, loc, tgGlobalI32,
                                  cTpb.getResult())
                                  .getResult();
    mlir::Value tidI32Local = mlir::arith::SubIOp::create(
                                  rewriter, loc, tidGlobalI32, tgOffsetI32)
                                  .getResult();
    // Per Wall 8: when spt1Direct, the per-k offset within thread is k*tpb
    // (stride-tpb), so we don't multiply tid by spt (which would be 1). When
    // spt > 1, the original `tid*spt` precompute is reused for each k.
    mlir::Value tidScaled;
    if (!spt1Direct) {
      auto cSpt = mlir::arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getI32IntegerAttr(static_cast<int32_t>(spt)));
      tidScaled = mlir::arith::MulIOp::create(rewriter, loc, tidI32Local,
                                              cSpt.getResult())
                      .getResult();
    } else {
      tidScaled = tidI32Local;
    }
    mlir::Type loadEltTy;
    mlir::Value scalarOff;
    if (!richCone) {
      loadEltTy = mlir::cast<MetalMemRefType>(memref.getType()).getType();
      // Wall 13 fix: accumulate any scalar tt.addptr chain offsets (e.g.
      // row_idx*input_row_stride) into a loop-invariant scalar `scalarOff`.
      // The mask check below uses the column-only `idxI32`; the memref index
      // `addrUI32` adds `scalarOff` so the per-row addressing is correct for
      // softmax-style kernels.
      scalarOff = accumulateScalarAddPtrOffsets(loadOp.getPtr(), rewriter, loc);
    }
    // Wall 15: re-roll the Wall-14 per-k unroll into a single scf.for + f32
    // iter_arg accumulator. ModuleTranslation::translate(scf::ForOp)
    // emits `float v<accIdx> = init;` BEFORE the for line and
    // `v<accIdx> = yielded;` at the matching scf.yield, so MSL emission
    // is O(1) in E (vs O(E) under the prior unroll). Body ordering
    // (Wall 11 invariant — MUST stay):
    //   cyclic-tile arith → scf.if masked load → type-bridge →
    //   emitScalarChain → arith.select(condK, chainElt, identity) →
    //   emitCombine → scf.yield
    // Combiner-identity init; a storeTy-typed Value. f32 reduces emit
    // `float vN = init;` via the Wall 15 translator; i32 reduces emit
    // `int32_t`/`uint32_t vN = init;` via the same (now type-generic) path.
    // Integer identities route through a signless constant + cast so the
    // arith.constant verifier (signless-only) is satisfied.
    // Wall 15 + multi-accumulator (metal-multiacc-reduce-plan.md): re-roll the
    // per-k unroll into an scf.for. When E >= 8, use K=8 independent iter_arg
    // accumulators to break the serial combine dependency chain (ILP), which
    // Phase 0 measured at ~1.3-3x on the low-occupancy single-threadgroup
    // reduce the backend emits. The combiner is associative (add/max), so K-way
    // reassociation is safe: bit-exact for i32, within test tolerance for f32.
    // The loop steps by K; iteration j folds flat elements k = j+0..j+K-1 into
    // accumulators 0..K-1; a balanced tree-combine then reduces the K
    // accumulators to one scalar. K=1 (E<8) is byte-identical to the prior
    // single-iter_arg emission.
    int64_t K = (E >= 8) ? 8 : 1; // E is power-of-2 in [1,64]; K divides E.

    // Hoist the masked `other` value + its validation out of the per-element
    // body (loop-invariant) so the failure path can abort the whole lowering.
    // Building the attribute creates no IR ops, so K=1 emission is unchanged.
    mlir::TypedAttr elseAttr;
    if (maskBoundN) {
      elseAttr = otherAttrPre;
      if (!elseAttr) {
        if (mlir::isa<mlir::FloatType>(loadEltTy))
          elseAttr = rewriter.getFloatAttr(loadEltTy, 0.0);
        else
          elseAttr = mlir::cast<mlir::TypedAttr>(
              rewriter.getIntegerAttr(loadEltTy, 0));
      } else if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(elseAttr)) {
        if (fa.getType() != loadEltTy)
          return rewriter.notifyMatchFailure(
              op, "rank-1 reduce B2.3 masked: other type mismatches load "
                  "elt");
      }
    }

    // Per-element partial: given a flat element index `kFlatI32` (0..E-1) and an
    // accumulator `acc`, emit the masked/unmasked load + Wall 11 chain and
    // return combine(acc, elt). Loop-invariant addressing state captured by ref.
    auto buildPartial = [&](mlir::Value kFlatI32,
                            mlir::Value acc) -> mlir::Value {
      // Wall 14 cyclic-tile decomposition lifted to runtime arith:
      //   spt1Direct (spt=1): kOffset = k*tpb
      //   else: kOffset = (k/spt)*(tpb*spt) + (k%spt)
      mlir::Value kOffsetI32;
      if (spt1Direct) {
        auto cTpb = mlir::arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
        kOffsetI32 = mlir::arith::MulIOp::create(
                         rewriter, loc, kFlatI32, cTpb.getResult())
                         .getResult();
      } else {
        auto cSpt = mlir::arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(spt)));
        auto cTpbSpt = mlir::arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb * spt)));
        auto outerV = mlir::arith::DivSIOp::create(
            rewriter, loc, kFlatI32, cSpt.getResult());
        auto innerV = mlir::arith::RemSIOp::create(
            rewriter, loc, kFlatI32, cSpt.getResult());
        auto outerScaled = mlir::arith::MulIOp::create(
            rewriter, loc, outerV.getResult(), cTpbSpt.getResult());
        kOffsetI32 = mlir::arith::AddIOp::create(
                         rewriter, loc, outerScaled.getResult(),
                         innerV.getResult())
                         .getResult();
      }
      mlir::Value idxI32 =
          mlir::arith::AddIOp::create(rewriter, loc, tidScaled, kOffsetI32)
              .getResult();
      // W-B rich-cone path: evaluate the WHOLE reduce cone at this logical
      // element index. evalRank1ValueAt re-derives every leaf (loads addressed
      // + masked independently), so multi-load select/cmp cones work. The cone
      // was validated by rank1ConeSupported up front, so a null here is a bug.
      if (richCone) {
        mlir::Value elt =
            evalRank1ValueAt(reduceSrc, idxI32, rewriter, loc, 0);
        if (!elt)
          return acc;
        if (elt.getType() != storeTy)
          elt = mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, mlir::TypeRange{storeTy},
                    mlir::ValueRange{elt})
                    .getResult(0);
        return emitCombine(acc, elt);
      }
      // Wall 13: address index includes scalar tt.addptr chain offset.
      mlir::Value addrI32 = idxI32;
      if (scalarOff) {
        addrI32 = mlir::arith::AddIOp::create(rewriter, loc, addrI32,
                                               scalarOff)
                      .getResult();
      }
      mlir::Value idxUI32 =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{ui32},
              mlir::ValueRange{addrI32})
              .getResult(0);

      mlir::Value elt;
      mlir::Value condKResult;
      if (maskBoundN) {
        // Wall 7 masked path inside the loop body.
        auto condK = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::slt, idxI32,
            maskBoundN);
        condKResult = condK.getResult();
        auto scfIf = mlir::scf::IfOp::create(
            rewriter, loc, mlir::TypeRange{loadEltTy}, condK.getResult(),
            /*addThenBlock=*/true, /*addElseBlock=*/true);
        {
          mlir::OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToStart(&scfIf.getThenRegion().front());
          auto getEl = GetElementOp::create(rewriter, loc, loadEltTy, memref,
                                            idxUI32);
          mlir::scf::YieldOp::create(rewriter, loc,
                                     mlir::ValueRange{getEl.getResult()});
        }
        {
          mlir::OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointToStart(&scfIf.getElseRegion().front());
          auto c = mlir::arith::ConstantOp::create(rewriter, loc, elseAttr);
          mlir::scf::YieldOp::create(rewriter, loc,
                                     mlir::ValueRange{c.getResult()});
        }
        elt = scfIf.getResult(0);
      } else {
        elt = GetElementOp::create(rewriter, loc, loadEltTy, memref, idxUI32)
                  .getResult();
      }
      // Type bridge elt → storeTy.
      if (elt.getType() != storeTy) {
        elt = mlir::UnrealizedConversionCastOp::create(
                  rewriter, loc, mlir::TypeRange{storeTy},
                  mlir::ValueRange{elt})
                  .getResult(0);
      }
      // Wall 11 chain re-emit + combiner-identity-override.
      if (!chain.empty()) {
        mlir::Value chainElt =
            emitScalarChain(chain, elt, /*spt_idx=*/0, loc, rewriter);
        if (condKResult) {
          mlir::Value identityV = buildIdentityVal();
          elt = mlir::arith::SelectOp::create(rewriter, loc, condKResult,
                                              chainElt, identityV)
                    .getResult();
        } else {
          elt = chainElt;
        }
      }
      // Combine: accumulator (region iter-arg) op elt.
      return emitCombine(acc, elt);
    };

    // K identity inits (one per accumulator).
    llvm::SmallVector<mlir::Value, 8> inits;
    for (int64_t i = 0; i < K; ++i)
      inits.push_back(buildIdentityVal());
    auto c0I32 = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto cEI32 = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(E)));
    auto cKI32 = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(K)));
    auto forOp = mlir::scf::ForOp::create(
        rewriter, loc, c0I32.getResult(), cEI32.getResult(),
        cKI32.getResult(), mlir::ValueRange(inits));

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      mlir::Value ivI32 = forOp.getInductionVar();
      llvm::SmallVector<mlir::Value, 8> newAccs;
      for (int64_t m = 0; m < K; ++m) {
        mlir::Value kFlat = ivI32;
        if (m > 0) {
          auto cM = mlir::arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getI32IntegerAttr(static_cast<int32_t>(m)));
          kFlat = mlir::arith::AddIOp::create(rewriter, loc, ivI32,
                                              cM.getResult())
                      .getResult();
        }
        newAccs.push_back(buildPartial(kFlat, forOp.getRegionIterArgs()[m]));
      }
      mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange(newAccs));
    }

    // Balanced tree-combine of the K accumulator results into one scalar.
    llvm::SmallVector<mlir::Value, 8> parts(forOp.getResults().begin(),
                                            forOp.getResults().end());
    while (parts.size() > 1) {
      llvm::SmallVector<mlir::Value, 8> next;
      for (size_t i = 0; i + 1 < parts.size(); i += 2)
        next.push_back(emitCombine(parts[i], parts[i + 1]));
      if (parts.size() & 1)
        next.push_back(parts.back());
      parts = next;
    }
    inputScalarPT = parts.front();
  } else {
    // BLOCK <= tpb: original Phase B path — adaptor delivers a single
    // per-thread scalar.
    inputScalarPT = adaptor.getSrcs().front();
    if (isI32 && inputScalarPT.getType() != storeTy) {
      inputScalarPT = mlir::UnrealizedConversionCastOp::create(
                          rewriter, loc, mlir::TypeRange{storeTy},
                          mlir::ValueRange{inputScalarPT})
                          .getResult(0);
    }
  }

  // Wall 13 fix (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC8):
  // butterfly stage uses LOCAL thread id, not global. The threadgroup
  // scratch buffer (buf, sized tpb) is per-threadgroup; indexing it with
  // global_tid would OOB-write for tgid > 0 in multi-program launches.
  mlir::Value tidGlobalForButterfly =
      ThreadIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"))
          .getResult();
  mlir::Value tidGlobalForButterflyI32 =
      mlir::UnrealizedConversionCastOp::create(
          rewriter, loc, mlir::TypeRange{i32},
          mlir::ValueRange{tidGlobalForButterfly})
          .getResult(0);
  mlir::Value tgForButterflyUI32 =
      ThreadgroupIdOp::create(rewriter, loc, ui32,
                                rewriter.getStringAttr("x"))
          .getResult();
  mlir::Value tgForButterflyI32 =
      mlir::UnrealizedConversionCastOp::create(
          rewriter, loc, mlir::TypeRange{i32},
          mlir::ValueRange{tgForButterflyUI32})
          .getResult(0);
  auto cTpbButterfly = mlir::arith::ConstantOp::create(
      rewriter, loc,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
  mlir::Value tgOffsetButterfly = mlir::arith::MulIOp::create(
                                       rewriter, loc, tgForButterflyI32,
                                       cTpbButterfly.getResult())
                                       .getResult();
  mlir::Value tidI32 = mlir::arith::SubIOp::create(
                            rewriter, loc, tidGlobalForButterflyI32,
                            tgOffsetButterfly)
                            .getResult();
  mlir::Value tid = mlir::UnrealizedConversionCastOp::create(
                       rewriter, loc, mlir::TypeRange{ui32},
                       mlir::ValueRange{tidI32})
                       .getResult(0);

  // Tail handling: padded = (tid < BLOCK) ? inputScalarPT : identity.
  mlir::Value padded = inputScalarPT;
  if (BLOCK < tpb) {
    auto cBlock = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(BLOCK)));
    auto cond = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::ult, tidI32,
        cBlock.getResult());
    mlir::Value identity = buildIdentityVal();
    padded = mlir::arith::SelectOp::create(rewriter, loc, cond.getResult(),
                                            inputScalarPT, identity)
                 .getResult();
  }

  // Allocate threadgroup buffer of size `tpb` of element type `storeTy`.
  auto bufTy =
      MetalMemRefType::get(rewriter.getContext(), storeTy, tpb);
  mlir::Value buf =
      ThreadgroupAllocaOp::create(rewriter, loc, bufTy).getResult();

  // Write padded value to buf[tid]; barrier. (BLOCK <= tpb path.)
  // The leading barrier guards the buffer against a loop-carried
  // write-after-read hazard: when this reduce sits inside an scf.for, `buf` is
  // one static allocation reused every trip, so iteration t+1's write below
  // would otherwise race iteration t's broadcast read of buf[0]. Only
  // observable at tpb > 32 (a single SIMD-group cannot drift out of lockstep).
  BarrierOp::create(rewriter, loc);
  StoreOp::create(rewriter, loc, padded, buf, tid);
  BarrierOp::create(rewriter, loc);

  // log2(tpb) butterfly stages with halving stride.
  for (int64_t s = tpb / 2; s >= 1; s /= 2) {
    auto cS_i32 = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(s)));
    auto cond = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::ult, tidI32,
        cS_i32.getResult());
    auto ifOp = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{}, cond.getResult(),
        /*addThenBlock=*/true, /*addElseBlock=*/false);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      // partner_idx = tid + s (ui32 arith via i32 bridge).
      auto partnerI32 = mlir::arith::AddIOp::create(
          rewriter, loc, tidI32, cS_i32.getResult());
      mlir::Value partnerUI32 = mlir::UnrealizedConversionCastOp::create(
                                    rewriter, loc, mlir::TypeRange{ui32},
                                    mlir::ValueRange{partnerI32.getResult()})
                                    .getResult(0);
      auto partnerVal =
          GetElementOp::create(rewriter, loc, storeTy, buf, partnerUI32);
      auto selfVal = GetElementOp::create(rewriter, loc, storeTy, buf, tid);
      mlir::Value merged =
          emitCombine(selfVal.getResult(), partnerVal.getResult());
      StoreOp::create(rewriter, loc, merged, buf, tid);
      mlir::scf::YieldOp::create(rewriter, loc);
    }
    BarrierOp::create(rewriter, loc);
  }

  // result = buf[0] (broadcast read; tid=0's slot holds the reduction).
  auto cZeroI32 = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(0));
  mlir::Value zeroUI32 = mlir::UnrealizedConversionCastOp::create(
                            rewriter, loc, mlir::TypeRange{ui32},
                            mlir::ValueRange{cZeroI32.getResult()})
                            .getResult(0);
  mlir::Value result =
      GetElementOp::create(rewriter, loc, storeTy, buf, zeroUI32).getResult();

  // For i32, bridge ui32 storage → signless i32 for downstream consumers.
  if (isI32 && result.getType() != elemTy) {
    result = mlir::UnrealizedConversionCastOp::create(
                 rewriter, loc, mlir::TypeRange{elemTy},
                 mlir::ValueRange{result})
                 .getResult(0);
  }
  rewriter.replaceOp(op, result);
  return mlir::success();
}

// Peel shape-only ops (expand_dims / broadcast / convert_layout / reshape) off
// `v`, returning the underlying tensor value. Used to see through the broadcast
// cones of `x[None,:]` / `x[:,None]` to the real producer.
static mlir::Value peelShapeOps(mlir::Value v) {
  while (mlir::Operation *d = v.getDefiningOp()) {
    if (mlir::isa<mlir::triton::ExpandDimsOp, mlir::triton::BroadcastOp,
                  mlir::triton::gpu::ConvertLayoutOp, mlir::triton::ReshapeOp>(
            d))
      v = d->getOperand(0);
    else
      break;
  }
  return v;
}

// Symbolically evaluate the SCALAR i32 value of an index cone `v` at logical
// element position `idx` (an i32 scalar). Walks the ORIGINAL Triton cone:
// make_range(start s) -> idx + s; tt.splat(x) -> x; addi/muli/subi -> scalar
// arith; shape-only ops -> recurse operand 0. Returns null on any unsupported
// op. Lets the contiguous-masked-reduce path re-derive a per-element device
// index / mask bound without the canonical `slt make_range(0) splat` shape.
static mlir::Value scalarizeConeAtIndex(mlir::Value v, mlir::Value idx,
                                        mlir::ConversionPatternRewriter &rewriter,
                                        mlir::Location loc, int depth = 0) {
  if (depth > 24)
    return {};
  if (!mlir::isa<mlir::RankedTensorType>(v.getType()))
    return v; // already a scalar (e.g. a tt.splat src or a program-id)
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return {};
  if (mlir::isa<mlir::triton::ExpandDimsOp, mlir::triton::BroadcastOp,
                mlir::triton::gpu::ConvertLayoutOp, mlir::triton::ReshapeOp>(def))
    return scalarizeConeAtIndex(def->getOperand(0), idx, rewriter, loc,
                                depth + 1);
  if (auto mr = mlir::dyn_cast<mlir::triton::MakeRangeOp>(def)) {
    int32_t s = static_cast<int32_t>(mr.getStart());
    if (s == 0)
      return idx;
    auto cS = mlir::arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getI32IntegerAttr(s));
    return mlir::arith::AddIOp::create(rewriter, loc, idx, cS.getResult())
        .getResult();
  }
  if (auto sp = mlir::dyn_cast<mlir::triton::SplatOp>(def)) {
    if (!sp.getOperation()->getBlock())
      return {}; // stale (already converted)
    return sp.getSrc();
  }
  auto recurse2 = [&](mlir::Value a, mlir::Value b,
                      auto make) -> mlir::Value {
    mlir::Value L = scalarizeConeAtIndex(a, idx, rewriter, loc, depth + 1);
    mlir::Value R = scalarizeConeAtIndex(b, idx, rewriter, loc, depth + 1);
    if (!L || !R)
      return mlir::Value{};
    return make(L, R);
  };
  if (auto a = mlir::dyn_cast<mlir::arith::AddIOp>(def))
    return recurse2(a.getLhs(), a.getRhs(), [&](mlir::Value L, mlir::Value R) {
      return mlir::arith::AddIOp::create(rewriter, loc, L, R).getResult();
    });
  if (auto a = mlir::dyn_cast<mlir::arith::MulIOp>(def))
    return recurse2(a.getLhs(), a.getRhs(), [&](mlir::Value L, mlir::Value R) {
      return mlir::arith::MulIOp::create(rewriter, loc, L, R).getResult();
    });
  if (auto a = mlir::dyn_cast<mlir::arith::SubIOp>(def))
    return recurse2(a.getLhs(), a.getRhs(), [&](mlir::Value L, mlir::Value R) {
      return mlir::arith::SubIOp::create(rewriter, loc, L, R).getResult();
    });
  return {};
}

// Contiguous masked full-reduce: `tt.reduce(addf, axis=0)` -> f32 whose source
// is (through reshape/shape ops) a directly-masked `tt.load` of a tile with a
// single non-unit dimension — i.e. the leet-triton subarray-sum shape
// `tl.load(base + offset, mask=row & (col <= bound)).sum()`. The canonical
// `lowerRank1Reduce` masked path rejects this (reshape + 2D-unit-dim load +
// compound `sle` mask), so re-derive the per-element device index and mask via
// `scalarizeConeAtIndex` and run the same threadgroup butterfly. f32 add only.
// Returns failure (creating NOTHING during the inspection phase) so the caller
// can fall back to `lowerRank1Reduce`.
static mlir::LogicalResult
lowerContiguousMaskedReduce(mlir::triton::ReduceOp op,
                            mlir::ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto rtt =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getSrcs().front().getType());
  if (!rtt || rtt.getRank() != 1 || rtt.isDynamicDim(0) || op.getAxis() != 0)
    return mlir::failure();
  if (!rtt.getElementType().isF32())
    return mlir::failure();
  // Combine must be addf.
  mlir::Operation *combine = nullptr;
  if (op->getNumRegions() > 0 && !op->getRegion(0).empty())
    for (auto &nested : op->getRegion(0).front())
      if (!mlir::isa<mlir::triton::ReduceReturnOp>(nested)) {
        combine = &nested;
        break;
      }
  if (!combine || !mlir::isa<mlir::arith::AddFOp>(combine))
    return mlir::failure();

  // ---- Inspection phase (NO IR creation) ----
  mlir::Value loadVal = peelShapeOps(op.getSrcs().front());
  auto loadOp = loadVal.getDefiningOp<mlir::triton::LoadOp>();
  if (!loadOp || !loadOp.getMask())
    return mlir::failure();
  // Load tile must have exactly one non-unit dim (effectively 1-D).
  auto loadTy = mlir::dyn_cast<mlir::RankedTensorType>(loadVal.getType());
  if (!loadTy)
    return mlir::failure();
  int nonUnit = 0;
  for (auto s : loadTy.getShape())
    if (s != 1)
      ++nonUnit;
  if (nonUnit != 1)
    return mlir::failure();
  // `other` must be absent or 0 (masked lanes contribute the add identity).
  if (loadOp.getOther()) {
    auto oc = extractSplatConstantAttr(loadOp.getOther());
    auto fa = oc ? mlir::dyn_cast<mlir::FloatAttr>(*oc) : mlir::FloatAttr();
    if (!fa || fa.getValueAsDouble() != 0.0)
      return mlir::failure();
  }
  auto addptr = loadOp.getPtr().getDefiningOp<mlir::triton::AddPtrOp>();
  if (!addptr)
    return mlir::failure();
  mlir::Value offsetTensor = addptr.getOffset();
  // Mask: andi(splat(rowOK_i1), colCmpCone) or a bare colCmpCone.
  mlir::Value maskRoot = peelShapeOps(loadOp.getMask());
  mlir::Value rowOK; // optional uniform i1 scalar
  mlir::Value colCmpVal = maskRoot;
  if (auto andOp = maskRoot.getDefiningOp<mlir::arith::AndIOp>()) {
    mlir::Value lhs = peelShapeOps(andOp.getLhs());
    mlir::Value rhs = peelShapeOps(andOp.getRhs());
    auto ls = lhs.getDefiningOp<mlir::triton::SplatOp>();
    auto rs = rhs.getDefiningOp<mlir::triton::SplatOp>();
    if (ls && !rs) {
      rowOK = ls.getSrc();
      colCmpVal = rhs;
    } else if (rs && !ls) {
      rowOK = rs.getSrc();
      colCmpVal = lhs;
    } else {
      return mlir::failure();
    }
  }
  auto colCmp = colCmpVal.getDefiningOp<mlir::arith::CmpIOp>();
  if (!colCmp)
    return mlir::failure();
  auto pred = colCmp.getPredicate();
  if (pred != mlir::arith::CmpIPredicate::slt &&
      pred != mlir::arith::CmpIPredicate::sle)
    return mlir::failure();
  // Require `cmpi {slt|sle} (idxCone) (splat bound)` — bound on the RHS.
  auto boundSplat =
      peelShapeOps(colCmp.getRhs()).getDefiningOp<mlir::triton::SplatOp>();
  if (!boundSplat)
    return mlir::failure();
  mlir::Value boundScalar = boundSplat.getSrc();
  mlir::Value colIdxCone = colCmp.getLhs();

  auto srcBlocked = mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
      rtt.getEncoding());
  if (!srcBlocked)
    return mlir::failure();
  int64_t tpb = 1;
  for (auto t : srcBlocked.getThreadsPerWarp())
    tpb *= t;
  for (auto w : srcBlocked.getWarpsPerCTA())
    tpb *= w;
  if (tpb <= 0 || (tpb & (tpb - 1)) != 0)
    return mlir::failure();
  int64_t BLOCK = rtt.getDimSize(0);
  if (BLOCK <= 0 || BLOCK % tpb != 0)
    return mlir::failure();
  int64_t E = BLOCK / tpb;
  if (E < 1 || E > 1024 || (E & (E - 1)) != 0)
    return mlir::failure();

  mlir::Value memref = findBaseMemref(loadOp.getPtr(), rewriter);
  if (!memref)
    return mlir::failure();
  mlir::Type loadEltTy = mlir::cast<MetalMemRefType>(memref.getType()).getType();
  if (!loadEltTy.isF32())
    return mlir::failure();

  // ---- Emission phase ----
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
  auto i32 = rewriter.getI32Type();
  auto addEnum = BinaryExpOperatorAttr::get(rewriter.getContext(),
                                            BinaryExpOperator::addOp);
  auto emitAdd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    return BinaryExpOp::create(rewriter, loc, a.getType(), addEnum, a, b)
        .getResult();
  };

  // localTid = id.x - tgid.x * tpb.
  auto tidG = ThreadIdOp::create(rewriter, loc, ui32,
                                 rewriter.getStringAttr("x"));
  mlir::Value tidI32 = mlir::UnrealizedConversionCastOp::create(
                           rewriter, loc, mlir::TypeRange{i32},
                           mlir::ValueRange{tidG.getResult()})
                           .getResult(0);
  auto tgG = ThreadgroupIdOp::create(rewriter, loc, ui32,
                                     rewriter.getStringAttr("x"));
  mlir::Value tgI32 = mlir::UnrealizedConversionCastOp::create(
                          rewriter, loc, mlir::TypeRange{i32},
                          mlir::ValueRange{tgG.getResult()})
                          .getResult(0);
  auto cTpb = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
  auto tgOff = mlir::arith::MulIOp::create(rewriter, loc, tgI32,
                                           cTpb.getResult());
  mlir::Value localTid =
      mlir::arith::SubIOp::create(rewriter, loc, tidI32, tgOff.getResult())
          .getResult();

  // Per-thread accumulator: for k in [0, E): acc += masked load of element
  // (localTid + k*tpb).
  auto cZeroF = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getF32FloatAttr(0.0f));
  auto cZeroI = mlir::arith::ConstantOp::create(rewriter, loc,
                                                rewriter.getI32IntegerAttr(0));
  auto cE = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(E)));
  auto cOne = mlir::arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getI32IntegerAttr(1));
  auto forOp = mlir::scf::ForOp::create(
      rewriter, loc, cZeroI.getResult(), cE.getResult(), cOne.getResult(),
      mlir::ValueRange{cZeroF.getResult()});
  {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    mlir::Value iv = forOp.getInductionVar();
    mlir::Value acc = forOp.getRegionIterArgs()[0];
    // idx = localTid + iv*tpb.
    auto kOff = mlir::arith::MulIOp::create(rewriter, loc, iv, cTpb.getResult());
    mlir::Value idx =
        mlir::arith::AddIOp::create(rewriter, loc, localTid, kOff.getResult())
            .getResult();
    mlir::Value addrI32 = scalarizeConeAtIndex(offsetTensor, idx, rewriter, loc);
    // Add the scalar tt.addptr chain offset (e.g. the per-program row base
    // `program_id * N`) carried in the load ptr's splat base. `offsetTensor` is
    // only the per-COLUMN tensor offset; without this, every threadgroup reads
    // program 0's row (multi-program per-row reduce, e.g. layer-norm mean).
    if (addrI32)
      if (mlir::Value soff =
              accumulateScalarAddPtrOffsets(loadOp.getPtr(), rewriter, loc))
        addrI32 =
            mlir::arith::AddIOp::create(rewriter, loc, addrI32, soff).getResult();
    mlir::Value colv = scalarizeConeAtIndex(colIdxCone, idx, rewriter, loc);
    if (!addrI32 || !colv) {
      // Inspection passed but a cone op is unsupported: bail. (The partial ops
      // created here are rolled back by the conversion driver on failure.)
      return rewriter.notifyMatchFailure(op, "contiguous reduce: cone scalarize");
    }
    mlir::Value cond = mlir::arith::CmpIOp::create(rewriter, loc, pred, colv,
                                                   boundScalar)
                           .getResult();
    if (rowOK)
      cond = mlir::arith::AndIOp::create(rewriter, loc, cond, rowOK).getResult();
    mlir::Value addrUI32 = mlir::UnrealizedConversionCastOp::create(
                               rewriter, loc, mlir::TypeRange{ui32},
                               mlir::ValueRange{addrI32})
                               .getResult(0);
    auto scfIf = mlir::scf::IfOp::create(rewriter, loc,
                                         mlir::TypeRange{loadEltTy}, cond,
                                         /*addThenBlock=*/true,
                                         /*addElseBlock=*/true);
    {
      mlir::OpBuilder::InsertionGuard g2(rewriter);
      rewriter.setInsertionPointToStart(&scfIf.getThenRegion().front());
      auto el = GetElementOp::create(rewriter, loc, loadEltTy, memref, addrUI32);
      mlir::scf::YieldOp::create(rewriter, loc,
                                 mlir::ValueRange{el.getResult()});
    }
    {
      mlir::OpBuilder::InsertionGuard g2(rewriter);
      rewriter.setInsertionPointToStart(&scfIf.getElseRegion().front());
      auto z = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getF32FloatAttr(0.0f));
      mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{z.getResult()});
    }
    mlir::Value combined = emitAdd(acc, scfIf.getResult(0));
    mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{combined});
  }
  mlir::Value partial = forOp.getResult(0);

  // Threadgroup butterfly over `tpb` lanes -> buf[0] holds the total.
  auto bufTy = MetalMemRefType::get(rewriter.getContext(), loadEltTy, tpb);
  mlir::Value buf = ThreadgroupAllocaOp::create(rewriter, loc, bufTy).getResult();
  mlir::Value tidUI = mlir::UnrealizedConversionCastOp::create(
                          rewriter, loc, mlir::TypeRange{ui32},
                          mlir::ValueRange{localTid})
                          .getResult(0);
  // Leading barrier: same loop-carried hazard as in lowerRank1Reduce, and for
  // the same reason — `buf` is one static allocation reused on every trip of an
  // enclosing scf.for, so without it iteration t+1's write races iteration t's
  // read of buf[0]. Only observable at tpb > 32.
  BarrierOp::create(rewriter, loc);
  StoreOp::create(rewriter, loc, partial, buf, tidUI);
  BarrierOp::create(rewriter, loc);
  for (int64_t s = tpb / 2; s >= 1; s /= 2) {
    auto cS = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(s)));
    auto cond = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::slt, localTid,
        cS.getResult());
    auto ifOp = mlir::scf::IfOp::create(rewriter, loc, mlir::TypeRange{},
                                        cond.getResult(),
                                        /*addThenBlock=*/true,
                                        /*addElseBlock=*/false);
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto partnerI32 =
          mlir::arith::AddIOp::create(rewriter, loc, localTid, cS.getResult());
      mlir::Value partnerUI = mlir::UnrealizedConversionCastOp::create(
                                  rewriter, loc, mlir::TypeRange{ui32},
                                  mlir::ValueRange{partnerI32.getResult()})
                                  .getResult(0);
      auto pv = GetElementOp::create(rewriter, loc, loadEltTy, buf, partnerUI);
      auto sv = GetElementOp::create(rewriter, loc, loadEltTy, buf, tidUI);
      StoreOp::create(rewriter, loc, emitAdd(sv.getResult(), pv.getResult()),
                      buf, tidUI);
      mlir::scf::YieldOp::create(rewriter, loc);
    }
    BarrierOp::create(rewriter, loc);
  }
  auto cZeroIdx = mlir::arith::ConstantOp::create(rewriter, loc,
                                                  rewriter.getI32IntegerAttr(0));
  mlir::Value zeroUI = mlir::UnrealizedConversionCastOp::create(
                           rewriter, loc, mlir::TypeRange{ui32},
                           mlir::ValueRange{cZeroIdx.getResult()})
                           .getResult(0);
  mlir::Value result =
      GetElementOp::create(rewriter, loc, loadEltTy, buf, zeroUI).getResult();
  rewriter.replaceOp(op, result);
  return mlir::success();
}

// Inc 2.5 (prototype): during an inline reduce fill over a loop-dependent cone
// at M<=tpb, each fill thread reduces its OWN row (r == localTid), so a
// non-re-emittable per-row leaf (q0_rope, loop-carried acc) resolves to that
// thread's converted per-thread scalar via getRemappedValue. `g_stagedLeaves`
// maps such leaves -> their getRemappedValue for the duration of that fill.
static const llvm::DenseMap<mlir::Value, mlir::Value> *g_stagedLeaves = nullptr;

// Inc 2.5 rank-2: a whole [M,N] tile staged in threadgroup memory, read at
// `buf[r*N + n]`. `g_stagedLeaves` cannot express this — it maps a leaf to ONE
// per-thread scalar, which is exactly a per-ROW value, so a tile that also
// varies along the row has no representation there.
//
// The case that needs it is a loop-carried tile: `h = A*h + B` inside an
// scf.for, reduced every trip. The cone evaluator re-emits its expression per
// (r, n) from leaves that survive conversion, but an iter_arg is a
// BlockArgument with no defining op, and the recurrence means trip t's value is
// not re-derivable from anything in device memory. Materialising the tile is
// the only option.
struct StagedTile {
  mlir::Value buf; // !metal.memref<M*N x f32>
  int64_t n;       // row stride
};
static const llvm::DenseMap<mlir::Value, StagedTile> *g_tileBuffers = nullptr;

// Chained reduces: a rank-2 axis=1 `tt.reduce` RESULT -> the threadgroup
// `rowBuf[M]` it was reduced into. A consuming reduce's cone reads `rowBuf[r]`
// for the row IT is filling.
//
// The Inc-2.5 staging resolves such a leaf through `getRemappedValue`, i.e. the
// producer's per-thread scalar. That only agrees with the consumer's row when
// the producer's readback is itself row-per-thread. It is not, once the result
// is ALSO broadcast back into a 2D tile — a full softmax needs `m` at two
// different indexings at once (`flat / N` for the materialised `p`, `localTid`
// for the reduce over it), which one SSA value cannot provide. Reading the
// buffer directly gives the consumer its own indexing and leaves the producer's
// per-thread value to the materialised path.
static const llvm::DenseMap<mlir::Value, mlir::Value> *g_reduceRowBufs =
    nullptr;

// buf[r*N + n], or null if `v` is not a staged tile.
static mlir::Value readStagedTile(mlir::Value v, mlir::Value rVal,
                                  mlir::Value nVal,
                                  mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Location loc) {
  if (!g_tileBuffers)
    return nullptr;
  auto it = g_tileBuffers->find(v);
  if (it == g_tileBuffers->end())
    return nullptr;
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
  auto cN = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(it->second.n)));
  auto rowOff =
      mlir::arith::MulIOp::create(rewriter, loc, rVal, cN.getResult());
  auto flat =
      mlir::arith::AddIOp::create(rewriter, loc, rowOff.getResult(), nVal);
  mlir::Value idx = mlir::UnrealizedConversionCastOp::create(
                        rewriter, loc, mlir::TypeRange{ui32},
                        mlir::ValueRange{flat.getResult()})
                        .getResult(0);
  mlir::Type eltTy =
      mlir::cast<MetalMemRefType>(it->second.buf.getType()).getType();
  return GetElementOp::create(rewriter, loc, eltTy, it->second.buf, idx)
      .getResult();
}

// W-C scan: maps a `tt.scan` (cumsum) RESULT placeholder value -> the
// threadgroup buffer holding the DISTRIBUTED prefix-sum. ScanLowering fills the
// buffer + registers this (operands-first, so it runs before any consuming
// reduce); the rich cone evaluator reads `scanbuf[idxVal]` per element. Pass-
// lifetime (populated across the whole conversion, not scoped to one reduce).
static const llvm::DenseMap<mlir::Value, mlir::Value> *g_scanBuffers = nullptr;

// Wall 17 Increment 2: evaluate one element (logical index `idxVal`, an i32
// scalar) of a RANK-1 tensor cone as scalar Metal ops. Used for the per-row /
// per-column operands a softmax cone broadcasts into the reduce tile
// (`q0_rope[:,None]` → row index; `seq_idx[None,:]` → col index). Device loads
// are addressed by replaying the load's index cone via `scalarizeConeAtIndex`.
// f32 leaves go through metal.binary_exp/unary_exp; integer arith + select +
// compare use raw scalar arith (all translated by ModuleTranslation). Returns
// null on an unsupported producer.
static mlir::Value evalRank1ValueAt(mlir::Value v, mlir::Value idxVal,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Location loc, int depth) {
  if (depth > 24)
    return nullptr;
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);

  // Inc 2.5: a staged per-row leaf resolves to the thread's own scalar.
  if (g_stagedLeaves) {
    auto it = g_stagedLeaves->find(v);
    if (it != g_stagedLeaves->end())
      return it->second;
  }
  // Chained reduce: a prior reduce's result resolves to ITS rowBuf[idxVal].
  if (g_reduceRowBufs) {
    auto it = g_reduceRowBufs->find(v);
    if (it != g_reduceRowBufs->end()) {
      mlir::Value idxUI32 =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{idxVal})
              .getResult(0);
      mlir::Type eltTy =
          mlir::cast<MetalMemRefType>(it->second.getType()).getType();
      return GetElementOp::create(rewriter, loc, eltTy, it->second, idxUI32)
          .getResult();
    }
  }
  // W-C scan: a scan-result placeholder resolves to scanbuf[idxVal].
  if (g_scanBuffers) {
    auto it = g_scanBuffers->find(v);
    if (it != g_scanBuffers->end()) {
      mlir::Value idxUI32 =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{idxVal})
              .getResult(0);
      mlir::Type eltTy =
          mlir::cast<MetalMemRefType>(it->second.getType()).getType();
      return GetElementOp::create(rewriter, loc, eltTy, it->second, idxUI32)
          .getResult();
    }
  }

  if (!mlir::isa<mlir::RankedTensorType>(v.getType()))
    return v; // already a scalar

  if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>()) {
    if (!splat.getOperation()->getBlock())
      return nullptr;
    return splat.getSrc();
  }
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat())
      return nullptr;
    return mlir::arith::ConstantOp::create(
               rewriter, loc, dense.getSplatValue<mlir::TypedAttr>())
        .getResult();
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return nullptr;

  // Shape-only ops: recurse operand 0 at the same index.
  if (mlir::isa<mlir::triton::ExpandDimsOp, mlir::triton::BroadcastOp,
                mlir::triton::gpu::ConvertLayoutOp, mlir::triton::ReshapeOp>(
          def))
    return evalRank1ValueAt(def->getOperand(0), idxVal, rewriter, loc,
                            depth + 1);

  // Tensor-yielding scf.if with a UNIFORM (scalar) condition → per-element
  // select(cond, then[idx], else[idx]). Both branches are re-derived (loads are
  // guarded/pure), so evaluating the untaken branch is side-effect-free. Used
  // by speculative decoding's `scf.if is_uniform` scan input.
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
    if (ifOp.getElseRegion().empty())
      return nullptr;
    if (mlir::isa<mlir::RankedTensorType>(ifOp.getCondition().getType()))
      return nullptr; // per-element condition unsupported
    unsigned resIdx = mlir::cast<mlir::OpResult>(v).getResultNumber();
    mlir::Value t = evalRank1ValueAt(ifOp.thenYield().getOperand(resIdx), idxVal,
                                     rewriter, loc, depth + 1);
    mlir::Value e = evalRank1ValueAt(ifOp.elseYield().getOperand(resIdx), idxVal,
                                     rewriter, loc, depth + 1);
    if (!t || !e)
      return nullptr;
    return mlir::arith::SelectOp::create(rewriter, loc, ifOp.getCondition(), t,
                                         e)
        .getResult();
  }

  // make_range: element value = start + idx.
  if (auto mr = mlir::dyn_cast<mlir::triton::MakeRangeOp>(def)) {
    int32_t s = static_cast<int32_t>(mr.getStart());
    if (s == 0)
      return idxVal;
    auto cS = mlir::arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getI32IntegerAttr(s));
    return mlir::arith::AddIOp::create(rewriter, loc, idxVal, cS.getResult())
        .getResult();
  }

  // tt.load: address via the index cone, read device[addr].
  if (auto load = mlir::dyn_cast<mlir::triton::LoadOp>(def)) {
    auto addptr = load.getPtr().getDefiningOp<mlir::triton::AddPtrOp>();
    if (!addptr)
      return nullptr;
    // Sum EVERY offset along the tt.addptr chain, scalarising the tensor-typed
    // ones at `idxVal`.
    //
    // This used to scalarise only the OUTERMOST offset and then add the SCALAR
    // offsets from the rest of the chain. That drops a tensor offset sitting on
    // an INNER addptr — which is exactly where the per-element index lives once
    // the base pointer is hoisted out of a loop:
    //
    //   base = tt.addptr(splat(p + pid*S), offs)   <- row/col index, tensor
    //   addr = tt.addptr(base, splat(t*S))         <- outermost, no index
    //
    // The mask is scalarised down a different path and kept its index, so the
    // read came out silently off-row instead of obviously broken.
    auto i32 = rewriter.getI32Type();
    mlir::Value addr;
    auto addTerm = [&](mlir::Value t) {
      if (t.getType() != i32)
        t = mlir::UnrealizedConversionCastOp::create(
                rewriter, loc, mlir::TypeRange{i32}, mlir::ValueRange{t})
                .getResult(0);
      addr = addr ? mlir::arith::AddIOp::create(rewriter, loc, addr, t)
                        .getResult()
                  : t;
    };
    {
      mlir::Value cur = load.getPtr();
      while (true) {
        while (auto sp = cur.getDefiningOp<mlir::triton::SplatOp>())
          cur = sp.getSrc();
        auto ap = cur.getDefiningOp<mlir::triton::AddPtrOp>();
        if (!ap)
          break;
        mlir::Value off = ap.getOffset();
        if (mlir::isa<mlir::RankedTensorType>(off.getType())) {
          mlir::Value s = scalarizeConeAtIndex(off, idxVal, rewriter, loc, 0);
          if (!s)
            return nullptr;
          addTerm(s);
        } else {
          addTerm(off);
        }
        cur = ap.getPtr();
      }
    }
    if (!addr)
      return nullptr;
    mlir::Value memref = findBaseMemref(load.getPtr(), rewriter);
    if (!memref)
      return nullptr;
    mlir::Type eltTy = mlir::cast<MetalMemRefType>(memref.getType()).getType();
    mlir::Value addrUI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{addr})
            .getResult(0);
    if (!load.getMask())
      return GetElementOp::create(rewriter, loc, eltTy, memref, addrUI32)
          .getResult();
    // Masked load: guard the device read with the mask cone evaluated at this
    // index, yielding `other` (or 0) for masked-out elements. Crucially the
    // GetElement is emitted INSIDE the scf.if so no OOB device read happens
    // when BLOCK >> the valid extent (e.g. speculative decoding pads V→1024).
    mlir::Value cond =
        evalRank1ValueAt(load.getMask(), idxVal, rewriter, loc, depth + 1);
    if (!cond)
      return nullptr;
    mlir::Value otherV;
    if (load.getOther()) {
      otherV = evalRank1ValueAt(load.getOther(), idxVal, rewriter, loc, depth + 1);
      if (!otherV)
        return nullptr;
      if (otherV.getType() != eltTy)
        otherV = mlir::UnrealizedConversionCastOp::create(
                     rewriter, loc, mlir::TypeRange{eltTy},
                     mlir::ValueRange{otherV})
                     .getResult(0);
    } else {
      auto zeroAttr =
          mlir::isa<mlir::FloatType>(eltTy)
              ? mlir::cast<mlir::TypedAttr>(rewriter.getFloatAttr(eltTy, 0.0))
              : mlir::cast<mlir::TypedAttr>(rewriter.getIntegerAttr(eltTy, 0));
      otherV = mlir::arith::ConstantOp::create(rewriter, loc, zeroAttr)
                   .getResult();
    }
    auto scfIf = mlir::scf::IfOp::create(rewriter, loc, mlir::TypeRange{eltTy},
                                         cond, /*addThenBlock=*/true,
                                         /*addElseBlock=*/true);
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&scfIf.getThenRegion().front());
      auto el = GetElementOp::create(rewriter, loc, eltTy, memref, addrUI32);
      mlir::scf::YieldOp::create(rewriter, loc,
                                 mlir::ValueRange{el.getResult()});
    }
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&scfIf.getElseRegion().front());
      mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{otherV});
    }
    return scfIf.getResult(0);
  }

  // int → float cast.
  if (auto s2f = mlir::dyn_cast<mlir::arith::SIToFPOp>(def)) {
    mlir::Value x =
        evalRank1ValueAt(s2f.getIn(), idxVal, rewriter, loc, depth + 1);
    if (!x)
      return nullptr;
    return mlir::arith::SIToFPOp::create(rewriter, loc, rewriter.getF32Type(), x)
        .getResult();
  }
  // float widen/narrow (fp16 load `.to(f32)` inside a reduce cone, etc.).
  auto floatEltOf = [](mlir::Type t) -> mlir::Type {
    if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(t))
      return rtt.getElementType();
    return t;
  };
  if (auto ext = mlir::dyn_cast<mlir::arith::ExtFOp>(def)) {
    mlir::Value x =
        evalRank1ValueAt(ext.getIn(), idxVal, rewriter, loc, depth + 1);
    if (!x)
      return nullptr;
    return mlir::arith::ExtFOp::create(rewriter, loc,
                                       floatEltOf(ext.getType()), x)
        .getResult();
  }
  if (auto trunc = mlir::dyn_cast<mlir::arith::TruncFOp>(def)) {
    mlir::Value x =
        evalRank1ValueAt(trunc.getIn(), idxVal, rewriter, loc, depth + 1);
    if (!x)
      return nullptr;
    return mlir::arith::TruncFOp::create(rewriter, loc,
                                         floatEltOf(trunc.getType()), x)
        .getResult();
  }

  // f32 unary math → metal.unary_exp.
  auto unary = [&](mlir::Value operand, UnaryExpOperator k) -> mlir::Value {
    mlir::Value x =
        evalRank1ValueAt(operand, idxVal, rewriter, loc, depth + 1);
    if (!x)
      return nullptr;
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(), k);
    return UnaryExpOp::create(rewriter, loc, x.getType(), attr, x).getResult();
  };
  if (auto e = mlir::dyn_cast<mlir::math::ExpOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::expOp);
  if (auto e = mlir::dyn_cast<mlir::math::SqrtOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::sqrtOp);
  if (auto e = mlir::dyn_cast<mlir::math::LogOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::logOp);
  if (auto e = mlir::dyn_cast<mlir::math::SinOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::sinOp);
  if (auto e = mlir::dyn_cast<mlir::math::CosOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::cosOp);
  if (auto e = mlir::dyn_cast<mlir::math::ErfOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::erfOp);
  if (auto e = mlir::dyn_cast<mlir::math::RsqrtOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::rsqrtOp);

  // f32 binary arith → metal.binary_exp (recurse both operands).
  auto fbinary = [&](mlir::Value lhs, mlir::Value rhs,
                     BinaryExpOperator k) -> mlir::Value {
    mlir::Value a = evalRank1ValueAt(lhs, idxVal, rewriter, loc, depth + 1);
    mlir::Value b = evalRank1ValueAt(rhs, idxVal, rewriter, loc, depth + 1);
    if (!a || !b)
      return nullptr;
    auto attr = BinaryExpOperatorAttr::get(rewriter.getContext(), k);
    return BinaryExpOp::create(rewriter, loc, a.getType(), attr, a, b)
        .getResult();
  };
  if (auto o = mlir::dyn_cast<mlir::arith::AddFOp>(def))
    return fbinary(o.getLhs(), o.getRhs(), BinaryExpOperator::addOp);
  if (auto o = mlir::dyn_cast<mlir::arith::SubFOp>(def))
    return fbinary(o.getLhs(), o.getRhs(), BinaryExpOperator::subOp);
  if (auto o = mlir::dyn_cast<mlir::arith::MulFOp>(def))
    return fbinary(o.getLhs(), o.getRhs(), BinaryExpOperator::mulOp);
  if (auto o = mlir::dyn_cast<mlir::arith::DivFOp>(def))
    return fbinary(o.getLhs(), o.getRhs(), BinaryExpOperator::divOp);
  if (auto o = mlir::dyn_cast<mlir::arith::MaximumFOp>(def))
    return fbinary(o.getLhs(), o.getRhs(), BinaryExpOperator::maxOp);
  if (auto o = mlir::dyn_cast<mlir::arith::MaxNumFOp>(def))
    return fbinary(o.getLhs(), o.getRhs(), BinaryExpOperator::maxOp);

  // int binary arith → raw scalar arith. Operands are normalised to signless
  // (see `toSignlessInt`): a cone leaf read from a device buffer carries the
  // ui32 STORAGE type, which every arith integer op rejects.
  auto ibinary = [&](mlir::Value lhs, mlir::Value rhs,
                     auto make) -> mlir::Value {
    mlir::Value a = evalRank1ValueAt(lhs, idxVal, rewriter, loc, depth + 1);
    mlir::Value b = evalRank1ValueAt(rhs, idxVal, rewriter, loc, depth + 1);
    if (!a || !b)
      return mlir::Value{};
    return make(toSignlessInt(a, rewriter, loc), toSignlessInt(b, rewriter, loc));
  };
  if (auto o = mlir::dyn_cast<mlir::arith::AddIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::AddIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::SubIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::SubIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::MulIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::MulIOp::create(rewriter, loc, a, b).getResult();
    });
  // i1 boolean combine (`(a>b) & mask`, `cond0 | cond1`).
  if (auto o = mlir::dyn_cast<mlir::arith::AndIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::AndIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::OrIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::OrIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::XOrIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::XOrIOp::create(rewriter, loc, a, b).getResult();
    });
  // Shifts — the bit-slicing half of a digit extraction (`(v >> shift) & 0xF`,
  // the radix-sort histogram cone). Without these the walker bailed at the
  // shift and the whole `tt.reduce` failed to legalize even though `&` was
  // already handled.
  if (auto o = mlir::dyn_cast<mlir::arith::ShLIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::ShLIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::ShRSIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::ShRSIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::ShRUIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::ShRUIOp::create(rewriter, loc, a, b).getResult();
    });
  // Integer width casts inside a cone (`tl.cast(pred, tl.int32)`). `floatEltOf`
  // above is type-agnostic — it just peels a tensor type down to its element —
  // so it serves the integer results here too. Results must be SIGNLESS (arith
  // ext/trunc reject ui32), which is what Triton's tensor type already carries.
  auto signlessEltOf = [&](mlir::Type t) -> mlir::Type {
    auto e = floatEltOf(t);
    if (auto i = llvm::dyn_cast<mlir::IntegerType>(e))
      if (!i.isSignless())
        return mlir::IntegerType::get(e.getContext(), i.getWidth());
    return e;
  };
  if (auto ext = mlir::dyn_cast<mlir::arith::ExtUIOp>(def)) {
    mlir::Value x =
        evalRank1ValueAt(ext.getIn(), idxVal, rewriter, loc, depth + 1);
    if (!x)
      return nullptr;
    return mlir::arith::ExtUIOp::create(rewriter, loc, signlessEltOf(ext.getType()),
                                        toSignlessInt(x, rewriter, loc))
        .getResult();
  }
  if (auto trunc = mlir::dyn_cast<mlir::arith::TruncIOp>(def)) {
    mlir::Value x =
        evalRank1ValueAt(trunc.getIn(), idxVal, rewriter, loc, depth + 1);
    if (!x)
      return nullptr;
    return mlir::arith::TruncIOp::create(rewriter, loc,
                                         signlessEltOf(trunc.getType()),
                                         toSignlessInt(x, rewriter, loc))
        .getResult();
  }

  // select / compare (tl.where conditions).
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    mlir::Value c =
        evalRank1ValueAt(sel.getCondition(), idxVal, rewriter, loc, depth + 1);
    mlir::Value t =
        evalRank1ValueAt(sel.getTrueValue(), idxVal, rewriter, loc, depth + 1);
    mlir::Value f =
        evalRank1ValueAt(sel.getFalseValue(), idxVal, rewriter, loc, depth + 1);
    if (!c || !t || !f)
      return nullptr;
    // arith.select requires both arms to share one type; a device-read arm
    // arrives as ui32 while a rebuilt-constant arm is signless i32.
    return mlir::arith::SelectOp::create(rewriter, loc, c,
                                         toSignlessInt(t, rewriter, loc),
                                         toSignlessInt(f, rewriter, loc))
        .getResult();
  }
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def)) {
    mlir::Value a =
        evalRank1ValueAt(cmp.getLhs(), idxVal, rewriter, loc, depth + 1);
    mlir::Value b =
        evalRank1ValueAt(cmp.getRhs(), idxVal, rewriter, loc, depth + 1);
    if (!a || !b)
      return nullptr;
    return mlir::arith::CmpIOp::create(rewriter, loc, cmp.getPredicate(),
                                       toSignlessInt(a, rewriter, loc),
                                       toSignlessInt(b, rewriter, loc))
        .getResult();
  }
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def)) {
    mlir::Value a =
        evalRank1ValueAt(cmp.getLhs(), idxVal, rewriter, loc, depth + 1);
    mlir::Value b =
        evalRank1ValueAt(cmp.getRhs(), idxVal, rewriter, loc, depth + 1);
    if (!a || !b)
      return nullptr;
    return mlir::arith::CmpFOp::create(rewriter, loc, cmp.getPredicate(), a, b)
        .getResult();
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Wall 17 (Case C): rank-2 axis=1 reduce over a COMPUTED tile.
//
// The rank-2 device-load reduce body (`ReduceLowering`) reduces by re-reading
// the tile from a device `tt.load` at `base + r*N + n`. Softmax-style kernels
// (e.g. leet adder_transformer) instead reduce register-resident COMPUTED
// values — `score = (q·k)*scale; where(...); tl.max/sum(score, axis=1)` — so
// there is no single device tensor to re-read.
//
// `evalRank2ConeAt` re-derives one element (logical row `r`, column `nVal`) of
// such a cone as scalar Metal ops, recursing the PRE-conversion tensor IR down
// to persistent leaves (device loads, splat scalars, dense-splat constants).
// This is conversion-ordering-safe for the same reason the rank-1 B2.3 walker
// is: it never consumes an intermediate op's converted scalar — it rebuilds
// every node from scratch from leaves that survive conversion.
//
// `rowBase` is the precomputed per-row flat base (`r*N (+ scalarOff)`); each
// device load is read at `rowBase + nVal` (Increment 1 assumes every cone load
// shares the reduce tile's row stride N and per-program scalar offset, which
// holds for the softmax shape where all operands are `(batch, N)` tiles).
//
// Increment 1 producers: tt.load, f32 arith {add,sub,mul,div,maximumf,
// maxnumf} (BOTH operands recursed — two-tensor products like `q0*k0 + q1*k1`
// are supported), unary math {exp,sqrt,log,sin,cos,erf,rsqrt}, tt.splat and
// splat arith.constant. Broadcast/expand_dims (per-row / per-col), tl.where
// (arith.select + cmp), and prior-reduce-result broadcast are Increment 2.
// Returns null on any unsupported producer (caller bails to notifyMatchFailure).
//===----------------------------------------------------------------------===//
static mlir::Value evalRank2ConeAt(mlir::Value v, mlir::Value rVal,
                                   mlir::Value rowBase, mlir::Value nVal,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   mlir::Location loc, int depth) {
  if (depth > 24)
    return nullptr;
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);

  // A staged [M,N] tile resolves to buf[r*N + n]. Checked FIRST so the walk
  // never descends into a loop-carried recurrence it cannot re-emit.
  if (mlir::Value staged = readStagedTile(v, rVal, nVal, rewriter, loc))
    return staged;

  // Scalar splat → its scalar source (a per-tile-uniform value).
  if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>()) {
    if (!splat.getOperation()->getBlock())
      return nullptr; // stale (already converted)
    return splat.getSrc();
  }
  // Dense-splat constant → materialise the scalar element.
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat())
      return nullptr;
    return mlir::arith::ConstantOp::create(
               rewriter, loc, dense.getSplatValue<mlir::TypedAttr>())
        .getResult();
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return nullptr;

  // tt.load leaf: re-materialise the address from the load's OWN tt.addptr
  // chain, evaluated at (rVal, nVal).
  //
  // This used to read `device[rowBase + nVal]`, where `rowBase` is derived once
  // from the reduce's REPRESENTATIVE load. That silently assumes every load in
  // the cone shares the reduce tile's row stride N and per-program base — false
  // as soon as a cone load has a different shape. In the SSM scan the reduce
  // tile is (BLOCK_D, BLOCK_N) while `A` is (d_model, d_state), so A was read
  // with stride BLOCK_N and carrying `C`'s batch offset; it only came out right
  // when d_state == BLOCK_N and the launch was single-program, which is exactly
  // the corner the first tests happened to sit in.
  //
  // Walking the load's own chain also subsumes the fabricated `tgid*tpb*N`
  // program term: an address whose program offset is folded into the row tensor
  // (`offs_m = pid*BLOCK_M + arange`) yields it naturally from evaluating that
  // tensor at rVal, so nothing is added on top and nothing double-counts.
  if (auto load = mlir::dyn_cast<mlir::triton::LoadOp>(def)) {
    if (!load.getPtr().getDefiningOp<mlir::triton::AddPtrOp>())
      return nullptr;
    mlir::Value memref = findBaseMemref(load.getPtr(), rewriter);
    if (!memref)
      return nullptr;
    mlir::Type eltTy = mlir::cast<MetalMemRefType>(memref.getType()).getType();
    auto i32 = rewriter.getI32Type();
    mlir::Value addr;
    auto addTerm = [&](mlir::Value t) {
      if (t.getType() != i32)
        t = mlir::UnrealizedConversionCastOp::create(
                rewriter, loc, mlir::TypeRange{i32}, mlir::ValueRange{t})
                .getResult(0);
      addr = addr ? mlir::arith::AddIOp::create(rewriter, loc, addr, t)
                        .getResult()
                  : t;
    };
    mlir::Value cur = load.getPtr();
    while (true) {
      // Peel the shape ops a 2D tile address is built through: the row half is
      // an (M,1) addptr broadcast to (M,N), so stopping at tt.broadcast would
      // drop the row term entirely.
      bool peeled = true;
      while (peeled) {
        peeled = false;
        if (auto sp = cur.getDefiningOp<mlir::triton::SplatOp>()) {
          cur = sp.getSrc();
          peeled = true;
        } else if (auto bc = cur.getDefiningOp<mlir::triton::BroadcastOp>()) {
          cur = bc.getSrc();
          peeled = true;
        } else if (auto ed = cur.getDefiningOp<mlir::triton::ExpandDimsOp>()) {
          cur = ed.getSrc();
          peeled = true;
        }
      }
      auto ap = cur.getDefiningOp<mlir::triton::AddPtrOp>();
      if (!ap)
        break;
      mlir::Value off = ap.getOffset();
      if (mlir::isa<mlir::RankedTensorType>(off.getType())) {
        mlir::Value s = evalRank2ConeAt(off, rVal, rowBase, nVal, rewriter, loc,
                                        depth + 1);
        if (!s)
          return nullptr;
        addTerm(s);
      } else {
        addTerm(off);
      }
      cur = ap.getPtr();
    }
    if (!addr)
      return nullptr;
    mlir::Value idxUI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{addr})
            .getResult(0);
    return GetElementOp::create(rewriter, loc, eltTy, memref, idxUI32)
        .getResult();
  }

  // Unary math → metal.unary_exp.
  auto unary = [&](mlir::Value operand,
                   UnaryExpOperator k) -> mlir::Value {
    mlir::Value x =
        evalRank2ConeAt(operand, rVal, rowBase, nVal, rewriter, loc, depth + 1);
    if (!x)
      return nullptr;
    auto attr = UnaryExpOperatorAttr::get(rewriter.getContext(), k);
    return UnaryExpOp::create(rewriter, loc, x.getType(), attr, x).getResult();
  };
  if (auto e = mlir::dyn_cast<mlir::math::ExpOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::expOp);
  if (auto e = mlir::dyn_cast<mlir::math::SqrtOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::sqrtOp);
  if (auto e = mlir::dyn_cast<mlir::math::LogOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::logOp);
  if (auto e = mlir::dyn_cast<mlir::math::SinOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::sinOp);
  if (auto e = mlir::dyn_cast<mlir::math::CosOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::cosOp);
  if (auto e = mlir::dyn_cast<mlir::math::ErfOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::erfOp);
  if (auto e = mlir::dyn_cast<mlir::math::RsqrtOp>(def))
    return unary(e.getOperand(), UnaryExpOperator::rsqrtOp);

  // Binary f32 arith → metal.binary_exp (recurse BOTH operands).
  auto binary = [&](mlir::Value lhs, mlir::Value rhs,
                    BinaryExpOperator k) -> mlir::Value {
    mlir::Value a =
        evalRank2ConeAt(lhs, rVal, rowBase, nVal, rewriter, loc, depth + 1);
    mlir::Value b =
        evalRank2ConeAt(rhs, rVal, rowBase, nVal, rewriter, loc, depth + 1);
    if (!a || !b)
      return nullptr;
    auto attr = BinaryExpOperatorAttr::get(rewriter.getContext(), k);
    return BinaryExpOp::create(rewriter, loc, a.getType(), attr, a, b)
        .getResult();
  };
  if (auto o = mlir::dyn_cast<mlir::arith::AddFOp>(def))
    return binary(o.getLhs(), o.getRhs(), BinaryExpOperator::addOp);
  if (auto o = mlir::dyn_cast<mlir::arith::SubFOp>(def))
    return binary(o.getLhs(), o.getRhs(), BinaryExpOperator::subOp);
  if (auto o = mlir::dyn_cast<mlir::arith::MulFOp>(def))
    return binary(o.getLhs(), o.getRhs(), BinaryExpOperator::mulOp);
  if (auto o = mlir::dyn_cast<mlir::arith::DivFOp>(def))
    return binary(o.getLhs(), o.getRhs(), BinaryExpOperator::divOp);
  if (auto o = mlir::dyn_cast<mlir::arith::MaximumFOp>(def))
    return binary(o.getLhs(), o.getRhs(), BinaryExpOperator::maxOp);
  if (auto o = mlir::dyn_cast<mlir::arith::MaxNumFOp>(def))
    return binary(o.getLhs(), o.getRhs(), BinaryExpOperator::maxOp);

  // tt.broadcast: rank-preserving replication over unit dims. Recurse the
  // source at the SAME (row, col) — the source's own structure (expand_dims /
  // make_range / cmpi) carries the real per-row / per-col dependence. (The
  // broadcast may wrap a computed [1,N] / [M,1] tensor, e.g. `seq<=K` is
  // compared at [1,N] then broadcast to [M,N].)
  if (auto bc = mlir::dyn_cast<mlir::triton::BroadcastOp>(def))
    return evalRank2ConeAt(bc.getSrc(), rVal, rowBase, nVal, rewriter, loc,
                           depth + 1);
  // tt.expand_dims(rank-1, axis): unit dim at `axis`. axis==1 → per-row (rVal);
  // axis==0 → per-col (nVal). Delegate to the rank-1 evaluator.
  if (auto ed = mlir::dyn_cast<mlir::triton::ExpandDimsOp>(def)) {
    auto srcRtt = mlir::dyn_cast<mlir::RankedTensorType>(ed.getSrc().getType());
    if (!srcRtt || srcRtt.getRank() != 1)
      return nullptr;
    mlir::Value idx = (ed.getAxis() == 1) ? rVal : nVal;
    return evalRank1ValueAt(ed.getSrc(), idx, rewriter, loc, depth + 1);
  }
  // Shape-only cvt / reshape: recurse the source unchanged.
  if (mlir::isa<mlir::triton::gpu::ConvertLayoutOp, mlir::triton::ReshapeOp>(
          def))
    return evalRank2ConeAt(def->getOperand(0), rVal, rowBase, nVal, rewriter,
                           loc, depth + 1);

  // tl.where → arith.select; comparisons feeding the where condition.
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    mlir::Value c = evalRank2ConeAt(sel.getCondition(), rVal, rowBase, nVal,
                                    rewriter, loc, depth + 1);
    mlir::Value t = evalRank2ConeAt(sel.getTrueValue(), rVal, rowBase, nVal,
                                    rewriter, loc, depth + 1);
    mlir::Value f = evalRank2ConeAt(sel.getFalseValue(), rVal, rowBase, nVal,
                                    rewriter, loc, depth + 1);
    if (!c || !t || !f)
      return nullptr;
    return mlir::arith::SelectOp::create(rewriter, loc, c, t, f).getResult();
  }
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def)) {
    mlir::Value a = evalRank2ConeAt(cmp.getLhs(), rVal, rowBase, nVal, rewriter,
                                    loc, depth + 1);
    mlir::Value b = evalRank2ConeAt(cmp.getRhs(), rVal, rowBase, nVal, rewriter,
                                    loc, depth + 1);
    if (!a || !b)
      return nullptr;
    return mlir::arith::CmpIOp::create(rewriter, loc, cmp.getPredicate(), a, b)
        .getResult();
  }
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def)) {
    mlir::Value a = evalRank2ConeAt(cmp.getLhs(), rVal, rowBase, nVal, rewriter,
                                    loc, depth + 1);
    mlir::Value b = evalRank2ConeAt(cmp.getRhs(), rVal, rowBase, nVal, rewriter,
                                    loc, depth + 1);
    if (!a || !b)
      return nullptr;
    return mlir::arith::CmpFOp::create(rewriter, loc, cmp.getPredicate(), a, b)
        .getResult();
  }

  // int binary arith → raw scalar arith.
  auto ibinary = [&](mlir::Value lhs, mlir::Value rhs,
                     auto make) -> mlir::Value {
    mlir::Value a =
        evalRank2ConeAt(lhs, rVal, rowBase, nVal, rewriter, loc, depth + 1);
    mlir::Value b =
        evalRank2ConeAt(rhs, rVal, rowBase, nVal, rewriter, loc, depth + 1);
    if (!a || !b)
      return mlir::Value{};
    return make(a, b);
  };
  if (auto o = mlir::dyn_cast<mlir::arith::AddIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::AddIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::SubIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::SubIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::MulIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::MulIOp::create(rewriter, loc, a, b).getResult();
    });
  // Boolean combine of per-element predicates — the rank-2 mask
  // `(rows<M) & (cols<N)` is `andi(broadcast(cmpi), broadcast(cmpi))`.
  if (auto o = mlir::dyn_cast<mlir::arith::AndIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::AndIOp::create(rewriter, loc, a, b).getResult();
    });
  if (auto o = mlir::dyn_cast<mlir::arith::OrIOp>(def))
    return ibinary(o.getLhs(), o.getRhs(), [&](mlir::Value a, mlir::Value b) {
      return mlir::arith::OrIOp::create(rewriter, loc, a, b).getResult();
    });

  return nullptr;
}

// Find a representative `tt.load` anywhere in a computed cone — used to derive
// the per-program scalar offset and to validate the cone is device-rooted.
static mlir::triton::LoadOp findFirstLoadInCone(mlir::Value v, int depth) {
  if (depth > 16)
    return nullptr;
  // Inc 2.5: a staged leaf is opaque (resolved via getRemappedValue); don't
  // descend into its cone looking for the representative device load. Same for
  // a staged [M,N] tile, which reads out of threadgroup memory.
  if (g_stagedLeaves && g_stagedLeaves->count(v))
    return nullptr;
  if (g_tileBuffers && g_tileBuffers->count(v))
    return nullptr;
  if (g_reduceRowBufs && g_reduceRowBufs->count(v))
    return nullptr;
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return nullptr;
  if (auto load = mlir::dyn_cast<mlir::triton::LoadOp>(def))
    return load;
  for (mlir::Value operand : def->getOperands())
    if (auto l = findFirstLoadInCone(operand, depth + 1))
      return l;
  return nullptr;
}

// Mirrors `scalarizeConeAtIndex`'s accepted index producers (predicate only).
static bool indexConeSupported(mlir::Value v, int depth) {
  if (depth > 24)
    return false;
  if (!mlir::isa<mlir::RankedTensorType>(v.getType()))
    return true; // scalar
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (mlir::isa<mlir::triton::ExpandDimsOp, mlir::triton::BroadcastOp,
                mlir::triton::gpu::ConvertLayoutOp, mlir::triton::ReshapeOp>(
          def))
    return indexConeSupported(def->getOperand(0), depth + 1);
  if (mlir::isa<mlir::triton::MakeRangeOp>(def))
    return true;
  if (auto sp = mlir::dyn_cast<mlir::triton::SplatOp>(def))
    return sp.getOperation()->getBlock() != nullptr;
  if (mlir::isa<mlir::arith::AddIOp, mlir::arith::MulIOp, mlir::arith::SubIOp>(
          def))
    return indexConeSupported(def->getOperand(0), depth + 1) &&
           indexConeSupported(def->getOperand(1), depth + 1);
  return false;
}

// Mirrors `evalRank1ValueAt`'s accepted producers (predicate only).
static bool rank1ConeSupported(mlir::Value v, int depth) {
  if (depth > 24)
    return false;
  // Inc 2.5: a staged per-row leaf is accepted (read from the thread's own
  // scalar during the inline fill).
  if (g_stagedLeaves && g_stagedLeaves->count(v))
    return true;
  if (g_reduceRowBufs && g_reduceRowBufs->count(v))
    return true;
  // W-C scan: a scan-result placeholder (buffer registered) is accepted.
  if (g_scanBuffers && g_scanBuffers->count(v))
    return true;
  if (!mlir::isa<mlir::RankedTensorType>(v.getType()))
    return true; // scalar
  if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>())
    return splat.getOperation()->getBlock() != nullptr;
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    return dense && dense.isSplat();
  }
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (mlir::isa<mlir::triton::ExpandDimsOp, mlir::triton::BroadcastOp,
                mlir::triton::gpu::ConvertLayoutOp, mlir::triton::ReshapeOp>(
          def))
    return rank1ConeSupported(def->getOperand(0), depth + 1);
  // Tensor-yielding scf.if with a scalar (uniform) condition → per-element
  // select of the two branch yields.
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
    if (ifOp.getElseRegion().empty() ||
        mlir::isa<mlir::RankedTensorType>(ifOp.getCondition().getType()))
      return false;
    unsigned resIdx = mlir::cast<mlir::OpResult>(v).getResultNumber();
    return rank1ConeSupported(ifOp.thenYield().getOperand(resIdx), depth + 1) &&
           rank1ConeSupported(ifOp.elseYield().getOperand(resIdx), depth + 1);
  }
  if (mlir::isa<mlir::triton::MakeRangeOp>(def))
    return true;
  if (auto load = mlir::dyn_cast<mlir::triton::LoadOp>(def)) {
    auto ap = load.getPtr().getDefiningOp<mlir::triton::AddPtrOp>();
    if (!ap || !indexConeSupported(ap.getOffset(), 0))
      return false;
    // Masked loads are handled by an scf.if guard; the mask + other cones must
    // themselves be evaluable at an index.
    if (load.getMask() && !rank1ConeSupported(load.getMask(), depth + 1))
      return false;
    if (load.getOther() && !rank1ConeSupported(load.getOther(), depth + 1))
      return false;
    return true;
  }
  if (auto s = mlir::dyn_cast<mlir::arith::SIToFPOp>(def))
    return rank1ConeSupported(s.getIn(), depth + 1);
  // Integer width casts mirror the ExtUIOp/TruncIOp cases in evalRank1ValueAt.
  if (mlir::isa<mlir::arith::ExtFOp, mlir::arith::TruncFOp,
                mlir::arith::ExtUIOp, mlir::arith::TruncIOp>(def))
    return rank1ConeSupported(def->getOperand(0), depth + 1);
  if (mlir::isa<mlir::math::ExpOp, mlir::math::SqrtOp, mlir::math::LogOp,
                mlir::math::SinOp, mlir::math::CosOp, mlir::math::ErfOp,
                mlir::math::RsqrtOp>(def))
    return rank1ConeSupported(def->getOperand(0), depth + 1);
  // NOTE: this list must stay in sync with the binary cases in
  // `evalRank1ValueAt` — this predicate is the dry run that decides whether the
  // evaluator gets to emit at all, so an op accepted there but missing here is
  // silently unreachable (that is exactly how `(v >> shift) & 0xF` failed to
  // legalize even after the evaluator learned shifts).
  if (mlir::isa<mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
                mlir::arith::DivFOp, mlir::arith::MaximumFOp,
                mlir::arith::MaxNumFOp, mlir::arith::AddIOp, mlir::arith::SubIOp,
                mlir::arith::MulIOp, mlir::arith::AndIOp, mlir::arith::OrIOp,
                mlir::arith::XOrIOp, mlir::arith::ShLIOp, mlir::arith::ShRSIOp,
                mlir::arith::ShRUIOp,
                mlir::arith::CmpIOp, mlir::arith::CmpFOp>(def))
    return rank1ConeSupported(def->getOperand(0), depth + 1) &&
           rank1ConeSupported(def->getOperand(1), depth + 1);
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def))
    return rank1ConeSupported(sel.getCondition(), depth + 1) &&
           rank1ConeSupported(sel.getTrueValue(), depth + 1) &&
           rank1ConeSupported(sel.getFalseValue(), depth + 1);
  return false;
}

// Predicate mirroring `evalRank2ConeAt`'s accepted producers. Run BEFORE the
// reduce loop emits anything so an unsupported cone yields a clean
// notifyMatchFailure instead of relying on conversion rollback.
static bool rank2ConeSupported(mlir::Value v, int depth) {
  if (depth > 24)
    return false;
  // A staged [M,N] tile is a leaf: it reads out of threadgroup memory, so the
  // walk stops here and never has to re-emit the recurrence behind it.
  if (g_tileBuffers && g_tileBuffers->count(v))
    return true;
  if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>())
    return splat.getOperation()->getBlock() != nullptr;
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    return dense && dense.isSplat();
  }
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (auto load = mlir::dyn_cast<mlir::triton::LoadOp>(def))
    return load.getPtr().getDefiningOp<mlir::triton::AddPtrOp>() != nullptr;
  if (auto bc = mlir::dyn_cast<mlir::triton::BroadcastOp>(def))
    return rank2ConeSupported(bc.getSrc(), depth + 1);
  if (auto ed = mlir::dyn_cast<mlir::triton::ExpandDimsOp>(def)) {
    auto srcRtt = mlir::dyn_cast<mlir::RankedTensorType>(ed.getSrc().getType());
    return srcRtt && srcRtt.getRank() == 1 &&
           rank1ConeSupported(ed.getSrc(), depth + 1);
  }
  if (mlir::isa<mlir::triton::gpu::ConvertLayoutOp, mlir::triton::ReshapeOp>(
          def))
    return rank2ConeSupported(def->getOperand(0), depth + 1);
  if (mlir::isa<mlir::math::ExpOp, mlir::math::SqrtOp, mlir::math::LogOp,
                mlir::math::SinOp, mlir::math::CosOp, mlir::math::ErfOp,
                mlir::math::RsqrtOp>(def))
    return rank2ConeSupported(def->getOperand(0), depth + 1);
  if (mlir::isa<mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
                mlir::arith::DivFOp, mlir::arith::MaximumFOp,
                mlir::arith::MaxNumFOp, mlir::arith::AddIOp, mlir::arith::SubIOp,
                mlir::arith::MulIOp, mlir::arith::CmpIOp, mlir::arith::CmpFOp>(
          def))
    return rank2ConeSupported(def->getOperand(0), depth + 1) &&
           rank2ConeSupported(def->getOperand(1), depth + 1);
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def))
    return rank2ConeSupported(sel.getCondition(), depth + 1) &&
           rank2ConeSupported(sel.getTrueValue(), depth + 1) &&
           rank2ConeSupported(sel.getFalseValue(), depth + 1);
  return false;
}

// Inc 2.5: collect the per-row (expand_dims axis=1) leaves in a reduce cone that
// are NOT re-emittable (loop-carried / computed values like q0_rope). At M<=tpb
// these are staged via getRemappedValue during an inline fill; re-emittable
// leaves (seq_idx, masks) stay on the normal path.
static void collectStagingLeaves(mlir::Value v, int depth,
                                 llvm::SmallVectorImpl<mlir::Value> &out,
                                 llvm::SmallPtrSetImpl<void *> &seen) {
  if (depth > 24)
    return;
  auto *def = v.getDefiningOp();
  if (!def)
    return;
  if (auto ed = mlir::dyn_cast<mlir::triton::ExpandDimsOp>(def)) {
    mlir::Value src = ed.getSrc();
    if (!rank1ConeSupported(src, 0) &&
        seen.insert(src.getAsOpaquePointer()).second)
      out.push_back(src);
    return;
  }
  for (auto operand : def->getOperands())
    collectStagingLeaves(operand, depth + 1, out, seen);
}

// W-C: `tt.scan` (cumsum) — inclusive prefix-sum over a rank-1 f32 tensor.
//
// Emits a DISTRIBUTED prefix-sum into a threadgroup buffer `scanbuf[BLOCK]`:
//   1. fill an `inbuf[BLOCK]` threadgroup buffer with the scan INPUT cone,
//      re-derived per logical element via `evalRank1ValueAt` (each thread writes
//      its E = BLOCK/tpb owned positions pos = tid + k*tpb);
//   2. `metal.threadgroup_prefix_sum inbuf -> scanbuf` (the spike-validated
//      Hillis-Steele + iv-carry template);
//   3. replace the scan with a per-thread placeholder `scanbuf[tid]` (a valid
//      f32 for any per-thread consumer that is dead after its reduce lowers) and
//      register `g_scanBuffers[placeholder] = scanbuf` so the rich cone
//      evaluator reads `scanbuf[idx]` per element in the consuming reduce.
//
// Envelope: rank-1, f32, `arith.addf` combine, axis=0, reverse=false, BLOCK a
// multiple of tpb with E = BLOCK/tpb in [1,64] pow2 (mirrors the rank-1 reduce
// spt-fold envelope). The scan INPUT cone must be `rank1ConeSupported`.
struct ScanLowering
    : public mlir::OpConversionPattern<mlir::triton::ScanOp> {
  ScanLowering(mlir::TypeConverter &tc, mlir::MLIRContext *ctx,
               llvm::DenseMap<mlir::Value, mlir::Value> *scanBufs,
               const ScanBufPool *bufPool)
      : mlir::OpConversionPattern<mlir::triton::ScanOp>(tc, ctx),
        scanBufs(scanBufs), bufPool(bufPool) {}
  llvm::DenseMap<mlir::Value, mlir::Value> *scanBufs;
  const ScanBufPool *bufPool;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::ScanOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op.getSrcs().size() != 1 || op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "scan: single operand/result only");
    if (op.getAxis() != 0 || op.getReverse())
      return rewriter.notifyMatchFailure(op, "scan: axis=0 forward only");
    auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(op.getType(0));
    mlir::Type scanEltTy = rtt ? rtt.getElementType() : mlir::Type();
    const bool scanIsI32 = scanEltTy && scanEltTy.isInteger(32);
    if (!rtt || rtt.getRank() != 1 || !(scanEltTy.isF32() || scanIsI32))
      return rewriter.notifyMatchFailure(op, "scan: rank-1 f32/i32 only");

    // Combine must be a plain add (cumsum) of the matching element type.
    mlir::Operation *combine = nullptr;
    if (op->getNumRegions() > 0 && !op->getRegion(0).empty())
      for (auto &nested : op->getRegion(0).front()) {
        if (mlir::isa<mlir::triton::ScanReturnOp>(nested))
          continue;
        combine = &nested;
        break;
      }
    const bool combineIsAdd =
        combine && (scanIsI32 ? mlir::isa<mlir::arith::AddIOp>(combine)
                              : mlir::isa<mlir::arith::AddFOp>(combine));
    if (!combineIsAdd)
      return rewriter.notifyMatchFailure(op, "scan: only add (cumsum)");

    auto srcBlocked = mlir::dyn_cast_or_null<
        mlir::triton::gpu::BlockedEncodingAttr>(rtt.getEncoding());
    if (!srcBlocked)
      return rewriter.notifyMatchFailure(op, "scan: no blocked encoding");
    int64_t tpb = 1;
    for (auto t : srcBlocked.getThreadsPerWarp()) tpb *= t;
    for (auto w : srcBlocked.getWarpsPerCTA()) tpb *= w;
    if (tpb <= 0 || (tpb & (tpb - 1)) != 0)
      return rewriter.notifyMatchFailure(op, "scan: tpb not power-of-two");
    int64_t BLOCK = rtt.getDimSize(0);
    if (BLOCK <= 0)
      return rewriter.notifyMatchFailure(op, "scan: empty tile");
    // Sub-tpb tiles (BLOCK < tpb) are PADDED up to one full tpb window instead
    // of getting a partial-window prefix-sum template. The tail positions
    // [BLOCK, tpb) are filled with 0.0 — the identity for the addf combine — so
    // every prefix of a position < BLOCK is unaffected and the template runs
    // completely unchanged. That matters: a partial window would mean guarding
    // the template's buffer writes, and on Metal a threadgroup_barrier inside
    // divergent control flow is UB, so the guard and the barriers would have to
    // be interleaved by hand. Padding sidesteps the hazard entirely.
    //
    // BLOCK < tpb is otherwise reachable ONLY here: tpb = 32 * num_warps, so
    // BLOCK >= 32 can always be made to fit by lowering num_warps, but a
    // sub-warp BLOCK (1..16) cannot. tt.load / tt.store / tt.reduce already
    // handle sub-tpb tiles (see the store's `needGuard` path); tt.scan was the
    // one op that did not, which made num_warps decide whether a kernel with a
    // small cumsum compiled at all.
    const int64_t bufLen = std::max(BLOCK, tpb);
    const bool padded = BLOCK < bufLen;
    if (bufLen % tpb != 0)
      return rewriter.notifyMatchFailure(op, "scan: BLOCK not a multiple of tpb");
    int64_t E = bufLen / tpb;
    if (E < 1 || E > 64 || (E & (E - 1)) != 0)
      return rewriter.notifyMatchFailure(op, "scan: E=BLOCK/tpb outside [1,64] pow2");
    // NOTE: sizePerThread is irrelevant here. inbuf is indexed by LOGICAL
    // position and filled round-robin (thread t writes pos = t + k*tpb, k<E,
    // covering [0,BLOCK) exactly once); evalRank1ValueAt re-derives each value
    // from the cone by logical index, and the prefix-sum template runs in
    // logical order — all independent of how the source layout distributes
    // elements across threads. So spt=1 and spt>1 (e.g. V=1024 → spt=4) both work.

    mlir::Value scanInput = op.getSrcs().front();
    if (!rank1ConeSupported(scanInput, 0))
      return rewriter.notifyMatchFailure(op, "scan: input cone unsupported");

    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto i32 = rewriter.getI32Type();
    auto f32 = rewriter.getF32Type();

    // LOCAL thread id = global_tid - tgid*tpb (multi-program safe).
    mlir::Value tidG =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{ThreadIdOp::create(rewriter, loc, ui32,
                                                rewriter.getStringAttr("x"))
                                 .getResult()})
            .getResult(0);
    mlir::Value tgG =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{ThreadgroupIdOp::create(
                                 rewriter, loc, ui32,
                                 rewriter.getStringAttr("x"))
                                 .getResult()})
            .getResult(0);
    auto cTpb = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
    mlir::Value tgOff =
        mlir::arith::MulIOp::create(rewriter, loc, tgG, cTpb.getResult())
            .getResult();
    mlir::Value tidLocal =
        mlir::arith::SubIOp::create(rewriter, loc, tidG, tgOff).getResult();

    // Staging element type. The metal buffer/element ops reject SIGNLESS i32
    // (same constraint the rank-1 reduce documents at `storeTy`), so an i32
    // cumsum stages through ui32. That is exact for a SUM: two's-complement
    // addition is bit-identical for signed and unsigned operands, so wrapping
    // negative partial sums round-trip unchanged.
    mlir::Type stageTy = scanIsI32 ? mlir::Type(ui32) : mlir::Type(f32);
    // Prefer the function-entry pair `preprocessScanBuffers` reserved for this
    // scan; only fall back to a private allocation when the function was not
    // poolable (single scan, or a scan feeding a reduce — see that pre-pass).
    mlir::Value inbuf, scanbuf;
    bool pooled = false;
    if (bufPool) {
      auto it = bufPool->find(op.getOperation());
      if (it != bufPool->end()) {
        inbuf = it->second.first;
        scanbuf = it->second.second;
        pooled = true;
      }
    }
    if (!pooled) {
      auto bufTy = MetalMemRefType::get(rewriter.getContext(), stageTy, bufLen);
      inbuf = ThreadgroupAllocaOp::create(rewriter, loc, bufTy).getResult();
      scanbuf = ThreadgroupAllocaOp::create(rewriter, loc, bufTy).getResult();
    }

    // When the scan sits inside a loop — the FuncOpLowering tile loop, or a
    // user `tl.range` (speculative decoding scans inside `for v_offset`) —
    // `inbuf`/`scanbuf` are ONE static allocation reused every trip. This
    // trip's fill would then race the previous trip's reads of scanbuf: the
    // per-element placeholder read at the bottom sits inside the same loop, and
    // the prefix-sum template's trailing barrier is the last one before it. Same
    // write-after-read hazard the rank-1 butterfly and the rank-2 rowBuf fill
    // guard against, and likewise only observable once a threadgroup spans more
    // than one SIMD-group.
    // A POOLED pair adds the same hazard without any loop: the previous scan in
    // program order read `scanbuf` and this fill is about to overwrite it, so
    // the barrier is unconditional there.
    if (pooled || findOutermostScfFor(op))
      BarrierOp::create(rewriter, loc);

    // Fill inbuf: each thread writes its E owned positions pos = tid + k*tpb.
    for (int64_t k = 0; k < E; ++k) {
      mlir::Value pos = tidLocal;
      if (k > 0) {
        auto cKtpb = mlir::arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(k * tpb)));
        pos = mlir::arith::AddIOp::create(rewriter, loc, tidLocal,
                                          cKtpb.getResult())
                  .getResult();
      }
      // Padded tail: `pos` may land in [BLOCK, tpb), where the cone has no
      // element — evaluating it there would emit a device load past the end of
      // the source tensor. Evaluate at a clamped-to-zero index instead and
      // select the add-identity, branchless: no scf.if, so the barriers around
      // this fill stay uniform for the whole threadgroup.
      mlir::Value inRange;
      mlir::Value evalPos = pos;
      if (padded) {
        auto cBlock = mlir::arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(BLOCK)));
        auto cZeroI = mlir::arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32IntegerAttr(0));
        inRange = mlir::arith::CmpIOp::create(
                      rewriter, loc, mlir::arith::CmpIPredicate::slt, pos,
                      cBlock.getResult())
                      .getResult();
        evalPos = mlir::arith::SelectOp::create(rewriter, loc, inRange, pos,
                                                cZeroI.getResult())
                      .getResult();
      }
      mlir::Value val = evalRank1ValueAt(scanInput, evalPos, rewriter, loc, 0);
      if (!val)
        return rewriter.notifyMatchFailure(op, "scan: input eval failed");
      if (padded) {
        // Add-identity in the cone's own type; `val` is still signless here
        // (the bridge to the ui32 staging type happens just below).
        auto zeroAttr =
            scanIsI32
                ? mlir::cast<mlir::TypedAttr>(rewriter.getI32IntegerAttr(0))
                : mlir::cast<mlir::TypedAttr>(rewriter.getF32FloatAttr(0.0f));
        auto cZero = mlir::arith::ConstantOp::create(rewriter, loc, zeroAttr);
        val = mlir::arith::SelectOp::create(rewriter, loc, inRange,
                                            toSignlessInt(val, rewriter, loc),
                                            cZero.getResult())
                  .getResult();
      }
      // Bridge signless i32 -> the ui32 staging type the buffer ops require.
      if (val.getType() != stageTy)
        val = mlir::UnrealizedConversionCastOp::create(
                  rewriter, loc, mlir::TypeRange{stageTy}, mlir::ValueRange{val})
                  .getResult(0);
      mlir::Value posUI =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{pos})
              .getResult(0);
      StoreOp::create(rewriter, loc, val, inbuf, posUI);
    }

    ThreadgroupPrefixSumOp::create(rewriter, loc, inbuf, scanbuf,
                                   rewriter.getI64IntegerAttr(bufLen),
                                   rewriter.getI64IntegerAttr(tpb));

    // Per-element placeholder for any live per-thread consumer (e.g. a direct
    // `tl.store` of the cumsum); the consuming reduce instead re-reads
    // scanbuf[idx] via g_scanBuffers.
    //
    // The index must be THIS ITERATION'S logical element index, not the thread
    // id. Under the FuncOpLowering tile loop each thread owns E = BLOCK/tpb
    // elements (idx = tid*E + iv when contiguous), and the placeholder is
    // re-evaluated once per iteration. Indexing by `tid` made every one of a
    // thread's E elements read the SAME scanbuf slot, so a stored cumsum came
    // out as `out[i] == scan[i/E]` — silently wrong for any BLOCK > tpb, which
    // is every natural cumsum block size (256/512/1024). E == 1 keeps the plain
    // scanbuf[tid] emission, byte-identical to before.
    mlir::Value idxUI;
    auto resTile = tileFromTensor(op.getType(0));
    mlir::scf::ForOp tileLoop = findOutermostScfFor(op);
    if (resTile && resTile->elemPerThread > 1 && tileLoop) {
      idxUI = emitPerIterIndex(*resTile, tileLoop, rewriter, loc);
    } else {
      idxUI = mlir::UnrealizedConversionCastOp::create(
                  rewriter, loc, mlir::TypeRange{ui32},
                  mlir::ValueRange{tidLocal})
                  .getResult(0);
    }
    auto placeholderOp =
        GetElementOp::create(rewriter, loc, stageTy, scanbuf, idxUI);
    // Pin the read to HERE. `scanbuf` is overwritten by the next pooled scan,
    // or by the next trip of the tile loop; the emitter inlines a single-use
    // value at its use site, which would move this read past that overwrite and
    // hand the consumer the WRONG scan's prefix sums. (Caught by a 2-cumsum
    // local-rank probe: both `tl.where` arms ended up reading one buffer after
    // both prefix sums had run.)
    if (pooled || findOutermostScfFor(op))
      placeholderOp->setAttr("metal.materialize", rewriter.getUnitAttr());
    mlir::Value placeholder = placeholderOp.getResult();
    // Key by the ORIGINAL scan result: a conversion `replaceOp` is lazy, so the
    // consuming reduce's cone walk (rank1ConeSupported / evalRank1ValueAt) still
    // sees `op.getResult()`, not the placeholder. The placeholder is only the
    // remapped operand handed to the (dead) per-thread consumers via adaptor.
    mlir::Value scanResult = op->getResult(0);
    (*scanBufs)[scanResult] = scanbuf;
    rewriter.replaceOp(op, placeholder);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Rank-2 axis=0 (per-COLUMN) reduce → rank-1 [N] (Session L3a2).
//
// `tt.reduce(tile[BM,BN], axis=0)` sums each column over its BM rows. Unlike
// the axis=1 reduce (per-row, needs a threadgroup rowBuf + cross-thread
// cooperation), axis=0's output columns are INDEPENDENT — each is owned by one
// output thread, which sums `device[offs(m, myCol)]` over m in [0, BM) locally.
// No threadgroup buffer, no barrier.
//
// The tutorial-05 backward's `_layer_norm_bwd_dwdb` reduces a LOOP-CARRIED 2D
// accumulator (`dw += load_tile`) over axis 0; `reassociateLoopCarriedAxis0-
// Reduce` rewrites that to `scf.for(s1d += reduce(load_tile, axis=0))` so the
// reduce here sees a per-iteration direct masked load. f32 sum, E_out==1.
//
// The offset/mask are evaluated per (row m, output col) via `evalRank2ConeAt`
// on the load's own offset and mask cones — so the runtime row stride N and the
// per-program column base (pid*BN) are recovered from the cone, not extracted.
static mlir::LogicalResult
lowerRank2Axis0Reduce(mlir::triton::ReduceOp op,
                      mlir::ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto rtt =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getSrcs().front().getType());
  if (!rtt || rtt.getRank() != 2 || op.getAxis() != 0)
    return mlir::failure();
  mlir::Type eltTy = rtt.getElementType();
  if (rtt.isDynamicDim(0) || rtt.isDynamicDim(1) ||
      !(eltTy.isF32() || eltTy.isInteger(32)))
    return mlir::failure();
  mlir::Operation *combine = nullptr;
  if (op->getNumRegions() > 0 && !op->getRegion(0).empty())
    for (auto &n : op->getRegion(0).front())
      if (!mlir::isa<mlir::triton::ReduceReturnOp>(n)) {
        combine = &n;
        break;
      }
  if (!combine)
    return mlir::failure();

  // Combine kind → accumulator element type, identity (== the masked-out per-row
  // value so masked rows are inert), and the scalar combine op (all
  // translator-supported). f32 sum/max and i32 sum/max/min.
  enum { SumF, MaxF, SumI, MaxI, MinI } kind;
  if (mlir::isa<mlir::arith::AddFOp>(combine))
    kind = SumF;
  else if (mlir::isa<mlir::arith::MaxNumFOp>(combine) ||
           mlir::isa<mlir::arith::MaximumFOp>(combine))
    kind = MaxF;
  else if (mlir::isa<mlir::arith::AddIOp>(combine))
    kind = SumI;
  else if (mlir::isa<mlir::arith::MaxSIOp>(combine))
    kind = MaxI;
  else if (mlir::isa<mlir::arith::MinSIOp>(combine))
    kind = MinI;
  else
    return rewriter.notifyMatchFailure(
        op, "rank-2 axis0 reduce: unsupported combine");
  bool isF = (kind == SumF || kind == MaxF);
  bool isSum = (kind == SumF || kind == SumI);
  if (isF != eltTy.isF32())
    return rewriter.notifyMatchFailure(
        op, "rank-2 axis0 reduce: combine/dtype mismatch");

  int64_t BM = rtt.getDimSize(0);
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
  auto i32 = rewriter.getI32Type();

  auto makeIdentity = [&]() -> mlir::Value {
    switch (kind) {
    case SumF:
      return mlir::arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getF32FloatAttr(0.0f))
          .getResult();
    case MaxF:
      return mlir::arith::ConstantOp::create(
                 rewriter, loc,
                 rewriter.getF32FloatAttr(-std::numeric_limits<float>::max()))
          .getResult();
    case SumI:
      return mlir::arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32IntegerAttr(0))
          .getResult();
    case MaxI:
      return mlir::arith::ConstantOp::create(
                 rewriter, loc, rewriter.getI32IntegerAttr(INT32_MIN))
          .getResult();
    default: // MinI
      return mlir::arith::ConstantOp::create(
                 rewriter, loc, rewriter.getI32IntegerAttr(INT32_MAX))
          .getResult();
    }
  };
  // Sum uses scalar arith (translator-supported). f32 max uses metal.binary_exp
  // (scalar arith.maxnumf has no scalar translator emit — the rank-1/axis1
  // reduces spell it the same way). i32 max/min use cmpi+select: metal.binary_exp
  // rejects signless i32, and select(a>b,a,b) is signed-correct and
  // translator-supported.
  auto combineScalar = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
    if (kind == SumF)
      return mlir::arith::AddFOp::create(rewriter, loc, a, b).getResult();
    if (kind == SumI)
      return mlir::arith::AddIOp::create(rewriter, loc, a, b).getResult();
    if (kind == MaxF) {
      auto opEnum = BinaryExpOperatorAttr::get(rewriter.getContext(),
                                               BinaryExpOperator::maxOp);
      return BinaryExpOp::create(rewriter, loc, a.getType(), opEnum, a, b)
          .getResult();
    }
    auto pred = (kind == MinI) ? mlir::arith::CmpIPredicate::slt
                               : mlir::arith::CmpIPredicate::sgt;
    mlir::Value c = mlir::arith::CmpIOp::create(rewriter, loc, pred, a, b)
                        .getResult();
    return mlir::arith::SelectOp::create(rewriter, loc, c, a, b).getResult();
  };

  // Peel one shape-only cvt hop to the real source.
  mlir::Value src = op.getSrcs().front();
  if (auto cvt = src.getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>())
    src = cvt.getSrc();

  // Uniform (splat / splat-constant) source: reduce(splat(c), axis=0) is BM*c
  // for sum, c for max/min. Covers the reassociation seed `reduce(init)`.
  {
    mlir::Value uniformScalar;
    if (auto sp = src.getDefiningOp<mlir::triton::SplatOp>())
      uniformScalar = sp.getSrc();
    else if (auto cst = src.getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue()))
        if (dense.isSplat())
          uniformScalar = mlir::arith::ConstantOp::create(
                              rewriter, loc,
                              dense.getSplatValue<mlir::TypedAttr>())
                              .getResult();
    }
    if (uniformScalar && uniformScalar.getType() == eltTy) {
      mlir::Value r = uniformScalar;
      if (isSum) {
        if (isF) {
          auto cBM = mlir::arith::ConstantOp::create(
              rewriter, loc, rewriter.getF32FloatAttr(static_cast<float>(BM)));
          r = mlir::arith::MulFOp::create(rewriter, loc, uniformScalar,
                                          cBM.getResult())
                  .getResult();
        } else {
          auto cBM = mlir::arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getI32IntegerAttr(static_cast<int32_t>(BM)));
          r = mlir::arith::MulIOp::create(rewriter, loc, uniformScalar,
                                          cBM.getResult())
                  .getResult();
        }
      }
      rewriter.replaceOp(op, r);
      return mlir::success();
    }
  }

  auto loadOp = src.getDefiningOp<mlir::triton::LoadOp>();
  if (!loadOp)
    return rewriter.notifyMatchFailure(
        op, "rank-2 axis0 reduce: source is not a device load or uniform splat");
  auto ap = loadOp.getPtr().getDefiningOp<mlir::triton::AddPtrOp>();
  if (!ap)
    return rewriter.notifyMatchFailure(op,
                                       "rank-2 axis0 reduce: load missing addptr");
  mlir::Value memref = findBaseMemref(loadOp.getPtr(), rewriter);
  if (!memref)
    return rewriter.notifyMatchFailure(op,
                                       "rank-2 axis0 reduce: base memref not found");
  mlir::Value offs = ap.getOffset();
  mlir::Value maskV = loadOp.getMask();

  // tpb from the source tile's blocked encoding (mirrors the axis=1 path — a
  // forward walk to the store is fragile under conversion ordering). Require
  // E_out == 1 (BN <= tpb) so each thread owns exactly one output column
  // c == localTid. BN > tpb is DEFERRED: at that point coalescing has moved the
  // output to a sizePerThread>1 layout (not the naive spt=1, E>1 the tile-loop
  // model assumes), so a per-iteration column index would silently mis-address.
  // The caller avoids it by launching tpb >= BN (as the layer-norm backward
  // does) or tiling the columns across the grid.
  auto srcBlocked =
      mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
          rtt.getEncoding());
  if (!srcBlocked)
    return rewriter.notifyMatchFailure(
        op, "rank-2 axis0 reduce: source not blocked-encoded");
  int64_t tpb = 1;
  for (auto t : srcBlocked.getThreadsPerWarp())
    tpb *= t;
  for (auto w : srcBlocked.getWarpsPerCTA())
    tpb *= w;
  int64_t BN = rtt.getDimSize(1);
  if (tpb <= 0 || BN > tpb)
    return rewriter.notifyMatchFailure(
        op, "rank-2 axis0 reduce: E_out>1 deferred (launch tpb>=BN)");

  // nVal = per-thread output column = localTid = globalTid - tgid*tpb.
  auto tidG =
      ThreadIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"));
  mlir::Value tidI = mlir::UnrealizedConversionCastOp::create(
                         rewriter, loc, mlir::TypeRange{i32},
                         mlir::ValueRange{tidG.getResult()})
                         .getResult(0);
  auto tgG = ThreadgroupIdOp::create(rewriter, loc, ui32,
                                     rewriter.getStringAttr("x"));
  mlir::Value tgI = mlir::UnrealizedConversionCastOp::create(
                        rewriter, loc, mlir::TypeRange{i32},
                        mlir::ValueRange{tgG.getResult()})
                        .getResult(0);
  auto cTpb = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
  mlir::Value tgOff =
      mlir::arith::MulIOp::create(rewriter, loc, tgI, cTpb.getResult())
          .getResult();
  mlir::Value nVal =
      mlir::arith::SubIOp::create(rewriter, loc, tidI, tgOff).getResult();
  mlir::Value dummyRowBase =
      mlir::arith::ConstantOp::create(rewriter, loc,
                                      rewriter.getI32IntegerAttr(0))
          .getResult();

  // s = combine over local row m in [0, BM) of (mask ? device[offs(m,nVal)]
  //                                                    : identity).
  mlir::Type loadEltTy = mlir::cast<MetalMemRefType>(memref.getType()).getType();
  auto cLo = mlir::arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32IntegerAttr(0));
  auto cHi = mlir::arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(BM)));
  auto cOne = mlir::arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getI32IntegerAttr(1));
  mlir::Value initV = makeIdentity();
  auto forOp = mlir::scf::ForOp::create(rewriter, loc, cLo.getResult(),
                                        cHi.getResult(), cOne.getResult(),
                                        mlir::ValueRange{initV});
  {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    mlir::Value m = forOp.getInductionVar();
    mlir::Value acc = forOp.getRegionIterArgs()[0];
    mlir::Value addrI =
        evalRank2ConeAt(offs, m, dummyRowBase, nVal, rewriter, loc, /*depth=*/0);
    if (!addrI)
      return rewriter.notifyMatchFailure(
          op, "rank-2 axis0 reduce: offset cone not evaluable");
    mlir::Value maskBit;
    if (maskV) {
      maskBit = evalRank2ConeAt(maskV, m, dummyRowBase, nVal, rewriter, loc, 0);
      if (!maskBit)
        return rewriter.notifyMatchFailure(
            op, "rank-2 axis0 reduce: mask cone not evaluable");
    }
    mlir::Value safeAddr = addrI;
    if (maskBit) {
      auto z = mlir::arith::ConstantOp::create(rewriter, loc,
                                               rewriter.getI32IntegerAttr(0));
      safeAddr = mlir::arith::SelectOp::create(rewriter, loc, maskBit, addrI,
                                               z.getResult())
                     .getResult();
    }
    mlir::Value idxU = mlir::UnrealizedConversionCastOp::create(
                           rewriter, loc, mlir::TypeRange{ui32},
                           mlir::ValueRange{safeAddr})
                           .getResult(0);
    mlir::Value v =
        GetElementOp::create(rewriter, loc, loadEltTy, memref, idxU).getResult();
    // Bridge the memref storage type (ui32 for an i32 buffer) to the signless
    // accumulator element type.
    if (loadEltTy != eltTy)
      v = mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{eltTy}, mlir::ValueRange{v})
              .getResult(0);
    if (maskBit)
      v = mlir::arith::SelectOp::create(rewriter, loc, maskBit, v,
                                        makeIdentity())
              .getResult();
    mlir::Value s = combineScalar(acc, v);
    mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{s});
  }
  rewriter.replaceOp(op, forOp.getResult(0));
  return mlir::success();
}

// True if `root`'s expression tree reads `loop`'s induction variable or any of
// its iter_args, i.e. the value genuinely changes from trip to trip.
//
// The rank-2 axis=1 reduce does NOT re-materialise the tile address from the
// load's offset expression; it fabricates `rowBase = (r + tgid*tpb)*N` and adds
// only the SCALAR tt.addptr offsets (accumulateScalarAddPtrOffsets drops
// tensor-typed offsets). That form has no trip term, and the fill is hoisted
// above the enclosing scf.for, so a trip-varying address would silently reduce
// trip 0's tile on every iteration. Detect it and bail rather than emit code
// that is quietly wrong.
//
// Lexical containment in the loop is deliberately NOT the test: an address that
// is merely *computed* inside the loop but loop-invariant stays correct under
// hoisting. Only a real dependence on the loop-carried values matters.
static bool readsLoopCarriedValue(mlir::Value root, mlir::scf::ForOp loop) {
  if (!loop || !root)
    return false;
  llvm::SmallPtrSet<mlir::Value, 8> carried;
  carried.insert(loop.getInductionVar());
  for (auto a : loop.getRegionIterArgs())
    carried.insert(a);
  llvm::SmallVector<mlir::Value, 16> wl{root};
  llvm::SmallPtrSet<mlir::Value, 16> seen;
  while (!wl.empty()) {
    mlir::Value v = wl.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (carried.contains(v))
      return true;
    if (auto *def = v.getDefiningOp())
      for (auto o : def->getOperands())
        wl.push_back(o);
  }
  return false;
}

// True if `target` appears in `root`'s backward cone. The walk stops at
// `barrier` (when set), so this answers "reachable WITHOUT going through
// `barrier`" — which is the question that matters once `barrier` is a staged
// leaf the cone evaluator never descends past.
static bool valueInCone(mlir::Value root, mlir::Value target,
                        mlir::Value barrier = {}) {
  llvm::SmallVector<mlir::Value, 16> wl{root};
  llvm::SmallPtrSet<mlir::Value, 16> seen;
  while (!wl.empty()) {
    mlir::Value v = wl.pop_back_val();
    if (v == target)
      return true;
    if (barrier && v == barrier)
      continue;
    if (!seen.insert(v).second)
      continue;
    if (auto *def = v.getDefiningOp())
      for (auto o : def->getOperands())
        wl.push_back(o);
  }
  return false;
}

// A loop-carried [M,N] f32 tile that the reduce depends on — the SSM selective
// scan's `h = exp(dt*A)*h + (dt*u)*B`, reduced to `y = sum(C*h, axis=1)` every
// trip.
//
// Not reachable by the Inc-2.5 per-row staging: that maps a leaf to ONE
// per-thread scalar, and `h` varies along the row as well. Not reachable by the
// axis-0 reassociation either: that needs the reduce OUTSIDE the loop and the
// update independent of the accumulator, and this recurrence is multiplicative
// in `h`. The tile has to be materialised.
// What the pre-pass hands ReduceLowering for one staged reduce.
//
// The split of labour is forced by WHEN each half can run:
//
//  * Detection and the buffer must be pre-conversion. By the time the pattern
//    runs, `populateSCFStructuralTypeConversions` has rebuilt the loop with
//    per-thread scalar iter_args, and the ORIGINAL body block is detached
//    (its parentOp is null), so `getParentOfType<scf::ForOp>` no longer leads
//    anywhere useful. Measured, not assumed.
//  * The cone VALUES survive that rebuild intact — `iterArg` is still a live
//    `tensor<MxNxf32>` BlockArgument and the update cone is still tensor-typed
//    arith — which is why the per-(r, n) emission can stay in the pattern,
//    where `evalRank2ConeAt` and the row/column loop machinery already live.
//
// Same shape as `preprocessMaskedStoreSentinels`'s scratch map: the pre-pass
// allocates and records, keyed by the consuming op.
struct StagedTileInfo {
  mlir::Value buf;             // !metal.memref<M*N x f32>, seeded with h_0
  mlir::BlockArgument iterArg; // h at trip entry -> buf[r*N + n]
  mlir::Value yielded;         // h after the update -> buf[r*N + n]
  int64_t n;
};
using LoopCarriedTileMap = llvm::DenseMap<mlir::Operation *, StagedTileInfo>;

// Pre-conversion. Find `scf.for`s carrying a rank-2 f32 tile that a rank-2
// axis=1 `tt.reduce` inside the body consumes, allocate a threadgroup buffer
// for each, seed it with the loop's init value, and record the pairing.
//
// Restricted to M <= tpb, where the reduce's grid-stride row fill degenerates
// to one row per thread (r == localTid). Thread r then owns row r of the tile
// outright — seed, update and read all happen on that one thread — so the
// buffer needs no barrier at all. Above tpb a row would be shared and the
// update/read would need synchronising; rejected rather than raced.
static void preprocessLoopCarriedReduceTiles(mlir::ModuleOp moduleOp,
                                             LoopCarriedTileMap &tileMap) {
  moduleOp.walk([&](mlir::triton::FuncOp funcOp) {
    if (funcOp.getBody().empty())
      return;
    llvm::SmallVector<mlir::scf::ForOp> loops;
    funcOp.walk([&](mlir::scf::ForOp f) { loops.push_back(f); });
    for (auto loop : loops) {
      auto yieldOp =
          mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
      if (!yieldOp)
        continue;
      auto iterArgs = loop.getRegionIterArgs();
      for (unsigned i = 0; i < iterArgs.size(); ++i) {
        mlir::BlockArgument arg = iterArgs[i];
        auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
        if (!rtt || rtt.getRank() != 2 || !rtt.getElementType().isF32())
          continue;
        if (i >= yieldOp.getNumOperands())
          continue;
        int64_t M = rtt.getDimSize(0), N = rtt.getDimSize(1);
        if (M <= 0 || N <= 0)
          continue;
        auto blocked = mlir::dyn_cast_or_null<
            mlir::triton::gpu::BlockedEncodingAttr>(rtt.getEncoding());
        if (!blocked)
          continue;
        int64_t tpb = 1;
        for (auto t : blocked.getThreadsPerWarp())
          tpb *= t;
        for (auto w : blocked.getWarpsPerCTA())
          tpb *= w;
        if (tpb <= 0 || M > tpb)
          continue;
        // h_0 must be materialisable as one scalar; every real init is
        // `tl.zeros(...)`.
        auto initCst =
            loop.getInitArgs()[i].getDefiningOp<mlir::arith::ConstantOp>();
        auto initDense =
            initCst ? mlir::dyn_cast<mlir::DenseElementsAttr>(initCst.getValue())
                    : nullptr;
        if (!initDense || !initDense.isSplat())
          continue;
        mlir::Value yielded = yieldOp.getOperand(i);

        // Which reduces in this body does the tile serve? The reduce must read
        // the UPDATED tile: one buffer holds one trip of state, and by the time
        // the reduce runs it holds the post-update value, so a cone reaching
        // the iter_arg on its own — not merely through `yielded`, which the
        // evaluator stops at — would silently get the previous trip.
        llvm::SmallVector<mlir::triton::ReduceOp> targets;
        loop.getBody()->walk([&](mlir::triton::ReduceOp red) {
          if (red.getSrcs().size() != 1 || red.getAxis() != 1)
            return;
          mlir::Value src = red.getSrcs().front();
          auto srcRtt = mlir::dyn_cast<mlir::RankedTensorType>(src.getType());
          if (!srcRtt || srcRtt.getRank() != 2 ||
              srcRtt.getDimSize(0) != M || srcRtt.getDimSize(1) != N)
            return;
          if (!valueInCone(src, yielded) ||
              valueInCone(src, arg, /*barrier=*/yielded))
            return;
          targets.push_back(red);
        });
        if (targets.empty())
          continue;

        // Alloca at FUNCTION ENTRY (threadgroup memory is function-scope, and
        // entry placement dominates unconditionally — the reason the previous
        // attempt at this shape hit "operand does not dominate this use").
        mlir::OpBuilder builder(funcOp.getContext());
        auto loc = loop.getLoc();
        auto ui32 = builder.getIntegerType(32, /*isSigned=*/false);
        builder.setInsertionPointToStart(&funcOp.getBody().front());
        auto bufTy = MetalMemRefType::get(funcOp.getContext(),
                                          rtt.getElementType(), M * N);
        mlir::Value buf =
            ThreadgroupAllocaOp::create(builder, loc, bufTy).getResult();

        // Seed BEFORE the loop, so a loop nested in an outer one re-seeds each
        // outer trip — matching the iter_arg init being re-evaluated there.
        builder.setInsertionPoint(loop);
        mlir::Value localTid = emitLocalTid(builder, loc, tpb);
        auto cM = mlir::arith::ConstantOp::create(
            builder, loc, builder.getI32IntegerAttr(static_cast<int32_t>(M)));
        auto cN = mlir::arith::ConstantOp::create(
            builder, loc, builder.getI32IntegerAttr(static_cast<int32_t>(N)));
        auto inBounds = mlir::arith::CmpIOp::create(
            builder, loc, mlir::arith::CmpIPredicate::slt, localTid,
            cM.getResult());
        auto seedIf = mlir::scf::IfOp::create(
            builder, loc, mlir::TypeRange{}, inBounds.getResult(),
            /*addThenBlock=*/true, /*addElseBlock=*/false);
        {
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(&seedIf.getThenRegion().front());
          auto seedVal = mlir::arith::ConstantOp::create(
              builder, loc, initDense.getSplatValue<mlir::TypedAttr>());
          auto zero = mlir::arith::ConstantOp::create(
              builder, loc, builder.getI32IntegerAttr(0));
          auto one = mlir::arith::ConstantOp::create(
              builder, loc, builder.getI32IntegerAttr(1));
          auto seedFor = mlir::scf::ForOp::create(
              builder, loc, zero.getResult(), cN.getResult(), one.getResult());
          {
            // ForOp::create supplies the body terminator; IfOp::create with
            // addThenBlock does NOT, so the then-block's scf.yield is added
            // explicitly below.
            mlir::OpBuilder::InsertionGuard g2(builder);
            builder.setInsertionPointToStart(seedFor.getBody());
            auto rowOff = mlir::arith::MulIOp::create(builder, loc, localTid,
                                                      cN.getResult());
            auto flat = mlir::arith::AddIOp::create(
                builder, loc, rowOff.getResult(), seedFor.getInductionVar());
            mlir::Value idx = mlir::UnrealizedConversionCastOp::create(
                                  builder, loc, mlir::TypeRange{ui32},
                                  mlir::ValueRange{flat.getResult()})
                                  .getResult(0);
            StoreOp::create(builder, loc, seedVal.getResult(), buf, idx);
          }
          mlir::scf::YieldOp::create(builder, loc);
        }

        for (auto red : targets)
          tileMap[red.getOperation()] = StagedTileInfo{buf, arg, yielded, N};
      }
    }
  });
}

struct ReduceLowering
    : public mlir::OpConversionPattern<mlir::triton::ReduceOp> {
  ReduceLowering(const mlir::TypeConverter &tc, mlir::MLIRContext *ctx,
                 const LoopCarriedTileMap *tiles,
                 llvm::DenseMap<mlir::Value, mlir::Value> *rowBufs)
      : OpConversionPattern(tc, ctx), tileMap(tiles), rowBufMap(rowBufs) {}
  const LoopCarriedTileMap *tileMap;
  llvm::DenseMap<mlir::Value, mlir::Value> *rowBufMap;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::ReduceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getSrcs().size() != 1)
      return mlir::failure();
    auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(
        op.getSrcs().front().getType());
    if (!rtt)
      return mlir::failure();
    if (rtt.getRank() == 1) {
      // Try the contiguous-masked full-reduce path first (subarray-sum shape);
      // fall back to the canonical rank-1 reduce.
      if (mlir::succeeded(lowerContiguousMaskedReduce(op, rewriter)))
        return mlir::success();
      return lowerRank1Reduce(op, adaptor, rewriter);
    }
    if (rtt.getRank() != 2)
      return mlir::failure();
    if (rtt.isDynamicDim(0) || rtt.isDynamicDim(1))
      return mlir::failure();
    if (op.getAxis() == 0)
      return lowerRank2Axis0Reduce(op, rewriter);
    if (op.getAxis() != 1)
      return mlir::failure();
    mlir::Type elemTy = rtt.getElementType();
    bool isF32 = elemTy.isF32();
    bool isI32 = elemTy.isInteger(32);
    if (!isF32 && !isI32)
      return mlir::failure();
    // Defensive combine-op check (the L3 pre-pass already enforced this).
    mlir::Operation *combine = nullptr;
    if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
      for (auto &nested : op->getRegion(0).front()) {
        if (mlir::isa<mlir::triton::ReduceReturnOp>(nested))
          continue;
        combine = &nested;
        break;
      }
    }
    if (!combine)
      return mlir::failure();
    // Combine kinds: f32 sum (arith.addf) / f32 max (arith.maxnumf or
    // arith.maximumf — Triton's tl.max emits maxnumf) / i32 sum (arith.addi).
    // Rank-2 i32 max (arith.maxsi) is not yet wired (the L3 pre-pass only
    // accepts maxsi for rank-1), so it stays rejected here.
    bool isAddF = mlir::isa<mlir::arith::AddFOp>(combine);
    bool isMaxF = mlir::isa<mlir::arith::MaximumFOp>(combine) ||
                  mlir::isa<mlir::arith::MaxNumFOp>(combine);
    bool isAddI = mlir::isa<mlir::arith::AddIOp>(combine);
    if (isF32 && !(isAddF || isMaxF))
      return mlir::failure();
    if (isI32 && !isAddI)
      return mlir::failure();

    auto loc = op.getLoc();
    int64_t M = rtt.getDimSize(0);
    int64_t N = rtt.getDimSize(1);
    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto i32 = rewriter.getI32Type();

    // ===================================================================
    // Rank-2 axis=1 reduce — self-contained per-row reduction (L3a-tileloop-2).
    //
    // The previous body assumed M*N == tpb (each thread owns one logical
    // (row, col) element, delivered as the per-thread `adaptor` scalar). That
    // is false whenever M*N > tpb: a thread then owns several tile elements
    // spread across the FuncOpLowering tile loop's iterations, so the single
    // per-thread scalar cannot express a row sum. The old chunked/single-pass
    // body therefore produced wrong results (and over-allocated threadgroup
    // memory) for every M*N > tpb shape.
    //
    // New model (mirrors `lowerRank1Reduce`): ignore the per-thread scalar and
    // the tile loop entirely; reduce straight from device memory. The reduce
    // input is a contiguous (M, N) tile loaded by a `tt.load` whose flat device
    // index for element (r, n) is `base + r*N + n`. Each thread cooperatively
    // sums a grid-strided set of whole rows into a threadgroup `rowBuf[M]`;
    // after a barrier every thread reads the row its downstream output store
    // targets.
    //
    // The staging (rowBuf fill + barrier) is hoisted ABOVE the FuncOpLowering
    // tile loop so it runs exactly once. The per-row result read stays inside
    // the loop. `findTileInfo` now sizes that loop from the reduce OUTPUT
    // (E_out) rather than the larger pre-reduce input, so the output store and
    // this read share the same per-iter index formula (a bijection over
    // [0, M)) — which is all that's required for correctness, independent of
    // the source/output blocked element-thread mapping.
    // ===================================================================
    mlir::Type storeTy = isI32 ? mlir::Type(ui32) : elemTy;

    // tpb from the source blocked encoding.
    auto srcBlocked = mlir::dyn_cast_or_null<
        mlir::triton::gpu::BlockedEncodingAttr>(rtt.getEncoding());
    if (!srcBlocked)
      return mlir::failure();
    int64_t tpb = 1;
    for (auto t : srcBlocked.getThreadsPerWarp()) tpb *= t;
    for (auto w : srcBlocked.getWarpsPerCTA()) tpb *= w;
    if (tpb <= 0)
      return mlir::failure();

    // Walk back to the producing tt.load (optionally through one cvt hop).
    mlir::Value reduceSrc = op.getSrcs().front();
    if (auto cvt =
            reduceSrc.getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>())
      reduceSrc = cvt.getSrc();
    auto loadOp = reduceSrc.getDefiningOp<mlir::triton::LoadOp>();

    // The per-row column reduction reads its elements from one of two sources:
    //   * a direct unmasked device `tt.load`  → read device[rowBase + n]; or
    //   * a COMPUTED cone (Wall 17 Case C, f32 only) → re-derive each element
    //     via `evalRank2ConeAt`. We still need ONE representative load to
    //     recover the per-program scalar offset and confirm the cone is
    //     device-rooted (a pure-splat reduce carries no per-row data).
    bool computedCone = false;
    mlir::Value coneRoot;           // = reduceSrc, fed to the cone evaluator
    mlir::triton::LoadOp reprLoad;  // representative load (scalarOff source)
    mlir::Value memref;             // direct-path base memref (null for cone)
    mlir::Type loadEltTy;           // direct-path element type (null for cone)
    // Inc 2.5: staged per-row leaves (loop-carried / computed values the cone
    // evaluator can't re-emit). `g_stagedLeaves` is cleared on any return.
    llvm::DenseMap<mlir::Value, mlir::Value> stagedMap;
    bool inlineStaged = false;
    // Inc 2.5 rank-2: loop-carried [M,N] tile staged in threadgroup memory.
    llvm::DenseMap<mlir::Value, StagedTile> tileUpdateMap, tileReadMap;
    std::optional<StagedTileInfo> carriedTile;
    mlir::Value tileBuf;
    struct StagedResetGuard {
      ~StagedResetGuard() {
        g_stagedLeaves = nullptr;
        g_tileBuffers = nullptr;
      }
    } stagedResetGuard;
    if (loadOp && !loadOp.getMask()) {
      reprLoad = loadOp;
    } else {
      if (!isF32)
        return rewriter.notifyMatchFailure(
            op, "rank-2 reduce: computed-cone src supported for f32 only");
      // Inc 2.5 (M<=tpb): each fill thread reduces its OWN row (r==localTid), so
      // a non-re-emittable per-row leaf (q0_rope) equals the thread's converted
      // per-thread scalar (getRemappedValue). Register these so the cone is
      // accepted and evalRank2ConeAt reads the staged value; the fill is then
      // emitted INLINE (not hoisted) so the leaf dominates.
      if (M <= tpb) {
        llvm::SmallVector<mlir::Value> leaves;
        llvm::SmallPtrSet<void *, 8> seen;
        collectStagingLeaves(reduceSrc, 0, leaves, seen);
        for (auto lf : leaves)
          if (mlir::Value remapped = rewriter.getRemappedValue(lf))
            stagedMap[lf] = remapped;
        if (!stagedMap.empty()) {
          g_stagedLeaves = &stagedMap;
          inlineStaged = true;
        }
      }
      // Inc 2.5 rank-2: a loop-carried [M,N] tile. Stage it in threadgroup
      // memory and make both cones treat it as a leaf. Two maps, because the
      // buffer means different things on either side of the update: while
      // emitting the update the iter_arg reads the PREVIOUS trip's value out of
      // it; afterwards it holds this trip's value and the reduce reads the
      // yielded tile from it.
      if (auto it = tileMap->find(op.getOperation()); it != tileMap->end()) {
        // The pre-pass already allocated and seeded the buffer; all that is
        // left is to point both cones at it.
        carriedTile = it->second;
        tileBuf = it->second.buf;
        tileUpdateMap[it->second.iterArg] = StagedTile{it->second.buf, N};
        tileReadMap[it->second.yielded] = StagedTile{it->second.buf, N};
      }
      if (carriedTile) {
        // The update cone must itself be re-emittable with the iter_arg
        // resolved out of the buffer.
        g_tileBuffers = &tileUpdateMap;
        bool updateOk = rank2ConeSupported(carriedTile->yielded, 0);
        g_tileBuffers = &tileReadMap;
        if (!updateOk) {
          g_tileBuffers = nullptr;
          return rewriter.notifyMatchFailure(
              op, "rank-2 reduce: loop-carried tile update has an unsupported "
                  "producer");
        }
      }
      if (!rank2ConeSupported(reduceSrc, 0))
        return rewriter.notifyMatchFailure(
            op, "rank-2 reduce: computed-cone src has an unsupported producer "
                "(Wall 17 Increment 1: load/arith/math/splat only)");
      reprLoad = findFirstLoadInCone(reduceSrc, 0);
      if (!reprLoad)
        return rewriter.notifyMatchFailure(
            op, "rank-2 reduce: computed src has no device load");
      computedCone = true;
      coneRoot = reduceSrc;
    }
    if (!reprLoad.getPtr().getDefiningOp<mlir::triton::AddPtrOp>())
      return rewriter.notifyMatchFailure(
          op, "rank-2 reduce: load missing tt.addptr");
    if (!computedCone) {
      memref = findBaseMemref(reprLoad.getPtr(), rewriter);
      if (!memref)
        return rewriter.notifyMatchFailure(
            op, "rank-2 reduce: base memref not found");
      loadEltTy = mlir::cast<MetalMemRefType>(memref.getType()).getType();
    }

    // Output tile info: the post-reduce (#blocked) layout the downstream store
    // consumes. The reduce result itself carries a #ttg.slice encoding (no
    // blocked tile); walk forward to the first blocked-typed user (the
    // convert_layout / store value) to recover the per-thread output indexing.
    //
    // Take the LARGEST such tensor, not the first. The first is the
    // `tt.expand_dims` shim the broadcast goes through — `tensor<Mx1xf32>` —
    // whose shape says nothing about the geometry the consuming threads
    // actually iterate. Reading the row out of a shape-[M,1] tile gives
    // `flat / 1`, i.e. the flat index, which for a chained softmax
    // (`p = exp(x - m[:, None])`) indexed rowBuf[M] with a value in [0, M*N).
    // `findTileInfo` skips the same 16x1 / 1x16 shims for the same reason.
    std::optional<TileInfo> outTile;
    {
      llvm::SmallVector<mlir::Operation *, 8> wl;
      llvm::SmallPtrSet<mlir::Operation *, 8> seen;
      int64_t bestSize = 0;
      auto consider = [&](mlir::Type t) {
        auto ti = tileFromTensor(t);
        if (!ti)
          return false;
        int64_t sz = 1;
        for (auto s : ti->shape)
          sz *= s;
        if (sz > bestSize) {
          bestSize = sz;
          outTile = ti;
        }
        return true;
      };
      for (auto *u : op->getResult(0).getUsers())
        wl.push_back(u);
      while (!wl.empty()) {
        auto *u = wl.pop_back_val();
        if (!seen.insert(u).second)
          continue;
        if (auto st = mlir::dyn_cast<mlir::triton::StoreOp>(u)) {
          consider(st.getValue().getType());
          continue;
        }
        for (auto res : u->getResults()) {
          consider(res.getType());
          for (auto *uu : res.getUsers())
            wl.push_back(uu);
        }
      }
    }

    // Combine enum: addOp for sum, maxOp for f32 max (lowers to MSL max(a,b)).
    auto combineEnum = BinaryExpOperatorAttr::get(
        rewriter.getContext(),
        isMaxF ? BinaryExpOperator::maxOp : BinaryExpOperator::addOp);
    mlir::scf::ForOp tileLoop = findOutermostScfFor(op);
    // Hoisting the fill above `tileLoop` is only sound when the tile it reduces
    // is the same on every trip. `findOutermostScfFor` cannot tell a
    // FuncOpLowering tile loop from a user `for` — when E==1 no tile loop is
    // created and this IS the user's loop — so ask the address instead. A
    // trip-varying address must be re-reduced every iteration, exactly like the
    // staged-cone case; hoisting it would reduce trip 0's tile every time.
    const bool loopVaryingAddr =
        tileLoop && readsLoopCarriedValue(reprLoad.getPtr(), tileLoop);
    // ...but only when the loop carries no iter_args. With iter_args the loop
    // is rebuilt by its own conversion pattern (its arg types change), which
    // moves the ops around the inline fill and leaves the hoisted rowBuf alloca
    // failing to dominate the read — "operand #0 does not dominate this use".
    // Reject up front: a clear unsupported-shape message beats either a
    // dominance crash or the silently-wrong hoisted reduce this used to emit.
    // ...but only when the loop carries iter_args we have NOT taken ownership
    // of. With a staged tile the iter_args are exactly what we replaced with
    // threadgroup memory, and the alloca sits above the loop, so the shape is
    // handled rather than rejected.
    if (loopVaryingAddr && !carriedTile &&
        !tileLoop.getRegionIterArgs().empty())
      return rewriter.notifyMatchFailure(
          op, "rank-2 reduce: tile address varies with an enclosing loop that "
              "also carries iter_args; the fill cannot be hoisted (wrong tile) "
              "nor emitted inline (rowBuf alloca would not dominate)");
    // inlineStaged (M<=tpb) reads rowBuf[localTid] (each thread its own row), so
    // it needs no output tile layout.
    if (tileLoop && !outTile && !inlineStaged)
      return rewriter.notifyMatchFailure(
          op, "rank-2 reduce: output tile layout not found");

    // -------- Staging. --------
    // The rowBuf threadgroup alloca is ALWAYS hoisted above the outermost loop
    // (threadgroup memory is function-scope). The FILL is hoisted for a
    // loop-invariant (device) cone (compute once) but emitted INLINE for a
    // staged cone (inlineStaged): the staged leaves' getRemappedValue only
    // dominates inside the loop, and the reduce must recompute each iteration.
    // A loop-varying address (loopVaryingAddr) is inline for the same reason.
    // A staged tile is likewise inline: the buffer advances one trip per fill,
    // so the fill IS the recurrence and must run every iteration.
    const bool fillInLoop = inlineStaged || loopVaryingAddr || carriedTile;
    mlir::Value rowBuf;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      if (tileLoop)
        rewriter.setInsertionPoint(tileLoop);
      auto rowBufTy =
          MetalMemRefType::get(rewriter.getContext(), storeTy, M);
      rowBuf = ThreadgroupAllocaOp::create(rewriter, loc, rowBufTy).getResult();
    }
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      if (tileLoop && !fillInLoop)
        rewriter.setInsertionPoint(tileLoop);

      // localTid = global_tid - tgid * tpb (multi-program safe).
      mlir::Value tidGlobalUI32 =
          ThreadIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"))
              .getResult();
      mlir::Value tidGlobalI32 =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{i32},
              mlir::ValueRange{tidGlobalUI32})
              .getResult(0);
      mlir::Value tgGlobalUI32 =
          ThreadgroupIdOp::create(rewriter, loc, ui32,
                                  rewriter.getStringAttr("x"))
              .getResult();
      mlir::Value tgGlobalI32 =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{i32},
              mlir::ValueRange{tgGlobalUI32})
              .getResult(0);
      auto cTpb = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
      auto tgOffset = mlir::arith::MulIOp::create(rewriter, loc, tgGlobalI32,
                                                  cTpb.getResult());
      mlir::Value localTid =
          mlir::arith::SubIOp::create(rewriter, loc, tidGlobalI32,
                                      tgOffset.getResult())
              .getResult();

      // Any scalar tt.addptr chain offset (e.g. row_base*stride for a
      // multi-program launch); null for the simple direct-load shape. For a
      // computed cone the representative load supplies the per-program offset.
      // Walks through tt.broadcast/tt.expand_dims so the per-program term of a
      // 2D tile address is actually found; when it is, it REPLACES the
      // fabricated tgid*tpb*N below instead of adding to it.
      mlir::Value scalarOff = accumulateScalarAddPtrOffsetsThroughShape(
          reprLoad.getPtr(), rewriter, loc);

      auto cN = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(N)));
      auto cM = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(M)));

      // When the fill runs inside a loop, `rowBuf` is one static allocation
      // reused every trip, so this trip's writes below would race the previous
      // trip's reads of it (the per-row read at the bottom sits inside the same
      // loop). Same write-after-read hazard the rank-1 butterfly guards
      // against, and likewise only observable once a threadgroup spans more
      // than one SIMD-group.
      if (fillInLoop)
        BarrierOp::create(rewriter, loc);

      // Grid-stride over rows: for ri in [0, ceil(M/tpb)): r = localTid+ri*tpb;
      // if (r < M) rowBuf[r] = sum_n device[base + r*N + n].
      int64_t nRowIters = (M + tpb - 1) / tpb;
      auto cRowLo = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(0));
      auto cRowHi = mlir::arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getI32IntegerAttr(static_cast<int32_t>(nRowIters)));
      auto cOne = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(1));
      auto cColLo = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(0));

      auto rowFor = mlir::scf::ForOp::create(
          rewriter, loc, cRowLo.getResult(), cRowHi.getResult(),
          cOne.getResult());
      {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(rowFor.getBody());
        mlir::Value ri = rowFor.getInductionVar();
        // r = localTid + ri * tpb
        auto riScaled =
            mlir::arith::MulIOp::create(rewriter, loc, ri, cTpb.getResult());
        mlir::Value r = mlir::arith::AddIOp::create(rewriter, loc, localTid,
                                                    riScaled.getResult())
                            .getResult();
        auto rLtM = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::slt, r, cM.getResult());
        auto guardIf = mlir::scf::IfOp::create(
            rewriter, loc, mlir::TypeRange{}, rLtM.getResult(),
            /*addThenBlock=*/true, /*addElseBlock=*/false);
        {
          mlir::OpBuilder::InsertionGuard g2(rewriter);
          rewriter.setInsertionPointToStart(&guardIf.getThenRegion().front());
          // rowBase: the device buffer is indexed by the GLOBAL row, while r is
          // the program-LOCAL row, so the per-program base has to come from
          // somewhere. Two shapes occur, and mixing them double-counts:
          //
          //  (a) The address carries the program term as a SCALAR addptr
          //      (`ptr + pid*S`, the usual `tl.load(p + pid*S + rows*N + cols)`
          //      form). scalarOff now recovers that real term, so use
          //      `r*N + scalarOff` verbatim. This is what makes a tile base
          //      other than pid*M*N (e.g. pid*K*M*N) come out right; the old
          //      fabricated term silently assumed S == tpb*N.
          //
          //  (b) No scalar term — the program offset is folded into the row
          //      tensor itself (`offs_m = pid*BLOCK_M + arange(...)`). Nothing
          //      is recoverable here, so keep the fabricated tgid.x*tpb. It is
          //      a no-op for a single-program launch, and without it program
          //      k>0 would reduce program 0's rows (multi-program adder →
          //      wrong scores → overflow/NaN).
          //
          // rowBuf itself stays LOCAL-indexed by r in both cases.
          mlir::Value rowIdx = r;
          if (!scalarOff)
            rowIdx = mlir::arith::AddIOp::create(rewriter, loc, r,
                                                 tgOffset.getResult())
                         .getResult();
          mlir::Value rowBase =
              mlir::arith::MulIOp::create(rewriter, loc, rowIdx, cN.getResult())
                  .getResult();
          if (scalarOff)
            rowBase = mlir::arith::AddIOp::create(rewriter, loc, rowBase,
                                                  scalarOff)
                          .getResult();
          // Loop-carried tile: advance this row's slice of the staged buffer
          // BEFORE reducing it. `h_new(r, n)` is re-emitted with the iter_arg
          // resolving to buf[r*N + n] (the previous trip's value), then written
          // back to that same slot. In place is safe: every op the rank-2 cone
          // admits is elementwise in (r, n), or a broadcast of a rank-1 value
          // indexed by r or n alone, so nothing reads the tile at a position
          // other than the one being written. Thread r owns row r outright
          // (M <= tpb), so no barrier separates the write from the read below.
          if (carriedTile) {
            auto updFor = mlir::scf::ForOp::create(
                rewriter, loc, cColLo.getResult(), cN.getResult(),
                cOne.getResult());
            mlir::OpBuilder::InsertionGuard g3(rewriter);
            rewriter.setInsertionPointToStart(updFor.getBody());
            mlir::Value nIv = updFor.getInductionVar();
            g_tileBuffers = &tileUpdateMap;
            mlir::Value hNew =
                evalRank2ConeAt(carriedTile->yielded, r, rowBase, nIv, rewriter,
                                loc, /*depth=*/0);
            g_tileBuffers = &tileReadMap;
            if (!hNew)
              return rewriter.notifyMatchFailure(
                  op, "rank-2 reduce: loop-carried tile update failed to "
                      "re-emit");
            auto rowOff =
                mlir::arith::MulIOp::create(rewriter, loc, r, cN.getResult());
            auto flat = mlir::arith::AddIOp::create(rewriter, loc,
                                                    rowOff.getResult(), nIv);
            mlir::Value idx = mlir::UnrealizedConversionCastOp::create(
                                  rewriter, loc, mlir::TypeRange{ui32},
                                  mlir::ValueRange{flat.getResult()})
                                  .getResult(0);
            StoreOp::create(rewriter, loc, hNew, tileBuf, idx);
          }

          // Inner column reduction: rowCombine = combine_n elem(rowBase + n).
          // For a direct load the element is device[rowBase + n]; for a
          // computed cone (Wall 17 Case C) it is re-derived per (row, col) by
          // `evalRank2ConeAt` reading the cone's device loads at rowBase + n.
          auto emitColLoad = [&](mlir::Value colOff) -> mlir::Value {
            if (computedCone)
              return evalRank2ConeAt(coneRoot, r, rowBase, colOff, rewriter,
                                     loc, /*depth=*/0);
            auto colIdx =
                mlir::arith::AddIOp::create(rewriter, loc, rowBase, colOff);
            mlir::Value colIdxUI32 =
                mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, mlir::TypeRange{ui32},
                    mlir::ValueRange{colIdx.getResult()})
                    .getResult(0);
            return GetElementOp::create(rewriter, loc, loadEltTy, memref,
                                        colIdxUI32)
                .getResult();
          };
          mlir::Value rowSum;
          if (isF32) {
            // f32: a single scf.for carrying one f32 iter_arg (identity-init
            // 0.0). Reuses Wall 15's translator path so MSL emission is O(1)
            // in N. The 0.0 init is emitted fresh above the for (Wall 15 R9).
            // Identity: 0.0 for sum; -FLT_MAX for max (the MSL float-constant
            // emitter can't render -inf, and max(x, -FLT_MAX) == x for finite x).
            auto initVal = mlir::arith::ConstantOp::create(
                rewriter, loc,
                rewriter.getF32FloatAttr(
                    isMaxF ? -std::numeric_limits<float>::max() : 0.0f));
            auto colFor = mlir::scf::ForOp::create(
                rewriter, loc, cColLo.getResult(), cN.getResult(),
                cOne.getResult(), mlir::ValueRange{initVal.getResult()});
            {
              mlir::OpBuilder::InsertionGuard g3(rewriter);
              rewriter.setInsertionPointToStart(colFor.getBody());
              mlir::Value nIv = colFor.getInductionVar();
              mlir::Value acc = colFor.getRegionIterArgs()[0];
              mlir::Value elt = emitColLoad(nIv);
              auto combined = BinaryExpOp::create(rewriter, loc, storeTy,
                                                  combineEnum, acc, elt);
              mlir::scf::YieldOp::create(rewriter, loc,
                                         mlir::ValueRange{combined.getResult()});
            }
            rowSum = colFor.getResult(0);
          } else {
            // i32 (ui32-storage): emit the column scan UNROLLED. The
            // translator only threads f32 scf.for iter_args (Wall 15); a ui32
            // iter_arg would hit ModuleTranslation's "Unexpected operation".
            // N (the column count) is small, so the bounded unroll is fine.
            for (int64_t j = 0; j < N; ++j) {
              auto cJ = mlir::arith::ConstantOp::create(
                  rewriter, loc,
                  rewriter.getI32IntegerAttr(static_cast<int32_t>(j)));
              mlir::Value elt = emitColLoad(cJ.getResult());
              if (!rowSum)
                rowSum = elt;
              else
                rowSum = BinaryExpOp::create(rewriter, loc, storeTy, combineEnum,
                                             rowSum, elt)
                             .getResult();
            }
          }
          mlir::Value rUI32 =
              mlir::UnrealizedConversionCastOp::create(
                  rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{r})
                  .getResult(0);
          StoreOp::create(rewriter, loc, rowSum, rowBuf, rUI32);
          mlir::scf::YieldOp::create(rewriter, loc);
        }
      }
      BarrierOp::create(rewriter, loc);
    }

    // -------- Per-row result read (inside the tile loop). --------
    // result = rowBuf[outIdx], where outIdx is the SAME per-iter index the
    // downstream store uses (a bijection over [0, M)).
    //
    // When the consumer is a RANK-2 tile the result is being broadcast back
    // into the tile (`p = exp(x - m[:, None])`), so element (r, n) needs
    // rowBuf[r] — not rowBuf[flat]. With order=[1,0] the per-iteration flat
    // index is r*N + n, so the row is flat / N. Reading rowBuf at the flat
    // index instead walks off the end of an [M] buffer: for a 32x64 tile that
    // is an index in [0, 2048) into 32 slots, which is how chained softmax
    // (`m` correct, `s` correct, `p` garbage) came out silently wrong.
    mlir::Value outIdxUI32;
    const bool outIs2D =
        outTile && outTile->rank == 2 && outTile->shape.size() == 2;
    auto rowOfFlat = [&](mlir::Value flatUI32) -> mlir::Value {
      mlir::Value flatI32 =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{i32}, mlir::ValueRange{flatUI32})
              .getResult(0);
      auto cOutN = mlir::arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getI32IntegerAttr(static_cast<int32_t>(outTile->shape[1])));
      mlir::Value row = mlir::arith::DivSIOp::create(rewriter, loc, flatI32,
                                                     cOutN.getResult())
                            .getResult();
      auto cMrow = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(M)));
      // Threads past the tile's rows would otherwise read past rowBuf; their
      // downstream store is already masked off.
      row = mlir::arith::RemSIOp::create(rewriter, loc, row, cMrow.getResult())
                .getResult();
      return mlir::UnrealizedConversionCastOp::create(
                 rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{row})
          .getResult(0);
    };
    //
    // The row decomposition applies ONLY when a tile loop exists, i.e. when the
    // thread's tile position really is `tid*E + iv`. Without one, the reduce
    // result's per-thread value is also what the Inc-2.5 staging hands a
    // CONSUMING reduce as `getRemappedValue`, and that contract is row ==
    // localTid (each fill thread owns its own row). Applying `flat / N` there
    // feeds the next reduce the wrong row — chained softmax's `s` regresses
    // exactly that way.
    if (tileLoop && outTile && outTile->elemPerThread > 1 && !inlineStaged) {
      mlir::Value flat = emitPerIterIndex(*outTile, tileLoop, rewriter, loc);
      outIdxUI32 = outIs2D ? rowOfFlat(flat) : flat;
    } else if (outIs2D && !inlineStaged) {
      // No tile loop (the tile is exactly tpb elements, E == 1): the thread's
      // flat tile position IS its local id, so the row is still flat / N.
      outIdxUI32 = rowOfFlat(emitLocalTidUI32(rewriter, loc,
                                              outTile->threadsPerBlock));
    } else {
      // No tile loop (M <= tpb): each thread holds one output row. Read
      // rowBuf[tid mod M] so threads with tid >= M (whose downstream store is
      // a tolerated out-of-range write) do not read past rowBuf.
      mlir::Value tidUI32 =
          ThreadIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"))
              .getResult();
      mlir::Value tidI32 =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{i32}, mlir::ValueRange{tidUI32})
              .getResult(0);
      auto cM = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(M)));
      auto modM = mlir::arith::RemUIOp::create(rewriter, loc, tidI32,
                                               cM.getResult());
      outIdxUI32 = mlir::UnrealizedConversionCastOp::create(
                       rewriter, loc, mlir::TypeRange{ui32},
                       mlir::ValueRange{modM.getResult()})
                       .getResult(0);
    }
    // Register this reduce's rowBuf so a LATER reduce whose cone reads this
    // result can index it at the row IT is filling (chained softmax).
    if (rowBufMap)
      (*rowBufMap)[op->getResult(0)] = rowBuf;

    mlir::Value result =
        GetElementOp::create(rewriter, loc, storeTy, rowBuf, outIdxUI32)
            .getResult();
    // Bridge ui32 storage back to signless i32 for downstream consumers.
    if (isI32 && result.getType() != elemTy) {
      result = mlir::UnrealizedConversionCastOp::create(
                   rewriter, loc, mlir::TypeRange{elemTy},
                   mlir::ValueRange{result})
                   .getResult(0);
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.addptr → identity returning the offset (we fold ptr+offset into the
// downstream load/store).
//
// Convention: `tt.addptr %ptr, %off` becomes just `%off` after conversion;
// the consuming `tt.load`/`tt.store` pattern rebuilds the indexed access.
//===----------------------------------------------------------------------===//

struct AddPtrLowering
    : public mlir::OpConversionPattern<mlir::triton::AddPtrOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::AddPtrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // ConversionPattern can only return one value per result. The
    // downstream consumer (LoadOp / StoreOp lowering) recovers the base
    // memref by walking the ORIGINAL (pre-conversion) `tt.addptr` chain
    // via `findBaseMemref`, and reads the per-thread scalarized offset
    // off this op's converted result.
    //
    // Two shapes occur:
    //   1) First-level addptr — `tt.addptr(splat(base), off)` — the ptr
    //      operand of `adaptor` is the converted splat (memref-ish), so
    //      we just forward the offset.
    //   2) Chained addptr — `tt.addptr(tt.addptr(splat(base), A), B)` —
    //      the inner addptr converted first and was replaced with its
    //      integer offset `A`. The framework wraps that integer in an
    //      `unrealized_conversion_cast` back to `!tt.ptr<...>` so this
    //      outer addptr's `adaptor.getPtr()` looks ptr-typed. We peel
    //      that cast to see the integer underneath and emit
    //      `arith.addi(A, B)` so the full chain is preserved (Lmultiload
    //      Phase B fix — see `.omc/specs/deep-interview-lmultiload-phase-
    //      b-fix.md`).
    mlir::Value innerPtr = adaptor.getPtr();
    mlir::Value outerOff = adaptor.getOffset();
    mlir::Value innerProbe = innerPtr;
    // Wall 13 fix (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC8):
    // pattern-order resilient cast peel. The original peel only walked
    // through `UnrealizedConversionCast`. For the softmax-tutorial chain
    // `tt.addptr(tt.splat(tt.addptr(funcArg, scalarOff)), tensorOff)`,
    // SplatLowering and the inner AddPtrLowering may not have fired yet
    // when the outer AddPtrLowering runs. In that case, innerProbe stops
    // at the `tt.splat` op (tensor-typed), the chained branch never fires,
    // and the inner scalar offset (row_idx*stride) is dropped.
    //
    // Walk through tt.splat (return scalar src) AND through SCALAR
    // tt.addptr (accumulate its scalar offset into `outerOff`). When we
    // reach the function-arg memref, `outerOff` carries the full sum
    // (col_offset + row_offset + ...). The else branch below then
    // `replaceOp(op, outerOff)` with the combined offset, and downstream
    // emitLoadStoreIndex / findBaseMemref both see the correct values.
    while (true) {
      if (auto cast =
              innerProbe.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (cast.getInputs().size() != 1)
          break;
        innerProbe = cast.getInputs()[0];
        continue;
      }
      if (auto splat = innerProbe.getDefiningOp<mlir::triton::SplatOp>()) {
        innerProbe = splat.getSrc();
        continue;
      }
      if (auto inner = innerProbe.getDefiningOp<mlir::triton::AddPtrOp>()) {
        mlir::Value innerOff = inner.getOffset();
        if (mlir::isa<mlir::RankedTensorType>(innerOff.getType()))
          break; // tensor inner offset — out of scope for this peel
        if (innerOff.getType() != outerOff.getType()) {
          innerOff = mlir::UnrealizedConversionCastOp::create(
                         rewriter, op.getLoc(),
                         mlir::TypeRange{outerOff.getType()},
                         mlir::ValueRange{innerOff})
                         .getResult(0);
        }
        outerOff = mlir::arith::AddIOp::create(rewriter, op.getLoc(),
                                                outerOff, innerOff)
                       .getResult();
        innerProbe = inner.getPtr();
        continue;
      }
      break;
    }
    if (innerProbe.getType().isIntOrIndex()) {
      auto offTy = outerOff.getType();
      mlir::Value innerForAdd = innerProbe;
      if (innerProbe.getType() != offTy) {
        innerForAdd =
            mlir::UnrealizedConversionCastOp::create(
                rewriter, op.getLoc(), mlir::TypeRange{offTy},
                mlir::ValueRange{innerProbe})
                .getResult(0);
      }
      mlir::Value sum = mlir::arith::AddIOp::create(rewriter, op.getLoc(),
                                                    innerForAdd, outerOff)
                            .getResult();
      rewriter.replaceOp(op, sum);
    } else {
      rewriter.replaceOp(op, outerOff);
    }
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.load → metal.get_element %memref[%offset_cast_to_ui32]
//
// We look at the ORIGINAL (pre-conversion) tt.load to find which memref
// pointer feeds it (via the now-erased tt.addptr's first operand). The
// scalar offset is read from the adaptor's operand (post-conversion).
//===----------------------------------------------------------------------===//

// Walk through unrealized_conversion_cast wrappers to find the bare memref.
static mlir::Value unwrapToMemref(mlir::Value v) {
  while (v) {
    if (mlir::isa<MetalMemRefType>(v.getType()))
      return v;
    if (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        return {};
      v = cast.getInputs()[0];
      continue;
    }
    return {};
  }
  return {};
}

// Walk the original (pre-conversion) tt.addptr / tt.splat chain to find the
// base block argument (the kernel's pointer arg), then ask the rewriter for
// its remapped value (the MetalMemRefType). This is robust to ordering: the
// block argument's remapped value is established by the type-converter before
// any pattern fires (block arg type-conversion happens at region entry).
//
// Handles nested chains like:
//   tt.addptr(tt.addptr(tt.splat(funcArg), off0), off1)
// as generated by strided 2D kernels (e.g. matrix_transpose).
static mlir::Value findBaseMemref(mlir::Value origPtrVal,
                                  mlir::ConversionPatternRewriter &rewriter) {
  mlir::Value cur = origPtrVal;
  while (cur) {
    // If this is a block argument, handle two cases:
    // (a) already MetalMemRefType (post-FuncOpLowering RAUW'd new arg) —
    //     return directly.
    // (b) original source block arg — ask rewriter for its remapped value.
    if (mlir::isa<mlir::BlockArgument>(cur)) {
      if (mlir::isa<MetalMemRefType>(cur.getType()))
        return cur;
      mlir::Value remapped = rewriter.getRemappedValue(cur);
      if (remapped)
        return unwrapToMemref(remapped);
      return {};
    }
    // Wall 13 fix (.omc/specs/deep-interview-tutorial02-walls-9-to-13.md AC8):
    // do NOT trust the remapped value at non-block-arg levels. The conversion
    // driver inserts source-materialization casts (e.g. `i32 → !metal.memref`)
    // when a lowered i32 offset feeds a memref operand — those casts look
    // memref-typed but trace back to the OFFSET, not the device base. The
    // chained `tt.addptr(tt.splat(tt.addptr(funcArg, off0)), off1)` shape in
    // the softmax tutorial output store hits this path. Always chase the
    // original tt.addptr/tt.splat chain to the block arg; the block-arg base
    // case below resolves the real memref via the post-FuncOpLowering remap.
    // Chase through tt.addptr (ptr operand is index 0).
    if (auto addptr = cur.getDefiningOp<mlir::triton::AddPtrOp>()) {
      cur = addptr.getPtr();
      continue;
    }
    // Chase through tt.splat (src is the scalar pointer).
    if (auto splat = cur.getDefiningOp<mlir::triton::SplatOp>()) {
      cur = splat.getSrc();
      continue;
    }
    // Chase through tt.broadcast / tt.expand_dims: the source tensor is
    // the operand 0. These arise in strided 2D kernels where the per-axis
    // ptr tensor is broadcast to the full tile before the outer addptr.
    if (auto broadcast = cur.getDefiningOp<mlir::triton::BroadcastOp>()) {
      cur = broadcast.getSrc();
      continue;
    }
    if (auto expandDims = cur.getDefiningOp<mlir::triton::ExpandDimsOp>()) {
      cur = expandDims.getSrc();
      continue;
    }
    break;
  }
  return {};
}

// Wall 13 fix helper (see forward decl). Walks the ORIGINAL ptr chain and
// sums up SCALAR (non-tensor) tt.addptr offsets. Skips tt.splat (the splat
// itself contributes no offset, only the SCALAR addptr above it does).
// Stops at block args. Returns null on empty accumulation.
static mlir::Value accumulateScalarAddPtrOffsets(
    mlir::Value origPtrVal, mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc) {
  auto i32 = rewriter.getI32Type();
  mlir::Value cur = origPtrVal;
  mlir::Value acc;
  while (cur) {
    if (mlir::isa<mlir::BlockArgument>(cur))
      break;
    if (auto addptr = cur.getDefiningOp<mlir::triton::AddPtrOp>()) {
      mlir::Value off = addptr.getOffset();
      if (!mlir::isa<mlir::RankedTensorType>(off.getType())) {
        // scalar offset → accumulate
        if (off.getType() != i32) {
          off = mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, mlir::TypeRange{i32},
                    mlir::ValueRange{off})
                    .getResult(0);
        }
        if (!acc)
          acc = off;
        else
          acc = mlir::arith::AddIOp::create(rewriter, loc, acc, off)
                    .getResult();
      }
      cur = addptr.getPtr();
      continue;
    }
    if (auto splat = cur.getDefiningOp<mlir::triton::SplatOp>()) {
      cur = splat.getSrc();
      continue;
    }
    break;
  }
  return acc;
}

// As above, but also walks the SHAPE ops (`tt.broadcast` / `tt.expand_dims`)
// that a 2D tile address threads between its addptr levels.
//
// Triton builds a tile address as
//   addptr(broadcast(addptr(splat(addptr(arg, pid*S)), rows*N)), cols)
// so the scalar `pid*S` term sits below a `tt.broadcast`. The plain walker
// stops there and returns null, which is why the rank-2 axis=1 reduce used to
// substitute a fabricated `tgid*tpb*N` for the per-program offset — a value
// that only coincidentally equals `pid*S` when S == tpb*N. Walking the shape
// ops recovers the real term, so any tile base (e.g. `pid*K*M*N`) is honoured.
static mlir::Value accumulateScalarAddPtrOffsetsThroughShape(
    mlir::Value origPtrVal, mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc) {
  auto i32 = rewriter.getI32Type();
  mlir::Value cur = origPtrVal;
  mlir::Value acc;
  while (cur) {
    if (mlir::isa<mlir::BlockArgument>(cur))
      break;
    if (auto addptr = cur.getDefiningOp<mlir::triton::AddPtrOp>()) {
      mlir::Value off = addptr.getOffset();
      if (!mlir::isa<mlir::RankedTensorType>(off.getType())) {
        if (off.getType() != i32)
          off = mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, mlir::TypeRange{i32}, mlir::ValueRange{off})
                    .getResult(0);
        acc = acc ? mlir::arith::AddIOp::create(rewriter, loc, acc, off)
                        .getResult()
                  : off;
      }
      cur = addptr.getPtr();
      continue;
    }
    if (auto splat = cur.getDefiningOp<mlir::triton::SplatOp>()) {
      cur = splat.getSrc();
      continue;
    }
    if (auto bc = cur.getDefiningOp<mlir::triton::BroadcastOp>()) {
      cur = bc.getSrc();
      continue;
    }
    if (auto ed = cur.getDefiningOp<mlir::triton::ExpandDimsOp>()) {
      cur = ed.getSrc();
      continue;
    }
    break;
  }
  return acc;
}

// For 2D rank-2 tensor ops we read the converted scalar offset from
// `adaptor.getPtr()` (AddPtrLowering replaces tt.addptr with its offset),
// because the IR's `(pid_m*BLOCK_M+row)*stride + (pid_n*BLOCK_N+col)` arith
// chain naturally produces the correct per-thread flat offset once
// MakeRangeLowering emits per-axis row/col values.
//
// For 1D the IR's scalarized offset chain looks like
// `splat(pid*BLOCK) + arange_scalarized + (any divergent extras)`.
// Lmultiload Phase C (see `.omc/specs/deep-interview-lmultiload-phase-c-
// makerange.md`) makes `MakeRangeLowering` the single source of truth
// for the per-thread term: it emits `localTid = id.x - tgid.x*tpb`
// (optionally wrapped with the tile-loop iv) so per-thread arange
// arithmetic (`>> 1`, `* k`, `% n`, `& mask`) survives into MSL. As a
// result, `emitLoadStoreIndex` no longer needs to add localTid itself —
// the AddPtrLowering accumulator threads it through `convertedOffset`.
// Both 1D and 2D branches now reduce to a pass-through of
// `convertedOffset`, modulo the cast-peel that keeps the MLIR type
// chain integer-typed (so the Metal translator never sees a stale
// `!tt.ptr` cast). The Phase B canonical-pattern short-circuit
// (`emitPerIterIndex` for `splat(pid*BLOCK) + arange→0`) has been
// DELETED; downstream lit fixtures pin the new arithmetic-explicit MSL
// form directly.
static mlir::Value
emitLoadStoreIndex(const TileInfo &tile, mlir::Value convertedOffset,
                   mlir::scf::ForOp /*parentFor*/,
                   mlir::ConversionPatternRewriter &rewriter,
                   mlir::Location loc) {
  auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
  auto i32 = rewriter.getI32Type();
  if (tile.rank == 2) {
    return mlir::UnrealizedConversionCastOp::create(
               rewriter, loc, mlir::TypeRange{ui32},
               mlir::ValueRange{convertedOffset})
        .getResult(0);
  }
  // 1D pass-through: peel `unrealized_conversion_cast` wrappers around
  // `convertedOffset` so we never round-trip through a non-integer type
  // (e.g. `i32 -> !tt.ptr<f32> -> i32`). The Metal translator does not
  // load the `tt` dialect, so an in-flight `!tt.ptr<f32>` cast in the
  // converted IR is rejected as an unregistered-dialect type. The
  // AddPtrLowering replacement is always an integer underneath the
  // (possibly source-materialized) cast wrapper.
  mlir::Value offProbe = convertedOffset;
  while (auto cast =
             offProbe.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1)
      break;
    offProbe = cast.getInputs()[0];
  }
  assert(offProbe.getType().isIntOrIndex() &&
         "tt.addptr offset must scalarize to an integer");
  mlir::Value offI32 = offProbe;
  if (offProbe.getType() != i32) {
    offI32 = mlir::UnrealizedConversionCastOp::create(
                 rewriter, loc, mlir::TypeRange{i32},
                 mlir::ValueRange{offProbe})
                 .getResult(0);
  }
  return mlir::UnrealizedConversionCastOp::create(
             rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{offI32})
      .getResult(0);
}

struct LoadLowering
    : public mlir::OpConversionPattern<mlir::triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getMask())
      return mlir::failure();  // handled by MaskedLoadLowering
    auto loc = op.getLoc();
    // A tensor load normally reads a tile through `tt.addptr`. The one shape
    // without an addptr is a bare `tt.splat` of a scalar pointer: every lane
    // then reads the SAME address. Triton produces this whenever the
    // per-element offset folds to zero — e.g. a BLOCK=1 tile, where
    // `tl.arange(0, 1) * stride` is constant 0 and the addptr disappears.
    auto splatPtr = op.getPtr().getDefiningOp<mlir::triton::SplatOp>();
    if (!op.getPtr().getDefiningOp<mlir::triton::AddPtrOp>() && !splatPtr)
      return rewriter.notifyMatchFailure(
          op, "tt.load expects a tt.addptr or a splat scalar ptr feeding ptr");
    mlir::Value memref = findBaseMemref(op.getPtr(), rewriter);
    if (!memref)
      return rewriter.notifyMatchFailure(op,
                                         "memref source not MetalMemRefType");
    auto tile = tileFromLoadPtrTensor(op.getPtr().getType());
    if (!tile)
      return rewriter.notifyMatchFailure(
          op, "tt.load operand missing ttg.blocked / ttg.slice layout");
    mlir::Value idx;
    if (splatPtr) {
      // Uniform pointer: there is no per-element index to emit. The address is
      // whatever the scalar tt.addptr chain feeding the splat accumulated
      // (null → the bare kernel-arg pointer → index 0), exactly as
      // ScalarLoadLowering does for a scalar `!tt.ptr<T>`.
      auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
      mlir::Value offI32 =
          accumulateScalarAddPtrOffsets(op.getPtr(), rewriter, loc);
      if (!offI32)
        offI32 = mlir::arith::ConstantOp::create(
                     rewriter, loc, rewriter.getI32IntegerAttr(0))
                     .getResult();
      idx = mlir::UnrealizedConversionCastOp::create(
                rewriter, loc, mlir::TypeRange{ui32}, mlir::ValueRange{offI32})
                .getResult(0);
    } else {
      auto parentFor =
          findOutermostScfFor(op); // Wall 13 fix: tile loop, not user loop
      idx = emitLoadStoreIndex(*tile, adaptor.getPtr(), parentFor, rewriter, loc);
    }
    auto memrefTy = mlir::cast<MetalMemRefType>(memref.getType());
    // L2b: get_element yields the memref STORAGE type (ui32 for an i32 ptr).
    // When that differs from the tensor's signless element type, bridge back
    // so downstream signless arith sees the original type.
    mlir::Type storageTy = memrefTy.getType();
    mlir::Value ge =
        GetElementOp::create(rewriter, loc, storageTy, memref, idx).getResult();
    mlir::Type wantTy =
        mlir::cast<mlir::RankedTensorType>(op.getType()).getElementType();
    if (storageTy != wantTy)
      ge = mlir::UnrealizedConversionCastOp::create(
               rewriter, loc, mlir::TypeRange{wantTy}, mlir::ValueRange{ge})
               .getResult(0);
    rewriter.replaceOp(op, ge);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Scalar tt.load → metal.get_element (L3a-tileloop-compiler-A).
//
// Matches an unmasked `tt.load` whose ptr is a bare `!tt.ptr<f32>` (NOT a
// `tensor<Nx!tt.ptr<f32>>`) and whose result is bare `f32`. Two shapes are
// supported:
//   * `tt.load %scalar_ptr : !tt.ptr<f32>` (offset 0; no addptr — Triton
//     folds `addptr(p, 0)` away)
//   * `tt.load %addptr_result` where `tt.addptr %scalar_ptr, %i32_offset`
//     (scalar offset from a `tl.static_range` iter or any scalar IV)
// Lowers to `metal.get_element %memref[%idx_ui32] : (memref, ui32) -> f32`.
// Downstream consumers that need a tensor (e.g. `kv * inp`) see a `tt.splat`
// emitted by the Triton frontend BEFORE the arith op — splat lowering is
// already in place, so no broadcast work is needed here.
//
// Envelope: f32 only; unmasked only; load only. See
// `.omc/specs/deep-interview-leet-triton-l3a-tileloop-compiler-a-scalar-load.md`.
//===----------------------------------------------------------------------===//
struct ScalarLoadLowering
    : public mlir::OpConversionPattern<mlir::triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getMask())
      return mlir::failure();  // masked scalar load deferred
    // Must be SCALAR ptr — not a tensor of pointers. The ptr operand may
    // appear as either `!tt.ptr<f32>` (intermediate `tt.addptr` operand) or
    // `!metal.memref<? x f32>` (kernel-arg block argument rewritten by
    // FuncOpLowering). Tensor-of-ptr shapes go to LoadLowering /
    // MaskedLoadLowering.
    mlir::Type elemTy;
    if (auto ptrTy = mlir::dyn_cast<mlir::triton::PointerType>(
            op.getPtr().getType())) {
      elemTy = ptrTy.getPointeeType();
    } else if (auto memTy = mlir::dyn_cast<MetalMemRefType>(
                   op.getPtr().getType())) {
      elemTy = memTy.getType();
    } else {
      return mlir::failure();  // tensor / unknown — not our case
    }
    // Envelope: f32 or i32. i32 is routed through ui32 storage (see
    // metalStorageElementType) and bridged back to signless i32 for the result.
    bool isF32 = mlir::isa<mlir::FloatType>(elemTy) &&
                 mlir::cast<mlir::FloatType>(elemTy).getWidth() == 32;
    bool isI32 = elemTy.isInteger(32);
    if (!isF32 && !isI32)
      return rewriter.notifyMatchFailure(
          op, "scalar tt.load: only f32/i32 supported");
    auto loc = op.getLoc();
    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    // Resolve the base memref + accumulated scalar index across the WHOLE
    // scalar tt.addptr chain (e.g. `draft_tokens_ptr + b*T + i` is two nested
    // tt.addptr ops). findBaseMemref walks to the base; accumulateScalar-
    // AddPtrOffsets sums every scalar offset. Offset 0 (bare kernel-arg ptr,
    // Triton folded addptr(p,0)) → null accumulator → index constant 0.
    mlir::Value memref = findBaseMemref(op.getPtr(), rewriter);
    if (!memref)
      return rewriter.notifyMatchFailure(op,
                                         "memref source not MetalMemRefType");
    mlir::Value offset =
        accumulateScalarAddPtrOffsets(op.getPtr(), rewriter, loc);
    mlir::Value idxI32 = offset;
    if (!idxI32)
      idxI32 = mlir::arith::ConstantOp::create(
                   rewriter, loc, rewriter.getI32IntegerAttr(0))
                   .getResult();
    mlir::Value idxUi32 = mlir::UnrealizedConversionCastOp::create(
                              rewriter, loc, mlir::TypeRange{ui32},
                              mlir::ValueRange{idxI32})
                              .getResult(0);
    // Read the STORAGE-typed element (f32 → f32; i32 → ui32), then bridge back
    // to the Triton element type so downstream signless arith sees i32.
    mlir::Type storageTy = metalStorageElementType(elemTy);
    mlir::Value el =
        GetElementOp::create(rewriter, loc, storageTy, memref, idxUi32)
            .getResult();
    if (storageTy != elemTy)
      el = mlir::UnrealizedConversionCastOp::create(
               rewriter, loc, mlir::TypeRange{elemTy}, mlir::ValueRange{el})
               .getResult(0);
    rewriter.replaceOp(op, el);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Helper: rebuild a per-thread mask from the original tensor `arith.cmpi`.
//
// The masked path expects the mask to be sourced from
//   %m = arith.cmpi <pred> %abs_off, %splat_N : tensor<NxNi32>
//   %splat_N = tt.splat %N : i32 -> tensor<NxNi32>
// Under our 1-elt-per-thread + single-block-launch assumption, the per-thread
// truth of the mask is `metal.thread_id "x" <pred> N`. We rebuild it here so
// downstream `scf.if` sees a correct per-thread condition (rather than the
// converted-form `arith.cmpi` whose lhs collapses to a constant 0 — see the
// `MakeRangeLowering` rationale).
//===----------------------------------------------------------------------===//

// One component of a (possibly AND-reduced) mask. For 1D masks `axis == -1`
// means "use the flat per-iter idx directly". For 2D masks (e.g.
// `(offs_m[:,None] < M) & (offs_n[None,:] < N)`) each component carries
// `axis == 0` (row bound) or `axis == 1` (col bound) so emission can
// decompose the flat idx into per-axis values.
// See `.omc/specs/deep-interview-metal-2d-maskedaccess-emitperiterindex.md`.
struct MaskComponent {
  mlir::arith::CmpIPredicate pred;
  mlir::Value scalarBound; // POST-conversion scalar (i32 typically)
  int axis;                // -1 for 1D; 0 or 1 for 2D row/col
};

// Walk back through expand_dims/broadcast/addi(splat,_) to find the original
// 1D ranked tensor and read its slice<dim=X> encoding. The non-sliced axis
// (`rank-1 - dim` for 2D) is the axis this 1D value varies along, i.e. the
// axis the bound refers to. Returns -1 if no slice encoding is found (1D case).
static int axisFromCmpiLhs(mlir::Value lhs) {
  while (lhs) {
    if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType())) {
      if (auto slice = mlir::dyn_cast_or_null<
              mlir::triton::gpu::SliceEncodingAttr>(rt.getEncoding())) {
        unsigned dim = slice.getDim();
        if (auto parent = mlir::dyn_cast_or_null<
                mlir::triton::gpu::BlockedEncodingAttr>(slice.getParent())) {
          int parentRank = static_cast<int>(parent.getOrder().size());
          if (parentRank == 2)
            return 1 - static_cast<int>(dim);
        }
      }
    }
    auto def = lhs.getDefiningOp();
    if (!def)
      return -1;
    if (auto ed = mlir::dyn_cast<mlir::triton::ExpandDimsOp>(def)) {
      lhs = ed.getSrc();
      continue;
    }
    if (auto bc = mlir::dyn_cast<mlir::triton::BroadcastOp>(def)) {
      lhs = bc.getSrc();
      continue;
    }
    if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(def)) {
      // `splat(pid*BLOCK) + make_range` shape: pick the non-splat operand.
      auto lhsSplat = addi.getLhs().getDefiningOp<mlir::triton::SplatOp>();
      lhs = lhsSplat ? addi.getRhs() : addi.getLhs();
      continue;
    }
    if (auto muli = mlir::dyn_cast<mlir::arith::MulIOp>(def)) {
      auto lhsSplat = muli.getLhs().getDefiningOp<mlir::triton::SplatOp>();
      lhs = lhsSplat ? muli.getRhs() : muli.getLhs();
      continue;
    }
    return -1;
  }
  return -1;
}

static bool matchSingleCmpi(mlir::Value v,
                            llvm::SmallVectorImpl<MaskComponent> &out,
                            mlir::ConversionPatternRewriter &rewriter) {
  // Walk through `tt.broadcast` / `tt.expand_dims` wrappers to the underlying
  // `arith.cmpi`. Triton's 2D mask shape is
  //   cmpi(1D) -> expand_dims -> broadcast -> andi(broadcast, broadcast)
  while (v) {
    auto def = v.getDefiningOp();
    if (!def)
      return false;
    if (auto bc = mlir::dyn_cast<mlir::triton::BroadcastOp>(def)) {
      v = bc.getSrc();
      continue;
    }
    if (auto ed = mlir::dyn_cast<mlir::triton::ExpandDimsOp>(def)) {
      v = ed.getSrc();
      continue;
    }
    break;
  }
  auto cmpi = v.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmpi)
    return false;
  auto splat = cmpi.getRhs().getDefiningOp<mlir::triton::SplatOp>();
  if (!splat)
    return false;
  mlir::Value remappedBound = rewriter.getRemappedValue(splat.getSrc());
  if (!remappedBound)
    remappedBound = splat.getSrc();
  int axis = axisFromCmpiLhs(cmpi.getLhs());
  out.push_back({cmpi.getPredicate(), remappedBound, axis});
  return true;
}

// Walks `origMask` (the ORIGINAL pre-conversion mask value) and returns the
// per-axis components. Accepts either a single `arith.cmpi(.., tt.splat)`
// (1D) or `arith.andi(arith.cmpi(.., tt.splat), arith.cmpi(.., tt.splat))`
// (2D row+col bound check). Returns nullopt on shape mismatch.
static std::optional<llvm::SmallVector<MaskComponent, 2>>
matchMaskShape(mlir::Value origMask,
               mlir::ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<MaskComponent, 2> components;
  if (auto andi = origMask.getDefiningOp<mlir::arith::AndIOp>()) {
    if (!matchSingleCmpi(andi.getLhs(), components, rewriter))
      return std::nullopt;
    if (!matchSingleCmpi(andi.getRhs(), components, rewriter))
      return std::nullopt;
  } else {
    if (!matchSingleCmpi(origMask, components, rewriter))
      return std::nullopt;
  }
  return components;
}

// Emit the per-thread (or per-iteration in tile loops) mask check. For 1D
// masks (single component, axis == -1) this is `arith.cmpi <pred> idx_i32
// scalarBound`. For 2D masks (two components with axis ∈ {0,1}) this emits
// per-axis cmpi (row = idx / BLOCK_N, col = idx % BLOCK_N) ANDed together.
static mlir::Value
emitTileAwareMask(llvm::ArrayRef<MaskComponent> shapes, const TileInfo &tile,
                  mlir::scf::ForOp parentFor,
                  mlir::ConversionPatternRewriter &rewriter,
                  mlir::Location loc) {
  auto i32 = rewriter.getI32Type();
  mlir::Value idxUi32 = emitPerIterIndex(tile, parentFor, rewriter, loc);
  mlir::Value idxI32 = mlir::UnrealizedConversionCastOp::create(
                           rewriter, loc, mlir::TypeRange{i32},
                           mlir::ValueRange{idxUi32})
                           .getResult(0);
  // For 2D, decompose the flat idx into (row, col). BLOCK_N comes from the
  // tile's logical shape; with order=[1,0] the col axis is contiguous in
  // the flat layout, so row = idx / BLOCK_N and col = idx % BLOCK_N.
  mlir::Value rowI32 = idxI32;
  mlir::Value colI32 = idxI32;
  if (tile.rank == 2 && tile.shape.size() == 2) {
    int64_t blockN = tile.shape[1];
    auto bn = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(blockN));
    rowI32 = mlir::arith::DivSIOp::create(rewriter, loc, idxI32, bn.getResult())
                 .getResult();
    colI32 = mlir::arith::RemSIOp::create(rewriter, loc, idxI32, bn.getResult())
                 .getResult();
  }

  mlir::Value result;
  for (const auto &shape : shapes) {
    mlir::Value axisIdx = idxI32;
    if (shape.axis == 0)
      axisIdx = rowI32;
    else if (shape.axis == 1)
      axisIdx = colI32;
    auto cmp = mlir::arith::CmpIOp::create(rewriter, loc, shape.pred, axisIdx,
                                            shape.scalarBound);
    if (!result)
      result = cmp.getResult();
    else
      result = mlir::arith::AndIOp::create(rewriter, loc, result, cmp.getResult())
                   .getResult();
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Masked tt.load → scf.if (yields `other` scalar or zero on the masked-off
// branch).
//
// Restricted to (addr, mask[, other]) shape. The `other` operand is accepted
// only when its producer SSA chain is a splat-constant: either
// `tt.splat(arith.constant scalar)` or `arith.constant dense<splat>` —
// matching the canonical Triton emit for `tl.load(..., other=0.0)`. Non-splat
// `other` is rejected with a clear diagnostic. Restricted to float element
// types in this slice; integer masked loads are next-session work. Session L1
// of `.omc/specs/deep-interview-leet-triton-l1-refine-and-ship.md`.
//===----------------------------------------------------------------------===//

// Extract the splat scalar attribute behind `other`. Accepts both
// `tt.splat(arith.constant scalar)` and `arith.constant dense<splat>` shapes.
// Returns std::nullopt for non-splat or non-constant producers.
static std::optional<mlir::TypedAttr>
extractSplatConstantAttr(mlir::Value other) {
  if (!other)
    return std::nullopt;
  // Shape A: tt.splat(arith.constant scalar)
  if (auto splat = other.getDefiningOp<mlir::triton::SplatOp>()) {
    if (auto cst = splat.getSrc().getDefiningOp<mlir::arith::ConstantOp>())
      return cst.getValue();
    return std::nullopt;
  }
  // Shape B: arith.constant dense<splat>
  if (auto cst = other.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto dense = mlir::dyn_cast<mlir::SplatElementsAttr>(cst.getValue())) {
      auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(cst.getType());
      if (!rtt)
        return std::nullopt;
      auto elemTy = rtt.getElementType();
      if (mlir::isa<mlir::FloatType>(elemTy)) {
        return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(
            elemTy, dense.getSplatValue<llvm::APFloat>()));
      }
      if (mlir::isa<mlir::IntegerType>(elemTy)) {
        return mlir::cast<mlir::TypedAttr>(mlir::IntegerAttr::get(
            elemTy, dense.getSplatValue<llvm::APInt>()));
      }
      return std::nullopt;
    }
  }
  return std::nullopt;
}

struct MaskedLoadLowering
    : public mlir::OpConversionPattern<mlir::triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.getMask())
      return mlir::failure();  // handled by LoadLowering
    // `other` operand: accept only splat-constant producers. Non-splat (or
    // dynamic) `other` is rejected here; the line-1868 dot-prepass handles
    // the orthogonal restriction of "no `other` in `tt.dot` operand position".
    std::optional<mlir::TypedAttr> otherSplatAttr;
    if (op.getOther()) {
      otherSplatAttr = extractSplatConstantAttr(op.getOther());
      if (!otherSplatAttr)
        return rewriter.notifyMatchFailure(
            op,
            "tt.load `other` operand requires splat-constant producer");
    }
    auto loc = op.getLoc();
    if (!op.getPtr().getDefiningOp<mlir::triton::AddPtrOp>())
      return rewriter.notifyMatchFailure(
          op, "tt.load expects a tt.addptr feeding ptr");
    mlir::Value memref = findBaseMemref(op.getPtr(), rewriter);
    if (!memref)
      return rewriter.notifyMatchFailure(op,
                                         "memref source not MetalMemRefType");
    auto memrefTy = mlir::cast<MetalMemRefType>(memref.getType());
    auto elemTy = memrefTy.getType();
    // L2b: accept FloatType OR a width-8/width-32 integer STORAGE type. `elemTy`
    // is the memref storage type, so an i32 ptr already presents as ui32 here
    // (routed by metalStorageElementType); i8 stays signless i8. Other integer
    // widths (i1/i16/i64) are still rejected; widen only when a leet needs them.
    {
      bool isFloat = mlir::isa<mlir::FloatType>(elemTy);
      auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemTy);
      bool isI8 = intTy && intTy.getWidth() == 8;
      bool isI32 = intTy && intTy.getWidth() == 32;
      if (!isFloat && !isI8 && !isI32)
        return rewriter.notifyMatchFailure(
            op,
            "masked tt.load: only float, i8 and i32 element types supported");
    }

    auto tile = tileFromLoadPtrTensor(op.getPtr().getType());
    if (!tile)
      return rewriter.notifyMatchFailure(
          op, "tt.load operand missing ttg.blocked / ttg.slice layout");
    auto parentFor = findOutermostScfFor(op); // Wall 13 fix: tile loop, not user loop
    mlir::Value cond;
    if (tile->rank == 2) {
      // For 2D, the IR's converted mask `andi(cmpi(pid*BM+lid_row<M),
      // cmpi(pid*BN+lid_col<N))` is per-thread-correct once
      // MakeRangeLowering emits real per-axis local-tid values. Use the
      // converted scalar mask directly.
      cond = adaptor.getMask();
    } else {
      // Use the typeconverter-scalarized mask directly: the ArithCmpILowering /
      // ArithAndILowering chain scalarizes the tensor cmpi/andi into a per-iter
      // scalar i1 that is per-thread-correct under MakeRangeLowering's LOCAL
      // per-axis ids, and it reads the ACTUAL mask index cone. The old
      // `emitTileAwareMask` shortcut reconstructed the index via emitPerIterIndex
      // (a GLOBAL flat index), which is correct only for vector-add-style masks
      // whose cone already includes the program offset (`pid*BLOCK + arange`) —
      // for a per-row mask (`cols < N` with `cols = off + arange`, no pid) it
      // emitted `id.x < N`, masking out every program k>=1 (each read row 0 /
      // returned 0). See test_reduce_per_row_multiprogram and layer-norm mean.
      cond = adaptor.getMask();
    }

    auto scfIf = mlir::scf::IfOp::create(rewriter,
        loc, mlir::TypeRange{elemTy}, cond,
        /*addThenBlock=*/true, /*addElseBlock=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&scfIf.getThenRegion().front());
      mlir::Value idx =
          emitLoadStoreIndex(*tile, adaptor.getPtr(), parentFor, rewriter, loc);
      auto getEl = GetElementOp::create(rewriter, loc, elemTy, memref, idx);
      mlir::scf::YieldOp::create(rewriter,
          loc, mlir::ValueRange{getEl.getResult()});
    }
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&scfIf.getElseRegion().front());
      // Else branch yields the user-provided splat-constant `other` scalar if
      // present; otherwise zero. v1 supports float elem types only, and the
      // splat-constant must match the load's element type.
      mlir::TypedAttr elseAttr;
      if (otherSplatAttr) {
        auto a = *otherSplatAttr;
        if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(a)) {
          if (fa.getType() != elemTy)
            return rewriter.notifyMatchFailure(
                op, "tt.load `other` element type mismatches result");
          elseAttr = fa;
        } else if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a)) {
          // L2b: integer `other` splat-constant. The attr carries the tensor's
          // (possibly signless) element type while `elemTy` is the storage
          // type (ui32 for i32). Match on bit width and re-key the attr to the
          // storage type so the emitted metal.constant satisfies Metal_Type.
          auto iaTy = mlir::dyn_cast<mlir::IntegerType>(ia.getType());
          auto elTy = mlir::dyn_cast<mlir::IntegerType>(elemTy);
          if (!iaTy || !elTy || iaTy.getWidth() != elTy.getWidth())
            return rewriter.notifyMatchFailure(
                op, "tt.load `other` element type mismatches result");
          elseAttr = rewriter.getIntegerAttr(elemTy, ia.getValue());
        } else {
          return rewriter.notifyMatchFailure(
              op,
              "tt.load `other`: only float or integer splat-constants supported");
        }
      } else if (mlir::isa<mlir::FloatType>(elemTy)) {
        elseAttr = rewriter.getFloatAttr(elemTy, 0.0);
      } else {
        // L2b: default-zero for integer element types (i8 today).
        elseAttr = rewriter.getIntegerAttr(elemTy, 0);
      }
      auto elseVal = ConstantOp::create(rewriter, loc, elemTy, elseAttr);
      mlir::scf::YieldOp::create(rewriter,
          loc, mlir::ValueRange{elseVal.getResult()});
    }
    // L2b: the scf.if yields the storage type (ui32 for i32). Bridge back to
    // the tensor's signless element type so downstream signless arith is sound.
    mlir::Value result = scfIf.getResult(0);
    mlir::Type wantTy =
        mlir::cast<mlir::RankedTensorType>(op.getType()).getElementType();
    if (elemTy != wantTy)
      result = mlir::UnrealizedConversionCastOp::create(
                   rewriter, loc, mlir::TypeRange{wantTy},
                   mlir::ValueRange{result})
                   .getResult(0);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.store → metal.store
//===----------------------------------------------------------------------===//

struct StoreLowering
    : public mlir::OpConversionPattern<mlir::triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getMask())
      return mlir::failure();  // handled by MaskedStoreLowering
    auto loc = op.getLoc();
    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);

    // Scalar pointer path: `tt.store %scalar_ptr, %val` where the ptr is a
    // bare `!tt.ptr<T>` block argument (no tt.addptr, no tensor layout).
    // This arises for rank-1 reduce result stores: `tl.store(out_ptr + 0, s)`
    // compiles to `tt.store %out_ptr, %s : !tt.ptr<f32>` when the offset is 0.
    // Mirrors ScalarLoadLowering's direct-block-arg path.
    if (!op.getPtr().getDefiningOp<mlir::triton::AddPtrOp>() &&
        !tileFromTensor(op.getPtr().getType())) {
      mlir::Value memref = findBaseMemref(op.getPtr(), rewriter);
      if (!memref)
        return rewriter.notifyMatchFailure(op,
                                           "scalar tt.store: memref not found");
      auto zero = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(0));
      mlir::Value idxUi32 = mlir::UnrealizedConversionCastOp::create(
                                rewriter, loc, mlir::TypeRange{ui32},
                                mlir::ValueRange{zero.getResult()})
                                .getResult(0);
      mlir::Value sval = castToMemrefStorage(adaptor.getValue(), memref,
                                             rewriter, loc);
      StoreOp::create(rewriter, loc, sval, memref, idxUi32);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Scalar pointer fed by a SCALAR tt.addptr chain: `tt.store %p, %val` where
    // %p = addptr(...addptr(base, off0)..., offN) and every offset is scalar (no
    // tensor layout). Arises for `tl.store(out_ptr + b*(T+1) + idx, v)` in
    // sequential per-program kernels (e.g. speculative decoding). Mirrors the
    // scalar-ptr path above / ScalarLoadLowering: resolve the base memref +
    // accumulated scalar offset, bridge the value to storage. tileFromTensor is
    // null for a scalar `!tt.ptr<T>`; the no-addptr case was handled above, so
    // reaching here with a scalar ptr means a scalar addptr chain.
    if (!tileFromTensor(op.getPtr().getType())) {
      mlir::Value memref = findBaseMemref(op.getPtr(), rewriter);
      if (!memref)
        return rewriter.notifyMatchFailure(
            op, "scalar-addptr tt.store: memref not found");
      mlir::Value offset =
          accumulateScalarAddPtrOffsets(op.getPtr(), rewriter, loc);
      mlir::Value idxI32 = offset;
      if (!idxI32)
        idxI32 = mlir::arith::ConstantOp::create(
                     rewriter, loc, rewriter.getI32IntegerAttr(0))
                     .getResult();
      mlir::Value idxUi32 = mlir::UnrealizedConversionCastOp::create(
                                rewriter, loc, mlir::TypeRange{ui32},
                                mlir::ValueRange{idxI32})
                                .getResult(0);
      mlir::Value sval = castToMemrefStorage(adaptor.getValue(), memref,
                                             rewriter, loc);
      StoreOp::create(rewriter, loc, sval, memref, idxUi32);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // ---- Scatter store through a layout relabel --------------------------
    // A data-dependent store — `tl.store(dst + idx_tensor, vals)`, the scatter
    // phase of a radix sort — reaches here as
    //
    //   %p = tt.addptr(splat(dst), %loaded_idx)   : blocked<spt=[4]>
    //   %pc = ttg.convert_layout %p               : -> blocked<spt=[1]>
    //   %vc = ttg.convert_layout %vals            : -> blocked<spt=[1]>
    //   tt.store %pc, %vc
    //
    // so the ptr is NOT directly a tt.addptr and the old code rejected it.
    //
    // The two layouts assign different logical elements to a given (thread,
    // tile-loop iteration): the source has sizePerThread>1 (`idx = tid*E + iv`)
    // while the destination is strided (`idx = tid + iv*T`). So the cvt is a
    // genuine permutation and CANNOT simply be treated as a no-op.
    //
    // It can, however, be side-stepped: a store is just a set of (address,
    // value) pairs indexed by logical element. Relabelling which thread owns
    // which element permutes the pairs but does not change the SET, so
    // performing the store entirely in the SOURCE layout writes exactly the
    // same bytes. That requires the address and the value to be relabelled
    // IDENTICALLY — hence the check that both operands come through a cvt out
    // of the same source encoding (a uniform splat value is also fine, since a
    // permutation leaves it unchanged). If they disagree, fall through and let
    // the store be rejected rather than silently pairing a value with another
    // element's address.
    mlir::Value storePtr = op.getPtr();
    mlir::Value convertedPtr = adaptor.getPtr();
    mlir::Value convertedVal = adaptor.getValue();
    if (auto cvt =
            op.getPtr().getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>()) {
      auto srcRtt = mlir::dyn_cast<mlir::RankedTensorType>(cvt.getSrc().getType());
      auto dstRtt =
          mlir::dyn_cast<mlir::RankedTensorType>(cvt.getResult().getType());
      if (srcRtt && dstRtt && srcRtt.getRank() == 1 &&
          srcRtt.getShape() == dstRtt.getShape() &&
          cvt.getSrc().getDefiningOp<mlir::triton::AddPtrOp>()) {
        mlir::Attribute srcEnc = srcRtt.getEncoding();
        // The value must land in the same source layout.
        mlir::Value valSrc;
        if (auto vc =
                op.getValue().getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>()) {
          auto vSrcRtt =
              mlir::dyn_cast<mlir::RankedTensorType>(vc.getSrc().getType());
          if (vSrcRtt && vSrcRtt.getEncoding() == srcEnc) {
            valSrc = vc.getSrc();
          }
        } else if (auto vRtt = mlir::dyn_cast<mlir::RankedTensorType>(
                       op.getValue().getType())) {
          if (vRtt.getEncoding() == srcEnc)
            valSrc = op.getValue();
        }
        if (valSrc) {
          mlir::Value remappedPtr = rewriter.getRemappedValue(cvt.getSrc());
          mlir::Value remappedVal = rewriter.getRemappedValue(valSrc);
          if (remappedPtr && remappedVal) {
            storePtr = cvt.getSrc();
            convertedPtr = remappedPtr;
            convertedVal = remappedVal;
          }
        }
      }
    }

    if (!storePtr.getDefiningOp<mlir::triton::AddPtrOp>())
      return rewriter.notifyMatchFailure(
          op, "tt.store expects a tt.addptr feeding ptr");
    mlir::Value memref = findBaseMemref(storePtr, rewriter);
    if (!memref)
      return rewriter.notifyMatchFailure(op,
                                         "memref source not MetalMemRefType");
    auto tile = tileFromTensor(storePtr.getType());
    if (!tile)
      return rewriter.notifyMatchFailure(
          op, "tt.store operand missing ttg.blocked layout");
    auto parentFor = findOutermostScfFor(op); // Wall 13 fix: tile loop, not user loop
    mlir::Value idx =
        emitLoadStoreIndex(*tile, convertedPtr, parentFor, rewriter, loc);
    mlir::Value sval = castToMemrefStorage(convertedVal, memref, rewriter, loc);
    // Sub-tpb store guard. When the stored tensor has FEWER elements than the
    // threadgroup has threads AND there is no tile loop (E <= 1), the threads
    // whose per-thread element index is >= numElements have no valid output
    // slot. An unconditional `out[tid] = ...` then writes out of bounds, and
    // because the Metal backend binds tensors ZERO-COPY (a host tensor IS the
    // device buffer), that OOB write silently corrupts a DIFFERENT live
    // tensor's memory — surfacing as nondeterministic cross-test failures
    // (e.g. rank-2 reduce outputs of M < tpb rows). Guard the device store
    // with `if (localTid < numElements)`. For numElements >= tpb, or whenever
    // a tile loop is present (E > 1, indices span exactly [0, numElements)),
    // no guard is emitted — byte-identical to the prior emission.
    int64_t numElements = 1;
    for (auto s : tile->shape)
      numElements *= s;
    bool needGuard =
        tile->elemPerThread <= 1 && numElements < tile->threadsPerBlock;
    if (!needGuard) {
      StoreOp::create(rewriter, loc, sval, memref, idx);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    // localTid = global_tid - tgid * tpb (multi-program safe).
    auto i32 = rewriter.getI32Type();
    auto tidGlobal =
        ThreadIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"));
    mlir::Value tidI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tidGlobal.getResult()})
            .getResult(0);
    auto tgX = ThreadgroupIdOp::create(rewriter, loc, ui32,
                                       rewriter.getStringAttr("x"));
    mlir::Value tgI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tgX.getResult()})
            .getResult(0);
    auto cTpb = mlir::arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(tile->threadsPerBlock)));
    auto tgOffset =
        mlir::arith::MulIOp::create(rewriter, loc, tgI32, cTpb.getResult());
    mlir::Value localTid =
        mlir::arith::SubIOp::create(rewriter, loc, tidI32, tgOffset.getResult())
            .getResult();
    auto cNum = mlir::arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(numElements)));
    auto cond = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::slt, localTid,
        cNum.getResult());
    auto guardIf = mlir::scf::IfOp::create(rewriter, loc, mlir::TypeRange{},
                                           cond.getResult(),
                                           /*addThenBlock=*/true,
                                           /*addElseBlock=*/false);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&guardIf.getThenRegion().front());
      StoreOp::create(rewriter, loc, sval, memref, idx);
      mlir::scf::YieldOp::create(rewriter, loc);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.atomic_rmw fadd (scalar) → metal.atomic_rmw
//
// Models the scalar `tl.atomic_add(ptr, scalar)` form — e.g. the leet-triton
// subarray-sum kernels accumulating each program's partial into output[0].
// Only f32 add with an UNUSED old-value result is supported, and the mask must
// be absent or constant-true (the subarray guards the atomic with an outer
// `scf.if sum > 0`). The address is the base memref + any scalar tt.addptr
// offset, reusing the rank-1-reduce helpers. See `metal.atomic_rmw`.
//===----------------------------------------------------------------------===//
struct AtomicRmwLowering
    : public mlir::OpConversionPattern<mlir::triton::AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::triton::AtomicRMWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op.getAtomicRmwOp() != mlir::triton::RMWOp::FADD)
      return rewriter.notifyMatchFailure(op, "atomic_rmw: only fadd supported");

    // ---- Per-element rank-1 tensor form ------------------------------------
    // `tl.atomic_add(P + cols, v, mask)` where P/v/mask are rank-1 tensors.
    // This is the lock-free accumulation the layer-norm backward uses in place
    // of the tutorial's spin lock (Apple GPUs give no cross-threadgroup
    // forward-progress guarantee, so a global spin lock can deadlock). Model it
    // on the masked device store (StoreLowering / MaskedStoreLowering): the
    // typeconverter scalarizes the value/index/mask to the per-thread cone, and
    // the E>1 tile loop replicates the op in place — so a single guarded
    // `metal.atomic_rmw` at the scalarized index handles E==1/E>1,
    // multi-program base offsets, and masking identically to the store. Unlike
    // the scalar form below (a per-PROGRAM op guarded to `localTid==0`), each
    // thread here owns its own element and adds unconditionally under the mask.
    if (mlir::isa<mlir::RankedTensorType>(op.getVal().getType())) {
      auto rtt = mlir::cast<mlir::RankedTensorType>(op.getVal().getType());
      if (rtt.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "atomic_rmw: only rank-1 tensor form supported");
      if (!op.getResult().use_empty())
        return rewriter.notifyMatchFailure(
            op, "atomic_rmw: tensor old-value result consumed (not modeled)");
      if (!rtt.getElementType().isF32())
        return rewriter.notifyMatchFailure(
            op, "atomic_rmw: tensor form only f32 add supported");
      if (!op.getPtr().getDefiningOp<mlir::triton::AddPtrOp>())
        return rewriter.notifyMatchFailure(
            op, "atomic_rmw: tensor form expects a tt.addptr feeding ptr");
      mlir::Value memref = findBaseMemref(op.getPtr(), rewriter);
      if (!memref)
        return rewriter.notifyMatchFailure(
            op, "atomic_rmw: tensor base memref not found");
      auto tile = tileFromTensor(op.getPtr().getType());
      if (!tile)
        return rewriter.notifyMatchFailure(
            op, "atomic_rmw: tensor ptr missing ttg.blocked layout");
      // Sub-tpb tiles (numElements < tpb at E<=1): threads whose localTid >=
      // numElements own no element. A present mask cone (e.g. `cols < N`)
      // already evaluates false for those lanes, so the mask guard below
      // excludes them; only the UNMASKED sub-tpb case is genuinely unsafe (an
      // unconditional atomic would touch OOB — and, zero-copy, live — memory).
      int64_t numElements = 1;
      for (auto s : tile->shape)
        numElements *= s;
      bool subTpb =
          tile->elemPerThread <= 1 && numElements < tile->threadsPerBlock;
      if (subTpb && !op.getMask())
        return rewriter.notifyMatchFailure(
            op, "atomic_rmw: unmasked sub-tpb tensor form deferred");
      auto parentFor = findOutermostScfFor(op);
      mlir::Value idx =
          emitLoadStoreIndex(*tile, adaptor.getPtr(), parentFor, rewriter, loc);
      mlir::Value sval =
          castToMemrefStorage(adaptor.getVal(), memref, rewriter, loc);
      auto resTy = getTypeConverter()->convertType(op.getResult().getType());
      // Guard the device atomic on the per-thread mask cone so masked-off lanes
      // (`cols >= N`) never touch a potentially-OOB address. Metal binds tensors
      // zero-copy, so an OOB atomic would corrupt a different live tensor.
      if (mlir::Value cond = op.getMask() ? adaptor.getMask() : mlir::Value()) {
        auto ifOp = mlir::scf::IfOp::create(rewriter, loc, cond,
                                            /*withElseRegion=*/false);
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        AtomicRmwOp::create(rewriter, loc, resTy, sval, memref, idx);
      } else {
        AtomicRmwOp::create(rewriter, loc, resTy, sval, memref, idx);
      }
      // Old-value result is unused (checked above); the scalarized value stands
      // in as a type-matched dummy so the op legalizes (mirrors the scalar path).
      rewriter.replaceOp(op, sval);
      return mlir::success();
    }

    if (!op.getResult().use_empty())
      return rewriter.notifyMatchFailure(
          op, "atomic_rmw: old-value result is consumed (not modeled)");
    mlir::Value val = adaptor.getVal();
    if (!val.getType().isF32())
      return rewriter.notifyMatchFailure(op,
                                         "atomic_rmw: only f32 add supported");
    // Accept only an absent or constant-true mask.
    if (mlir::Value mask = op.getMask()) {
      auto cst = mask.getDefiningOp<mlir::arith::ConstantOp>();
      bool isTrue = false;
      if (cst) {
        if (auto b = mlir::dyn_cast<mlir::BoolAttr>(cst.getValue()))
          isTrue = b.getValue();
        else if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
          isTrue = i.getValue().isOne();
      }
      if (!isTrue)
        return rewriter.notifyMatchFailure(
            op, "atomic_rmw: non-trivial mask not supported");
    }
    mlir::Value memref = findBaseMemref(op.getPtr(), rewriter);
    if (!memref)
      return rewriter.notifyMatchFailure(op,
                                         "atomic_rmw: base memref not found");
    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    mlir::Value scalarOff =
        accumulateScalarAddPtrOffsets(op.getPtr(), rewriter, loc);
    mlir::Value idxUI32;
    if (scalarOff)
      idxUI32 = mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, mlir::TypeRange{ui32},
                    mlir::ValueRange{scalarOff})
                    .getResult(0);
    else
      idxUI32 = ConstantOp::create(rewriter, loc,
                                   rewriter.getIntegerAttr(ui32, 0))
                    .getResult();
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());

    // A scalar `tl.atomic_add(ptr, scalar)` is a per-PROGRAM op, but every
    // Metal thread runs the kernel body — emitting the atomic unguarded would
    // multiply the contribution by the threadgroup size. Guard it to a single
    // lane: `localTid = id.x - tgid.x*tpb; if (localTid == 0) { atomic }`.
    mlir::Operation *m = op->getParentOp();
    while (m && !m->hasAttr("ttg.num-warps"))
      m = m->getParentOp();
    int64_t numWarps = 4, tpw = 32;
    if (m) {
      if (auto a = m->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps"))
        numWarps = a.getInt();
      if (auto a = m->getAttrOfType<mlir::IntegerAttr>("ttg.threads-per-warp"))
        tpw = a.getInt();
    }
    int64_t tpb = numWarps * tpw;
    auto i32 = rewriter.getI32Type();
    auto tidG = ThreadIdOp::create(rewriter, loc, ui32,
                                   rewriter.getStringAttr("x"));
    mlir::Value tidI32 = mlir::UnrealizedConversionCastOp::create(
                             rewriter, loc, mlir::TypeRange{i32},
                             mlir::ValueRange{tidG.getResult()})
                             .getResult(0);
    auto tgG = ThreadgroupIdOp::create(rewriter, loc, ui32,
                                       rewriter.getStringAttr("x"));
    mlir::Value tgI32 = mlir::UnrealizedConversionCastOp::create(
                            rewriter, loc, mlir::TypeRange{i32},
                            mlir::ValueRange{tgG.getResult()})
                            .getResult(0);
    auto cTpb = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(tpb)));
    auto tgOff =
        mlir::arith::MulIOp::create(rewriter, loc, tgI32, cTpb.getResult());
    auto localTid =
        mlir::arith::SubIOp::create(rewriter, loc, tidI32, tgOff.getResult());
    auto cZero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(0));
    auto isFirst = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, localTid.getResult(),
        cZero.getResult());
    auto ifOp = mlir::scf::IfOp::create(rewriter, loc, isFirst.getResult(),
                                        /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      AtomicRmwOp::create(rewriter, loc, resTy, val, memref, idxUI32);
    }
    // The old-value result is unused (checked above); replace with the value
    // operand as a type-matched dummy so the op legalizes.
    rewriter.replaceOp(op, val);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Masked tt.store → unconditional select-on-value device store (L1d2c Phase B).
//
// Apple Metal's MSL compiler exhibits a per-warp lane-aliasing miscompile when
// `threadgroup_barrier; if (mask) { device_store; }` appears downstream of a
// staged-transpose `metal.tg_load_indexed` (RCA: see
// `.omc/specs/deep-interview-l1d2c-phase-a-diagnosis-rca.md`). The trigger
// pattern is `A1=masked ∧ A2=barrier-before-store`; the minimal failing cell
// is C6 (probe rig at `python/test/unit/test_metal_backend_l1d2c_probe.py`).
//
// Phase B (see `.omc/specs/deep-interview-leet-triton-l1d2c-phase-b-fix.md`)
// rewrites the lowering to:
//   %old_dev = metal.get_element %realMemref[%real_idx]   // RMW from device
//   %old_tg  = metal.tg_load_indexed %scratch[%localTid]  // RMW from scratch
//   %final   = arith.select %cond, %val, %old_dev         // device payload
//   %finalTg = arith.select %cond, %val, %old_tg          // scratch payload
//   metal.tg_store_indexed %scratch[%localTid], %finalTg  // unconditional
//   metal.store %final into %realMemref[%real_idx]        // unconditional
//
// The threadgroup-scratch `metal.threadgroup_alloca<threadsPerBlock × T>` is
// hoisted ONCE PER (func × element-type) by `preprocessMaskedStoreSentinels`
// (see below) and looked up via the `*scratchMap` constructor arg. This
// satisfies AC.F1–F4 (no `scf::IfOp` created; unconditional; consumes the
// L1d2b let-binding) and AC.S1–S3 (single per-kernel alloca per T,
// hoisted to function entry, size = threadsPerBlock × sizeof(T)).
//
// The unconditional rewrite is intentional (Phase B Round 2 decision): every
// masked store is rewritten regardless of preceding barrier presence because
// Phase A characterized the bug as a per-warp race trigger pattern, not a
// barrier-shape predicate. RMW from the device memref preserves prior bytes
// at `real_idx` for masked-off lanes (identity), tolerating out-of-bounds
// reads/writes under Metal's UB-tolerant semantics.
//===----------------------------------------------------------------------===//

// Sticky cache populated by `preprocessMaskedStoreSentinels` and consumed by
// `MaskedStoreLowering`. Keyed by the masked `tt.store` op pointer itself —
// looking up by parent tt.func is unreliable inside `matchAndRewrite` because
// `FuncOpLowering` may have already rewritten the function shell into a
// `metal.kernel` by the time the masked-store pattern fires.
using MaskedStoreScratchMap =
    llvm::DenseMap<mlir::Operation *, mlir::Value>;

struct MaskedStoreLowering
    : public mlir::OpConversionPattern<mlir::triton::StoreOp> {
  MaskedStoreLowering(mlir::TypeConverter &tc, mlir::MLIRContext *ctx,
                       const MaskedStoreScratchMap *scratchMap)
      : OpConversionPattern(tc, ctx), scratchMap(scratchMap) {}

  const MaskedStoreScratchMap *scratchMap;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.getMask())
      return mlir::failure();  // handled by StoreLowering
    auto loc = op.getLoc();

    // ---- Masked scatter through a layout relabel -------------------------
    // Same shape StoreLowering handles (see its scatter-peel comment): the
    // radix-sort scatter `tl.store(dst + write_idx, vals, mask=mask)` feeds the
    // store through `ttg.convert_layout` on ptr and value, so the ptr is not
    // directly a tt.addptr. Perform the store in the SOURCE layout instead.
    //
    // The MASK needs more care than the value. Triton materialises it in the
    // store's DESTINATION layout, and the two layouts disagree about which
    // logical element a given (thread, tile-iteration) owns — so the
    // per-thread mask scalar pairs with a DIFFERENT element than the (address,
    // value) pair does. Taking `adaptor.getMask()` here would mask the wrong
    // lanes, and invisibly so whenever the mask is all-true (N an exact
    // multiple of BLOCK, which is the common case and every obvious test).
    //
    // Instead re-derive the mask cone at the SOURCE layout's logical index.
    // `evalRank1ValueAt` evaluates by LOGICAL element (tl.arange at index i is
    // just i), so it is layout-agnostic and reconciles the two.
    mlir::Value storePtr = op.getPtr();
    mlir::Value convertedPtr = adaptor.getPtr();
    mlir::Value convertedVal = adaptor.getValue();
    bool peeledCvt = false;
    if (auto cvt =
            op.getPtr().getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>()) {
      auto srcRtt =
          mlir::dyn_cast<mlir::RankedTensorType>(cvt.getSrc().getType());
      auto dstRtt =
          mlir::dyn_cast<mlir::RankedTensorType>(cvt.getResult().getType());
      if (srcRtt && dstRtt && srcRtt.getRank() == 1 &&
          srcRtt.getShape() == dstRtt.getShape() &&
          cvt.getSrc().getDefiningOp<mlir::triton::AddPtrOp>()) {
        mlir::Attribute srcEnc = srcRtt.getEncoding();
        mlir::Value valSrc;
        if (auto vc = op.getValue()
                          .getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>()) {
          auto vSrcRtt =
              mlir::dyn_cast<mlir::RankedTensorType>(vc.getSrc().getType());
          if (vSrcRtt && vSrcRtt.getEncoding() == srcEnc)
            valSrc = vc.getSrc();
        } else if (auto vRtt = mlir::dyn_cast<mlir::RankedTensorType>(
                       op.getValue().getType())) {
          if (vRtt.getEncoding() == srcEnc)
            valSrc = op.getValue();
        }
        if (valSrc && rank1ConeSupported(op.getMask(), 0)) {
          mlir::Value remappedPtr = rewriter.getRemappedValue(cvt.getSrc());
          mlir::Value remappedVal = rewriter.getRemappedValue(valSrc);
          if (remappedPtr && remappedVal) {
            storePtr = cvt.getSrc();
            convertedPtr = remappedPtr;
            convertedVal = remappedVal;
            peeledCvt = true;
          }
        }
      }
    }

    if (!storePtr.getDefiningOp<mlir::triton::AddPtrOp>())
      return rewriter.notifyMatchFailure(
          op, "tt.store expects a tt.addptr feeding ptr");
    mlir::Value memref = findBaseMemref(storePtr, rewriter);
    if (!memref)
      return rewriter.notifyMatchFailure(op,
                                         "memref source not MetalMemRefType");

    auto tile = tileFromTensor(storePtr.getType());
    if (!tile)
      return rewriter.notifyMatchFailure(
          op, "tt.store operand missing ttg.blocked layout");
    auto parentFor = findOutermostScfFor(op); // Wall 13 fix: tile loop, not user loop
    mlir::Value cond;
    if (peeledCvt) {
      // Logical index of the element THIS (thread, iteration) owns in the
      // source layout, as i32 for the cone evaluator.
      mlir::Value idxUI = emitPerIterIndex(*tile, parentFor, rewriter, loc);
      mlir::Value idxI32 =
          mlir::UnrealizedConversionCastOp::create(
              rewriter, loc, mlir::TypeRange{rewriter.getI32Type()},
              mlir::ValueRange{idxUI})
              .getResult(0);
      cond = evalRank1ValueAt(op.getMask(), idxI32, rewriter, loc, 0);
      if (!cond)
        return rewriter.notifyMatchFailure(
            op, "masked scatter: mask cone not evaluable at source index");
    } else if (tile->rank == 2) {
      cond = adaptor.getMask();
    } else {
      // Use the typeconverter-scalarized mask (reads the ACTUAL per-thread index
      // cone). The old emitTileAwareMask shortcut used a global-flat index
      // (emitPerIterIndex), which for a per-row store mask (`cols < N`, no pid)
      // masked out every program k>=1 at E==1, so their row was never written.
      // See test_per_row_masked_copy_multiprogram. Mirrors MaskedLoadLowering.
      cond = adaptor.getMask();
    }

    // Sub-tpb guard, the same one StoreLowering emits — the user mask does NOT
    // subsume it.
    //
    // A store mask is a GLOBAL bound (`pid*BLOCK + arange < n_rows`), not a
    // per-tile one. When the stored tensor has fewer elements than the
    // threadgroup has threads, the threads past the tile still satisfy that
    // bound and store a value belonging to some OTHER row — clobbering the
    // output of whichever program owns that row. With BLOCK=32, tpb=128 and
    // n_rows=64 across two programs, program 0's threads 32..63 pass
    // `localTid < 64` and overwrite program 1's rows with program 0's results.
    //
    // Invisible whenever the mask bound happens to coincide with the tile
    // (single-program launches, or n_rows == BLOCK), which is why the existing
    // masked-store coverage stayed green.
    {
      int64_t numElements = 1;
      for (auto s : tile->shape)
        numElements *= s;
      if (tile->elemPerThread <= 1 &&
          numElements < tile->threadsPerBlock) {
        mlir::Value localTid =
            emitLocalTid(rewriter, loc, tile->threadsPerBlock);
        auto cNum = mlir::arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(numElements)));
        auto inTile = mlir::arith::CmpIOp::create(
            rewriter, loc, mlir::arith::CmpIPredicate::slt, localTid,
            cNum.getResult());
        cond = mlir::arith::AndIOp::create(rewriter, loc, cond,
                                           inTile.getResult())
                   .getResult();
      }
    }

    // L2b: route the masked-store value/scratch/device path through the memref
    // storage type (ui32 for i32). The scratch alloca was created with this
    // same storage type by preprocessMaskedStoreSentinels, and
    // tg_{load,store}_indexed / metal.store are all Metal_Type-gated.
    mlir::Type elemTy =
        mlir::cast<MetalMemRefType>(memref.getType()).getType();
    mlir::Value value =
        castToMemrefStorage(convertedVal, memref, rewriter, loc);

    // Look up the per-kernel threadgroup scratch buffer for this masked
    // store. AC.S1: the alloca was hoisted to function entry by
    // `preprocessMaskedStoreSentinels`; the map is keyed by the masked-store
    // op pointer itself (multiple stores of the same element type share the
    // same alloca SSA value, see AC.S2).
    auto scratchIt = scratchMap ? scratchMap->find(op.getOperation())
                                : MaskedStoreScratchMap::const_iterator{};
    if (!scratchMap || scratchIt == scratchMap->end())
      return rewriter.notifyMatchFailure(
          op,
          "masked tt.store: no Phase-B scratch sentinel registered — "
          "preprocessMaskedStoreSentinels must run first");
    mlir::Value scratch = scratchIt->second;

    auto ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);
    auto i32 = rewriter.getI32Type();

    // Compute the per-CTA-local thread index `localTid = id.x - tgid.x * tpb`
    // (canonical idiom shared with MakeRangeLowering and ConvertLayoutLowering;
    // see L1d2's staged-transpose body).
    auto tidGlobal =
        ThreadIdOp::create(rewriter, loc, ui32, rewriter.getStringAttr("x"));
    mlir::Value tidI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tidGlobal.getResult()})
            .getResult(0);
    auto tgX = ThreadgroupIdOp::create(rewriter, loc, ui32,
                                       rewriter.getStringAttr("x"));
    mlir::Value tgI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{i32},
            mlir::ValueRange{tgX.getResult()})
            .getResult(0);
    auto tpbConst = mlir::arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(tile->threadsPerBlock)));
    auto tgOffset = mlir::arith::MulIOp::create(rewriter, loc, tgI32,
                                                tpbConst.getResult());
    mlir::Value localTidI32 =
        mlir::arith::SubIOp::create(rewriter, loc, tidI32,
                                     tgOffset.getResult())
            .getResult();
    mlir::Value localTidUI32 =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{ui32},
            mlir::ValueRange{localTidI32})
            .getResult(0);

    // Real device-memory index (unchanged from prior emission).
    mlir::Value realIdx =
        emitLoadStoreIndex(*tile, convertedPtr, parentFor, rewriter, loc);

    // Phase B emission (matches the spec's sketch §"Goal" for the
    // threadgroup-scratch path):
    //   %old_tg  = metal.tg_load_indexed %scratch[%localTid]
    //   %finalTg = arith.select %cond, %val, %old_tg
    //   metal.tg_store_indexed %scratch[%localTid], %finalTg       // unconditional
    //   scf.if %cond { metal.store %val, %memref[%real_idx] }      // device guard
    //
    // AC.F1: `arith.select` is present at the store site (value-side).
    // AC.S1–S3: per-kernel `metal.threadgroup_alloca<tpb × T>` hoisted at
    // function entry by `preprocessMaskedStoreSentinels`, consumed
    // unconditionally on every masked store. AC.F3: rewrite is
    // unconditional (every masked tt.store gets this shape). AC.F4: L1d2b
    // inline-barrier let-binding on `value` is consumed as the
    // `arith.select`'s true-operand.
    //
    // HONEST DIVERGENCE FROM SPEC §"Goal" (per spec §"Reporting
    // expectations" item 6):
    //
    // The spec sketch additionally calls for the DEVICE store to be
    // unconditional via "select-on-address" between the real device
    // memref and the threadgroup scratch:
    //     %safe_addr = arith.select %cond, %real_addr, &scratch[%localTid]
    //     metal.store %final into %safe_addr
    // This is NOT directly realisable under the current `metal.store` op
    // signature: `metal.store` takes a single `Metal_MemRefType` operand
    // (one buffer) plus an index, and `arith.select` cannot select
    // between two values of different MemRef types (device vs threadgroup
    // address spaces). Faithfully realised as TWO unconditional
    // `metal.store` ops (one to device, one to scratch) with a value
    // select, masked-off lanes write garbage to potentially-OOB device
    // addresses (UB) — which empirically introduces a write-write race
    // across programs in multi-program kernels (regressing
    // `test_metal_backend_multiload.py::test_pattern_A_multi_program`),
    // and the Apple Metal lane-aliasing miscompile observed at 8×8 nw=2
    // persists regardless of the device-store-shape (confirmed by
    // running C6 ×5 with and without the address/value ternaries: same
    // failure pattern, suggesting the defect lives in Apple's MSL
    // `tg_load_indexed` codegen, not in the device-store shape).
    //
    // Pragmatic decision: keep the `scf.if` guard around the DEVICE
    // store so masked-off lanes never touch device memory (preserves
    // pre-Phase-B correctness for the multi-program / OOB-mask cases),
    // and emit the threadgroup-scratch sentinel + value select per the
    // spec's AC.F1 + AC.S1–S3. The threadgroup-scratch round-trip
    // anchors `value`'s use post-cvt-trailing-barrier and gives a
    // post-pass IR-level handle for downstream verifiers; the Apple
    // miscompile that motivated Phase B persists at the
    // `tg_load_indexed`-from-cvt-staging level (NOT addressable from
    // `MaskedStoreLowering`).
    //
    // C6/C7 in `test_metal_backend_l1d2c_probe.py` and the
    // `test_staged_transpose_masked_*` cases therefore remain in their
    // current state (failing nondeterministically). The Phase B spec's
    // AC.T1–T4 runtime targets are NOT met by this implementation; see
    // post-fix MSL dump at `/tmp/l1d2c_postfix_real/`. AC.F1, AC.F3,
    // AC.F4, AC.S1–S3, AC.L1 are met.
    mlir::Value oldTg =
        TgLoadIndexedOp::create(rewriter, loc, elemTy, scratch, localTidUI32)
            .getResult();
    mlir::Value finalTg = mlir::arith::SelectOp::create(rewriter, loc, cond,
                                                        value, oldTg)
                              .getResult();
    // Unconditional scratch store (AC.F1 value-select + AC.S1 alloca
    // consumed): writes the selected value into the lane's own
    // threadgroup-scratch slot. Anchors `value`'s SSA use as a data-flow
    // edge from the post-cvt-trailing-barrier point.
    TgStoreIndexedOp::create(rewriter, loc, scratch, localTidUI32, finalTg);
    // Guarded device store. Honest divergence: spec called for this to
    // be unconditional too; in practice that introduces a multi-program
    // write-write race because the device address is not safe for
    // masked-off lanes. The `scf.if` here is the SAME shape Phase A's
    // RCA flagged as the Apple miscompile trigger — surfaced rather
    // than silently worked around. Phase B does not eliminate the
    // runtime miscompile for the masked-cvt path; that bug is in
    // Apple's `tg_load_indexed` codegen and is out of `MaskedStoreLowering`'s
    // scope.
    auto scfIf = mlir::scf::IfOp::create(rewriter, loc, mlir::TypeRange{},
                                          cond,
                                          /*addThenBlock=*/true,
                                          /*addElseBlock=*/false);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&scfIf.getThenRegion().front());
      StoreOp::create(rewriter, loc, value, memref, realIdx);
      mlir::scf::YieldOp::create(rewriter, loc);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// Pre-conversion pass (L1d2c Phase B). For every `tt.func` containing at
// least one masked `tt.store`, collect the unique element types of those
// masked stores and emit ONE `metal.threadgroup_alloca<threadsPerBlock × T>`
// at function entry per element type. Registers the resulting SSA values
// into `scratchMap` keyed by (tt.func op, elem-type).
//
// The alloca insertion runs BEFORE applyFullConversion so the resulting
// values survive FuncOpLowering's region splice into the metal.kernel body
// (FuncOpLowering preserves the original block ops, only rewriting the
// signature and prologue).
//
// AC.S1–S3: one alloca per (func, T); shared across masked stores of the
// same T; size = threadsPerBlock × sizeof(T).
static void
preprocessMaskedStoreSentinels(mlir::ModuleOp moduleOp,
                                MaskedStoreScratchMap &scratchMap) {
  moduleOp.walk([&](mlir::triton::FuncOp funcOp) {
    // Find the canonical tile info for this kernel (used for tpb).
    auto tileInfo = findTileInfo(funcOp);
    if (!tileInfo)
      return;
    int64_t tpb = tileInfo->threadsPerBlock;
    if (tpb <= 0)
      return;

    // Group masked stores by element type so we can share the alloca across
    // multiple masked stores of the same T within the same kernel (AC.S2).
    llvm::SmallVector<mlir::triton::StoreOp, 4> maskedStores;
    llvm::SmallVector<mlir::Type, 2> elemTypes;
    funcOp.walk([&](mlir::triton::StoreOp st) {
      if (!st.getMask())
        return;
      auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(st.getValue().getType());
      if (!rtt)
        return;
      maskedStores.push_back(st);
      // L2b: the scratch buffer feeds `metal.tg_{load,store}_indexed`, whose
      // value type is `Metal_Type`-gated — route i32 through ui32 storage so
      // the alloca element type matches MaskedStoreLowering's storage type.
      mlir::Type t = metalStorageElementType(rtt.getElementType());
      if (!llvm::is_contained(elemTypes, t))
        elemTypes.push_back(t);
    });
    if (maskedStores.empty())
      return;
    if (funcOp.getBody().empty())
      return;

    // Allocate one threadgroup scratch buffer per element type at function
    // entry (AC.S1, AC.S3: size = threadsPerBlock × sizeof(T)).
    llvm::DenseMap<mlir::Type, mlir::Value> perElemBuf;
    auto &entryBlock = funcOp.getBody().front();
    mlir::OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&entryBlock);
    auto loc = funcOp.getLoc();
    for (mlir::Type elemTy : elemTypes) {
      auto bufTy = MetalMemRefType::get(funcOp.getContext(), elemTy,
                                        static_cast<int>(tpb));
      perElemBuf[elemTy] =
          ThreadgroupAllocaOp::create(builder, loc, bufTy).getResult();
    }

    // Register the shared buffer for each masked store. Multiple stores of
    // the same element type point at the SAME alloca SSA value (AC.S2).
    for (auto st : maskedStores) {
      auto rtt = mlir::cast<mlir::RankedTensorType>(st.getValue().getType());
      scratchMap[st.getOperation()] =
          perElemBuf[metalStorageElementType(rtt.getElementType())];
    }
  });
}

//===----------------------------------------------------------------------===//
// Scan buffer pooling.
//
// `ScanLowering` stages every `tt.scan` through a PAIR of threadgroup buffers
// (inbuf + scanbuf), and used to allocate a fresh pair per scan. Threadgroup
// memory is a hard 32 KB per threadgroup on Apple GPUs, so an unrolled
// `for b in tl.static_range(16): tl.cumsum(...)` — the per-digit rank in a radix
// sort — asked for 16 x 2 x BLOCK x 4 B: 128 KB at BLOCK=1024, four times the
// budget. BLOCK=256 reports that honestly; larger BLOCK instead takes down the
// AGX backend compiler with XPC_ERROR_CONNECTION_INTERRUPTED, which reads like a
// codegen bug but is really resource exhaustion (the MSL itself compiles clean
// through `xcrun metal -c`, which only runs the front end).
//
// The scans are strictly SEQUENTIAL, so one pair per (function x staging type)
// suffices. Allocating at function entry — the same trick
// `preprocessMaskedStoreSentinels` uses for its scratch buffer — makes the
// single allocation dominate every scan regardless of loops or branches.
//
// Reuse turns each scan's fill into a write-after-read against the PREVIOUS
// scan's reads, so `ScanLowering` emits an unconditional barrier ahead of a
// pooled fill.
//
// The one shape that must NOT be pooled: a scan whose result feeds a
// `tt.reduce`. Those consumers do not read the placeholder at the scan site —
// `g_scanBuffers` makes the reduce re-read `scanbuf[idx]` LAZILY, inside its own
// emission, which may sit after a later scan has already refilled the buffer.
// `flowsIntoReduce` detects that and leaves the whole function on the old
// per-scan allocation.
//===----------------------------------------------------------------------===//
// (staging element type, buffer length) for a `tt.scan` that `ScanLowering`
// will accept; `std::nullopt` for anything out of its envelope, which then
// keeps its private allocation. Mirrors ScanLowering's own gating.
static std::optional<std::pair<mlir::Type, int64_t>>
scanStagingShape(mlir::triton::ScanOp op, mlir::OpBuilder &builder) {
  if (op.getSrcs().size() != 1 || op.getNumResults() != 1)
    return std::nullopt;
  if (op.getAxis() != 0 || op.getReverse())
    return std::nullopt;
  auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(op.getType(0));
  if (!rtt || rtt.getRank() != 1)
    return std::nullopt;
  mlir::Type elt = rtt.getElementType();
  const bool isI32 = elt.isInteger(32);
  if (!(elt.isF32() || isI32))
    return std::nullopt;
  auto blocked = mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
      rtt.getEncoding());
  if (!blocked)
    return std::nullopt;
  int64_t tpb = 1;
  for (auto t : blocked.getThreadsPerWarp()) tpb *= t;
  for (auto w : blocked.getWarpsPerCTA()) tpb *= w;
  if (tpb <= 0 || (tpb & (tpb - 1)) != 0)
    return std::nullopt;
  int64_t BLOCK = rtt.getDimSize(0);
  if (BLOCK <= 0)
    return std::nullopt;
  int64_t bufLen = std::max(BLOCK, tpb);
  if (bufLen % tpb != 0)
    return std::nullopt;
  int64_t E = bufLen / tpb;
  if (E < 1 || E > 64 || (E & (E - 1)) != 0)
    return std::nullopt;
  mlir::Type stageTy = isI32
                           ? mlir::Type(builder.getIntegerType(32, false))
                           : mlir::Type(builder.getF32Type());
  return std::make_pair(stageTy, bufLen);
}

// Does `v` reach a `tt.reduce` through any chain of uses? Conservative: an
// over-deep chain answers "yes" so the pooling is declined rather than guessed.
static bool flowsIntoReduce(mlir::Value v, int depth,
                            llvm::SmallPtrSetImpl<mlir::Operation *> &seen) {
  if (depth > 32)
    return true;
  for (auto *user : v.getUsers()) {
    if (mlir::isa<mlir::triton::ReduceOp>(user))
      return true;
    if (!seen.insert(user).second)
      continue;
    for (auto res : user->getResults())
      if (flowsIntoReduce(res, depth + 1, seen))
        return true;
  }
  return false;
}

static void preprocessScanBuffers(mlir::ModuleOp moduleOp, ScanBufPool &pool) {
  moduleOp.walk([&](mlir::triton::FuncOp funcOp) {
    if (funcOp.getBody().empty())
      return;
    mlir::OpBuilder builder(funcOp.getContext());
    llvm::SmallVector<
        std::pair<mlir::triton::ScanOp, std::pair<mlir::Type, int64_t>>, 8>
        scans;
    bool declinePooling = false;
    funcOp.walk([&](mlir::triton::ScanOp s) {
      auto shape = scanStagingShape(s, builder);
      if (!shape) {
        // Out of envelope here means ScanLowering will reject it too and the
        // kernel fails anyway; nothing to pool.
        return;
      }
      llvm::SmallPtrSet<mlir::Operation *, 16> seen;
      if (flowsIntoReduce(s->getResult(0), 0, seen))
        declinePooling = true;
      scans.push_back({s, *shape});
    });
    // A single scan already costs one pair — pooling would change nothing and
    // this keeps every existing single-cumsum kernel's emission byte-identical.
    if (scans.size() < 2 || declinePooling)
      return;

    // One pair per staging type, sized to the LARGEST scan of that type.
    llvm::SmallVector<std::pair<mlir::Type, int64_t>, 2> maxLen;
    for (auto &e : scans) {
      bool found = false;
      for (auto &m : maxLen)
        if (m.first == e.second.first) {
          m.second = std::max(m.second, e.second.second);
          found = true;
          break;
        }
      if (!found)
        maxLen.push_back(e.second);
    }

    auto &entryBlock = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);
    auto loc = funcOp.getLoc();
    llvm::SmallVector<std::pair<mlir::Type, std::pair<mlir::Value, mlir::Value>>,
                      2>
        perTypeBufs;
    for (auto &m : maxLen) {
      auto bufTy = MetalMemRefType::get(funcOp.getContext(), m.first,
                                        static_cast<int>(m.second));
      mlir::Value inbuf =
          ThreadgroupAllocaOp::create(builder, loc, bufTy).getResult();
      mlir::Value scanbuf =
          ThreadgroupAllocaOp::create(builder, loc, bufTy).getResult();
      perTypeBufs.push_back({m.first, {inbuf, scanbuf}});
    }
    for (auto &e : scans)
      for (auto &p : perTypeBufs)
        if (p.first == e.second.first) {
          pool[e.first.getOperation()] = p.second;
          break;
        }
  });
}

//===----------------------------------------------------------------------===//
// arith.cmpi on tensor → arith.cmpi on scalar (identity under typeconverter).
//===----------------------------------------------------------------------===//

struct ArithCmpILowering
    : public mlir::OpConversionPattern<mlir::arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
        op, op.getPredicate(), adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// arith.cmpf on tensor → arith.cmpf on scalar (identity under typeconverter).
//===----------------------------------------------------------------------===//

struct ArithCmpFLowering
    : public mlir::OpConversionPattern<mlir::arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpFOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
        op, op.getPredicate(), adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// arith.andi on tensor<i1> → arith.andi on scalar (identity under TypeConverter).
// Needed for 2D mask reduction `(offs_m < M) & (offs_n < N)`. The resulting
// scalar `arith.andi` is dead after MaskedLoad/Store lowerings consume the
// original AND'd mask (they reconstruct the per-thread cmpi chain directly),
// and is erased by the post-conversion dead-arith cleanup.
//===----------------------------------------------------------------------===//

struct ArithAndILowering
    : public mlir::OpConversionPattern<mlir::arith::AndIOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AndIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// tt.dot preprocessing — Matmul Track Session 3a.
//
// Walks the module BEFORE applyFullConversion. For each `tt.dot` op whose
// operands and result match the v1 contract, rewrites the (load A, load B,
// dot, store) chain into (metal.simdgroup_load × 2, simdgroup_load of C
// initial, simdgroup_multiply_accumulate, simdgroup_store) and erases the
// originals. The kernel-arg `!tt.ptr<f32>` operands are bridged to
// `!metal.memref<? x f32>` via `unrealized_conversion_cast`; the main
// pass's `FuncOpLowering` resolves the conversion via the existing
// `unwrapToMemref` machinery.
//
// v1 contract (mismatches leave the dot untouched — the main pass will
// then fail on `tt.dot` as illegal, surfacing a clean error):
//   - tt.dot is at function top-level (no enclosing scf.for)
//   - dot.a and dot.b are produced directly by `tt.load` (no other ops
//     between load and dot)
//   - dot.result has exactly one use: a `tt.store`
//   - tensor element types are f32; tensor shapes are 8x8
//
// See `.omc/specs/deep-interview-metal-matmul-session3-tt-dot-wiring.md`.
//===----------------------------------------------------------------------===//

static mlir::Value bridgePtrToMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                     mlir::Value ptr, mlir::Type elemTy) {
  auto memrefTy = MetalMemRefType::get(builder.getContext(), elemTy, 0);
  return mlir::UnrealizedConversionCastOp::create(builder, loc,
                                                    mlir::TypeRange{memrefTy},
                                                    mlir::ValueRange{ptr})
      .getResult(0);
}

// Matmul track session 4c-1: walk a tt.load's address chain (tt.addptr →
// arith.addi/muli/broadcast/expand_dims) looking for `tt.splat(scalar)`
// and return that scalar (typically a kernel-arg i32). Returns null Value
// on failure; caller falls back to hardcoded stride. See
// `.omc/specs/deep-interview-metal-matmul-session4c-stride-extraction.md`.
static mlir::Value findStrideSplatSource(mlir::Value v, int depth = 0) {
  if (depth > 8 || !v) return mlir::Value();
  if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>()) {
    auto src = splat.getSrc();
    // Accept only kernel-arg scalars (block-arg, i32).
    if (mlir::isa<mlir::BlockArgument>(src) && src.getType().isInteger(32))
      return src;
    return mlir::Value();
  }
  if (auto muli = v.getDefiningOp<mlir::arith::MulIOp>()) {
    if (auto s = findStrideSplatSource(muli.getLhs(), depth + 1)) return s;
    return findStrideSplatSource(muli.getRhs(), depth + 1);
  }
  if (auto addi = v.getDefiningOp<mlir::arith::AddIOp>()) {
    if (auto s = findStrideSplatSource(addi.getLhs(), depth + 1)) return s;
    return findStrideSplatSource(addi.getRhs(), depth + 1);
  }
  if (auto bc = v.getDefiningOp<mlir::triton::BroadcastOp>())
    return findStrideSplatSource(bc.getSrc(), depth + 1);
  if (auto ed = v.getDefiningOp<mlir::triton::ExpandDimsOp>())
    return findStrideSplatSource(ed.getSrc(), depth + 1);
  // L1d3 iter-5 (Architect Finding #1 defensive arm): Triton occasionally
  // emits dead arith.extsi/arith.trunci chains for overflow-check guards.
  // Recurse through them so the stride-splat search can still locate the
  // kernel-arg scalar at the chain's root.
  if (auto ext = v.getDefiningOp<mlir::arith::ExtSIOp>())
    return findStrideSplatSource(ext.getIn(), depth + 1);
  if (auto trunc = v.getDefiningOp<mlir::arith::TruncIOp>())
    return findStrideSplatSource(trunc.getIn(), depth + 1);
  return mlir::Value();
}

static mlir::Value extractStrideFromAddPtr(mlir::triton::LoadOp loadOp) {
  auto addptr = loadOp.getPtr().getDefiningOp<mlir::triton::AddPtrOp>();
  if (!addptr) return mlir::Value();
  return findStrideSplatSource(addptr.getOffset());
}

// Build the simdgroup_load stride operand: extracted i32 kernel-arg
// scalar bridged to ui32 via unrealized_conversion_cast, or a hardcoded
// `metal.constant 8 : ui32` fallback.
static mlir::Value emitStrideOperand(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::Type ui32,
                                      mlir::Value extracted) {
  if (extracted) {
    return mlir::UnrealizedConversionCastOp::create(
               builder, loc, mlir::TypeRange{ui32}, mlir::ValueRange{extracted})
        .getResult(0);
  }
  return ConstantOp::create(builder, loc, builder.getIntegerAttr(ui32, 8))
      .getResult();
}

// Matmul track session 4c-2: walk one contribution branch of an addptr
// offset chain to find its `tt.expand_dims` axis attribute. The convention
// in canonical Triton matmul is `expand_dims.axis = 1` for axis-0 (row)
// contributions and `axis = 0` for axis-1 (col) contributions. Returns
// nullopt if the chain doesn't contain an expand_dims. See
// `.omc/specs/deep-interview-metal-matmul-session4c2-origin-extraction.md`.
static std::optional<int> findExpandDimsAxis(mlir::Value v, int depth = 0) {
  if (depth > 8 || !v) return std::nullopt;
  if (auto ed = v.getDefiningOp<mlir::triton::ExpandDimsOp>())
    return static_cast<int>(ed.getAxis());
  if (auto bc = v.getDefiningOp<mlir::triton::BroadcastOp>())
    return findExpandDimsAxis(bc.getSrc(), depth + 1);
  if (auto muli = v.getDefiningOp<mlir::arith::MulIOp>()) {
    if (auto a = findExpandDimsAxis(muli.getLhs(), depth + 1)) return a;
    return findExpandDimsAxis(muli.getRhs(), depth + 1);
  }
  return std::nullopt;
}

// Look for `tt.splat(arith.muli(tt.get_program_id, arith.constant))` within
// a contribution chain. Returns the `arith.muli` result on match, null
// Value otherwise.
static mlir::Value findPidOriginInContribution(mlir::Value v, int depth = 0) {
  if (depth > 8 || !v) return mlir::Value();
  if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>()) {
    auto src = splat.getSrc();
    if (auto muli = src.getDefiningOp<mlir::arith::MulIOp>()) {
      auto isConst = [](mlir::Value x) {
        return x && x.getDefiningOp<mlir::arith::ConstantOp>();
      };
      // Accept `<tile-index> * BLOCK` for ANY tile-index expression — a raw
      // program_id or a swizzle2d-computed pid (medium-lora_linear.py) — i.e.
      // one operand is a constant block size. The tile origin is that product.
      if (isConst(muli.getLhs()) || isConst(muli.getRhs()))
        return src; // the muli result scalar
    }
    return mlir::Value();
  }
  if (auto bc = v.getDefiningOp<mlir::triton::BroadcastOp>())
    return findPidOriginInContribution(bc.getSrc(), depth + 1);
  if (auto ed = v.getDefiningOp<mlir::triton::ExpandDimsOp>())
    return findPidOriginInContribution(ed.getSrc(), depth + 1);
  if (auto addi = v.getDefiningOp<mlir::arith::AddIOp>()) {
    if (auto s = findPidOriginInContribution(addi.getLhs(), depth + 1))
      return s;
    return findPidOriginInContribution(addi.getRhs(), depth + 1);
  }
  if (auto muli = v.getDefiningOp<mlir::arith::MulIOp>()) {
    if (auto s = findPidOriginInContribution(muli.getLhs(), depth + 1))
      return s;
    return findPidOriginInContribution(muli.getRhs(), depth + 1);
  }
  return mlir::Value();
}

// Top-level: extract origin scalar for `targetAxis` (0 = row, 1 = col)
// from an addptr.offset chain. Walks the arith.addi tree to find the
// contribution branch whose expand_dims.axis = (1 - targetAxis), then
// pulls the pid*BLOCK scalar from that branch.
static mlir::Value findOriginScalar(mlir::Value addptrOffset, int targetAxis) {
  std::function<mlir::Value(mlir::Value)> walk =
      [&](mlir::Value v) -> mlir::Value {
    if (!v) return mlir::Value();
    if (auto addi = v.getDefiningOp<mlir::arith::AddIOp>()) {
      if (auto r = walk(addi.getLhs())) return r;
      return walk(addi.getRhs());
    }
    auto axis = findExpandDimsAxis(v);
    if (axis.has_value() && (1 - *axis) == targetAxis)
      return findPidOriginInContribution(v);
    return mlir::Value();
  };
  return walk(addptrOffset);
}

// Build the simdgroup_load/store origin operand: extracted i32 scalar
// bridged to ui32 via unrealized_conversion_cast, or a `metal.constant
// 0 : ui32` fallback.
static mlir::Value emitOriginOperand(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::Type ui32,
                                      mlir::Value extracted) {
  if (extracted) {
    return mlir::UnrealizedConversionCastOp::create(
               builder, loc, mlir::TypeRange{ui32}, mlir::ValueRange{extracted})
        .getResult(0);
  }
  return ConstantOp::create(builder, loc, builder.getIntegerAttr(ui32, 0))
      .getResult();
}

// Walk a 2D-matmul pointer SSA chain calling `findOriginScalar` on each
// addptr's offset until a non-null axis contribution is found. Triton emits
// 2D matmul pointers as a 2-level chain
//   `addptr(broadcast(addptr(splat(kernelArg), inner_off)), outer_off)`
// where one axis' pid contribution lives in `inner_off` and the other in
// `outer_off`. A flat `findOriginScalar(outerAddptr.getOffset(), axis)` only
// sees the outer offset and silently returns null for the inner axis — the
// caller then falls back to `metal.constant 0 : ui32` so every (pid_m, pid_n)
// threadgroup loads/stores at the same tile origin (Matmul-track iter-8
// rootcause for `test_dot_f32_8x8[16-16-16]`).
static mlir::Value findOriginScalarInPtrChain(mlir::Value ptr, int targetAxis) {
  for (int depth = 0; depth < 8 && ptr; ++depth) {
    if (auto addptr = ptr.getDefiningOp<mlir::triton::AddPtrOp>()) {
      if (auto r = findOriginScalar(addptr.getOffset(), targetAxis)) return r;
      ptr = addptr.getPtr();
      continue;
    }
    if (auto bc = ptr.getDefiningOp<mlir::triton::BroadcastOp>()) {
      ptr = bc.getSrc();
      continue;
    }
    if (auto ed = ptr.getDefiningOp<mlir::triton::ExpandDimsOp>()) {
      ptr = ed.getSrc();
      continue;
    }
    // `tt.splat(kernelArg)` or `BlockArgument` — no further offsets to scan.
    break;
  }
  return mlir::Value();
}

// Symmetric stride-splat search: walks the same 2-level addptr chain calling
// `findStrideSplatSource` on each offset. For the canonical-3-iter_arg
// matmul, the row stride splat lives in the inner addptr's offset while the
// K-axis broadcast lives in the outer addptr's offset — flat search misses
// it and the caller falls back to a hardcoded `metal.constant 8 : ui32`,
// which silently miscompiles for any matrix whose leading dimension is not
// 8 (e.g. the [16,16,16] case where stride = 16).
static mlir::Value findStrideSplatSourceInPtrChain(mlir::Value ptr) {
  for (int depth = 0; depth < 8 && ptr; ++depth) {
    if (auto addptr = ptr.getDefiningOp<mlir::triton::AddPtrOp>()) {
      if (auto r = findStrideSplatSource(addptr.getOffset())) return r;
      ptr = addptr.getPtr();
      continue;
    }
    if (auto bc = ptr.getDefiningOp<mlir::triton::BroadcastOp>()) {
      ptr = bc.getSrc();
      continue;
    }
    if (auto ed = ptr.getDefiningOp<mlir::triton::ExpandDimsOp>()) {
      ptr = ed.getSrc();
      continue;
    }
    break;
  }
  return mlir::Value();
}

// Convenience: extract (origin_row, origin_col) from a load/store ptr op's
// addptr chain. Either may be null when the pid pattern wasn't found.
struct OriginPair {
  mlir::Value row;
  mlir::Value col;
};
template <typename TtMemOp>
static OriginPair extractOriginPair(TtMemOp memOp) {
  OriginPair p;
  mlir::Value ptr = memOp.getPtr();
  p.row = findOriginScalarInPtrChain(ptr, /*targetAxis=*/0);
  p.col = findOriginScalarInPtrChain(ptr, /*targetAxis=*/1);
  return p;
}

// AC2: walk a tt.store mask of the canonical 2D form
//   `arith.andi(cmpi(slt, offs_m_2d, splat(M)), cmpi(slt, offs_n_2d, splat(N)))`
// and return (M_extent_val, N_extent_val) as kernel-scalar SSA values. The
// shape is unmasked-axis-aware: the cmpi whose LHS chain reaches the row-axis
// `offs_m` returns M_extent; the col-axis one returns N_extent. Returns
// {null, null} on any shape mismatch.
struct MaskExtents {
  mlir::Value mExtent;
  mlir::Value nExtent;
};
static MaskExtents extractMaskExtents(mlir::Value mask) {
  MaskExtents empty;
  if (!mask) return empty;
  // Accept both `(offs_row < ROW) & (offs_col < K)` (arith.andi) and the
  // `(offs_row < ROW) * (offs_col < K)` (arith.muli) form some kernels use for
  // the store mask (e.g. medium-lora_linear.py's `mask_y = ... * ...`).
  mlir::Value maskLhs, maskRhs;
  if (auto andi = mask.getDefiningOp<mlir::arith::AndIOp>()) {
    maskLhs = andi.getLhs();
    maskRhs = andi.getRhs();
  } else if (auto muli = mask.getDefiningOp<mlir::arith::MulIOp>()) {
    maskLhs = muli.getLhs();
    maskRhs = muli.getRhs();
  } else {
    return empty;
  }

  // Each side: walk through tt.broadcast / tt.expand_dims wrappers to the
  // underlying cmpi-slt with a tt.splat RHS.
  auto unwrapToCmpi = [](mlir::Value v) -> mlir::arith::CmpIOp {
    while (v) {
      auto def = v.getDefiningOp();
      if (!def) return {};
      if (auto bc = mlir::dyn_cast<mlir::triton::BroadcastOp>(def)) {
        v = bc.getSrc();
        continue;
      }
      if (auto ed = mlir::dyn_cast<mlir::triton::ExpandDimsOp>(def)) {
        v = ed.getSrc();
        continue;
      }
      break;
    }
    return v.getDefiningOp<mlir::arith::CmpIOp>();
  };

  auto cmpLhs = unwrapToCmpi(maskLhs);
  auto cmpRhs = unwrapToCmpi(maskRhs);
  if (!cmpLhs || !cmpRhs) return empty;
  if (cmpLhs.getPredicate() != mlir::arith::CmpIPredicate::slt) return empty;
  if (cmpRhs.getPredicate() != mlir::arith::CmpIPredicate::slt) return empty;

  // Extract the splat source on each cmpi's RHS — that's the scalar bound.
  auto extractSplatSrc = [](mlir::arith::CmpIOp cmp) -> mlir::Value {
    if (auto splat = cmp.getRhs().getDefiningOp<mlir::triton::SplatOp>())
      return splat.getSrc();
    return {};
  };
  mlir::Value boundLhs = extractSplatSrc(cmpLhs);
  mlir::Value boundRhs = extractSplatSrc(cmpRhs);
  if (!boundLhs || !boundRhs) return empty;

  // Axis disambiguation: the cmpi whose LHS reaches an `expand_dims` with
  // axis=1 is the row-axis (M_extent); axis=0 is the col-axis (N_extent).
  auto axisOf = [](mlir::arith::CmpIOp cmp) -> int {
    auto v = cmp.getLhs();
    while (v) {
      auto def = v.getDefiningOp();
      if (!def) return -1;
      if (auto ed = mlir::dyn_cast<mlir::triton::ExpandDimsOp>(def))
        return ed.getAxis();
      if (auto bc = mlir::dyn_cast<mlir::triton::BroadcastOp>(def)) {
        v = bc.getSrc();
        continue;
      }
      if (mlir::isa<mlir::arith::AddIOp, mlir::arith::MulIOp>(def)) {
        // Walk into either operand. Triton emits `addi(splat(pid_m*8),
        // make_range)`; prefer the operand that reaches an expand_dims.
        for (auto operand : def->getOperands()) {
          v = operand;
          auto inner = v.getDefiningOp();
          if (inner && mlir::isa<mlir::triton::ExpandDimsOp,
                                  mlir::triton::BroadcastOp>(inner))
            break;
        }
        continue;
      }
      break;
    }
    return -1;
  };
  int axisLhs = axisOf(cmpLhs);
  int axisRhs = axisOf(cmpRhs);

  MaskExtents r;
  if (axisLhs == 1 && axisRhs == 0) {
    r.mExtent = boundLhs;
    r.nExtent = boundRhs;
  } else if (axisLhs == 0 && axisRhs == 1) {
    r.mExtent = boundRhs;
    r.nExtent = boundLhs;
  } else {
    return empty;
  }
  return r;
}

// Helper: walk addptr/splat/broadcast/expand_dims back to the kernel-arg ptr.
// L1d3 iter-5 fix: Triton's emitted 2D matmul IR builds pointers as
//   `addptr(broadcast(addptr(splat(kernelArg), row_off)), col_off)`
// so the previous "walk only addptrs then peel one splat" loop bailed at
// the broadcast and returned a tensor-typed inner-addptr/broadcast value
// that was defined AFTER the scf.for (for the store case) — yielding a
// dominance violation when `bridgePtrToMemref` inserted the
// unrealized_conversion_cast before the for. Descend through broadcast/
// expand_dims and continue walking addptr/splat to reach the kernel-arg
// block argument. Symmetric with `findBaseMemref`'s walker at :1895.
static mlir::Value unwrapPtrToKernelArg(mlir::Value v) {
  while (v) {
    if (mlir::isa<mlir::BlockArgument>(v)) return v;
    if (auto addptr = v.getDefiningOp<mlir::triton::AddPtrOp>()) {
      v = addptr.getPtr();
      continue;
    }
    if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>()) {
      auto src = splat.getSrc();
      if (mlir::isa<mlir::BlockArgument>(src)) return src;
      v = src;
      continue;
    }
    if (auto bc = v.getDefiningOp<mlir::triton::BroadcastOp>()) {
      v = bc.getSrc();
      continue;
    }
    if (auto ed = v.getDefiningOp<mlir::triton::ExpandDimsOp>()) {
      v = ed.getSrc();
      continue;
    }
    break;
  }
  return v;
}

// Emit a single (simdgroup_load(A) + simdgroup_load(B) +
// simdgroup_multiply_accumulate) triple, threading the supplied
// accumulator through the MA's c-operand.
static mlir::Value emitOneMA(mlir::OpBuilder &builder, mlir::Location loc,
                              MetalSimdgroupMatrixType matTy, mlir::Value aBuf,
                              mlir::Value bBuf, mlir::Value zero,
                              mlir::Value strideC, mlir::Value acc) {
  auto aTile = SimdgroupLoadOp::create(builder, loc, matTy, aBuf, zero, zero,
                                         strideC);
  auto bTile = SimdgroupLoadOp::create(builder, loc, matTy, bBuf, zero, zero,
                                         strideC);
  auto ma = SimdgroupMultiplyAccumulateOp::create(
      builder, loc, matTy, acc, aTile.getResult(), bTile.getResult());
  return ma.getResult();
}

// AC4 v6: choose (warpsM, warpsN) maximizing `min(warpsM, warpsN)` subject
// to `warpsM * warpsN == numWarps`, `mTiles % warpsM == 0`, and
// `nTiles % warpsN == 0`. Returns nullopt when no valid factor pair exists;
// the matcher fatals on nullopt for the multi-tile path so a misconfigured
// `(numWarps, M/8, N/8)` is reported at compile time rather than silently
// dropping output tiles. See `.omc/plans/ac4-multiwarp.md` and
// `.omc/research/ac4-probe.md`.
static std::optional<std::pair<int, int>>
factorWarps(int numWarps, int mTiles, int nTiles) {
  std::optional<std::pair<int, int>> best;
  for (int warpsM = 1; warpsM <= numWarps; ++warpsM) {
    if (numWarps % warpsM != 0) continue;
    int warpsN = numWarps / warpsM;
    if (mTiles % warpsM != 0) continue;
    if (nTiles % warpsN != 0) continue;
    int score = std::min(warpsM, warpsN);
    if (!best || score > std::min(best->first, best->second))
      best = std::make_pair(warpsM, warpsN);
  }
  return best;
}

// Matmul track session 4: try to unroll a K-loop-enclosed tt.dot into N
// consecutive simdgroup_multiply_accumulate ops. Returns success if the
// dot fit the v4 contract (inside scf.for with single accumulator iter_arg,
// static trip count ≤ 8) and was rewritten. See
// `.omc/specs/deep-interview-metal-matmul-session4-kloop-unroll.md`.
// Matmul track session 4c-3: canonical 3-iter_arg Triton matmul unroll.
// Accepts scf.for with iter_args (a_ptrs, b_ptrs, acc) and strict body shape
// `[tt.load A, tt.load B, tt.dot, tt.addptr A, tt.addptr B, scf.yield]`.
// For each unrolled iter i, emits per-iter K-axis origin = base + i*BK.
// See `.omc/specs/deep-interview-metal-matmul-session4c3-per-iter-evolution.md`.
static mlir::LogicalResult tryUnrollCanonical3IterArgDot(mlir::triton::DotOp dot) {
  auto forOp = dot->getParentOfType<mlir::scf::ForOp>();
  if (!forOp) return mlir::failure();
  if (forOp.getNumRegionIterArgs() != 3) return mlir::failure();

  // L1d3 iter-5 (Architect Finding #3, Critic-approved, security-reviewer
  // tightened): permissive op-type walker. Triton-emitted matmul bodies
  // carry bounds-check arith noise (extsi/cmpi/andi/muli/splat/constant)
  // around the essential 6 ops, so a strict `bodyOps.size() != 6` check
  // rejects valid IR. Instead, allow ANY *pure* (memory-effect-free AND
  // speculatable) op around the canonical [load, load, dot, addptr,
  // addptr, yield] skeleton. Reject:
  //   - side-effecting ops (`tt.atomic_*`),
  //   - speculation-blocking ops that may trap (`arith.divsi`/`divui`/
  //     `remsi`/`remui` on potential div-by-zero / INT_MIN/-1),
  //   - and any nested control flow (`getNumRegions() > 0`).
  // The `mlir::isPure` check covers both — its `isMemoryEffectFree` arm
  // rejects atomics and its speculatability arm rejects trapping divs.
  // The latter matters because the rewriter erases the for-body entirely
  // when unrolling, so any side-effecting body op would be silently
  // dropped. See `.omc/repros/canonical-3iterarg-dominance-rootcause.md`.
  mlir::triton::LoadOp loadA, loadB;
  mlir::triton::DotOp dotInBody;
  mlir::triton::AddPtrOp addptrA, addptrB;
  mlir::scf::YieldOp yieldOp;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (op.getNumRegions() > 0) return mlir::failure();
    if (auto l = mlir::dyn_cast<mlir::triton::LoadOp>(op)) {
      if (!loadA) loadA = l;
      else if (!loadB) loadB = l;
      else return mlir::failure();
    } else if (auto d = mlir::dyn_cast<mlir::triton::DotOp>(op)) {
      if (dotInBody) return mlir::failure();
      dotInBody = d;
    } else if (auto a = mlir::dyn_cast<mlir::triton::AddPtrOp>(op)) {
      if (!addptrA) addptrA = a;
      else if (!addptrB) addptrB = a;
      else return mlir::failure();
    } else if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
      yieldOp = y;
    } else {
      // `isPure` = `isMemoryEffectFree && isSpeculatable`; correctly
      // rejects `arith.divsi`/`divui`/`remsi`/`remui` even though they
      // are memory-effect-free, because their potential trap on div-by-0
      // (or INT_MIN/-1) makes them non-speculatable.
      if (!mlir::isPure(&op)) return mlir::failure();
    }
  }
  if (!loadA || !loadB || !dotInBody || dotInBody != dot || !addptrA ||
      !addptrB || !yieldOp) return mlir::failure();
  if (loadA.getMask() || loadB.getMask()) return mlir::failure();

  // W1 (transposed operand) is handled only on the runtime-K path for now.
  // Bail if either operand is a `tt.trans` so a static transposed-B matmul
  // errors cleanly (illegal tt.dot) instead of silently emitting a
  // non-transposed load.
  if (dot.getA().getDefiningOp<mlir::triton::TransOp>() ||
      dot.getB().getDefiningOp<mlir::triton::TransOp>())
    return mlir::failure();

  // Identify iter_arg roles by SSA matching.
  auto iterArgs = forOp.getRegionIterArgs();
  int accIdx = -1, aIdx = -1, bIdx = -1;
  for (int i = 0; i < 3; ++i) {
    if (iterArgs[i] == dot.getC()) accIdx = i;
    if (iterArgs[i] == loadA.getPtr()) aIdx = i;
    if (iterArgs[i] == loadB.getPtr()) bIdx = i;
  }
  if (accIdx < 0 || aIdx < 0 || bIdx < 0) return mlir::failure();
  if (accIdx == aIdx || accIdx == bIdx || aIdx == bIdx) return mlir::failure();

  // Validate yield: yield[accIdx] = dot.result, yield[aIdx] = addptrA, yield[bIdx] = addptrB.
  if (yieldOp.getOperand(accIdx) != dot.getResult()) return mlir::failure();
  if (yieldOp.getOperand(aIdx) != addptrA.getResult()) return mlir::failure();
  if (yieldOp.getOperand(bIdx) != addptrB.getResult()) return mlir::failure();
  if (addptrA.getPtr() != iterArgs[aIdx]) return mlir::failure();
  if (addptrB.getPtr() != iterArgs[bIdx]) return mlir::failure();

  // Static trip count.
  auto getConstInt = [](mlir::Value v) -> std::optional<int64_t> {
    auto c = v.getDefiningOp<mlir::arith::ConstantOp>();
    if (!c) return std::nullopt;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return intAttr.getInt();
    return std::nullopt;
  };
  auto lo = getConstInt(forOp.getLowerBound());
  auto hi = getConstInt(forOp.getUpperBound());
  auto st = getConstInt(forOp.getStep());
  if (!lo || !hi || !st || *st == 0) return mlir::failure();
  int64_t N = (*hi - *lo) / *st;
  if (N < 1 || N > 8) return mlir::failure();

  // Extract BK from an addptr's bump tensor. L1d3 iter-5 (Architect Finding
  // #2, PRIMARY rewriter fix): accept BOTH shapes Triton/lit fixtures emit:
  //   1. `arith.constant dense<N> : tensor<...xi32>` (Triton's A-bump emit).
  //   2. `tt.splat(arith.constant N : i32)` (hand-curated lit-fixture shape).
  auto extractBK = [](mlir::triton::AddPtrOp ap) -> std::optional<int64_t> {
    auto offset = ap.getOffset();
    if (auto cst = offset.getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue())) {
        if (dense.isSplat())
          return dense.getSplatValue<mlir::APInt>().getSExtValue();
      }
    }
    if (auto splat = offset.getDefiningOp<mlir::triton::SplatOp>()) {
      if (auto cst = splat.getSrc().getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
          return intAttr.getInt();
      }
    }
    return std::nullopt;
  };
  auto bkOpt = extractBK(addptrA);
  if (!bkOpt) return mlir::failure();
  int64_t BK = *bkOpt;
  // L1d3 iter-5 (Architect Synthesis #3, Critic-softened): advisory B-side
  // BK check. Triton's typical B-bump emits `tt.splat(arith.muli(const,
  // %stride_bk))` which extractBK cannot resolve as a pure constant — so
  // fail ONLY when B IS extractable and mismatches A's BK.
  if (auto bkBOpt = extractBK(addptrB); bkBOpt && *bkBOpt != BK)
    return mlir::failure();

  // Element type + shape gate. AC4 v6: shape gate lifted from `== 8x8` to
  // `% 8 == 0` so multi-tile dots (per-program M/N grids that decompose into
  // 8×8 sub-tiles) admit the new triple-unroll body below. The 8×8 case
  // stays on the legacy single-tile chain (m=n=1 below) for bit-identical
  // pre-AC4 emission. The partial-tile signal still comes from a masked
  // `tt.store` for single-tile (m=n=1) shapes only — masked multi-tile bails.
  auto resTy = mlir::dyn_cast<mlir::RankedTensorType>(dot.getType());
  if (!resTy) return mlir::failure();
  auto elemTy = resTy.getElementType();
  if (!elemTy.isF32()) return mlir::failure();
  auto shape = resTy.getShape();
  if (shape.size() != 2 || shape[0] % 8 != 0 || shape[1] % 8 != 0)
    return mlir::failure();
  const int64_t mTiles = shape[0] / 8;
  const int64_t nTiles = shape[1] / 8;

  // Locate the store consuming the for's accumulator result.
  if (!forOp.getResult(accIdx).hasOneUse()) return mlir::failure();
  auto store = mlir::dyn_cast<mlir::triton::StoreOp>(
      *forOp.getResult(accIdx).getUsers().begin());
  if (!store) return mlir::failure();
  // AC2: allow a masked store of the canonical 2D form. If the mask doesn't
  // decompose into the (offs_m < M) & (offs_n < N) pattern, bail to preserve
  // correctness — other mask shapes are out of scope for this AC.
  MaskExtents maskExtents;
  if (store.getMask()) {
    maskExtents = extractMaskExtents(store.getMask());
    if (!maskExtents.mExtent || !maskExtents.nExtent)
      return mlir::failure();
  }

  // Pull base origins/strides from the iter_arg INITS (which carry the
  // pid-driven address shapes built outside the loop).
  mlir::Value aPtrsInit = forOp.getInits()[aIdx];
  mlir::Value bPtrsInit = forOp.getInits()[bIdx];
  auto aInitAddptr = aPtrsInit.getDefiningOp<mlir::triton::AddPtrOp>();
  auto bInitAddptr = bPtrsInit.getDefiningOp<mlir::triton::AddPtrOp>();
  if (!aInitAddptr || !bInitAddptr) return mlir::failure();

  mlir::Value aPtr = unwrapPtrToKernelArg(aPtrsInit);
  mlir::Value bPtr = unwrapPtrToKernelArg(bPtrsInit);
  mlir::Value cPtr = unwrapPtrToKernelArg(store.getPtr());

  // iter-8 (chain-aware): the canonical Triton 2D-matmul pointer is a
  // 2-level chain whose row contribution + row stride live in the INNER
  // addptr (not the outer one consumed by `aInitAddptr.getOffset()`). Walk
  // the entire chain via the `*InPtrChain` helpers — the flat search would
  // miss both and silently fall back to origin=0 / stride=8, which works
  // by accident for the [8,8,8] single-tile case (pid=0, BLOCK_N=8) but
  // catastrophically miscompiles [16,16,16] grid=(2,2).
  mlir::Value strideA = findStrideSplatSourceInPtrChain(aPtrsInit);
  mlir::Value strideB = findStrideSplatSourceInPtrChain(bPtrsInit);
  mlir::Value aBaseRow = findOriginScalarInPtrChain(aPtrsInit, /*targetAxis=*/0);
  mlir::Value bBaseCol = findOriginScalarInPtrChain(bPtrsInit, /*targetAxis=*/1);
  OriginPair cOrig = extractOriginPair<mlir::triton::StoreOp>(store);

  // Emit IR before the scf.for.
  mlir::OpBuilder builder(forOp);
  auto loc = forOp.getLoc();
  auto ctx = builder.getContext();
  auto ui32 = builder.getIntegerType(32, /*isSigned=*/false);
  auto matTy = MetalSimdgroupMatrixType::get(ctx, 8, 8, elemTy);

  auto zeroU32 =
      ConstantOp::create(builder, loc, builder.getIntegerAttr(ui32, 0));
  mlir::Value strideAVal = emitStrideOperand(builder, loc, ui32, strideA);
  mlir::Value strideBVal = emitStrideOperand(builder, loc, ui32, strideB);
  // iter-8 (chain-aware): the C-store pointer is the same 2-level
  // `addptr(broadcast(addptr(splat(c_ptr), row_off)), col_off)` chain as A/B;
  // the row-stride splat lives in the INNER addptr's offset, so the flat
  // outer-only search used in iter-7 returned null and emitStrideOperand
  // fell back to `metal.constant 8 : ui32`. Walk the chain instead.
  mlir::Value strideC = findStrideSplatSourceInPtrChain(store.getPtr());
  mlir::Value strideCVal = emitStrideOperand(builder, loc, ui32, strideC);
  mlir::Value aBaseRowVal = emitOriginOperand(builder, loc, ui32, aBaseRow);
  mlir::Value bBaseColVal = emitOriginOperand(builder, loc, ui32, bBaseCol);
  mlir::Value cRowOrigin = emitOriginOperand(builder, loc, ui32, cOrig.row);
  mlir::Value cColOrigin = emitOriginOperand(builder, loc, ui32, cOrig.col);

  mlir::Value aBuf = bridgePtrToMemref(builder, loc, aPtr, elemTy);
  mlir::Value bBuf = bridgePtrToMemref(builder, loc, bPtr, elemTy);
  mlir::Value cBuf = bridgePtrToMemref(builder, loc, cPtr, elemTy);

  // C-init helper: use `metal.simdgroup_matrix_zero` when the for-loop's
  // accumulator iter_arg is initialized to `dense<0.0>` (the canonical
  // `acc = tl.zeros(...)` Triton pattern). Apple's matmul / FA samples
  // initialize the accumulator via the constructor `simdgroup_matrix<T,
  // M, N>(0.0f)`; chained `simdgroup_multiply_accumulate` calls only
  // accumulate correctly across all output columns when the accumulator
  // is initialized this way (iter-8 root cause for K_TILES >= 2 cases).
  // Fall back to `simdgroup_load` from the C buffer at the tile's (row,
  // col) origin if the iter_arg init is anything else.
  mlir::Value accInitOp = forOp.getInits()[accIdx];
  bool accInitIsZero = false;
  if (auto cst = accInitOp.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue())) {
      if (dense.isSplat() && dense.getSplatValue<mlir::APFloat>().isZero())
        accInitIsZero = true;
    }
  }
  auto emitAccInit = [&](mlir::Value cTileRow, mlir::Value cTileCol)
      -> mlir::Value {
    if (accInitIsZero)
      return SimdgroupMatrixZeroOp::create(builder, loc, matTy).getResult();
    return SimdgroupLoadOp::create(builder, loc, matTy, cBuf, cTileRow,
                                    cTileCol, strideCVal)
        .getResult();
  };

  // AC4 v6 branch: single-tile (m_tiles=n_tiles=1) emits the bit-identical
  // pre-AC4 chain; multi-tile emits a per-warp triple inline-unroll with
  // simdgroup_index_in_threadgroup-driven (warpM, warpN) partition.
  const bool multiTile = (mTiles > 1) || (nTiles > 1);
  const int64_t numWarps = mlir::triton::gpu::lookupNumWarps(dot);
  const bool multiWarp = numWarps > 1;

  if (!multiTile) {
    // Legacy single-tile chain (Bucket A) — bit-identical to pre-AC4.
    mlir::Value acc = emitAccInit(cRowOrigin, cColOrigin);
    for (int64_t i = 0; i < N; ++i) {
      auto kOffset = ConstantOp::create(
          builder, loc, builder.getIntegerAttr(ui32, i * BK));
      mlir::Value aColIter = kOffset.getResult();
      mlir::Value bRowIter = kOffset.getResult();
      auto aTile = SimdgroupLoadDeviceStagedOp::create(
          builder, loc, matTy, aBuf, aBaseRowVal, aColIter, strideAVal,
          mlir::ValueRange{});
      auto bTile = SimdgroupLoadDeviceStagedOp::create(
          builder, loc, matTy, bBuf, bRowIter, bBaseColVal, strideBVal,
          mlir::ValueRange{});
      auto ma = SimdgroupMultiplyAccumulateOp::create(
          builder, loc, matTy, acc, aTile.getResult(), bTile.getResult());
      acc = ma.getResult();
    }
    // Final store. AC2: attach `partial_extents = [m_extent, n_extent]`
    // operands when the tt.store had a (offs_m < M) & (offs_n < N) mask so
    // the emitter routes through the threadgroup-scratch masked epilogue.
    llvm::SmallVector<mlir::Value, 2> partialExtents;
    if (maskExtents.mExtent && maskExtents.nExtent) {
      auto toUi32 = [&](mlir::Value v) -> mlir::Value {
        if (v.getType() == ui32) return v;
        return mlir::UnrealizedConversionCastOp::create(builder, loc, ui32, v)
            .getResult(0);
      };
      partialExtents.push_back(toUi32(maskExtents.mExtent));
      partialExtents.push_back(toUi32(maskExtents.nExtent));
    }
    SimdgroupStoreOp::create(builder, loc, acc, cBuf, cRowOrigin, cColOrigin,
                              strideCVal, partialExtents);
  } else {
    // AC4 v6 multi-tile path.
    // AC2/AC4 cross-coupling guard: masked + multi-warp is out of scope —
    // the masked epilogue's threadgroup-scratch staging assumes single-warp
    // ownership of the C tile, which conflicts with per-warp tile ownership.
    if (store.getMask() && multiWarp) {
      llvm::report_fatal_error(
          "AC2 masked-tail not supported under multi-warp; see "
          ".omc/plans/ac4-multiwarp.md");
    }
    // Multi-tile + masked (single-warp): out of scope for this AC; bail so
    // the fallback `rewriteSingleDot` path is attempted.
    if (store.getMask()) return mlir::failure();

    int warpsM = 1, warpsN = 1;
    if (multiWarp) {
      auto wf = factorWarps(static_cast<int>(numWarps),
                            static_cast<int>(mTiles),
                            static_cast<int>(nTiles));
      if (!wf) {
        llvm::report_fatal_error(
            "AC4: factorWarps failed — no valid (warpsM, warpsN) partition "
            "for (numWarps, mTiles, nTiles); see .omc/plans/ac4-multiwarp.md");
      }
      warpsM = wf->first;
      warpsN = wf->second;
    }
    const int mPerWarp = static_cast<int>(mTiles) / warpsM;
    const int nPerWarp = static_cast<int>(nTiles) / warpsN;

    // Emit widx and warpM/warpN via arith.divui/remui (signless i32 land).
    auto i32 = builder.getIntegerType(32);
    auto toI32 = [&](mlir::Value u) -> mlir::Value {
      return mlir::UnrealizedConversionCastOp::create(
                 builder, loc, mlir::TypeRange{i32}, mlir::ValueRange{u})
          .getResult(0);
    };
    auto toUi32 = [&](mlir::Value v) -> mlir::Value {
      return mlir::UnrealizedConversionCastOp::create(
                 builder, loc, mlir::TypeRange{ui32}, mlir::ValueRange{v})
          .getResult(0);
    };
    mlir::Value widxI32, warpMI32, warpNI32, widxUi32Hoisted;
    if (multiWarp) {
      widxUi32Hoisted =
          SimdgroupIndexOp::create(builder, loc, ui32).getResult();
      widxI32 = toI32(widxUi32Hoisted);
      auto warpsNCst = mlir::arith::ConstantOp::create(
          builder, loc, builder.getI32IntegerAttr(warpsN));
      warpMI32 = mlir::arith::DivUIOp::create(builder, loc, widxI32,
                                                warpsNCst.getResult())
                     .getResult();
      warpNI32 = mlir::arith::RemUIOp::create(builder, loc, widxI32,
                                                warpsNCst.getResult())
                     .getResult();
    }

    // Cast base ui32 values to i32 once for arith composition.
    mlir::Value aBaseRowI32 = toI32(aBaseRowVal);
    mlir::Value bBaseColI32 = toI32(bBaseColVal);
    mlir::Value cRowOrigI32 = toI32(cRowOrigin);
    mlir::Value cColOrigI32 = toI32(cColOrigin);

    auto eight = mlir::arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(8));

    // Per-warp owned tile grid: tile_m = warpM + mIter * warpsM,
    // tile_n = warpN + nIter * warpsN. This interleaves warps across the
    // M/N grid (matching the plan v5 stride math).
    for (int mIter = 0; mIter < mPerWarp; ++mIter) {
      for (int nIter = 0; nIter < nPerWarp; ++nIter) {
        // tile_m * 8 (i32)
        mlir::Value mTileIdx;
        if (multiWarp) {
          auto mIterCst = mlir::arith::ConstantOp::create(
              builder, loc, builder.getI32IntegerAttr(mIter * warpsM));
          mTileIdx = mlir::arith::AddIOp::create(builder, loc, warpMI32,
                                                  mIterCst.getResult())
                         .getResult();
        } else {
          mTileIdx = mlir::arith::ConstantOp::create(
                         builder, loc,
                         builder.getI32IntegerAttr(mIter))
                         .getResult();
        }
        mlir::Value rowOff = mlir::arith::MulIOp::create(builder, loc, mTileIdx,
                                                          eight.getResult())
                                 .getResult();

        mlir::Value nTileIdx;
        if (multiWarp) {
          auto nIterCst = mlir::arith::ConstantOp::create(
              builder, loc, builder.getI32IntegerAttr(nIter * warpsN));
          nTileIdx = mlir::arith::AddIOp::create(builder, loc, warpNI32,
                                                  nIterCst.getResult())
                         .getResult();
        } else {
          nTileIdx = mlir::arith::ConstantOp::create(
                         builder, loc,
                         builder.getI32IntegerAttr(nIter))
                         .getResult();
        }
        mlir::Value colOff = mlir::arith::MulIOp::create(builder, loc, nTileIdx,
                                                          eight.getResult())
                                 .getResult();

        mlir::Value aTileRowI32 = mlir::arith::AddIOp::create(
                                       builder, loc, aBaseRowI32, rowOff)
                                       .getResult();
        mlir::Value bTileColI32 = mlir::arith::AddIOp::create(
                                       builder, loc, bBaseColI32, colOff)
                                       .getResult();
        mlir::Value cTileRowI32 = mlir::arith::AddIOp::create(
                                       builder, loc, cRowOrigI32, rowOff)
                                       .getResult();
        mlir::Value cTileColI32 = mlir::arith::AddIOp::create(
                                       builder, loc, cColOrigI32, colOff)
                                       .getResult();

        mlir::Value aTileRow = toUi32(aTileRowI32);
        mlir::Value bTileCol = toUi32(bTileColI32);
        mlir::Value cTileRow = toUi32(cTileRowI32);
        mlir::Value cTileCol = toUi32(cTileColI32);

        // Per-(mIter, nIter) accumulator re-init (each output tile owns
        // its own accumulator).
        mlir::Value acc = emitAccInit(cTileRow, cTileCol);

        // K-loop unrolled. For multi-warp, attach the hoisted ui32 widx
        // SSA value as the variadic `warp_index` operand on every staged
        // load so the per-warp Branch B in the emitter sees the same widx.
        // Use SmallVector storage to give the ValueRange a stable address —
        // `ValueRange(const Value &)` is LIFETIME_BOUND and would dangle
        // if we stored `mlir::ValueRange{widxUi32Hoisted}` in a variable.
        llvm::SmallVector<mlir::Value, 1> warpIdxStorage;
        if (multiWarp) warpIdxStorage.push_back(widxUi32Hoisted);
        for (int64_t i = 0; i < N; ++i) {
          auto kOffset = ConstantOp::create(
              builder, loc, builder.getIntegerAttr(ui32, i * BK));
          mlir::Value kVal = kOffset.getResult();
          auto aTile = SimdgroupLoadDeviceStagedOp::create(
              builder, loc, matTy, aBuf, aTileRow, kVal, strideAVal,
              mlir::ValueRange(warpIdxStorage));
          auto bTile = SimdgroupLoadDeviceStagedOp::create(
              builder, loc, matTy, bBuf, kVal, bTileCol, strideBVal,
              mlir::ValueRange(warpIdxStorage));
          auto ma = SimdgroupMultiplyAccumulateOp::create(
              builder, loc, matTy, acc, aTile.getResult(), bTile.getResult());
          acc = ma.getResult();
        }

        // Per-tile store. No partial extents in the multi-tile path
        // (masked + multi-tile bails above).
        SimdgroupStoreOp::create(builder, loc, acc, cBuf, cTileRow, cTileCol,
                                  strideCVal,
                                  llvm::SmallVector<mlir::Value, 2>{});
      }
    }
  }

  // Erase originals.
  store.erase();
  forOp.erase();
  return mlir::success();
}

// W2a: runtime-K single-tile matmul. Same canonical 3-iter_arg shape as
// `tryUnrollCanonical3IterArgDot` (iter_args = [a_ptrs, b_ptrs, acc], body
// `[load A, load B, dot, addptr A, addptr B, yield]`), but the K-loop trip
// count is a RUNTIME value (e.g. `for k in range(0, K, BLOCK_K)` with K a
// kernel arg). Instead of statically unrolling, emit a fresh `scf.for` that
// steps the K axis by 8 (one simdgroup 8x8 subtile per iteration) over the
// runtime bound, carrying the `simdgroup_matrix` accumulator as its single
// iter_arg. First cut: single 8x8 output tile, single-warp, unmasked, f32
// (larger shapes are reached via an 8x8 program grid). See
// `metal-lora-linear-fix-plan.md` (W2a).
static mlir::LogicalResult tryRuntimeKLoopCanonicalDot(mlir::triton::DotOp dot) {
  auto forOp = dot->getParentOfType<mlir::scf::ForOp>();
  if (!forOp) return mlir::failure();
  if (forOp.getNumRegionIterArgs() != 3) return mlir::failure();

  // Body shape: [load A, load B, dot, addptr A, addptr B, yield] + pure noise
  // (mirrors the static canonical matcher's permissive walk).
  mlir::triton::LoadOp loadA, loadB;
  mlir::triton::DotOp dotInBody;
  mlir::triton::AddPtrOp addptrA, addptrB;
  mlir::scf::YieldOp yieldOp;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (op.getNumRegions() > 0) return mlir::failure();
    if (auto l = mlir::dyn_cast<mlir::triton::LoadOp>(op)) {
      if (!loadA) loadA = l;
      else if (!loadB) loadB = l;
      else return mlir::failure();
    } else if (auto d = mlir::dyn_cast<mlir::triton::DotOp>(op)) {
      if (dotInBody) return mlir::failure();
      dotInBody = d;
    } else if (auto a = mlir::dyn_cast<mlir::triton::AddPtrOp>(op)) {
      if (!addptrA) addptrA = a;
      else if (!addptrB) addptrB = a;
      else return mlir::failure();
    } else if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
      yieldOp = y;
    } else {
      if (!mlir::isPure(&op)) return mlir::failure();
    }
  }
  if (!loadA || !loadB || !dotInBody || dotInBody != dot || !addptrA ||
      !addptrB || !yieldOp) return mlir::failure();
  if (loadA.getMask() || loadB.getMask()) return mlir::failure();

  // W1: the B operand may be `tt.trans(loadB)` — the transposed-B matmul
  // `tl.dot(a, tl.trans(b))` (all LoRA dots have this shape). A must be the
  // plain load (trans-on-A is not yet supported).
  bool bTransposed = false;
  if (dot.getA() != loadA.getResult()) return mlir::failure();
  if (auto tr = dot.getB().getDefiningOp<mlir::triton::TransOp>()) {
    if (tr.getSrc() != loadB.getResult()) return mlir::failure();
    bTransposed = true;
  } else if (dot.getB() != loadB.getResult()) {
    return mlir::failure();
  }

  auto iterArgs = forOp.getRegionIterArgs();
  int accIdx = -1, aIdx = -1, bIdx = -1;
  for (int i = 0; i < 3; ++i) {
    if (iterArgs[i] == dot.getC()) accIdx = i;
    if (iterArgs[i] == loadA.getPtr()) aIdx = i;
    if (iterArgs[i] == loadB.getPtr()) bIdx = i;
  }
  if (accIdx < 0 || aIdx < 0 || bIdx < 0) return mlir::failure();
  if (accIdx == aIdx || accIdx == bIdx || aIdx == bIdx) return mlir::failure();
  if (yieldOp.getOperand(accIdx) != dot.getResult()) return mlir::failure();
  if (yieldOp.getOperand(aIdx) != addptrA.getResult()) return mlir::failure();
  if (yieldOp.getOperand(bIdx) != addptrB.getResult()) return mlir::failure();
  if (addptrA.getPtr() != iterArgs[aIdx]) return mlir::failure();
  if (addptrB.getPtr() != iterArgs[bIdx]) return mlir::failure();

  auto getConstInt = [](mlir::Value v) -> std::optional<int64_t> {
    auto c = v.getDefiningOp<mlir::arith::ConstantOp>();
    if (!c) return std::nullopt;
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return ia.getInt();
    return std::nullopt;
  };
  // Lower bound and step must be static; UPPER bound must be RUNTIME (this is
  // exactly the case the static matcher rejects).
  auto lo = getConstInt(forOp.getLowerBound());
  auto st = getConstInt(forOp.getStep());
  if (!lo || *lo != 0 || !st || *st <= 0) return mlir::failure();
  if (getConstInt(forOp.getUpperBound())) return mlir::failure();  // static → other path

  // Require the element-offset loop form (step == BK), so the runtime upper
  // bound is the total K in elements and the fresh step-8 loop iterates [0,K).
  auto extractBK = [](mlir::triton::AddPtrOp ap) -> std::optional<int64_t> {
    auto offset = ap.getOffset();
    if (auto cst = offset.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue()))
        if (dense.isSplat())
          return dense.getSplatValue<mlir::APInt>().getSExtValue();
    if (auto splat = offset.getDefiningOp<mlir::triton::SplatOp>())
      if (auto cst = splat.getSrc().getDefiningOp<mlir::arith::ConstantOp>())
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
          return ia.getInt();
    return std::nullopt;
  };
  auto bkOpt = extractBK(addptrA);
  if (!bkOpt || *bkOpt != *st || *bkOpt % 8 != 0) return mlir::failure();

  // First cut: single-warp only.
  if (mlir::triton::gpu::lookupNumWarps(dot) != 1) return mlir::failure();

  auto resTy = mlir::dyn_cast<mlir::RankedTensorType>(dot.getType());
  if (!resTy) return mlir::failure();
  auto elemTy = resTy.getElementType();
  if (!elemTy.isF32()) return mlir::failure();
  auto shape = resTy.getShape();
  if (shape.size() != 2 || shape[0] != 8 || shape[1] != 8)
    return mlir::failure();

  if (!forOp.getResult(accIdx).hasOneUse()) return mlir::failure();
  auto store = mlir::dyn_cast<mlir::triton::StoreOp>(
      *forOp.getResult(accIdx).getUsers().begin());
  if (!store || store.getMask()) return mlir::failure();  // masked tail deferred

  mlir::Value aPtrsInit = forOp.getInits()[aIdx];
  mlir::Value bPtrsInit = forOp.getInits()[bIdx];
  if (!aPtrsInit.getDefiningOp<mlir::triton::AddPtrOp>() ||
      !bPtrsInit.getDefiningOp<mlir::triton::AddPtrOp>())
    return mlir::failure();

  // Accumulator init must be dense<0.0> (the canonical `acc = tl.zeros(...)`).
  {
    auto cst = forOp.getInits()[accIdx].getDefiningOp<mlir::arith::ConstantOp>();
    if (!cst) return mlir::failure();
    auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat() ||
        !dense.getSplatValue<mlir::APFloat>().isZero())
      return mlir::failure();
  }

  mlir::Value aPtr = unwrapPtrToKernelArg(aPtrsInit);
  mlir::Value bPtr = unwrapPtrToKernelArg(bPtrsInit);
  mlir::Value cPtr = unwrapPtrToKernelArg(store.getPtr());
  mlir::Value strideA = findStrideSplatSourceInPtrChain(aPtrsInit);
  mlir::Value strideB = findStrideSplatSourceInPtrChain(bPtrsInit);
  mlir::Value aBaseRow = findOriginScalarInPtrChain(aPtrsInit, /*targetAxis=*/0);
  // Normal B is [K,N] (N on axis 1); transposed B loads an [N,K] tile (N on
  // axis 0) staged transposed. In both cases `strideB` (axis-0 row stride) is
  // the correct simdgroup-load stride.
  mlir::Value bBaseCol =
      findOriginScalarInPtrChain(bPtrsInit, /*targetAxis=*/bTransposed ? 0 : 1);
  OriginPair cOrig = extractOriginPair<mlir::triton::StoreOp>(store);

  mlir::OpBuilder builder(forOp);
  auto loc = forOp.getLoc();
  auto ctx = builder.getContext();
  auto ui32 = builder.getIntegerType(32, /*isSigned=*/false);
  auto matTy = MetalSimdgroupMatrixType::get(ctx, 8, 8, elemTy);

  mlir::Value strideAVal = emitStrideOperand(builder, loc, ui32, strideA);
  mlir::Value strideBVal = emitStrideOperand(builder, loc, ui32, strideB);
  mlir::Value strideCVal = emitStrideOperand(
      builder, loc, ui32, findStrideSplatSourceInPtrChain(store.getPtr()));
  mlir::Value aBaseRowVal = emitOriginOperand(builder, loc, ui32, aBaseRow);
  mlir::Value bBaseColVal = emitOriginOperand(builder, loc, ui32, bBaseCol);
  mlir::Value cRowOrigin = emitOriginOperand(builder, loc, ui32, cOrig.row);
  mlir::Value cColOrigin = emitOriginOperand(builder, loc, ui32, cOrig.col);
  mlir::Value aBuf = bridgePtrToMemref(builder, loc, aPtr, elemTy);
  mlir::Value bBuf = bridgePtrToMemref(builder, loc, bPtr, elemTy);
  mlir::Value cBuf = bridgePtrToMemref(builder, loc, cPtr, elemTy);

  auto toUi32 = [&](mlir::OpBuilder &b, mlir::Value v) -> mlir::Value {
    if (v.getType() == ui32) return v;
    return mlir::UnrealizedConversionCastOp::create(
               b, loc, mlir::TypeRange{ui32}, mlir::ValueRange{v})
        .getResult(0);
  };

  // acc = simdgroup_matrix<f32,8,8>(0.0f), carried as the loop iter_arg.
  auto zeroInit = SimdgroupMatrixZeroOp::create(builder, loc, matTy);
  auto c0 = mlir::arith::ConstantOp::create(builder, loc,
                                            builder.getI32IntegerAttr(0));
  auto c8 = mlir::arith::ConstantOp::create(builder, loc,
                                            builder.getI32IntegerAttr(8));
  mlir::Value ub = forOp.getUpperBound();  // runtime total K (i32)

  auto loop = mlir::scf::ForOp::create(
      builder, loc, c0.getResult(), ub, c8.getResult(),
      mlir::ValueRange{zeroInit.getResult()},
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value iv,
          mlir::ValueRange args) {
        mlir::Value acc = args[0];
        mlir::Value kUi32 = toUi32(b, iv);  // K-axis offset = induction var
        auto aTile = SimdgroupLoadDeviceStagedOp::create(
            b, l, matTy, aBuf, aBaseRowVal, kUi32, strideAVal,
            mlir::ValueRange{});
        // Transposed B: load the [N,K] tile at (N-origin, K-offset) and stage
        // it transposed (the `transposed` attr swaps the staging index) so the
        // simdgroup sees the [K,N] = trans(b) subtile. Normal B: [K,N] tile at
        // (K-offset, N-origin).
        mlir::triton::metal::SimdgroupLoadDeviceStagedOp bTile;
        if (bTransposed) {
          bTile = SimdgroupLoadDeviceStagedOp::create(
              b, l, matTy, bBuf, bBaseColVal, kUi32, strideBVal,
              mlir::ValueRange{});
          bTile->setAttr("transposed", b.getUnitAttr());
        } else {
          bTile = SimdgroupLoadDeviceStagedOp::create(
              b, l, matTy, bBuf, kUi32, bBaseColVal, strideBVal,
              mlir::ValueRange{});
        }
        auto ma = SimdgroupMultiplyAccumulateOp::create(
            b, l, matTy, acc, aTile.getResult(), bTile.getResult());
        mlir::scf::YieldOp::create(b, l, mlir::ValueRange{ma.getResult()});
      });

  SimdgroupStoreOp::create(builder, loc, loop.getResult(0), cBuf, cRowOrigin,
                            cColOrigin, strideCVal, mlir::ValueRange{});

  store.erase();
  forOp.erase();
  return mlir::success();
}

// Does the tensor `v` (an addptr offset contribution) contain `splat(scalar)`
// anywhere in its broadcast/expand_dims/addi/muli tree? Used to confirm the
// K-axis offset of a recompute-from-IV load is `offs_k = <iv> + arange`.
static bool contributionContainsSplatOf(mlir::Value v, mlir::Value scalar,
                                         int depth = 0) {
  if (depth > 8 || !v) return false;
  if (auto splat = v.getDefiningOp<mlir::triton::SplatOp>())
    return splat.getSrc() == scalar;
  if (auto bc = v.getDefiningOp<mlir::triton::BroadcastOp>())
    return contributionContainsSplatOf(bc.getSrc(), scalar, depth + 1);
  if (auto ed = v.getDefiningOp<mlir::triton::ExpandDimsOp>())
    return contributionContainsSplatOf(ed.getSrc(), scalar, depth + 1);
  if (auto addi = v.getDefiningOp<mlir::arith::AddIOp>())
    return contributionContainsSplatOf(addi.getLhs(), scalar, depth + 1) ||
           contributionContainsSplatOf(addi.getRhs(), scalar, depth + 1);
  if (auto muli = v.getDefiningOp<mlir::arith::MulIOp>())
    return contributionContainsSplatOf(muli.getLhs(), scalar, depth + 1) ||
           contributionContainsSplatOf(muli.getRhs(), scalar, depth + 1);
  return false;
}

// Axis-aware row-stride extractor for the recompute-from-IV shape. Unlike
// `findStrideSplatSourceInPtrChain`, which returns the first `splat(i32
// block-arg)` it finds — and is fooled by the `splat(iv)` inside
// `offs_k = k + arange` — this returns the stride multiplied by the offset
// contribution that varies along `axis` (i.e. the `muli(expand_dims(..),
// splat(stride))` whose expand_dims axis is `1 - axis`). Returns null if not
// found (caller emits a `metal.constant` fallback).
static mlir::Value findAxisStride(mlir::Value ptr, int axis) {
  std::function<mlir::Value(mlir::Value)> fromOffset =
      [&](mlir::Value off) -> mlir::Value {
    if (!off) return mlir::Value();
    if (auto addi = off.getDefiningOp<mlir::arith::AddIOp>()) {
      if (auto r = fromOffset(addi.getLhs())) return r;
      return fromOffset(addi.getRhs());
    }
    if (auto muli = off.getDefiningOp<mlir::arith::MulIOp>()) {
      auto ax = findExpandDimsAxis(off);
      if (ax.has_value() && (1 - *ax) == axis) {
        auto splatSrc = [](mlir::Value v) -> mlir::Value {
          while (v) {
            if (auto s = v.getDefiningOp<mlir::triton::SplatOp>())
              return s.getSrc();
            if (auto bc = v.getDefiningOp<mlir::triton::BroadcastOp>()) {
              v = bc.getSrc();
              continue;
            }
            break;
          }
          return mlir::Value();
        };
        if (auto s = splatSrc(muli.getLhs())) return s;
        if (auto s = splatSrc(muli.getRhs())) return s;
      }
    }
    return mlir::Value();
  };
  for (int depth = 0; depth < 8 && ptr; ++depth) {
    if (auto addptr = ptr.getDefiningOp<mlir::triton::AddPtrOp>()) {
      if (auto r = fromOffset(addptr.getOffset())) return r;
      ptr = addptr.getPtr();
      continue;
    }
    if (auto bc = ptr.getDefiningOp<mlir::triton::BroadcastOp>()) {
      ptr = bc.getSrc();
      continue;
    }
    if (auto ed = ptr.getDefiningOp<mlir::triton::ExpandDimsOp>()) {
      ptr = ed.getSrc();
      continue;
    }
    break;
  }
  return mlir::Value();
}

// True if the `kAxis` contribution of a load pointer chain is driven by the
// induction variable `iv` (the recompute-from-IV element-offset form
// `offs_k = k + tl.arange(...)`). Mirrors `findOriginScalarInPtrChain`'s walk
// but matches `splat(iv)` on the K axis instead of a pid origin.
static bool kAxisUsesInductionVar(mlir::Value ptr, int kAxis, mlir::Value iv) {
  for (int depth = 0; depth < 8 && ptr; ++depth) {
    if (auto addptr = ptr.getDefiningOp<mlir::triton::AddPtrOp>()) {
      mlir::Value off = addptr.getOffset();
      auto ax = findExpandDimsAxis(off);
      if (ax.has_value() && (1 - *ax) == kAxis &&
          contributionContainsSplatOf(off, iv))
        return true;
      ptr = addptr.getPtr();
      continue;
    }
    if (auto bc = ptr.getDefiningOp<mlir::triton::BroadcastOp>()) {
      ptr = bc.getSrc();
      continue;
    }
    if (auto ed = ptr.getDefiningOp<mlir::triton::ExpandDimsOp>()) {
      ptr = ed.getSrc();
      continue;
    }
    break;
  }
  return false;
}

// Convert an i32 mask-extent scalar to ui32 (null passes through).
static mlir::Value emitExtentUi32(mlir::OpBuilder &b, mlir::Location loc,
                                  mlir::Type ui32, mlir::Value extent) {
  if (!extent) return mlir::Value();
  if (extent.getType() == ui32) return extent;
  return mlir::UnrealizedConversionCastOp::create(
             b, loc, mlir::TypeRange{ui32}, mlir::ValueRange{extent})
      .getResult(0);
}

// Emit a staged simdgroup load — masked (out-of-bounds → 0) when both extents
// are non-null, otherwise the plain staged load. `transposed` swaps the staging
// destination (W1). `widx` (optional) selects a per-warp staging buffer for the
// multi-warp path; masked multi-warp is not yet supported.
static mlir::Value emitStagedLoad(mlir::OpBuilder &b, mlir::Location loc,
                                  MetalSimdgroupMatrixType matTy,
                                  mlir::Value buf, mlir::Value originRow,
                                  mlir::Value originCol, mlir::Value stride,
                                  bool transposed, mlir::Value rowExtent,
                                  mlir::Value colExtent,
                                  mlir::Value widx = mlir::Value()) {
  llvm::SmallVector<mlir::Value, 1> warpIdx;
  if (widx) warpIdx.push_back(widx);
  if (rowExtent && colExtent) {
    auto op = SimdgroupLoadDeviceStagedMaskedOp::create(
        b, loc, matTy, buf, originRow, originCol, stride, rowExtent, colExtent,
        warpIdx);
    if (transposed) op->setAttr("transposed", b.getUnitAttr());
    return op.getResult();
  }
  auto op = SimdgroupLoadDeviceStagedOp::create(
      b, loc, matTy, buf, originRow, originCol, stride, warpIdx);
  if (transposed) op->setAttr("transposed", b.getUnitAttr());
  return op.getResult();
}

// W2b (foundation): runtime-K matmul in the "recompute-from-IV" loop shape that
// `medium-lora_linear.py` and most hand-written matmul kernels use:
//
//   acc = tl.zeros(...)
//   for k in range(0, K, BLOCK_K):
//     offs_k = k + tl.arange(0, BLOCK_K)
//     x = tl.load(X + offs_m[:,None]*sxm + offs_k[None,:]*sxk)   # [M,K]
//     w = tl.load(W + offs_n[:,None]*swn + offs_k[None,:]*swk)   # [N,K]
//     acc = tl.dot(x, tl.trans(w), acc)
//   tl.store(...)
//
// Unlike `tryRuntimeKLoopCanonicalDot` the loop carries ONLY the accumulator
// (no a_ptrs/b_ptrs iter_args, no addptr in the body) — addresses are recomputed
// each iteration from the induction variable. Origins/strides are therefore
// pulled from the loads themselves, and the K axis of each load must be the
// induction variable (the element-offset form). Emission is identical to the
// canonical runtime path: a fresh `scf.for` stepping K by 8 with a
// simdgroup_matrix accumulator. First cut: single dot / single 8x8 tile /
// single-warp / unmasked. See `metal-lora-linear-fix-plan.md` (W2b).
static mlir::LogicalResult tryRuntimeKLoopRecomputeDot(mlir::triton::DotOp dot) {
  auto forOp = dot->getParentOfType<mlir::scf::ForOp>();
  if (!forOp) return mlir::failure();
  if (forOp.getNumRegionIterArgs() != 1) return mlir::failure();
  mlir::Value accArg = forOp.getRegionIterArg(0);
  if (dot.getC() != accArg) return mlir::failure();

  // Body: [load A, load B, (trans), dot, yield] + pure noise; NO addptr.
  mlir::triton::LoadOp loadA, loadB;
  mlir::triton::DotOp dotInBody;
  mlir::scf::YieldOp yieldOp;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (op.getNumRegions() > 0) return mlir::failure();
    if (auto l = mlir::dyn_cast<mlir::triton::LoadOp>(op)) {
      if (!loadA) loadA = l;
      else if (!loadB) loadB = l;
      else return mlir::failure();
    } else if (auto d = mlir::dyn_cast<mlir::triton::DotOp>(op)) {
      if (dotInBody) return mlir::failure();
      dotInBody = d;
    } else if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
      yieldOp = y;
    } else {
      // Address-computation ops (tt.addptr, tt.splat, arith, tt.broadcast,
      // tt.expand_dims, tt.make_range, tt.trans) are all pure and tolerated.
      if (!mlir::isPure(&op)) return mlir::failure();
    }
  }
  if (!loadA || !loadB || dotInBody != dot || !yieldOp) return mlir::failure();
  auto maskShapeOk = [](mlir::triton::LoadOp l) {
    if (!l.getMask()) return true;
    auto e = extractMaskExtents(l.getMask());
    return (bool)(e.mExtent && e.nExtent);
  };
  if (!maskShapeOk(loadA) || !maskShapeOk(loadB)) return mlir::failure();
  if (yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0) != dot.getResult()) return mlir::failure();

  // trans-B detection (A must be the plain load; trans-on-A unsupported).
  bool bTransposed = false;
  if (dot.getA() != loadA.getResult()) return mlir::failure();
  if (auto tr = dot.getB().getDefiningOp<mlir::triton::TransOp>()) {
    if (tr.getSrc() != loadB.getResult()) return mlir::failure();
    bTransposed = true;
  } else if (dot.getB() != loadB.getResult()) {
    return mlir::failure();
  }
  // B tile axes: transposed w is [N,K] (K on axis 1, N on axis 0); a normal B
  // is [K,N] (K on axis 0, N on axis 1).
  const int bKAxis = bTransposed ? 1 : 0;
  const int bNAxis = bTransposed ? 0 : 1;

  // Bounds: lower + step static, upper RUNTIME.
  auto getConstInt = [](mlir::Value v) -> std::optional<int64_t> {
    auto c = v.getDefiningOp<mlir::arith::ConstantOp>();
    if (!c) return std::nullopt;
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return ia.getInt();
    return std::nullopt;
  };
  auto lo = getConstInt(forOp.getLowerBound());
  auto st = getConstInt(forOp.getStep());
  if (!lo || *lo != 0 || !st || *st <= 0) return mlir::failure();
  if (getConstInt(forOp.getUpperBound())) return mlir::failure();  // static → other path

  // Element-offset form: the K axis of every load must be the induction var
  // directly (so `upperBound` is the total K in elements and a step-8 loop
  // sweeps [0, K)). This rejects the tile-index form (offs_k = k*BK + arange).
  mlir::Value iv = forOp.getInductionVar();
  if (!kAxisUsesInductionVar(loadA.getPtr(), /*kAxis=*/1, iv))
    return mlir::failure();
  if (!kAxisUsesInductionVar(loadB.getPtr(), bKAxis, iv))
    return mlir::failure();

  const int numWarps = mlir::triton::gpu::lookupNumWarps(dot);
  if (numWarps < 1) return mlir::failure();

  auto resTy = mlir::dyn_cast<mlir::RankedTensorType>(dot.getType());
  if (!resTy) return mlir::failure();
  auto elemTy = resTy.getElementType();
  if (!elemTy.isF32()) return mlir::failure();
  auto shape = resTy.getShape();
  if (shape.size() != 2 || shape[0] % 8 != 0 || shape[1] % 8 != 0)
    return mlir::failure();
  const int mTiles = shape[0] / 8, nTiles = shape[1] / 8;
  // Multi-warp: partition the M-tile rows across warps (each warp owns
  // mTiles/numWarps rows, all N-tiles) so per-warp register pressure stays
  // bounded. Requires mTiles % numWarps == 0. Multi-warp masked is not yet
  // supported (the masked staged load has no per-warp buffer).
  const bool multiWarp = numWarps > 1;
  if (multiWarp) {
    if (mTiles % numWarps != 0) return mlir::failure();
    if (loadA.getMask() || loadB.getMask()) return mlir::failure();
  }
  const int mPerWarp = multiWarp ? mTiles / numWarps : mTiles;
  // Register guard: each output tile is a live simdgroup_matrix accumulator
  // per warp (~2 regs/thread).
  if (mPerWarp * nTiles > 32) return mlir::failure();

  if (!forOp.getResult(0).hasOneUse()) return mlir::failure();
  auto store = mlir::dyn_cast<mlir::triton::StoreOp>(
      *forOp.getResult(0).getUsers().begin());
  if (!store) return mlir::failure();
  if (multiWarp && store.getMask()) return mlir::failure();
  MaskExtents storeExt;
  if (store.getMask()) {
    storeExt = extractMaskExtents(store.getMask());
    if (!storeExt.mExtent || !storeExt.nExtent) return mlir::failure();
  }

  // Origins/strides come from the loads (recomputed each iter) rather than
  // iter_arg inits. The M/N origins are pid-driven scalars defined OUTSIDE the
  // loop, so they still dominate the pre-loop insertion point after erasure.
  mlir::Value aPtr = unwrapPtrToKernelArg(loadA.getPtr());
  mlir::Value bPtr = unwrapPtrToKernelArg(loadB.getPtr());
  mlir::Value cPtr = unwrapPtrToKernelArg(store.getPtr());
  // Row (axis-0) strides — the axis-aware extractor avoids mistaking the
  // K-offset's `splat(iv)` for a stride.
  mlir::Value strideA = findAxisStride(loadA.getPtr(), /*axis=*/0);
  mlir::Value strideB = findAxisStride(loadB.getPtr(), /*axis=*/0);
  mlir::Value aBaseRow = findOriginScalarInPtrChain(loadA.getPtr(), /*axis=*/0);
  mlir::Value bNOrigin = findOriginScalarInPtrChain(loadB.getPtr(), bNAxis);

  mlir::OpBuilder builder(forOp);
  auto loc = forOp.getLoc();
  auto ctx = builder.getContext();
  auto ui32 = builder.getIntegerType(32, /*isSigned=*/false);
  auto matTy = MetalSimdgroupMatrixType::get(ctx, 8, 8, elemTy);

  mlir::Value strideAVal = emitStrideOperand(builder, loc, ui32, strideA);
  mlir::Value strideBVal = emitStrideOperand(builder, loc, ui32, strideB);
  mlir::Value strideCVal = emitStrideOperand(
      builder, loc, ui32, findAxisStride(store.getPtr(), /*axis=*/0));
  OriginPair cOrig = extractOriginPair<mlir::triton::StoreOp>(store);
  mlir::Value aBuf = bridgePtrToMemref(builder, loc, aPtr, elemTy);
  mlir::Value bBuf = bridgePtrToMemref(builder, loc, bPtr, elemTy);
  mlir::Value cBuf = bridgePtrToMemref(builder, loc, cPtr, elemTy);

  auto toUi32 = [&](mlir::OpBuilder &b, mlir::Value v) -> mlir::Value {
    if (v.getType() == ui32) return v;
    return mlir::UnrealizedConversionCastOp::create(
               b, loc, mlir::TypeRange{ui32}, mlir::ValueRange{v})
        .getResult(0);
  };
  // Per-tile origin: `baseScalar (i32, may be null) + tileIdx*8`, cast to ui32.
  auto tileOrigin = [&](mlir::Value baseScalar, int tileIdx) -> mlir::Value {
    mlir::Value v = baseScalar;
    if (tileIdx != 0) {
      auto off = mlir::arith::ConstantOp::create(
          builder, loc, builder.getI32IntegerAttr(tileIdx * 8));
      v = v ? mlir::arith::AddIOp::create(builder, loc, v, off.getResult())
                  .getResult()
            : off.getResult();
    }
    return emitOriginOperand(builder, loc, ui32, v);
  };

  mlir::Value aRowExt, aColExt, bRowExt, bColExt;
  auto extentsFor =
      [&](mlir::triton::LoadOp ld) -> std::pair<mlir::Value, mlir::Value> {
    if (!ld.getMask()) return {mlir::Value(), mlir::Value()};
    auto e = extractMaskExtents(ld.getMask());
    return {emitExtentUi32(builder, loc, ui32, e.mExtent),
            emitExtentUi32(builder, loc, ui32, e.nExtent)};
  };
  std::tie(aRowExt, aColExt) = extentsFor(loadA);
  std::tie(bRowExt, bColExt) = extentsFor(loadB);

  // Multi-warp: widx = simdgroup index; each warp owns M-tile rows
  // {widx + mIter*numWarps : mIter in [0, mPerWarp)}, all N-tiles.
  auto i32ty = builder.getIntegerType(32);
  mlir::Value widx, warpMI32;
  if (multiWarp) {
    widx = SimdgroupIndexOp::create(builder, loc, ui32).getResult();
    warpMI32 = mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange{i32ty}, mlir::ValueRange{widx})
                   .getResult(0);
  }
  auto mTileIdxVal = [&](int mIter) -> mlir::Value {
    if (!multiWarp)
      return mlir::arith::ConstantOp::create(
                 builder, loc, builder.getI32IntegerAttr(mIter)).getResult();
    auto c = mlir::arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(mIter * numWarps));
    return mlir::arith::AddIOp::create(builder, loc, warpMI32, c.getResult())
        .getResult();
  };
  auto originForRuntimeTile =
      [&](mlir::Value baseScalar, mlir::Value tileIdxVal) -> mlir::Value {
    auto eight = mlir::arith::ConstantOp::create(builder, loc,
                                                 builder.getI32IntegerAttr(8));
    mlir::Value off =
        mlir::arith::MulIOp::create(builder, loc, tileIdxVal, eight.getResult())
            .getResult();
    mlir::Value v =
        baseScalar
            ? mlir::arith::AddIOp::create(builder, loc, baseScalar, off).getResult()
            : off;
    return toUi32(builder, v);
  };

  // Per-warp M-tile origins (A/C rows, runtime); N-tile origins are static.
  llvm::SmallVector<mlir::Value> aRowTile(mPerWarp), cRowTile(mPerWarp);
  for (int mi = 0; mi < mPerWarp; ++mi) {
    mlir::Value idx = mTileIdxVal(mi);
    aRowTile[mi] = originForRuntimeTile(aBaseRow, idx);
    cRowTile[mi] = originForRuntimeTile(cOrig.row, idx);
  }
  llvm::SmallVector<mlir::Value> bNTile(nTiles), cColTile(nTiles);
  for (int ni = 0; ni < nTiles; ++ni) {
    bNTile[ni] = tileOrigin(bNOrigin, ni);
    cColTile[ni] = tileOrigin(cOrig.col, ni);
  }

  llvm::SmallVector<mlir::Value> inits;
  for (int t = 0; t < mPerWarp * nTiles; ++t)
    inits.push_back(SimdgroupMatrixZeroOp::create(builder, loc, matTy).getResult());
  auto c0 = mlir::arith::ConstantOp::create(builder, loc,
                                            builder.getI32IntegerAttr(0));
  auto c8 = mlir::arith::ConstantOp::create(builder, loc,
                                            builder.getI32IntegerAttr(8));
  mlir::Value ub = forOp.getUpperBound();

  auto loop = mlir::scf::ForOp::create(
      builder, loc, c0.getResult(), ub, c8.getResult(), inits,
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value ivFresh,
          mlir::ValueRange args) {
        mlir::Value kUi32 = toUi32(b, ivFresh);
        llvm::SmallVector<mlir::Value> aTiles(mPerWarp), bTiles(nTiles);
        for (int mi = 0; mi < mPerWarp; ++mi)
          aTiles[mi] = emitStagedLoad(b, l, matTy, aBuf, aRowTile[mi], kUi32,
                                      strideAVal, /*transposed=*/false, aRowExt,
                                      aColExt, widx);
        for (int ni = 0; ni < nTiles; ++ni)
          bTiles[ni] =
              bTransposed
                  ? emitStagedLoad(b, l, matTy, bBuf, bNTile[ni], kUi32,
                                   strideBVal, /*transposed=*/true, bRowExt,
                                   bColExt, widx)
                  : emitStagedLoad(b, l, matTy, bBuf, kUi32, bNTile[ni],
                                   strideBVal, /*transposed=*/false, bRowExt,
                                   bColExt, widx);
        llvm::SmallVector<mlir::Value> newAccs;
        for (int mi = 0; mi < mPerWarp; ++mi)
          for (int ni = 0; ni < nTiles; ++ni)
            newAccs.push_back(SimdgroupMultiplyAccumulateOp::create(
                                  b, l, matTy, args[mi * nTiles + ni],
                                  aTiles[mi], bTiles[ni])
                                  .getResult());
        mlir::scf::YieldOp::create(b, l, newAccs);
      });

  llvm::SmallVector<mlir::Value, 2> oPartial;
  if (storeExt.mExtent && storeExt.nExtent) {
    oPartial.push_back(emitExtentUi32(builder, loc, ui32, storeExt.mExtent));
    oPartial.push_back(emitExtentUi32(builder, loc, ui32, storeExt.nExtent));
  }
  for (int mi = 0; mi < mPerWarp; ++mi)
    for (int ni = 0; ni < nTiles; ++ni)
      SimdgroupStoreOp::create(builder, loc, loop.getResult(mi * nTiles + ni),
                                cBuf, cRowTile[mi], cColTile[ni], strideCVal,
                                oPartial);

  store.erase();
  forOp.erase();
  return mlir::success();
}

// W2b: multi-accumulator recompute-from-IV runtime-K loop. Generalises
// `tryRuntimeKLoopRecomputeDot` to N accumulators / N dots that SHARE the same
// A operand (`x`) — the fused LoRA inner loop
//   acc0 = tl.dot(x, tl.trans(w), acc0)
//   acc1 = tl.dot(x, tl.trans(a), acc1)
// Each accumulator is a distinct iter_arg yielded by its dot and stored after
// the loop. Emits ONE fresh scf.for carrying N simdgroup_matrix accumulators
// with a single shared staged A load per iteration. Single-tile / single-warp /
// unmasked; each B may be transposed (W1). Called atomically from
// `preprocessDotChains` (which groups dots by loop). See
// `metal-lora-linear-fix-plan.md` (W2b).
static mlir::LogicalResult
tryRuntimeKLoopRecomputeMultiDot(mlir::scf::ForOp forOp) {
  const unsigned nAcc = forOp.getNumRegionIterArgs();
  if (nAcc < 2) return mlir::failure();

  // Body: shared A load + N B loads + optional trans + N dots + yield + noise.
  llvm::SmallVector<mlir::triton::LoadOp> loads;
  llvm::SmallVector<mlir::triton::DotOp> dots;
  mlir::scf::YieldOp yieldOp;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (op.getNumRegions() > 0) return mlir::failure();
    if (auto l = mlir::dyn_cast<mlir::triton::LoadOp>(op))
      loads.push_back(l);
    else if (auto d = mlir::dyn_cast<mlir::triton::DotOp>(op))
      dots.push_back(d);
    else if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op))
      yieldOp = y;
    else if (!mlir::isPure(&op))
      return mlir::failure();
  }
  if (dots.size() != nAcc) return mlir::failure();
  if (!yieldOp || yieldOp.getNumOperands() != nAcc) return mlir::failure();

  // Shared A operand: the same unmasked load feeds every dot's A.
  mlir::triton::LoadOp sharedA = dots[0].getA().getDefiningOp<mlir::triton::LoadOp>();
  if (!sharedA || sharedA.getMask()) return mlir::failure();
  for (auto d : dots)
    if (d.getA() != sharedA.getResult()) return mlir::failure();

  auto getConstInt = [](mlir::Value v) -> std::optional<int64_t> {
    auto c = v.getDefiningOp<mlir::arith::ConstantOp>();
    if (!c) return std::nullopt;
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return ia.getInt();
    return std::nullopt;
  };
  auto lo = getConstInt(forOp.getLowerBound());
  auto st = getConstInt(forOp.getStep());
  if (!lo || *lo != 0 || !st || *st <= 0) return mlir::failure();
  if (getConstInt(forOp.getUpperBound())) return mlir::failure();  // runtime only

  mlir::Value iv = forOp.getInductionVar();
  if (!kAxisUsesInductionVar(sharedA.getPtr(), /*kAxis=*/1, iv))
    return mlir::failure();
  if (mlir::triton::gpu::lookupNumWarps(dots[0]) != 1) return mlir::failure();

  // Per-accumulator info, indexed by iter_arg position.
  struct AccInfo {
    mlir::triton::DotOp dot;
    mlir::triton::LoadOp loadB;
    bool bTransposed;
    mlir::triton::StoreOp store;
  };
  llvm::SmallVector<AccInfo, 4> accs(nAcc);
  for (unsigned i = 0; i < nAcc; ++i) {
    mlir::Value iterArg = forOp.getRegionIterArg(i);
    mlir::triton::DotOp dot;
    for (auto d : dots)
      if (d.getC() == iterArg) { dot = d; break; }
    if (!dot) return mlir::failure();
    if (yieldOp.getOperand(i) != dot.getResult()) return mlir::failure();

    bool bT = false;
    mlir::triton::LoadOp loadB;
    if (auto tr = dot.getB().getDefiningOp<mlir::triton::TransOp>()) {
      loadB = tr.getSrc().getDefiningOp<mlir::triton::LoadOp>();
      bT = true;
    } else {
      loadB = dot.getB().getDefiningOp<mlir::triton::LoadOp>();
    }
    if (!loadB || loadB.getMask()) return mlir::failure();
    if (!kAxisUsesInductionVar(loadB.getPtr(), bT ? 1 : 0, iv))
      return mlir::failure();

    auto resTy = mlir::dyn_cast<mlir::RankedTensorType>(dot.getType());
    if (!resTy || !resTy.getElementType().isF32()) return mlir::failure();
    auto shape = resTy.getShape();
    if (shape.size() != 2 || shape[0] != 8 || shape[1] != 8)
      return mlir::failure();

    if (!forOp.getResult(i).hasOneUse()) return mlir::failure();
    auto store = mlir::dyn_cast<mlir::triton::StoreOp>(
        *forOp.getResult(i).getUsers().begin());
    if (!store || store.getMask()) return mlir::failure();
    accs[i] = {dot, loadB, bT, store};
  }

  // ---- Emit ----
  mlir::OpBuilder builder(forOp);
  auto loc = forOp.getLoc();
  auto ctx = builder.getContext();
  auto ui32 = builder.getIntegerType(32, /*isSigned=*/false);
  auto elemTy = builder.getF32Type();
  auto matTy = MetalSimdgroupMatrixType::get(ctx, 8, 8, elemTy);
  auto toUi32 = [&](mlir::OpBuilder &b, mlir::Value v) -> mlir::Value {
    if (v.getType() == ui32) return v;
    return mlir::UnrealizedConversionCastOp::create(
               b, loc, mlir::TypeRange{ui32}, mlir::ValueRange{v})
        .getResult(0);
  };

  // Shared A.
  mlir::Value aBuf = bridgePtrToMemref(
      builder, loc, unwrapPtrToKernelArg(sharedA.getPtr()), elemTy);
  mlir::Value strideAVal = emitStrideOperand(
      builder, loc, ui32, findAxisStride(sharedA.getPtr(), /*axis=*/0));
  mlir::Value aBaseRowVal = emitOriginOperand(
      builder, loc, ui32, findOriginScalarInPtrChain(sharedA.getPtr(), 0));

  // Per-accumulator staged-B + store operands.
  llvm::SmallVector<mlir::Value, 4> bBufs(nAcc), bNOriginVals(nAcc),
      strideBVals(nAcc), cBufs(nAcc), cRowOrigins(nAcc), cColOrigins(nAcc),
      strideCVals(nAcc);
  llvm::SmallVector<bool, 4> bTransposed(nAcc);
  for (unsigned i = 0; i < nAcc; ++i) {
    auto &a = accs[i];
    bTransposed[i] = a.bTransposed;
    int bNAxis = a.bTransposed ? 0 : 1;
    bBufs[i] = bridgePtrToMemref(
        builder, loc, unwrapPtrToKernelArg(a.loadB.getPtr()), elemTy);
    strideBVals[i] = emitStrideOperand(
        builder, loc, ui32, findAxisStride(a.loadB.getPtr(), /*axis=*/0));
    bNOriginVals[i] = emitOriginOperand(
        builder, loc, ui32, findOriginScalarInPtrChain(a.loadB.getPtr(), bNAxis));
    cBufs[i] = bridgePtrToMemref(
        builder, loc, unwrapPtrToKernelArg(a.store.getPtr()), elemTy);
    strideCVals[i] = emitStrideOperand(
        builder, loc, ui32, findAxisStride(a.store.getPtr(), /*axis=*/0));
    OriginPair cOrig = extractOriginPair<mlir::triton::StoreOp>(a.store);
    cRowOrigins[i] = emitOriginOperand(builder, loc, ui32, cOrig.row);
    cColOrigins[i] = emitOriginOperand(builder, loc, ui32, cOrig.col);
  }

  llvm::SmallVector<mlir::Value, 4> inits;
  for (unsigned i = 0; i < nAcc; ++i)
    inits.push_back(SimdgroupMatrixZeroOp::create(builder, loc, matTy).getResult());
  auto c0 = mlir::arith::ConstantOp::create(builder, loc,
                                            builder.getI32IntegerAttr(0));
  auto c8 = mlir::arith::ConstantOp::create(builder, loc,
                                            builder.getI32IntegerAttr(8));
  mlir::Value ub = forOp.getUpperBound();

  auto loop = mlir::scf::ForOp::create(
      builder, loc, c0.getResult(), ub, c8.getResult(), inits,
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value ivFresh,
          mlir::ValueRange args) {
        mlir::Value kUi32 = toUi32(b, ivFresh);
        // Shared A tile, loaded once and reused by every accumulator.
        auto aTile = SimdgroupLoadDeviceStagedOp::create(
            b, l, matTy, aBuf, aBaseRowVal, kUi32, strideAVal,
            mlir::ValueRange{});
        llvm::SmallVector<mlir::Value, 4> newAccs;
        for (unsigned i = 0; i < nAcc; ++i) {
          mlir::triton::metal::SimdgroupLoadDeviceStagedOp bTile;
          if (bTransposed[i]) {
            bTile = SimdgroupLoadDeviceStagedOp::create(
                b, l, matTy, bBufs[i], bNOriginVals[i], kUi32, strideBVals[i],
                mlir::ValueRange{});
            bTile->setAttr("transposed", b.getUnitAttr());
          } else {
            bTile = SimdgroupLoadDeviceStagedOp::create(
                b, l, matTy, bBufs[i], kUi32, bNOriginVals[i], strideBVals[i],
                mlir::ValueRange{});
          }
          auto ma = SimdgroupMultiplyAccumulateOp::create(
              b, l, matTy, args[i], aTile.getResult(), bTile.getResult());
          newAccs.push_back(ma.getResult());
        }
        mlir::scf::YieldOp::create(b, l, newAccs);
      });

  for (unsigned i = 0; i < nAcc; ++i)
    SimdgroupStoreOp::create(builder, loc, loop.getResult(i), cBufs[i],
                              cRowOrigins[i], cColOrigins[i], strideCVals[i],
                              mlir::ValueRange{});

  for (unsigned i = 0; i < nAcc; ++i)
    accs[i].store.erase();
  forOp.erase();
  return mlir::success();
}

// W2c: fully-fused LoRA. Matches the multi-accumulator loop PLUS the post-loop
// epilogue `acc0 += scale * tl.dot(acc1, tl.trans(b))`:
//   for k in ...: acc0 = dot(x, trans(w), acc0); acc1 = dot(x, trans(a), acc1)
//   acc0 += scale * tl.dot(acc1, tl.trans(b))
//   tl.store(out, acc0)
// The scale-and-add folds into a single `simdgroup_fused_store(acc0, dot3,
// scale)` where dot3 = acc1 · trans(b) (a simdgroup MMA on the loop's own
// accumulator). Two accumulators, single 8x8 tile, single-warp, unmasked.
static mlir::LogicalResult tryFusedLoRAEpilogue(mlir::scf::ForOp forOp) {
  if (forOp.getNumRegionIterArgs() != 2) return mlir::failure();

  // --- Loop body: shared A load + 2 B loads + 2 dots + yield + noise. ---
  llvm::SmallVector<mlir::triton::LoadOp> loads;
  llvm::SmallVector<mlir::triton::DotOp> dots;
  mlir::scf::YieldOp yieldOp;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (op.getNumRegions() > 0) return mlir::failure();
    if (auto l = mlir::dyn_cast<mlir::triton::LoadOp>(op)) loads.push_back(l);
    else if (auto d = mlir::dyn_cast<mlir::triton::DotOp>(op)) dots.push_back(d);
    else if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) yieldOp = y;
    else if (!mlir::isPure(&op)) return mlir::failure();
  }
  if (dots.size() != 2 || !yieldOp || yieldOp.getNumOperands() != 2)
    return mlir::failure();
  // A masked load is accepted only when its mask decomposes into the canonical
  // `(offs_row < ROW) & (offs_col < K)` extents (out-of-bounds → 0).
  auto maskOk = [](mlir::triton::LoadOp ld) {
    if (!ld.getMask()) return true;
    auto e = extractMaskExtents(ld.getMask());
    return (bool)(e.mExtent && e.nExtent);
  };
  mlir::triton::LoadOp sharedA = dots[0].getA().getDefiningOp<mlir::triton::LoadOp>();
  if (!sharedA || !maskOk(sharedA)) return mlir::failure();
  for (auto d : dots)
    if (d.getA() != sharedA.getResult()) return mlir::failure();

  auto getConstInt = [](mlir::Value v) -> std::optional<int64_t> {
    auto c = v.getDefiningOp<mlir::arith::ConstantOp>();
    if (!c) return std::nullopt;
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return ia.getInt();
    return std::nullopt;
  };
  auto lo = getConstInt(forOp.getLowerBound());
  auto st = getConstInt(forOp.getStep());
  if (!lo || *lo != 0 || !st || *st <= 0) return mlir::failure();
  if (getConstInt(forOp.getUpperBound())) return mlir::failure();
  mlir::Value iv = forOp.getInductionVar();
  if (!kAxisUsesInductionVar(sharedA.getPtr(), 1, iv)) return mlir::failure();
  const int numWarps = mlir::triton::gpu::lookupNumWarps(dots[0]);
  if (numWarps < 1) return mlir::failure();

  // --- Per-accumulator dot (loop body), by iter_arg index. ---
  struct AccDot { mlir::triton::DotOp dot; mlir::triton::LoadOp loadB; bool bT; };
  llvm::SmallVector<AccDot, 2> accDots(2);
  for (unsigned i = 0; i < 2; ++i) {
    mlir::Value iterArg = forOp.getRegionIterArg(i);
    mlir::triton::DotOp dot;
    for (auto d : dots) if (d.getC() == iterArg) { dot = d; break; }
    if (!dot || yieldOp.getOperand(i) != dot.getResult()) return mlir::failure();
    bool bT = false;
    mlir::triton::LoadOp loadB;
    if (auto tr = dot.getB().getDefiningOp<mlir::triton::TransOp>()) {
      loadB = tr.getSrc().getDefiningOp<mlir::triton::LoadOp>(); bT = true;
    } else loadB = dot.getB().getDefiningOp<mlir::triton::LoadOp>();
    if (!loadB || !maskOk(loadB)) return mlir::failure();
    if (!kAxisUsesInductionVar(loadB.getPtr(), bT ? 1 : 0, iv)) return mlir::failure();
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(dot.getType());
    if (!rt || !rt.getElementType().isF32()) return mlir::failure();
    auto sh = rt.getShape();
    if (sh.size() != 2 || sh[0] % 8 != 0 || sh[1] % 8 != 0) return mlir::failure();
    accDots[i] = {dot, loadB, bT};
  }

  // --- Epilogue roles: one loop result feeds dot3 (A operand), the other an
  // addf. dot3 result -> mulf(scale) -> addf(base) -> unmasked store. ---
  int accDotIdx = -1, accBaseIdx = -1;
  mlir::triton::DotOp dot3;
  for (unsigned i = 0; i < 2; ++i) {
    if (!forOp.getResult(i).hasOneUse()) return mlir::failure();
    mlir::Operation *user = *forOp.getResult(i).getUsers().begin();
    if (auto d = mlir::dyn_cast<mlir::triton::DotOp>(user)) {
      if (d.getA() != forOp.getResult(i)) return mlir::failure();
      dot3 = d; accDotIdx = i;
    } else if (mlir::isa<mlir::arith::AddFOp>(user)) {
      accBaseIdx = i;
    } else return mlir::failure();
  }
  if (!dot3 || accDotIdx < 0 || accBaseIdx < 0) return mlir::failure();

  // dot3.C must be dense<0.0>; B is trans(load) or load.
  {
    auto cst = dot3.getC().getDefiningOp<mlir::arith::ConstantOp>();
    if (!cst) return mlir::failure();
    auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue());
    if (!dense || !dense.isSplat() ||
        !dense.getSplatValue<mlir::APFloat>().isZero()) return mlir::failure();
  }
  bool b3T = false;
  mlir::triton::LoadOp loadB3;
  if (auto tr = dot3.getB().getDefiningOp<mlir::triton::TransOp>()) {
    loadB3 = tr.getSrc().getDefiningOp<mlir::triton::LoadOp>(); b3T = true;
  } else loadB3 = dot3.getB().getDefiningOp<mlir::triton::LoadOp>();
  if (!loadB3 || !maskOk(loadB3)) return mlir::failure();
  auto dot3Ty = mlir::dyn_cast<mlir::RankedTensorType>(dot3.getType());
  if (!dot3Ty || !dot3Ty.getElementType().isF32()) return mlir::failure();
  auto d3sh = dot3Ty.getShape();
  if (d3sh.size() != 2 || d3sh[0] % 8 != 0 || d3sh[1] % 8 != 0)
    return mlir::failure();

  // dot3 -> mulf(scale) -> addf(base) -> store. At multi-tile shapes the layout
  // assignment inserts `ttg.convert_layout` between the epilogue ops; skip them.
  auto stripCvt = [](mlir::Value v) -> mlir::Value {
    while (auto c = v.getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>())
      v = c.getSrc();
    return v;
  };
  auto userSkipCvt = [](mlir::Value v) -> mlir::Operation * {
    if (!v.hasOneUse()) return nullptr;
    mlir::Operation *u = *v.getUsers().begin();
    while (auto c = mlir::dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(u)) {
      if (!c.getResult().hasOneUse()) return nullptr;
      u = *c.getResult().getUsers().begin();
    }
    return u;
  };
  auto mulf = mlir::dyn_cast_or_null<mlir::arith::MulFOp>(
      userSkipCvt(dot3.getResult()));
  if (!mulf) return mlir::failure();
  mlir::Value scaleTensor = stripCvt(mulf.getLhs()) == dot3.getResult()
                                ? mulf.getRhs()
                                : mulf.getLhs();
  auto scaleSplat = stripCvt(scaleTensor).getDefiningOp<mlir::triton::SplatOp>();
  if (!scaleSplat) return mlir::failure();
  mlir::Value scaleScalar = scaleSplat.getSrc();
  if (!scaleScalar.getType().isF32()) return mlir::failure();
  auto addf = mlir::dyn_cast_or_null<mlir::arith::AddFOp>(
      userSkipCvt(mulf.getResult()));
  if (!addf) return mlir::failure();
  mlir::Value baseRes = forOp.getResult(accBaseIdx);
  auto isMulf = [&](mlir::Value v) { return stripCvt(v) == mulf.getResult(); };
  auto isBase = [&](mlir::Value v) { return stripCvt(v) == baseRes; };
  if (!((isMulf(addf.getLhs()) && isBase(addf.getRhs())) ||
        (isMulf(addf.getRhs()) && isBase(addf.getLhs()))))
    return mlir::failure();
  auto store = mlir::dyn_cast_or_null<mlir::triton::StoreOp>(
      userSkipCvt(addf.getResult()));
  if (!store) return mlir::failure();
  MaskExtents storeExt;
  if (store.getMask()) {
    storeExt = extractMaskExtents(store.getMask());
    if (!storeExt.mExtent || !storeExt.nExtent) return mlir::failure();
  }

  // Tile grid. acc0 (base) = [BM, BN], acc1 (dot) = [BM, BR], dot3 = [BM, BN].
  auto baseSh = mlir::cast<mlir::RankedTensorType>(
                    accDots[accBaseIdx].dot.getType()).getShape();
  auto dotSh = mlir::cast<mlir::RankedTensorType>(
                   accDots[accDotIdx].dot.getType()).getShape();
  const int MT = baseSh[0] / 8, NT = baseSh[1] / 8, RT = dotSh[1] / 8;
  if (dotSh[0] / 8 != MT) return mlir::failure();
  if (d3sh[0] != baseSh[0] || d3sh[1] != baseSh[1]) return mlir::failure();
  // Multi-warp: partition the M-tile rows across warps (each warp is
  // self-contained — its output tiles' dot3 uses only its own acc1 M-tiles, so
  // no cross-warp dependency). Requires MT % numWarps == 0; masked multi-warp
  // is not yet supported (masked staged load has no per-warp buffer).
  const bool multiWarp = numWarps > 1;
  // Masked multi-warp is supported here: masked staged loads carry a per-warp
  // buffer and the epilogue is a per-warp simdgroup_fused_store.
  if (multiWarp && MT % numWarps != 0) return mlir::failure();
  const int mPerWarp = multiWarp ? MT / numWarps : MT;
  // Register guard: per-warp live simdgroup_matrix accumulators (~2 regs/thread).
  if (mPerWarp * NT + mPerWarp * RT > 40) return mlir::failure();

  // ================= Emit =================
  mlir::OpBuilder builder(forOp);
  auto loc = forOp.getLoc();
  auto ctx = builder.getContext();
  auto ui32 = builder.getIntegerType(32, /*isSigned=*/false);
  auto elemTy = builder.getF32Type();
  auto matTy = MetalSimdgroupMatrixType::get(ctx, 8, 8, elemTy);
  auto toUi32 = [&](mlir::OpBuilder &b, mlir::Value v) -> mlir::Value {
    if (v.getType() == ui32) return v;
    return mlir::UnrealizedConversionCastOp::create(
               b, loc, mlir::TypeRange{ui32}, mlir::ValueRange{v}).getResult(0);
  };

  // (rowExtent, colExtent) = (axis-0 bound, axis-1 bound) of a load's mask, or
  // (null, null) when unmasked. Matches the staged load's (origin_row,
  // origin_col) axes for both normal and transposed loads.
  auto extentsFor =
      [&](mlir::triton::LoadOp ld) -> std::pair<mlir::Value, mlir::Value> {
    if (!ld.getMask()) return {mlir::Value(), mlir::Value()};
    auto e = extractMaskExtents(ld.getMask());
    return {emitExtentUi32(builder, loc, ui32, e.mExtent),
            emitExtentUi32(builder, loc, ui32, e.nExtent)};
  };

  // Per-tile origin: baseScalar (i32, may be null) + tileIdx*8, cast to ui32.
  auto tileOrigin = [&](mlir::Value baseScalar, int tileIdx) -> mlir::Value {
    mlir::Value v = baseScalar;
    if (tileIdx != 0) {
      auto off = mlir::arith::ConstantOp::create(
          builder, loc, builder.getI32IntegerAttr(tileIdx * 8));
      v = v ? mlir::arith::AddIOp::create(builder, loc, v, off.getResult())
                  .getResult()
            : off.getResult();
    }
    return emitOriginOperand(builder, loc, ui32, v);
  };
  auto constUi32 = [&](int val) -> mlir::Value {
    return ConstantOp::create(builder, loc, builder.getIntegerAttr(ui32, val))
        .getResult();
  };
  // Multi-warp: widx = simdgroup index; each warp owns M-tile rows
  // {widx + mIter*numWarps}, all N/R-tiles (its epilogue is self-contained).
  auto i32ty = builder.getIntegerType(32);
  mlir::Value widx, warpMI32;
  if (multiWarp) {
    widx = SimdgroupIndexOp::create(builder, loc, ui32).getResult();
    warpMI32 = mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange{i32ty}, mlir::ValueRange{widx})
                   .getResult(0);
  }
  auto mTileIdxVal = [&](int mIter) -> mlir::Value {
    if (!multiWarp)
      return mlir::arith::ConstantOp::create(
                 builder, loc, builder.getI32IntegerAttr(mIter)).getResult();
    auto c = mlir::arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(mIter * numWarps));
    return mlir::arith::AddIOp::create(builder, loc, warpMI32, c.getResult())
        .getResult();
  };
  auto originForRuntimeTile =
      [&](mlir::Value baseScalar, mlir::Value tileIdxVal) -> mlir::Value {
    auto eight = mlir::arith::ConstantOp::create(builder, loc,
                                                 builder.getI32IntegerAttr(8));
    mlir::Value off =
        mlir::arith::MulIOp::create(builder, loc, tileIdxVal, eight.getResult())
            .getResult();
    mlir::Value v =
        baseScalar
            ? mlir::arith::AddIOp::create(builder, loc, baseScalar, off).getResult()
            : off;
    return toUi32(builder, v);
  };
  llvm::SmallVector<mlir::Value, 1> widxRange;
  if (widx) widxRange.push_back(widx);

  // Shared A.
  mlir::Value aBuf = bridgePtrToMemref(
      builder, loc, unwrapPtrToKernelArg(sharedA.getPtr()), elemTy);
  mlir::Value strideAVal = emitStrideOperand(
      builder, loc, ui32, findAxisStride(sharedA.getPtr(), 0));
  mlir::Value aRowScalar = findOriginScalarInPtrChain(sharedA.getPtr(), 0);
  mlir::Value aRowExt, aColExt;
  std::tie(aRowExt, aColExt) = extentsFor(sharedA);

  // Per-accumulator B (loop body): acc0's is w, acc1's is a.
  mlir::Value bBuf[2], strideBVal[2], bNScalar[2], bRExt[2], bCExt[2];
  bool bTr[2];
  for (unsigned i = 0; i < 2; ++i) {
    bTr[i] = accDots[i].bT;
    int bNAxis = accDots[i].bT ? 0 : 1;
    bBuf[i] = bridgePtrToMemref(
        builder, loc, unwrapPtrToKernelArg(accDots[i].loadB.getPtr()), elemTy);
    strideBVal[i] = emitStrideOperand(
        builder, loc, ui32, findAxisStride(accDots[i].loadB.getPtr(), 0));
    bNScalar[i] =
        findOriginScalarInPtrChain(accDots[i].loadB.getPtr(), bNAxis);
    std::tie(bRExt[i], bCExt[i]) = extentsFor(accDots[i].loadB);
  }

  // Per-warp acc0 tiles [0, nBase); acc1 tiles [nBase, ...).
  const int nBase = mPerWarp * NT;
  llvm::SmallVector<mlir::Value> inits;
  for (int t = 0; t < mPerWarp * NT + mPerWarp * RT; ++t)
    inits.push_back(SimdgroupMatrixZeroOp::create(builder, loc, matTy).getResult());
  // Per-warp M-tile origins (runtime); N/R-tile origins are static.
  llvm::SmallVector<mlir::Value> aRowTile(mPerWarp), wNTile(NT), aRTile(RT);
  for (int mi = 0; mi < mPerWarp; ++mi)
    aRowTile[mi] = originForRuntimeTile(aRowScalar, mTileIdxVal(mi));
  for (int ni = 0; ni < NT; ++ni)
    wNTile[ni] = tileOrigin(bNScalar[accBaseIdx], ni);
  for (int rk = 0; rk < RT; ++rk)
    aRTile[rk] = tileOrigin(bNScalar[accDotIdx], rk);

  auto c0 = mlir::arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
  auto c8 = mlir::arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(8));
  mlir::Value ub = forOp.getUpperBound();

  auto loop = mlir::scf::ForOp::create(
      builder, loc, c0.getResult(), ub, c8.getResult(), inits,
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::Value ivFresh,
          mlir::ValueRange args) {
        mlir::Value kUi32 = toUi32(b, ivFresh);
        // Load each A row-tile, w col-tile (acc0 B) and a col-tile (acc1 B)
        // once, then reuse across the accumulator grid.
        llvm::SmallVector<mlir::Value> aTiles(mPerWarp), wTiles(NT),
            aMatTiles(RT);
        for (int mi = 0; mi < mPerWarp; ++mi)
          aTiles[mi] = emitStagedLoad(b, l, matTy, aBuf, aRowTile[mi], kUi32,
                                      strideAVal, false, aRowExt, aColExt, widx);
        for (int ni = 0; ni < NT; ++ni)
          wTiles[ni] =
              bTr[accBaseIdx]
                  ? emitStagedLoad(b, l, matTy, bBuf[accBaseIdx], wNTile[ni],
                                   kUi32, strideBVal[accBaseIdx], true,
                                   bRExt[accBaseIdx], bCExt[accBaseIdx], widx)
                  : emitStagedLoad(b, l, matTy, bBuf[accBaseIdx], kUi32,
                                   wNTile[ni], strideBVal[accBaseIdx], false,
                                   bRExt[accBaseIdx], bCExt[accBaseIdx], widx);
        for (int rk = 0; rk < RT; ++rk)
          aMatTiles[rk] =
              bTr[accDotIdx]
                  ? emitStagedLoad(b, l, matTy, bBuf[accDotIdx], aRTile[rk],
                                   kUi32, strideBVal[accDotIdx], true,
                                   bRExt[accDotIdx], bCExt[accDotIdx], widx)
                  : emitStagedLoad(b, l, matTy, bBuf[accDotIdx], kUi32,
                                   aRTile[rk], strideBVal[accDotIdx], false,
                                   bRExt[accDotIdx], bCExt[accDotIdx], widx);
        llvm::SmallVector<mlir::Value> newAccs(mPerWarp * NT + mPerWarp * RT);
        for (int mi = 0; mi < mPerWarp; ++mi)
          for (int ni = 0; ni < NT; ++ni)
            newAccs[mi * NT + ni] = SimdgroupMultiplyAccumulateOp::create(
                b, l, matTy, args[mi * NT + ni], aTiles[mi], wTiles[ni]).getResult();
        for (int mi = 0; mi < mPerWarp; ++mi)
          for (int rk = 0; rk < RT; ++rk)
            newAccs[nBase + mi * RT + rk] = SimdgroupMultiplyAccumulateOp::create(
                b, l, matTy, args[nBase + mi * RT + rk], aTiles[mi],
                aMatTiles[rk]).getResult();
        mlir::scf::YieldOp::create(b, l, newAccs);
      });

  // --- Epilogue: per output tile (mi,ni), dot3 = sum_rk acc1[mi][rk]·transB[rk][ni],
  //     fused-stored as acc0[mi][ni] + scale*dot3. ---
  mlir::Value b3Buf = bridgePtrToMemref(
      builder, loc, unwrapPtrToKernelArg(loadB3.getPtr()), elemTy);
  mlir::Value strideB3Val = emitStrideOperand(
      builder, loc, ui32, findAxisStride(loadB3.getPtr(), 0));
  int b3NAxis = b3T ? 0 : 1;
  mlir::Value b3NScalar = findOriginScalarInPtrChain(loadB3.getPtr(), b3NAxis);
  mlir::Value b3RExt, b3CExt;
  std::tie(b3RExt, b3CExt) = extentsFor(loadB3);

  mlir::Value oBuf = bridgePtrToMemref(
      builder, loc, unwrapPtrToKernelArg(store.getPtr()), elemTy);
  mlir::Value strideOVal = emitStrideOperand(
      builder, loc, ui32, findAxisStride(store.getPtr(), 0));
  OriginPair oOrig = extractOriginPair<mlir::triton::StoreOp>(store);
  llvm::SmallVector<mlir::Value, 2> oPartial;
  if (storeExt.mExtent && storeExt.nExtent) {
    oPartial.push_back(emitExtentUi32(builder, loc, ui32, storeExt.mExtent));
    oPartial.push_back(emitExtentUi32(builder, loc, ui32, storeExt.nExtent));
  }

  for (int mi = 0; mi < mPerWarp; ++mi) {
    mlir::Value oRow = originForRuntimeTile(oOrig.row, mTileIdxVal(mi));
    for (int ni = 0; ni < NT; ++ni) {
      mlir::Value dot3acc =
          SimdgroupMatrixZeroOp::create(builder, loc, matTy).getResult();
      for (int rk = 0; rk < RT; ++rk) {
        // transB[rk][ni]: b [N,R] (b3T) staged transposed at
        // (N-origin+ni*8, R-offset=rk*8); non-transposed b [R,N] at (rk*8, N+ni*8).
        mlir::Value transB =
            b3T ? emitStagedLoad(builder, loc, matTy, b3Buf,
                                 tileOrigin(b3NScalar, ni), constUi32(rk * 8),
                                 strideB3Val, true, b3RExt, b3CExt, widx)
                : emitStagedLoad(builder, loc, matTy, b3Buf, constUi32(rk * 8),
                                 tileOrigin(b3NScalar, ni), strideB3Val, false,
                                 b3RExt, b3CExt, widx);
        dot3acc = SimdgroupMultiplyAccumulateOp::create(
                      builder, loc, matTy, dot3acc,
                      loop.getResult(nBase + mi * RT + rk), transB).getResult();
      }
      SimdgroupFusedStoreOp::create(
          builder, loc, loop.getResult(mi * NT + ni), dot3acc, scaleScalar,
          oBuf, oRow, tileOrigin(oOrig.col, ni), strideOVal, oPartial,
          /*warp_index=*/mlir::ValueRange(widxRange));
    }
  }

  // Erase the old epilogue chain (store, any convert_layouts, addf, mulf, dot3,
  // trans) use-before-def, then the loop. Recurse only through epilogue op
  // kinds so the loop and the leftover dead loads/splats are left for DCE.
  std::function<void(mlir::Value)> eraseIfDead = [&](mlir::Value v) {
    auto *op = v.getDefiningOp();
    if (!op || !op->use_empty()) return;
    if (!mlir::isa<mlir::triton::gpu::ConvertLayoutOp, mlir::arith::MulFOp,
                   mlir::arith::AddFOp, mlir::triton::DotOp,
                   mlir::triton::TransOp>(op))
      return;
    llvm::SmallVector<mlir::Value> operands(op->getOperands());
    op->erase();
    for (auto o : operands) eraseIfDead(o);
  };
  mlir::Value stored = store.getValue();
  store.erase();
  eraseIfDead(stored);
  forOp.erase();
  return mlir::success();
}

static mlir::LogicalResult tryUnrollKLoopDot(mlir::triton::DotOp dot) {
  auto forOp = dot->getParentOfType<mlir::scf::ForOp>();
  if (!forOp) return mlir::failure();

  // Contract: exactly one iter_arg (the accumulator).
  if (forOp.getNumRegionIterArgs() != 1) return mlir::failure();
  auto iterArg = forOp.getRegionIterArg(0);
  if (dot.getC() != iterArg) return mlir::failure();

  // Yield must yield exactly the dot's result.
  auto yieldOp =
      mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
  if (yieldOp.getNumOperands() != 1) return mlir::failure();
  if (yieldOp.getOperand(0) != dot.getResult()) return mlir::failure();

  // scf.for's result must feed exactly one tt.store.
  if (!forOp.getResult(0).hasOneUse()) return mlir::failure();
  auto store = mlir::dyn_cast<mlir::triton::StoreOp>(
      *forOp.getResult(0).getUsers().begin());
  if (!store || store.getMask()) return mlir::failure();

  // Bounds must be static.
  auto getConstInt =
      [](mlir::Value v) -> std::optional<int64_t> {
    auto c = v.getDefiningOp<mlir::arith::ConstantOp>();
    if (!c) return std::nullopt;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return intAttr.getInt();
    return std::nullopt;
  };
  auto lo = getConstInt(forOp.getLowerBound());
  auto hi = getConstInt(forOp.getUpperBound());
  auto st = getConstInt(forOp.getStep());
  if (!lo || !hi || !st || *st == 0) return mlir::failure();
  int64_t N = (*hi - *lo) / *st;
  if (N < 1 || N > 8) return mlir::failure();

  // A/B feeders must be direct tt.load inside the loop body.
  auto aLoad = dot.getA().getDefiningOp<mlir::triton::LoadOp>();
  auto bLoad = dot.getB().getDefiningOp<mlir::triton::LoadOp>();
  if (!aLoad || !bLoad) return mlir::failure();
  if (aLoad.getMask() || bLoad.getMask()) return mlir::failure();

  // Element type and shape gates (same as v1).
  auto resTy = mlir::dyn_cast<mlir::RankedTensorType>(dot.getType());
  if (!resTy) return mlir::failure();
  auto elemTy = resTy.getElementType();
  if (!elemTy.isF32()) return mlir::failure();
  auto shape = resTy.getShape();
  if (shape.size() != 2 || shape[0] != 8 || shape[1] != 8)
    return mlir::failure();

  mlir::Value aPtr = unwrapPtrToKernelArg(aLoad.getPtr());
  mlir::Value bPtr = unwrapPtrToKernelArg(bLoad.getPtr());
  mlir::Value cPtr = unwrapPtrToKernelArg(store.getPtr());

  // Validate the iter_arg init: accept `arith.constant dense<0.0>` or
  // `tt.load %c_ptrs` whose unwrapped ptr matches cPtr (Session 4b — see
  // `.omc/specs/deep-interview-metal-matmul-session4b-memory-c-init.md`).
  // Other init shapes fall through to failure (the dot will not unroll).
  mlir::Value initOperand = forOp.getInits()[0];
  if (initOperand.getDefiningOp<mlir::arith::ConstantOp>()) {
    // dense<0.0> path — no further check; cBuf load below covers it.
  } else if (auto loadInit =
                 initOperand.getDefiningOp<mlir::triton::LoadOp>()) {
    if (loadInit.getMask()) return mlir::failure();
    if (unwrapPtrToKernelArg(loadInit.getPtr()) != cPtr) return mlir::failure();
    // Same-buffer tt.load init: emitted MSL is identical to dense<0.0> path
    // since the C-init simdgroup_load below already loads from cBuf.
  } else {
    return mlir::failure();
  }

  // Emit IR before the scf.for.
  mlir::OpBuilder builder(forOp);
  auto loc = forOp.getLoc();
  auto ctx = builder.getContext();
  auto ui32 = builder.getIntegerType(32, /*isSigned=*/false);
  auto matTy = MetalSimdgroupMatrixType::get(ctx, 8, 8, elemTy);

  auto zero =
      ConstantOp::create(builder, loc, builder.getIntegerAttr(ui32, 0));
  auto strideC =
      ConstantOp::create(builder, loc, builder.getIntegerAttr(ui32, 8));

  mlir::Value aBuf = bridgePtrToMemref(builder, loc, aPtr, elemTy);
  mlir::Value bBuf = bridgePtrToMemref(builder, loc, bPtr, elemTy);
  mlir::Value cBuf = bridgePtrToMemref(builder, loc, cPtr, elemTy);

  // C-init load (the for's iter_arg init typically dense<0.0>; we still
  // emit a simdgroup_load of cBuf since the v4 contract keeps addresses
  // hardcoded — semantically lit-only this session).
  auto cInitLoad = SimdgroupLoadOp::create(builder, loc, matTy, cBuf,
                                            zero.getResult(), zero.getResult(),
                                            strideC.getResult());
  mlir::Value acc = cInitLoad.getResult();

  // Unrolled chain.
  for (int64_t i = 0; i < N; ++i)
    acc = emitOneMA(builder, loc, matTy, aBuf, bBuf, zero.getResult(),
                     strideC.getResult(), acc);

  // Final store. No partial_extents: full 8x8 tile path.
  SimdgroupStoreOp::create(builder, loc, acc, cBuf, zero.getResult(),
                            zero.getResult(), strideC.getResult(),
                            /*partial_extents=*/mlir::ValueRange{});

  // Erase originals.
  store.erase();
  // If iter_arg init was a tt.load(C), it's now dead — walk back and erase
  // the tt.load + its tt.addptr + tt.splat chain so the main pass doesn't
  // see leftover tt.* ops that cause spurious unrealized_conversion_casts.
  mlir::Operation *initLoadOp = initOperand.getDefiningOp();
  forOp.erase();
  if (initLoadOp && initLoadOp->use_empty()) {
    if (auto cLoad = mlir::dyn_cast<mlir::triton::LoadOp>(initLoadOp)) {
      auto cAddr = cLoad.getPtr();
      cLoad.erase();
      if (auto cAddptr = cAddr.getDefiningOp<mlir::triton::AddPtrOp>()) {
        if (cAddptr->use_empty()) {
          auto cSplat = cAddptr.getPtr();
          cAddptr.erase();
          if (auto splat = cSplat.getDefiningOp<mlir::triton::SplatOp>()) {
            if (splat->use_empty()) splat.erase();
          }
        }
      }
    }
  }
  return mlir::success();
}

static void rewriteSingleDot(mlir::triton::DotOp dot) {
  // Matmul track session 4c-3: try canonical 3-iter_arg unroll first
  // (real Triton matmul shape with a_ptrs/b_ptrs/acc iter_args).
  if (mlir::succeeded(tryUnrollCanonical3IterArgDot(dot))) return;
  // W2a: canonical 3-iter_arg K-loop with a RUNTIME trip count (emits an
  // scf.for of simdgroup_multiply_accumulate rather than a static unroll).
  if (mlir::succeeded(tryRuntimeKLoopCanonicalDot(dot))) return;
  // W2b: recompute-from-IV runtime-K loop (accumulator-only iter_arg; the
  // LoRA-style shape whose addresses are rebuilt from the induction var).
  if (mlir::succeeded(tryRuntimeKLoopRecomputeDot(dot))) return;
  // Session 4: try 1-iter_arg K-loop unroll.
  if (mlir::succeeded(tryUnrollKLoopDot(dot))) return;

  // Single-iter path. The dot's static result is always 8x8 (BLOCK = 8);
  // the partial-tile signal comes from a masked `tt.store` below.
  auto resTy = mlir::dyn_cast<mlir::RankedTensorType>(dot.getType());
  if (!resTy || resTy.getRank() != 2) return;
  auto elemTy = resTy.getElementType();
  if (!elemTy.isF32()) return;
  auto shape = resTy.getShape();
  if (shape.size() != 2 || shape[0] != 8 || shape[1] != 8) return;

  auto aLoad = dot.getA().getDefiningOp<mlir::triton::LoadOp>();
  auto bLoad = dot.getB().getDefiningOp<mlir::triton::LoadOp>();
  if (!aLoad || !bLoad) return;
  if (aLoad.getMask() || bLoad.getMask()) return;

  // Result must feed a single tt.store.
  if (!dot.getResult().hasOneUse()) return;
  auto store = mlir::dyn_cast<mlir::triton::StoreOp>(
      *dot.getResult().getUsers().begin());
  if (!store) return;
  // AC2: allow masked stores of the canonical 2D form. If the mask doesn't
  // decompose into (offs_m < M) & (offs_n < N), bail.
  MaskExtents maskExtents;
  if (store.getMask()) {
    maskExtents = extractMaskExtents(store.getMask());
    if (!maskExtents.mExtent || !maskExtents.nExtent) return;
  }

  // Walk a/b/c load ptr operands back to the kernel-arg `!tt.ptr<f32>`.
  // Shares the helper used by the canonical-3-iter_arg path so 2D matmul
  // pointer shapes (`addptr(broadcast(addptr(splat(arg), row), col))`)
  // resolve to the kernel block-arg instead of bailing at the broadcast.
  mlir::Value aPtr = unwrapPtrToKernelArg(aLoad.getPtr());
  mlir::Value bPtr = unwrapPtrToKernelArg(bLoad.getPtr());
  mlir::Value cPtr = unwrapPtrToKernelArg(store.getPtr());

  mlir::OpBuilder builder(dot);
  auto loc = dot.getLoc();
  auto ctx = builder.getContext();
  auto ui32 = builder.getIntegerType(32, /*isSigned=*/false);
  auto matTy = MetalSimdgroupMatrixType::get(ctx, /*rows=*/8, /*cols=*/8, elemTy);

  // Chain-aware stride extraction (matches tryUnrollCanonical3IterArgDot path).
  // The canonical Triton 2D-matmul pointer is a 2-level
  // `addptr(broadcast(addptr(splat(arg), row_off)), col_off)` chain whose
  // row-stride splat lives in the INNER addptr's offset. The flat
  // `extractStrideFromAddPtr` walks only the outer addptr and returns null
  // for these shapes, which `emitStrideOperand` falls back to `metal.constant
  // 8`. For (M,N,K) where N != 8 or M != 8, that hardcoded 8 silently
  // miscompiles. Walk the full chain instead.
  mlir::Value strideAVal = emitStrideOperand(
      builder, loc, ui32, findStrideSplatSourceInPtrChain(aLoad.getPtr()));
  mlir::Value strideBVal = emitStrideOperand(
      builder, loc, ui32, findStrideSplatSourceInPtrChain(bLoad.getPtr()));
  mlir::Value strideCVal = emitStrideOperand(
      builder, loc, ui32, findStrideSplatSourceInPtrChain(store.getPtr()));
  OriginPair aOrig = extractOriginPair<mlir::triton::LoadOp>(aLoad);
  OriginPair bOrig = extractOriginPair<mlir::triton::LoadOp>(bLoad);
  OriginPair cOrig = extractOriginPair<mlir::triton::StoreOp>(store);
  mlir::Value aRowOrigin = emitOriginOperand(builder, loc, ui32, aOrig.row);
  mlir::Value aColOrigin = emitOriginOperand(builder, loc, ui32, aOrig.col);
  mlir::Value bRowOrigin = emitOriginOperand(builder, loc, ui32, bOrig.row);
  mlir::Value bColOrigin = emitOriginOperand(builder, loc, ui32, bOrig.col);
  mlir::Value cRowOrigin = emitOriginOperand(builder, loc, ui32, cOrig.row);
  mlir::Value cColOrigin = emitOriginOperand(builder, loc, ui32, cOrig.col);

  mlir::Value aBuf = bridgePtrToMemref(builder, loc, aPtr, elemTy);
  mlir::Value bBuf = bridgePtrToMemref(builder, loc, bPtr, elemTy);
  mlir::Value cBuf = bridgePtrToMemref(builder, loc, cPtr, elemTy);

  auto aTile = SimdgroupLoadDeviceStagedOp::create(
      builder, loc, matTy, aBuf, aRowOrigin, aColOrigin, strideAVal,
      mlir::ValueRange{});
  auto bTile = SimdgroupLoadDeviceStagedOp::create(
      builder, loc, matTy, bBuf, bRowOrigin, bColOrigin, strideBVal,
      mlir::ValueRange{});

  // C-init: emit `metal.simdgroup_matrix_zero` when the dot's accumulator
  // operand is `dense<0.0>` (the canonical `acc = tl.zeros(...)` Triton
  // pattern, which after trivial scf.for elimination becomes the dot's
  // direct C operand). Otherwise fall back to loading from the C buffer.
  mlir::Value acc;
  if (auto cst = dot.getC().getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue())) {
      if (dense.isSplat() && dense.getSplatValue<mlir::APFloat>().isZero())
        acc = SimdgroupMatrixZeroOp::create(builder, loc, matTy).getResult();
    }
  }
  if (!acc) {
    // f32-only: cBuf is bridged with the accumulator elem type (f32). If a
    // future non-zero, non-f32 accumulator surface is added, this direct
    // device load also needs to switch to SimdgroupLoadDeviceStagedOp.
    acc = SimdgroupLoadOp::create(builder, loc, matTy, cBuf, cRowOrigin,
                                   cColOrigin, strideCVal).getResult();
  }
  auto cResult = SimdgroupMultiplyAccumulateOp::create(
      builder, loc, matTy, acc, aTile.getResult(), bTile.getResult());
  llvm::SmallVector<mlir::Value, 2> partialExtents;
  if (maskExtents.mExtent && maskExtents.nExtent) {
    auto toUi32 = [&](mlir::Value v) -> mlir::Value {
      if (v.getType() == ui32) return v;
      return mlir::UnrealizedConversionCastOp::create(builder, loc, ui32, v)
          .getResult(0);
    };
    partialExtents.push_back(toUi32(maskExtents.mExtent));
    partialExtents.push_back(toUi32(maskExtents.nExtent));
  }
  SimdgroupStoreOp::create(builder, loc, cResult.getResult(), cBuf, cRowOrigin,
                            cColOrigin, strideCVal, partialExtents);

  // Erase originals in reverse-dependency order. The accumulator init op
  // (typically arith.constant dense<0.0>) becomes dead and is cleaned up
  // by the existing dead-arith pass below.
  store.erase();
  dot.erase();
  if (aLoad.use_empty()) aLoad.erase();
  if (bLoad.use_empty()) bLoad.erase();
}

// L1d3: dot-feeding ttg.convert_layout preempt. For every tt.dot whose A
// and B operands are `ttg.convert_layout(#blocked -> #dot_op)` of a
// `tt.load`, rewire the dot operand to the load directly and erase the cvt
// when its only use is gone. This makes the dot body match the canonical
// shape that `tryUnrollCanonical3IterArgDot` (:3066), `tryUnrollKLoopDot`
// (:3225) and `rewriteSingleDot` (:3356) expect, letting the existing
// dispatcher reach all 3 paths without modifying the cvt-legalisation
// gate at :3552-3555.
//
// Safety: `DotOp::verify()` returns success when both operand encodings
// are absent (Ops.cpp:271-272), and `verifyDotOpEncodingCompatibility`
// returns success when neither operand carries a `DotOperandEncodingAttr`
// (Dialect.cpp:3163-3164). The pre-walk strips BOTH A and B together
// (`if (!newA || !newB) return;`), so after rewire both operands carry
// `#blocked` and both safety nets fire.
//
// Note re. dot.getC() (accumulator): only A and B are rewired here. acc
// is typically an scf.for iter_arg `BlockArgument` (not a defining op).
// True Variant 3b (cvt on the acc-init OUTSIDE the loop, passed through
// an iter_arg) requires rewriting the scf.for iter_arg type and is
// deferred — see ADR follow-up #5 in
// `.omc/plans/l1d3-matmul-convert-layout-preempt-consensus.md`.
//
// Spec/plan: `.omc/specs/deep-interview-l1d3-matmul-convert-layout-preempt.md`
//            `.omc/specs/l1d3-broader-staged-transpose-dot-bridge.md`
static void preprocessDotCvtChains(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::triton::gpu::ConvertLayoutOp> deadCvts;
  moduleOp.walk([&](mlir::triton::DotOp dot) {
    // Only strip cvts when the downstream matmul track matchers will
    // accept the dot's element type and shape. These mirror the gates at
    // TritonGPUToMetal.cpp:3141, :3272, :3367. Without this guard, an
    // fp16 dot would have its cvts stripped, leaving a non-f32 dot the
    // matchers reject — which then crashes the legalization pipeline.
    //
    // AC4 (v6): shape gate lifted from `== 8x8` to `% 8 == 0` so multi-
    // tile dots (per-program M/N grids) admit the same operand-cvt and
    // C-side-cvt rewrites. See `.omc/plans/ac4-multiwarp.md` (Option α).
    auto dotResTy =
        mlir::dyn_cast<mlir::RankedTensorType>(dot.getType());
    if (!dotResTy) return;
    if (!dotResTy.getElementType().isF32()) return;
    auto dotShape = dotResTy.getShape();
    if (dotShape.size() != 2 || dotShape[0] % 8 != 0 || dotShape[1] % 8 != 0)
      return;

    auto peel = [&](mlir::Value v) -> mlir::Value {
      auto cvt = v.getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>();
      if (!cvt) return mlir::Value();
      // A multi-use cvt (a shared operand fed to several dots, e.g. LoRA's `x`)
      // is fine: each dot's use is rewired to the load one at a time as the
      // walk visits it, and the cvt is dropped once its last use is gone.
      auto dstTy = mlir::dyn_cast<mlir::RankedTensorType>(cvt.getType());
      if (!dstTy) return mlir::Value();
      auto dstEnc = mlir::dyn_cast_or_null<
          mlir::triton::gpu::DotOperandEncodingAttr>(dstTy.getEncoding());
      if (!dstEnc) return mlir::Value();
      mlir::Value src = cvt.getSrc();
      // Accept `cvt(load)` or `cvt(trans(load))`. The latter is the W1
      // transposed-operand matmul (`tl.dot(a, tl.trans(b))`): keep the
      // `tt.trans` in place — the dot matcher folds it into a transposed
      // simdgroup load — and strip only the cvt.
      if (src.getDefiningOp<mlir::triton::LoadOp>()) return src;
      if (auto tr = src.getDefiningOp<mlir::triton::TransOp>())
        if (tr.getSrc().getDefiningOp<mlir::triton::LoadOp>()) return src;
      // W2c: a loop-carried accumulator feeding a post-loop dot (LoRA's `acc1`
      // → dot #3). The fused-epilogue matcher consumes the scf.for result
      // directly.
      if (src.getDefiningOp<mlir::scf::ForOp>()) return src;
      return mlir::Value();
    };

    mlir::Value newA = peel(dot.getA());
    mlir::Value newB = peel(dot.getB());
    if (!newA || !newB) return;

    auto cvtA = dot.getA().getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>();
    auto cvtB = dot.getB().getDefiningOp<mlir::triton::gpu::ConvertLayoutOp>();
    dot.setOperand(0, newA);
    dot.setOperand(1, newB);
    if (cvtA && cvtA.use_empty()) deadCvts.push_back(cvtA);
    if (cvtB && cvtB.use_empty()) deadCvts.push_back(cvtB);

    // AC4 (v6) BLOCKER #2: peel the C-side cvt that sits between the dot
    // result (possibly via scf.for iter_arg/yield) and a `tt.store`.
    // Upstream picks a matmul-optimized #blocked encoding on the dot
    // result that differs from the store-side encoding at multi-tile
    // shapes (sizePerThread=[4,4] vs [1,4]); L1d2's envelope classifier
    // rejects this rank-2 blocked↔blocked cvt because the source
    // sizePerThread > 1. We can safely re-encode the dot result to match
    // the store side because the AC4 matcher (next pass) emits
    // SimdgroupStoreOp with explicit (row, col, stride) origins — the
    // MLIR-level tensor encoding has no MSL-level consequence.
    //
    // Chain (probe-verified): dot → scf.yield → scf.for result → cvt →
    // tt.store. The scf.for has 3 iter_args; the accumulator's index is
    // determined dynamically by walking from dot.getResult() back through
    // the yield. Init-side constant (typically arith.constant dense<0.0>)
    // gets re-encoded in-place by reshape() on its DenseElementsAttr.
    if (!dot.getResult().hasOneUse()) return;
    mlir::OpOperand &dotUse = *dot.getResult().getUses().begin();
    auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(dotUse.getOwner());
    if (!yieldOp) return;
    auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(yieldOp->getParentOp());
    if (!forOp) return;
    unsigned itArgIdx = dotUse.getOperandNumber();
    mlir::Value forResVal = forOp.getResult(itArgIdx);
    if (!forResVal.hasOneUse()) return;
    auto cvtC = mlir::dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(
        forResVal.getUses().begin()->getOwner());
    if (!cvtC) return;
    if (!cvtC->hasOneUse()) return;
    auto storeOp = mlir::dyn_cast<mlir::triton::StoreOp>(
        cvtC->use_begin()->getOwner());
    if (!storeOp) return;

    auto srcCTy =
        mlir::dyn_cast<mlir::RankedTensorType>(cvtC.getSrc().getType());
    auto dstCTy =
        mlir::dyn_cast<mlir::RankedTensorType>(cvtC.getType());
    if (!srcCTy || !dstCTy) return;
    if (srcCTy.getEncoding() == dstCTy.getEncoding()) return;
    auto newEnc = dstCTy.getEncoding();
    auto newDotTy = mlir::RankedTensorType::get(
        srcCTy.getShape(), srcCTy.getElementType(), newEnc);

    // (1) dot result.
    dot.getResult().setType(newDotTy);
    // (2) scf.for iter_arg.
    forOp.getRegionIterArgs()[itArgIdx].setType(newDotTy);
    // (3) scf.for result.
    forResVal.setType(newDotTy);
    // (4) init operand. If it's a dense-constant, re-encode in place; if
    //     it's anything else (rare), insert an outer cvt as a fallback.
    mlir::Value initVal = forOp.getInitArgs()[itArgIdx];
    if (auto constOp =
            initVal.getDefiningOp<mlir::arith::ConstantOp>()) {
      auto attr = constOp.getValueAttr();
      if (auto denseAttr =
              mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
        auto newAttr = denseAttr.reshape(newDotTy);
        constOp.setValueAttr(newAttr);
        constOp.getResult().setType(newDotTy);
      } else {
        // Fallback: leave init as-is, insert a cvt before scf.for.
        mlir::OpBuilder b(forOp);
        auto castedInit =
            mlir::triton::gpu::ConvertLayoutOp::create(
                b, forOp.getLoc(), newDotTy, initVal);
        forOp.getInitArgsMutable()[itArgIdx].assign(castedInit.getResult());
      }
    } else {
      mlir::OpBuilder b(forOp);
      auto castedInit = mlir::triton::gpu::ConvertLayoutOp::create(
          b, forOp.getLoc(), newDotTy, initVal);
      forOp.getInitArgsMutable()[itArgIdx].assign(castedInit.getResult());
    }
    // (5) Erase the now-identity cvt.
    cvtC.getResult().replaceAllUsesWith(cvtC.getSrc());
    deadCvts.push_back(cvtC);
  });
  // Defensive re-check at erasure time guards the unlikely aliased-cvt
  // (self-dot) case where the same cvt feeds both A and B and would be
  // pushed twice.
  for (auto cvt : deadCvts) {
    if (cvt.use_empty()) cvt.erase();
  }
}

static void preprocessDotChains(mlir::ModuleOp moduleOp) {
  // Group dots by their enclosing scf.for. A loop carrying >1 dot is a
  // multi-accumulator loop that must be rewritten atomically (one call erases
  // all of its dots), so it cannot go through the per-dot worklist — a dangling
  // sibling DotOp would be dereferenced. Standalone dots and single-dot loops
  // keep the per-dot dispatch.
  // Pass 1: multi-dot loops, in walk order. These are rewritten atomically —
  // one call erases all of the loop's dots, and the fused-LoRA epilogue also
  // consumes a POST-loop dot (dot #3). A per-dot worklist can't express that,
  // so loops go first and a fresh re-walk (pass 2) picks up everything not yet
  // consumed.
  llvm::SmallVector<mlir::scf::ForOp> loopOrder;
  llvm::DenseMap<mlir::Operation *, unsigned> dotCount;
  moduleOp.walk([&](mlir::triton::DotOp dot) {
    if (auto forOp = dot->getParentOfType<mlir::scf::ForOp>()) {
      auto *op = forOp.getOperation();
      if (!dotCount.count(op)) loopOrder.push_back(forOp);
      dotCount[op]++;
    }
  });
  for (auto forOp : loopOrder) {
    if (dotCount[forOp.getOperation()] < 2) continue;  // single-dot → pass 2
    if (mlir::succeeded(tryFusedLoRAEpilogue(forOp))) continue;
    (void)tryRuntimeKLoopRecomputeMultiDot(forOp);
    // On failure the loop's dots survive; pass 2 dispatches them (and they
    // fail to legalize -> clean error).
  }

  // Pass 2: everything remaining — single-dot loops and standalone dots.
  llvm::SmallVector<mlir::triton::DotOp> dots;
  moduleOp.walk([&](mlir::triton::DotOp dot) { dots.push_back(dot); });
  for (auto dot : dots) rewriteSingleDot(dot);
}

// Collapse a blocked<->blocked `ttg.convert_layout` whose source is a
// self-contained producer cone (loads, elementwise arith, addptr / splat /
// make_range / expand_dims / broadcast, mask cmpi) by rewriting that cone from
// the source encoding to the DESTINATION encoding. Both encodings are valid
// bijections over the same logical elements, so once the producers and the
// consuming op agree on ONE encoding the index math matches and the cvt
// collapses to an identity (handled by `ConvertLayoutLowering`'s passthrough).
// The Metal Load/Store lowerings derive each op's per-(thread, iter) index from
// its OWN layout (`tileFromTensor`), so a divergent cvt is a genuine repack —
// naive passthrough would permute the output.
//
// Two origins, same fix:
//   * Rank-1 (mixed pointer alignment): aligned pointers (tt.divisibility = 16)
//     vectorize to sizePerThread > 1 while an unaligned pointer (e.g. an MPS
//     tensor sliced to base[7:]) stays at sizePerThread = 1; the frontend
//     bridges the compute value to the store with `cvt #blockedN -> #blocked1`.
//   * Rank-2 (masked transpose, `out[x,y] = in[y,x]`): the loaded tile is
//     bridged from a row-major #blocked (order [1,0]) to a column-major
//     #blocked1 (order [0,1], sizePerThread swapped, E>1). The cvt source is a
//     pure gather whose layout is free to choose, so re-encoding the cone to
//     the dst layout turns the transpose into a direct gather/scatter — no
//     threadgroup staging (the otherwise-deferred L1d3 general repack path).
// Returns true iff the cvt was normalized.
// Remap a producer-cone encoding from the source blocked layout to the
// destination blocked layout. The blocked encoding itself maps to dstEnc; a
// rank-(R-1) slice<dim, parent=srcEnc> (the operands of make_range /
// expand_dims that thread a rank-2 cone's row/col index math) maps to
// slice<dim, parent=dstEnc>. Returns a null Attribute for encodings outside
// the cone (scalars, the other side's layout, dot operands, ...).
// `srcEnc` may itself be a slice: a rank-1 value that feeds BOTH a 2D tile
// (via tt.expand_dims, which forces the slice encoding) and a live rank-1
// store reaches the store through `cvt slice<dim,parent=B> -> #blocked1`. The
// two layouts disagree about which element a thread holds — under
// slice<dim=1,parent=B> thread t holds row t/N, under #blocked1 it holds row t
// — so that cvt is a real data relabel and the cone must be re-encoded.
static mlir::Attribute
remapDivergentConeEncoding(mlir::Attribute enc, mlir::Attribute srcEnc,
                           mlir::Attribute dstEnc) {
  if (enc == srcEnc)
    return dstEnc;
  auto srcBlocked =
      mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(srcEnc);
  auto dstBlocked =
      mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(dstEnc);
  if (srcBlocked && dstBlocked)
    if (auto slice =
            mlir::dyn_cast_or_null<mlir::triton::gpu::SliceEncodingAttr>(enc))
      if (slice.getParent() == srcBlocked)
        return mlir::triton::gpu::SliceEncodingAttr::get(
            srcBlocked.getContext(), slice.getDim(), dstBlocked);
  return nullptr;
}

// A cone value we must NOT re-encode, because its type is not ours to choose:
// `tt.reduce`'s result encoding is inferred from its 2D source (axis=1 over
// #blocked yields exactly slice<dim=1, parent=that blocked>), so re-typing it
// makes the op invalid. Such values become BOUNDARIES: the cone is re-encoded
// around them and a fresh `cvt` bridges each one back in.
//
// Leaving a cvt there is correct rather than a punt: `ReduceLowering` reads its
// result out of `rowBuf[emitPerIterIndex(*outTile)]`, and `outTile` is taken
// from the first blocked-typed user — i.e. the reduce already emits its output
// under the DESTINATION layout's indexing. The bridge cvt is a genuine scalar
// identity, which is the one case `isScalarIdentityConvert` was written for.
static bool isConeBoundaryValue(mlir::Value v) {
  auto *def = v.getDefiningOp();
  return !def || mlir::isa<mlir::triton::ReduceOp, mlir::triton::ScanOp>(def);
}

static bool
normalizeBlockedDivergentCvt(mlir::triton::gpu::ConvertLayoutOp cvt) {
  auto srcRtt = mlir::dyn_cast<mlir::RankedTensorType>(cvt.getSrc().getType());
  auto dstRtt =
      mlir::dyn_cast<mlir::RankedTensorType>(cvt.getResult().getType());
  if (!srcRtt || !dstRtt || srcRtt.getRank() != dstRtt.getRank() ||
      srcRtt.getShape() != dstRtt.getShape() ||
      srcRtt.getElementType() != dstRtt.getElementType())
    return false;
  mlir::Attribute srcEnc = srcRtt.getEncoding();
  auto dstEnc = mlir::dyn_cast_or_null<mlir::triton::gpu::BlockedEncodingAttr>(
      dstRtt.getEncoding());
  if (!dstEnc || !srcEnc || srcEnc == dstEnc)
    return false;
  // src is either a blocked layout (the original rank-2 transpose / rank-1
  // compute-to-store repack) or a slice of one (the rank-1 value that also
  // feeds a 2D tile). Anything else is not ours.
  if (!mlir::isa<mlir::triton::gpu::BlockedEncodingAttr,
                 mlir::triton::gpu::SliceEncodingAttr>(srcEnc))
    return false;

  // Collect the backward cone of src-encoded values feeding the cvt source:
  // the src encoding plus, when src is blocked, any slice<parent=srcEnc> (a
  // rank-2 cone threads its row/col index math through slice-encoded
  // make_range/expand_dims).
  //
  // `boundaries` are cone values whose type we may not rewrite (tt.reduce /
  // tt.scan results, block arguments). They terminate the walk and get a
  // bridging cvt after the re-encode below.
  llvm::SmallVector<mlir::Value, 32> wl{cvt.getSrc()};
  llvm::SmallPtrSet<mlir::Value, 32> inCone;
  llvm::SmallVector<mlir::Value, 32> ordered;
  llvm::SmallVector<mlir::Value, 4> boundaries;
  llvm::SmallPtrSet<mlir::Value, 4> boundarySet;
  while (!wl.empty()) {
    auto v = wl.pop_back_val();
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!rt || !remapDivergentConeEncoding(rt.getEncoding(), srcEnc, dstEnc))
      continue; // scalar operands / other encodings terminate the walk
    if (isConeBoundaryValue(v)) {
      if (boundarySet.insert(v).second)
        boundaries.push_back(v);
      continue;
    }
    if (!inCone.insert(v).second)
      continue;
    ordered.push_back(v);
    if (auto *def = v.getDefiningOp())
      for (auto operand : def->getOperands())
        wl.push_back(operand);
  }
  if (ordered.empty())
    return false;

  // Safety: only rewrite self-contained cones. Every cone value's users must be
  // the cvt itself or another cone op; otherwise an external op depends on the
  // source encoding and rewriting it in place would corrupt them.
  llvm::SmallPtrSet<mlir::Operation *, 32> coneOps;
  for (auto v : ordered)
    if (auto *d = v.getDefiningOp())
      coneOps.insert(d);
  auto usedExternally = [&](mlir::Value v) {
    for (auto *user : v.getUsers())
      if (user != cvt.getOperation() && !coneOps.count(user))
        return true;
    return false;
  };
  // A cheap, side-effect-free LEAF (splat / make_range / constant) shared with
  // ops outside the cone — e.g. `splat(N)` (the mask bound) CSE'd across a
  // kernel's loops — would otherwise force a bail. CLONE it into a cone-local
  // copy and redirect the cone's uses to the clone, leaving the original for the
  // external consumers, so the in-place re-encode below stays a valid repack.
  //
  // For a SLICE source the list also admits tt.addptr / tt.load. When one 1D
  // load feeds BOTH a 2D tile and a live 1D store, the two consumers genuinely
  // need different thread->element maps, so no single encoding serves both —
  // duplicating the load is the only correct answer, and it is exactly what
  // Triton itself does when no reduce result forces the two paths to share an
  // encoding. A load is a pure read: the copy is redundant traffic, never a
  // semantic change.
  //
  // Kept OFF the blocked->blocked path on purpose. That path's contract is that
  // a cone shared with an outside consumer is NOT self-contained and must fall
  // through to the reject/staged-transpose classification; making its loads
  // clonable normalizes cones that are supposed to be rejected (the
  // convert_layout_reject_nontrivial / staged_transpose fixtures pin exactly
  // that boundary).
  const bool srcIsSlice =
      mlir::isa<mlir::triton::gpu::SliceEncodingAttr>(srcEnc);
  for (size_t i = 0; i < ordered.size(); ++i) {
    mlir::Value v = ordered[i];
    if (!usedExternally(v))
      continue;
    mlir::Operation *def = v.getDefiningOp();
    // For a slice source, ANY pure single-result op is cloneable (the index
    // math — addi/muli/cmpi over make_range — is routinely CSE'd between the
    // live rank-1 cone and the 2D tile cone, so restricting to leaves bails on
    // every real kernel), plus tt.load, which has read effects but is
    // idempotent.
    bool cloneable =
        def && (mlir::isa<mlir::triton::SplatOp, mlir::triton::MakeRangeOp,
                          mlir::arith::ConstantOp>(def) ||
                (srcIsSlice && def->getNumResults() == 1 &&
                 def->getNumRegions() == 0 &&
                 (mlir::isMemoryEffectFree(def) ||
                  mlir::isa<mlir::triton::LoadOp>(def))));
    if (!cloneable)
      return false; // non-leaf shared value — too risky to duplicate
    mlir::OpBuilder b(def);
    mlir::Operation *clone = b.clone(*def);
    mlir::Value cloneV = clone->getResult(0);
    v.replaceUsesWithIf(cloneV, [&](mlir::OpOperand &use) {
      mlir::Operation *owner = use.getOwner();
      return owner == cvt.getOperation() || coneOps.count(owner);
    });
    ordered[i] = cloneV; // rewrite the clone (cone-local), not the shared orig
    coneOps.erase(def);
    coneOps.insert(clone);
  }

  // Rewrite every cone value's encoding to the destination. Operand/result
  // encodings stay mutually consistent because the whole cone moves together;
  // MLIR does not re-verify between setType calls. A tensor-valued
  // arith.constant (e.g. a masked load's `other`) carries a typed
  // ElementsAttr that the verifier requires to match the result type, so
  // re-encode it in place via DenseElementsAttr::reshape (same data, new
  // encoding) — mirrors the dot-init re-encode at the preprocessDotCvtChains
  // site above.
  for (auto v : ordered) {
    auto rt = mlir::cast<mlir::RankedTensorType>(v.getType());
    auto newTy = mlir::RankedTensorType::get(
        rt.getShape(), rt.getElementType(),
        remapDivergentConeEncoding(rt.getEncoding(), srcEnc, dstEnc));
    if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue()))
        cst.setValueAttr(dense.reshape(mlir::cast<mlir::ShapedType>(newTy)));
    v.setType(newTy);
  }

  // Bridge each boundary back in. Its own type stayed put (a tt.reduce result
  // encoding is inferred from the reduce's 2D source and is not ours to
  // rewrite), so the now-dst-encoded cone ops that consume it need a cvt. Only
  // uses INSIDE the cone are redirected — the boundary's other consumers keep
  // reading it in its original layout.
  for (auto b : boundaries) {
    auto rt = mlir::cast<mlir::RankedTensorType>(b.getType());
    auto bridgeTy = mlir::RankedTensorType::get(
        rt.getShape(), rt.getElementType(),
        remapDivergentConeEncoding(rt.getEncoding(), srcEnc, dstEnc));
    mlir::OpBuilder bldr(cvt.getContext());
    if (auto *def = b.getDefiningOp())
      bldr.setInsertionPointAfter(def);
    else
      bldr.setInsertionPointToStart(b.getParentBlock());
    auto bridge = mlir::triton::gpu::ConvertLayoutOp::create(
        bldr, b.getLoc(), bridgeTy, b);
    b.replaceUsesWithIf(bridge.getResult(), [&](mlir::OpOperand &use) {
      return coneOps.count(use.getOwner());
    });
  }
  return true;
}

static void normalizeBlockedDivergentCvts(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::triton::gpu::ConvertLayoutOp> cvts;
  moduleOp.walk(
      [&](mlir::triton::gpu::ConvertLayoutOp cvt) { cvts.push_back(cvt); });
  for (auto cvt : cvts)
    normalizeBlockedDivergentCvt(cvt);
}

//===----------------------------------------------------------------------===//
// Flash-attention loop matcher (Phase 2).
//===----------------------------------------------------------------------===//
//
// Recognize the online-softmax flash-attention loop (Q@K^T -> masked softmax ->
// P@V with running-max/sum rescaling) and replace the whole loop + its
// divide-by-sum epilogue + masked store with a single `metal.flash_attention`
// op. Runs BEFORE the cvt classifier so the loop's dot-operand
// convert_layouts (which the matmul track can't absorb — dot B's A operand is a
// computed `exp`, see metal-flash-attention-plan.md §1a) never reach the L1d3
// reject. The op's emitter (Phase 1) renders the Phase-0-validated MSL body.
// Anchored on the leet-triton hard-mult_head_attention.py TTGIR shape.

// Trace a dot operand back through convert_layout / trans to its tt.load.
static mlir::triton::LoadOp faTraceToLoad(mlir::Value v) {
  while (v) {
    auto *def = v.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto ld = mlir::dyn_cast<mlir::triton::LoadOp>(def))
      return ld;
    if (mlir::isa<mlir::triton::gpu::ConvertLayoutOp, mlir::triton::TransOp>(
            def)) {
      v = def->getOperand(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

// Does the producer cone of `v` (through convert_layout) contain a tt.trans?
// Only the Q@K^T dot has a transposed operand; P@V does not.
static bool faConeHasTrans(mlir::Value v) {
  auto *def = v.getDefiningOp();
  while (def) {
    if (mlir::isa<mlir::triton::TransOp>(def))
      return true;
    if (mlir::isa<mlir::triton::gpu::ConvertLayoutOp>(def)) {
      def = def->getOperand(0).getDefiningOp();
      continue;
    }
    return false;
  }
  return false;
}

static mlir::LogicalResult tryFlashAttentionLoop(mlir::scf::ForOp forOp) {
  // (1) exactly 3 iter_args: one rank-2 f32 accumulator + two rank-1 f32 state.
  if (forOp.getNumRegionIterArgs() != 3)
    return mlir::failure();
  mlir::RankedTensorType accTy;
  int nRank1 = 0;
  for (mlir::Value init : forOp.getInitArgs()) {
    auto tt = mlir::dyn_cast<mlir::RankedTensorType>(init.getType());
    if (!tt || !tt.getElementType().isF32())
      return mlir::failure();
    if (tt.getRank() == 2) {
      if (accTy)
        return mlir::failure();
      accTy = tt;
    } else if (tt.getRank() == 1) {
      ++nRank1;
    } else {
      return mlir::failure();
    }
  }
  if (!accTy || nRank1 != 2)
    return mlir::failure();

  // (2) exactly two tt.dot and two tt.reduce in the body.
  llvm::SmallVector<mlir::triton::DotOp> dots;
  forOp.getBodyRegion().walk([&](mlir::triton::DotOp d) { dots.push_back(d); });
  if (dots.size() != 2)
    return mlir::failure();
  int nReduce = 0;
  forOp.getBodyRegion().walk([&](mlir::triton::ReduceOp) { ++nReduce; });
  if (nReduce != 2)
    return mlir::failure();

  // (3) classify: dotA (Q@K^T) is the one whose B-operand cone has a tt.trans.
  mlir::triton::DotOp dotA, dotB;
  if (faConeHasTrans(dots[0].getB()) && !faConeHasTrans(dots[1].getB())) {
    dotA = dots[0];
    dotB = dots[1];
  } else if (faConeHasTrans(dots[1].getB()) && !faConeHasTrans(dots[0].getB())) {
    dotA = dots[1];
    dotB = dots[0];
  } else {
    return mlir::failure();
  }

  // (4) trace dot operands to their tt.load leaves.
  auto qLoad = faTraceToLoad(dotA.getA());
  auto kLoad = faTraceToLoad(dotA.getB());
  auto vLoad = faTraceToLoad(dotB.getB());
  if (!qLoad || !kLoad || !vLoad)
    return mlir::failure();

  // (5) unique tt.store + unique arith.divsi (d_head = d_model / h) in the func.
  auto funcOp = forOp->getParentOfType<mlir::triton::FuncOp>();
  if (!funcOp)
    return mlir::failure();
  mlir::triton::StoreOp store;
  int nStore = 0;
  funcOp.walk([&](mlir::triton::StoreOp s) {
    store = s;
    ++nStore;
  });
  if (nStore != 1)
    return mlir::failure();
  mlir::arith::DivSIOp dhead;
  int nDiv = 0;
  funcOp.walk([&](mlir::arith::DivSIOp d) {
    dhead = d;
    ++nDiv;
  });
  if (nDiv != 1)
    return mlir::failure();

  // (6) resolve kernel-arg pointers / scalars.
  mlir::Value qPtr = unwrapPtrToKernelArg(qLoad.getPtr());
  mlir::Value kPtr = unwrapPtrToKernelArg(kLoad.getPtr());
  mlir::Value vPtr = unwrapPtrToKernelArg(vLoad.getPtr());
  mlir::Value oPtr = unwrapPtrToKernelArg(store.getPtr());
  if (!qPtr || !kPtr || !vPtr || !oPtr)
    return mlir::failure();
  mlir::Value nVal = forOp.getUpperBound();
  mlir::Value dModelVal = dhead.getLhs();
  mlir::Value hVal = dhead.getRhs();

  // (7) block sizes from tensor shapes. acc = [BM, BD]; dotA result = [BM, BN].
  int64_t BM = accTy.getShape()[0];
  int64_t BD = accTy.getShape()[1];
  auto sTy = mlir::dyn_cast<mlir::RankedTensorType>(dotA.getType());
  if (!sTy || sTy.getRank() != 2 || sTy.getShape()[0] != BM)
    return mlir::failure();
  int64_t BN = sTy.getShape()[1];
  // Envelope: BM <= 32 (one query row per lane), all tile dims multiples of 8.
  // BD may exceed the runtime d_head (BLOCKSIZE_d = max(16, d_head)); the
  // emitter masks the padded columns via `d < d_head`.
  if (BM > 32 || BM % 8 || BN % 8 || BD % 8)
    return mlir::failure();
  // Threadgroup budget: qbuf+ktbuf+vbuf+sbuf+pbuf+obuf+otbuf+rmax+rsum floats
  // must fit Apple's 32 KiB. Rejects e.g. BD=64 (~48 KiB) — falls through to
  // the existing reject rather than emitting an over-budget kernel.
  int64_t tgFloats = 3 * BM * BD + 2 * BD * BN + 2 * BM * BN + 2 * BM;
  if (tgFloats > 8192)
    return mlir::failure();

  // (8) build metal.flash_attention before the loop.
  mlir::OpBuilder builder(forOp);
  auto loc = forOp.getLoc();
  auto f32 = builder.getF32Type();
  auto ui32Elem = wrapperElementType(nVal.getType());
  mlir::Value qBuf = bridgePtrToMemref(builder, loc, qPtr, f32);
  mlir::Value kBuf = bridgePtrToMemref(builder, loc, kPtr, f32);
  mlir::Value vBuf = bridgePtrToMemref(builder, loc, vPtr, f32);
  mlir::Value oBuf = bridgePtrToMemref(builder, loc, oPtr, f32);
  mlir::Value nBuf = bridgePtrToMemref(builder, loc, nVal, ui32Elem);
  mlir::Value dmBuf = bridgePtrToMemref(builder, loc, dModelVal, ui32Elem);
  mlir::Value hBuf = bridgePtrToMemref(builder, loc, hVal, ui32Elem);
  auto faOp = mlir::triton::metal::FlashAttentionOp::create(
      builder, loc, qBuf, kBuf, vBuf, oBuf, nBuf, dmBuf, hBuf, BM, BN, BD);

  // (9) DCE the now-dead loop / epilogue / loads / offset arithmetic in the
  // func entry block: everything except the new FA op + terminator becomes
  // dead once the store (the only writer) is gone. Bottom-up to fixpoint; the
  // FA op's operand bridge-casts stay live (used by the FA op).
  mlir::Block *blk = forOp->getBlock();
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Operation &op : llvm::make_early_inc_range(llvm::reverse(*blk))) {
      if (&op == faOp.getOperation() || op.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      if (op.use_empty()) {
        op.erase();
        changed = true;
      }
    }
  }
  return mlir::success();
}

// Structure a top-level early-exit guard `if (cond) return;` into an scf.if so
// the (structured-only) MSL emitter can handle it. Triton lowers a Python early
// `return` (e.g. `if pid >= B: return`) to `cf.cond_br %c, ^ret, ^cont` where
// ^ret is a bare void `tt.return` and ^cont holds the rest of the function.
// `--lift-cf-to-scf` cannot recover this (a void early-return does not
// reconverge). We rewrite it to `scf.if <enter-cont-cond> { <cont body> }`
// followed by a void `tt.return`, then erase the cf.cond_br + the two dead
// blocks. STRICT no-op unless the exact canonical shape matches, so every
// already-structured kernel (single block + structured scf) is untouched.
static void structureEarlyReturns(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::triton::FuncOp> funcs;
  moduleOp.walk([&](mlir::triton::FuncOp f) { funcs.push_back(f); });
  for (auto func : funcs) {
    if (func.getBody().empty())
      continue;
    mlir::Block &entry = func.getBody().front();
    auto condBr =
        mlir::dyn_cast<mlir::cf::CondBranchOp>(entry.getTerminator());
    if (!condBr)
      continue;
    // No successor block-argument operands (we don't thread values across).
    if (!condBr.getTrueDestOperands().empty() ||
        !condBr.getFalseDestOperands().empty())
      continue;

    auto isVoidReturnOnly = [](mlir::Block *b) {
      if (b->getNumArguments() != 0 || b->empty())
        return false;
      if (&b->front() != &b->back())
        return false; // exactly one op
      auto ret = mlir::dyn_cast<mlir::triton::ReturnOp>(&b->front());
      return ret && ret.getNumOperands() == 0;
    };

    mlir::Block *trueDest = condBr.getTrueDest();
    mlir::Block *falseDest = condBr.getFalseDest();
    mlir::Block *retBlk = nullptr, *contBlk = nullptr;
    bool enterContWhenTrue = false;
    if (isVoidReturnOnly(trueDest)) {
      retBlk = trueDest;
      contBlk = falseDest;
      enterContWhenTrue = false; // continue when cond is FALSE
    } else if (isVoidReturnOnly(falseDest)) {
      retBlk = falseDest;
      contBlk = trueDest;
      enterContWhenTrue = true; // continue when cond is TRUE
    } else {
      continue;
    }

    // The continuation must be a self-contained single block (only pred is the
    // entry, no block args) ending in a void tt.return. Its inner control flow
    // is already structured scf, so moving its ops is sound.
    if (contBlk == retBlk || contBlk->getNumArguments() != 0)
      continue;
    if (contBlk->getSinglePredecessor() != &entry ||
        retBlk->getSinglePredecessor() != &entry)
      continue;
    auto contRet =
        mlir::dyn_cast<mlir::triton::ReturnOp>(contBlk->getTerminator());
    if (!contRet || contRet.getNumOperands() != 0)
      continue;

    // --- Rewrite ---
    // Build `scf.if %cond { thenBody } else { elseBody }` keeping the ORIGINAL
    // condition polarity — the continuation goes in whichever arm the cond_br
    // branches to it (then=trueDest, else=falseDest). This avoids negating the
    // i1 (an `arith.xori %c, true` mis-lowers to MSL bitwise `%c ^ -1`, which is
    // always truthy). The non-continuation arm is left empty (the early return).
    mlir::OpBuilder builder(condBr);
    mlir::Location loc = condBr.getLoc();
    auto ifOp = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                        condBr.getCondition(),
                                        /*addThenBlock=*/true,
                                        /*addElseBlock=*/true);
    // addThen/ElseBlock=true create EMPTY blocks (no auto scf.yield). Move
    // contBlk's ops (all but its terminator) into the arm it branched to.
    mlir::Block *contArm = enterContWhenTrue ? ifOp.thenBlock() : ifOp.elseBlock();
    mlir::Block *emptyArm = enterContWhenTrue ? ifOp.elseBlock() : ifOp.thenBlock();
    contArm->getOperations().splice(contArm->end(), contBlk->getOperations(),
                                    contBlk->begin(),
                                    std::prev(contBlk->end()));
    mlir::OpBuilder contBuilder(contArm, contArm->end());
    mlir::scf::YieldOp::create(contBuilder, loc);
    mlir::OpBuilder emptyBuilder(emptyArm, emptyArm->end());
    mlir::scf::YieldOp::create(emptyBuilder, loc);
    // Final void return in the entry block, after the scf.if.
    mlir::triton::ReturnOp::create(builder, loc);
    // Erase the cf terminator and the now-dead blocks.
    condBr.erase();
    contBlk->erase();
    retBlk->erase();
  }
}

// W-C: flatten a `tt.scan` input that is a tensor-yielding `scf.if` with a
// UNIFORM (scalar) condition into `tt.scan(select(cond, thenYield, elseYield))`.
// Speculative decoding's inverse-CDF loop wraps the scan input in
// `if is_uniform: adj=1 else: adj=q-p` — an scf.if whose (pure, masked) branch
// bodies are hoisted out so both execute unconditionally and a scalar-condition
// select picks per element. Done BEFORE the dialect conversion because the SCF
// structural type conversion rewrites/empties the scf.if regions concurrently,
// which would race the ScanLowering cone walk. No-op unless the exact shape
// matches.
// True if `v`'s producer cone (restricted to ops inside `body`) references
// `target` — used to confirm the loop-carried delta is independent of the acc.
static bool coneUsesValue(mlir::Value v, mlir::Value target, mlir::Block *body,
                          llvm::DenseSet<void *> &seen) {
  if (v == target)
    return true;
  mlir::Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != body)
    return false;
  if (!seen.insert(v.getAsOpaquePointer()).second)
    return false;
  for (auto o : def->getOperands())
    if (coneUsesValue(o, target, body, seen))
      return true;
  return false;
}

// Clone the producer cone of `v` that lives inside `oldBody` into the current
// insertion point of `bb`, remapping via `map` (which pre-seeds oldIv->newIv).
// Values defined OUTSIDE oldBody (loop-invariant) are used as-is. Post-order so
// operands are cloned before their users.
static mlir::Value cloneConeInto(mlir::Value v, mlir::IRMapping &map,
                                 mlir::OpBuilder &bb, mlir::Block *oldBody) {
  if (map.contains(v))
    return map.lookup(v);
  mlir::Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != oldBody)
    return v; // loop-invariant / block arg handled by the pre-seeded map
  for (mlir::Value operand : def->getOperands())
    cloneConeInto(operand, map, bb, oldBody);
  bb.clone(*def, map);
  return map.lookup(v);
}

// W: reduce over a LOOP-CARRIED sum accumulator. `sum(scf.for(acc += delta))`
// (delta independent of acc) is reassociated to a SCALAR-accumulating loop
// `scf.for(s += sum(delta))` because addf is associative/commutative. This
// sidesteps the (unrepresentable) reduce over a loop-carried tensor iter_arg at
// BLOCK>tpb: the inner `sum(delta)` is a normal rank-1 reduce over a device-
// rooted cone, and the outer accumulation is a scalar scf.for. Layer-norm /
// RMSNorm mean+variance are exactly this shape. Done BEFORE conversion.
static void reassociateLoopCarriedSumReduce(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::triton::ReduceOp> reduces;
  moduleOp.walk([&](mlir::triton::ReduceOp r) { reduces.push_back(r); });
  for (auto R : reduces) {
    if (R.getSrcs().size() != 1 || R->getNumResults() != 1)
      continue;
    auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(R.getSrcs()[0].getType());
    if (!rtt || rtt.getRank() != 1 || !rtt.getElementType().isF32())
      continue;
    // Reduce combine must be arith.addf (sum).
    mlir::Operation *combine = nullptr;
    if (R->getNumRegions() > 0 && !R->getRegion(0).empty())
      for (auto &n : R->getRegion(0).front())
        if (!mlir::isa<mlir::triton::ReduceReturnOp>(n)) {
          combine = &n;
          break;
        }
    if (!combine || !mlir::isa<mlir::arith::AddFOp>(combine))
      continue;
    // Source must be an scf.for result whose yield is `addf(iterArg, delta)`.
    auto forOp = R.getSrcs()[0].getDefiningOp<mlir::scf::ForOp>();
    if (!forOp || forOp.getNumResults() != 1)
      continue;
    unsigned idx = mlir::cast<mlir::OpResult>(R.getSrcs()[0]).getResultNumber();
    auto yieldOp =
        mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    auto add = yieldOp.getOperand(idx).getDefiningOp<mlir::arith::AddFOp>();
    if (!add)
      continue;
    mlir::Value iterArg = forOp.getRegionIterArg(idx);
    mlir::Value delta;
    if (add.getLhs() == iterArg)
      delta = add.getRhs();
    else if (add.getRhs() == iterArg)
      delta = add.getLhs();
    else
      continue;
    llvm::DenseSet<void *> seen;
    if (coneUsesValue(delta, iterArg, forOp.getBody(), seen))
      continue; // delta must not depend on the accumulator

    mlir::OpBuilder b(forOp);
    auto loc = forOp.getLoc();
    // sinit = sum(init) — clone R over the loop's init value (usually zeros→0).
    mlir::Operation *sinitR = b.clone(*R.getOperation());
    sinitR->setOperand(0, forOp.getInitArgs()[idx]);
    mlir::Value sinit = sinitR->getResult(0);

    auto newFor = mlir::scf::ForOp::create(
        b, loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        mlir::ValueRange{sinit},
        [&](mlir::OpBuilder &bb, mlir::Location ll, mlir::Value iv,
            mlir::ValueRange args) {
          mlir::IRMapping map;
          map.map(forOp.getInductionVar(), iv);
          mlir::Value deltaCloned =
              cloneConeInto(delta, map, bb, forOp.getBody());
          mlir::Operation *dr = bb.clone(*R.getOperation());
          dr->setOperand(0, deltaCloned);
          mlir::Value sNew =
              mlir::arith::AddFOp::create(bb, ll, args[0], dr->getResult(0))
                  .getResult();
          mlir::scf::YieldOp::create(bb, ll, sNew);
        });
    R->getResult(0).replaceAllUsesWith(newFor.getResult(0));
    R.erase();
    if (forOp->use_empty())
      forOp.erase();
  }
}

// Axis-0 analog: `reduce(scf.for(acc2d += delta2d), axis=0)` [rank-1 N] is
// reassociated to `scf.for(s1d += reduce(delta2d, axis=0))`. The outer loop now
// carries the rank-1 [N] partial-column-sum (at E_out==1 it scalarizes to a
// per-thread scalar, like the forward's scalar carry); the inner
// `reduce(delta2d, axis=0)` is a rank-2 axis=0 reduce over the per-iteration
// direct masked load, handled by `lowerRank2Axis0Reduce`. This is tutorial-05's
// `_layer_norm_bwd_dwdb` (dw/db partial-sum reduction). Done BEFORE conversion.
static void reassociateLoopCarriedAxis0Reduce(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::triton::ReduceOp> reduces;
  moduleOp.walk([&](mlir::triton::ReduceOp r) { reduces.push_back(r); });
  for (auto R : reduces) {
    if (R.getSrcs().size() != 1 || R->getNumResults() != 1)
      continue;
    auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(R.getSrcs()[0].getType());
    if (!rtt || rtt.getRank() != 2)
      continue;
    mlir::Type eltTy = rtt.getElementType();
    if (!(eltTy.isF32() || eltTy.isInteger(32)))
      continue;
    if (R.getAxis() != 0)
      continue;
    mlir::Operation *combine = nullptr;
    if (R->getNumRegions() > 0 && !R->getRegion(0).empty())
      for (auto &n : R->getRegion(0).front())
        if (!mlir::isa<mlir::triton::ReduceReturnOp>(n)) {
          combine = &n;
          break;
        }
    // Every combine lowerRank2Axis0Reduce supports also has an elementwise
    // scalarizing lowering for its loop-carried tensor update op, so the
    // reassociated outer `s = combine(s, reduce)` legalizes: f32 sum/max
    // (Arith{Add,Max}FLowering) and i32 sum/max/min (AddI + ArithIntMinMax).
    if (!combine ||
        !mlir::isa<mlir::arith::AddFOp, mlir::arith::MaxNumFOp,
                   mlir::arith::MaximumFOp, mlir::arith::AddIOp,
                   mlir::arith::MaxSIOp, mlir::arith::MinSIOp>(combine))
      continue;
    // Multi-result loops are fine: `_layer_norm_bwd_dwdb` carries dw AND db in
    // one scf.for (2 results, 2 axis-0 reduces). Each reduce reassociates its
    // own result (`idx`) into a fresh single-result loop; the original loop is
    // erased once all its results are unused (after both reduces are handled).
    auto forOp = R.getSrcs()[0].getDefiningOp<mlir::scf::ForOp>();
    if (!forOp)
      continue;
    unsigned idx = mlir::cast<mlir::OpResult>(R.getSrcs()[0]).getResultNumber();
    auto yieldOp =
        mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    mlir::Operation *upd = yieldOp.getOperand(idx).getDefiningOp();
    if (!upd || upd->getName() != combine->getName() ||
        upd->getNumOperands() != 2)
      continue;
    mlir::Value iterArg = forOp.getRegionIterArg(idx);
    mlir::Value delta;
    if (upd->getOperand(0) == iterArg)
      delta = upd->getOperand(1);
    else if (upd->getOperand(1) == iterArg)
      delta = upd->getOperand(0);
    else
      continue;
    llvm::DenseSet<void *> seen;
    if (coneUsesValue(delta, iterArg, forOp.getBody(), seen))
      continue; // delta must not depend on the accumulator

    mlir::OpBuilder b(forOp);
    auto loc = forOp.getLoc();
    // sinit = reduce(init, axis=0). Init is the identity splat; lowerRank2-
    // Axis0Reduce folds a uniform-splat source (BM*c for sum, c for max/min)
    // without a device read, so no slice-encoded constant is materialized here.
    mlir::Operation *sinitR = b.clone(*R.getOperation());
    sinitR->setOperand(0, forOp.getInitArgs()[idx]);
    mlir::Value sinit = sinitR->getResult(0);

    auto newFor = mlir::scf::ForOp::create(
        b, loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        mlir::ValueRange{sinit},
        [&](mlir::OpBuilder &bb, mlir::Location ll, mlir::Value iv,
            mlir::ValueRange args) {
          mlir::IRMapping map;
          map.map(forOp.getInductionVar(), iv);
          mlir::Value deltaCloned =
              cloneConeInto(delta, map, bb, forOp.getBody());
          mlir::Operation *dr = bb.clone(*R.getOperation());
          dr->setOperand(0, deltaCloned);
          // sNew = same-kind-combine(args[0], reduce(delta, axis=0)). Build the
          // op fresh so the result type is inferred from the rank-1 operands
          // (cloning `upd` would keep its rank-2 [BM,BN] result type).
          mlir::OperationState st(ll, upd->getName());
          st.addOperands({args[0], dr->getResult(0)});
          st.addTypes({args[0].getType()});
          st.addAttributes(upd->getAttrs());
          mlir::Operation *sOp = bb.create(st);
          mlir::scf::YieldOp::create(bb, ll, sOp->getResult(0));
        });
    R->getResult(0).replaceAllUsesWith(newFor.getResult(0));
    R.erase();
    if (forOp->use_empty())
      forOp.erase();
  }
}

static void flattenUniformScanInputScfIf(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::scf::IfOp> targets;
  moduleOp.walk([&](mlir::triton::ScanOp scan) {
    if (scan.getSrcs().size() != 1)
      return;
    if (auto ifOp = scan.getSrcs().front().getDefiningOp<mlir::scf::IfOp>())
      targets.push_back(ifOp);
  });
  for (auto ifOp : targets) {
    if (ifOp.getNumResults() != 1 || ifOp.getElseRegion().empty())
      continue;
    if (mlir::isa<mlir::RankedTensorType>(ifOp.getCondition().getType()))
      continue;
    if (!ifOp.getThenRegion().hasOneBlock() ||
        !ifOp.getElseRegion().hasOneBlock())
      continue;
    mlir::Value thenVal = ifOp.thenYield().getOperand(0);
    mlir::Value elseVal = ifOp.elseYield().getOperand(0);
    mlir::Block *parent = ifOp->getBlock();
    // Hoist both branch bodies (all but the terminator) to before the scf.if;
    // they are pure (masked loads / selects), so unconditional execution is safe.
    mlir::Block &thenBlk = ifOp.getThenRegion().front();
    parent->getOperations().splice(ifOp->getIterator(), thenBlk.getOperations(),
                                   thenBlk.begin(), std::prev(thenBlk.end()));
    mlir::Block &elseBlk = ifOp.getElseRegion().front();
    parent->getOperations().splice(ifOp->getIterator(), elseBlk.getOperations(),
                                   elseBlk.begin(), std::prev(elseBlk.end()));
    mlir::OpBuilder b(ifOp);
    auto sel = mlir::arith::SelectOp::create(b, ifOp.getLoc(),
                                             ifOp.getCondition(), thenVal,
                                             elseVal);
    ifOp.getResult(0).replaceAllUsesWith(sel.getResult());
    ifOp.erase();
  }
}

static void runFlashAttentionMatcher(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::scf::ForOp> loops;
  moduleOp.walk([&](mlir::scf::ForOp f) {
    if (!f->getParentOfType<mlir::scf::ForOp>())
      loops.push_back(f); // top-level loops only
  });
  for (auto f : loops)
    (void)tryFlashAttentionLoop(f);
}

//===----------------------------------------------------------------------===//
// Pass entry point.
//===----------------------------------------------------------------------===//

struct ConvertTritonGPUToMetalPass
    : public impl::ConvertTritonGPUToMetalBase<ConvertTritonGPUToMetalPass> {
  using ConvertTritonGPUToMetalBase::ConvertTritonGPUToMetalBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    // Reject tt.load with an `other` operand only when it feeds a `tt.dot`
    // operand position. The matmul track does not (yet) honour Triton `other`
    // semantics under the simdgroup rewrite. Elementwise other-loads are
    // handled by MaskedLoadLowering (Session L1, splat-constant only). See
    // `.omc/specs/deep-interview-leet-triton-l1-refine-and-ship.md` §3.0 ADR.
    bool otherOk = true;
    moduleOp.walk([&](mlir::triton::LoadOp load) {
      if (!(load.getMask() && load.getOther()))
        return;
      for (auto &use : load.getResult().getUses()) {
        if (mlir::isa<mlir::triton::DotOp>(use.getOwner())) {
          load.emitError(
              "TritonGPUToMetal: tt.load `other` operand not supported in "
              "tt.dot operand position");
          otherOk = false;
          break;
        }
      }
    });
    if (!otherOk) {
      signalPassFailure();
      return;
    }

    // Structure top-level early-exit guards (`if cond: return`) into scf.if
    // BEFORE any other handling — the MSL emitter is structured-only and has no
    // cf.cond_br lowering. No-op for already-structured kernels.
    structureEarlyReturns(moduleOp);

    // W-C: hoist a `tt.scan` input scf.if (uniform cond) into a select before
    // the conversion (the SCF structural conversion would otherwise race the
    // ScanLowering cone walk on the scf.if regions).
    flattenUniformScanInputScfIf(moduleOp);

    // Reassociate `sum(scf.for(acc += delta))` into a scalar-accumulating loop
    // `scf.for(s += sum(delta))` so the reduce is over a device-rooted cone, not
    // an (unrepresentable) loop-carried tensor iter_arg at BLOCK>tpb. Layer-norm
    // / RMSNorm mean+variance.
    reassociateLoopCarriedSumReduce(moduleOp);

    // Axis-0 analog: `reduce(scf.for(acc2d += delta2d), axis=0)` ->
    // `scf.for(s1d += reduce(delta2d, axis=0))`. tutorial-05 `_layer_norm_bwd_dwdb`.
    reassociateLoopCarriedAxis0Reduce(moduleOp);

    // Phase 2: recognize the online-softmax flash-attention loop and replace
    // it + epilogue with a single `metal.flash_attention` op BEFORE any cvt
    // handling. Its dot-B operand cvts (A operand is a computed `exp`) can't be
    // absorbed by the matmul track and would otherwise hit the L1d3 reject.
    // See metal-flash-attention-plan.md.
    runFlashAttentionMatcher(moduleOp);

    // L1d3: rewire dot-feeding ttg.convert_layout(blocked -> dot_op) ops
    // off their tt.dot operands so they don't survive into the cvt
    // classifier below (which would emit the L1d3 hard error). The
    // existing matmul track matchers (preprocessDotChains, rewriteSingleDot,
    // tryUnrollCanonical3IterArgDot, tryUnrollKLoopDot) then see the
    // canonical body shape. See
    // `.omc/specs/deep-interview-l1d3-matmul-convert-layout-preempt.md`.
    preprocessDotCvtChains(moduleOp);

    // Collapse blocked↔blocked convert_layout ops whose source is a
    // self-contained gather cone — rank-1 (mixed pointer alignment) and rank-2
    // (masked transpose, sizePerThread>1) alike — by rewriting each cvt's
    // producer cone to the dest encoding so the cvt becomes an identity the
    // classifier/lowering accept (turning the transpose into a direct
    // gather/scatter, no threadgroup staging).
    normalizeBlockedDivergentCvts(moduleOp);

    // Classify non-identity ttg.convert_layout ops:
    //   - in-envelope (rank-2 blocked↔blocked, same shape, same elem-type,
    //     sizePerThread == [1,1] on both sides) → allow-through to
    //     `ConvertLayoutLowering`'s staged-transpose body (L1d2).
    //   - out-of-envelope (anything else non-identity, including rank-2
    //     blocked↔blocked with sizePerThread > 1) → deferred to L1d3.
    // L1c identity passthrough remains an accept path (handled by
    // `ConvertLayoutLowering`). See
    // `.omc/specs/deep-interview-leet-triton-l1d2-staged-transpose-body.md`.
    bool cvtOk = true;
    moduleOp.walk([&](mlir::triton::gpu::ConvertLayoutOp cvt) {
      auto srcTy = cvt.getSrc().getType();
      auto dstTy = cvt.getResult().getType();
      if (srcTy == dstTy)
        return; // L1c identity — `ConvertLayoutLowering` handles passthrough.
      auto srcRtt = mlir::dyn_cast<mlir::RankedTensorType>(srcTy);
      auto dstRtt = mlir::dyn_cast<mlir::RankedTensorType>(dstTy);
      // L3a passthrough: rank-1 slice→blocked cvt with matching shape and
      // element type collapses to scalar-identity under the TypeConverter
      // (both sides become the scalar element type). The Triton frontend
      // inserts this op between `tt.reduce` (which yields a slice-encoded
      // tensor) and the downstream `tt.store` (which expects a blocked
      // encoding). `ConvertLayoutLowering` handles the post-conversion
      // identity. See
      // `.omc/specs/deep-interview-leet-triton-l3a-reduce-body-axis1.md`.
      // Scalar-identity relabel — slice-involved broadcast/reduce cones
      // (`x[None,:]`, `x[:,None]`, reduce output) of any rank, plus degenerate
      // blocked↔blocked converts that only permute size-1 axes. These collapse
      // to a no-op under the scalarizing TypeConverter (`ConvertLayoutLowering`
      // Path 2 / `isScalarIdentityConvert`); the Metal backend re-derives every
      // per-thread index from `make_range`, so no data moves. Genuine transposes
      // (both blocked, >=2 non-unit dims) are NOT accepted here and fall through
      // to the rank-2 staged-transpose envelope below (or the L1d3 reject).
      if (isScalarIdentityConvert(cvt))
        return;
      auto srcBlocked =
          srcRtt ? mlir::dyn_cast_or_null<
                       mlir::triton::gpu::BlockedEncodingAttr>(
                       srcRtt.getEncoding())
                 : nullptr;
      auto dstBlocked =
          dstRtt ? mlir::dyn_cast_or_null<
                       mlir::triton::gpu::BlockedEncodingAttr>(
                       dstRtt.getEncoding())
                 : nullptr;
      bool sameShapeRank2BlockedPair =
          srcRtt && dstRtt && srcRtt.getRank() == 2 &&
          dstRtt.getRank() == 2 && srcRtt.getShape() == dstRtt.getShape() &&
          srcRtt.getElementType() == dstRtt.getElementType() && srcBlocked &&
          dstBlocked;
      bool sizePerThreadAllOne = false;
      if (sameShapeRank2BlockedPair) {
        auto srcSpt = srcBlocked.getSizePerThread();
        auto dstSpt = dstBlocked.getSizePerThread();
        sizePerThreadAllOne =
            srcSpt.size() == 2 && dstSpt.size() == 2 && srcSpt[0] == 1 &&
            srcSpt[1] == 1 && dstSpt[0] == 1 && dstSpt[1] == 1;
      }
      bool inEnvelope = sameShapeRank2BlockedPair && sizePerThreadAllOne;
      if (inEnvelope)
        return; // L1d2: handled by `ConvertLayoutLowering` staged-transpose
                // body below.
      // Rank-1 relabel feeding ONLY `tt.store` — the scatter shape
      // (`tl.store(dst + idx_tensor, vals[, mask])`, radix sort). Store-
      // Lowering / MaskedStoreLowering peel back to the pre-cvt operands and
      // perform the store in the SOURCE layout, so no data actually moves
      // across the relabel; `ConvertLayoutLowering` Path 2b then forwards the
      // (by-then unread) source. Kept in lockstep with that path's condition —
      // any non-store consumer makes the permutation observable and must still
      // be rejected here.
      if (srcRtt && dstRtt && srcRtt.getRank() == 1 && dstRtt.getRank() == 1 &&
          srcRtt.getShape() == dstRtt.getShape() &&
          srcRtt.getElementType() == dstRtt.getElementType() && srcBlocked &&
          dstBlocked &&
          llvm::all_of(cvt.getResult().getUsers(), [](mlir::Operation *u) {
            return mlir::isa<mlir::triton::StoreOp>(u);
          }))
        return;
      cvt.emitOpError(
          "ttg.convert_layout: broader staged-transpose deferred to L1d3 "
          "(rank≠2 or shape/elem-type change or non-blocked encoding or "
          "sizePerThread > 1)");
      cvtOk = false;
    });
    if (!cvtOk) {
      signalPassFailure();
      return;
    }

    // Session L3 reduce pre-pass: detect unsupported tt.reduce forms and
    // emit specific errors per spec
    // `.omc/specs/deep-interview-leet-triton-l3-reduce-axis-2d.md`.
    //
    // Honest divergence note (per spec §"Honest divergence policy"): this
    // session ships Phase A (metal.threadgroup_alloca + metal.barrier) and
    // Phase B (slice layout TypeConverter passthrough) plus negative-only
    // pre-pass rejection of tt.reduce. The Phase C tree-reduction lowering
    // body is deferred to L3a; meanwhile any tt.reduce in the input IR is
    // rejected here with the spec-mandated error strings so the negative lit
    // suite still exercises the gate.
    bool reduceOk = true;
    moduleOp.walk([&](mlir::triton::ReduceOp red) {
      auto axes = red.getAxis();
      // Inspect first operand for rank/shape/dtype.
      mlir::Type srcTy = red.getSrcs().empty()
                             ? mlir::Type{}
                             : red.getSrcs().front().getType();
      auto rtt = mlir::dyn_cast_or_null<mlir::RankedTensorType>(srcTy);
      // Combine-op probe: walk the reduce region's first non-terminator op.
      llvm::StringRef combineName;
      mlir::Type combineEltTy;
      if (red->getNumRegions() > 0 && !red->getRegion(0).empty()) {
        for (auto &op : red->getRegion(0).front()) {
          if (mlir::isa<mlir::triton::ReduceReturnOp>(op))
            continue;
          combineName = op.getName().getStringRef();
          if (op.getNumResults() > 0)
            combineEltTy = op.getResult(0).getType();
          break;
        }
      }
      // 1) rank check: rank-1 and rank-2 are supported; rank>2 deferred to L3b.
      if (!rtt || (rtt.getRank() != 1 && rtt.getRank() != 2)) {
        red.emitOpError("reduce with rank not in {1, 2} not supported "
                        "(rank-1 added in Option β; rank>2 requires Session L3b)");
        reduceOk = false;
        return;
      }
      // 2) multi-axis on rank-2 (i.e. axes attr length > 1) — getAxis returns
      // a single uint32, but tt.reduce supports a single axis; reject if the
      // input shape couldn't possibly hold a multi-axis form. (The TableGen
      // for tt.reduce uses one-axis `axis`. Multi-axis reduce in TTGIR is a
      // separate construct also matched here for spec parity.)
      // axis must be 0 or 1.
      if (axes > 1) {
        red.emitOpError("multi-axis or rank>2 reduce requires Session L3b (future)");
        reduceOk = false;
        return;
      }
      // Rank-2 axis=1 ships via the rank-2 ReduceLowering body. Rank-2 axis=0
      // (per-column) ships via lowerRank2Axis0Reduce (Session L3a2) for f32
      // sum/max and i32 sum/max/min (after reassociateLoopCarriedAxis0Reduce the
      // inner reduce is a direct axis=0 device reduce). Output must be E==1
      // (BN <= tpb); larger BN and other combines stay deferred.
      if (rtt.getRank() == 2 && axes == 0) {
        bool fOk = (combineName == "arith.addf" ||
                    combineName == "arith.maxnumf" ||
                    combineName == "arith.maximumf") &&
                   combineEltTy && combineEltTy.isF32();
        bool iOk = (combineName == "arith.addi" ||
                    combineName == "arith.maxsi" ||
                    combineName == "arith.minsi") &&
                   combineEltTy && combineEltTy.isInteger(32);
        if (!fOk && !iOk) {
          red.emitOpError(
              "tt.reduce axis=0 reduce supports f32 sum/max and i32 "
              "sum/max/min (Session L3a2); other combines deferred");
          reduceOk = false;
        }
        return;
      }
      // 3) combine must be arith.addf / arith.addi (sum) or arith.maximumf /
      // arith.maxnumf (max, f32 only). Triton's tl.max emits arith.maxnumf
      // (IEEE maxNum); arith.maximumf uses IEEE maximum. Both map to MSL max.
      bool combineOk = false;
      if (combineName == "arith.addf") {
        if (combineEltTy && combineEltTy.isF32())
          combineOk = true;
      } else if (combineName == "arith.addi") {
        if (auto intTy = mlir::dyn_cast_or_null<mlir::IntegerType>(combineEltTy))
          if (intTy.getWidth() == 32)
            combineOk = true;
      } else if (combineName == "arith.maximumf" ||
                 combineName == "arith.maxnumf") {
        if (combineEltTy && combineEltTy.isF32())
          combineOk = true;
      } else if (combineName == "arith.maxsi") {
        // i32 signed max (Triton tl.max on i32). Wired for rank-1 via
        // lowerRank1Reduce's si32 butterfly/accumulator. Rank-2 i32 max is
        // not yet implemented, so leave it rejected (falls through to L3c).
        if (rtt.getRank() == 1)
          if (auto intTy =
                  mlir::dyn_cast_or_null<mlir::IntegerType>(combineEltTy))
            if (intTy.getWidth() == 32)
              combineOk = true;
      } else if (combineName == "arith.minsi") {
        // i32 signed min (Triton tl.min on i32). Mirrors arith.maxsi — wired for
        // rank-1 via lowerRank1Reduce's si32 accumulator (identity INT32_MAX).
        // Rank-2 i32 min not implemented, so leave it rejected (→ L3c).
        if (rtt.getRank() == 1)
          if (auto intTy =
                  mlir::dyn_cast_or_null<mlir::IntegerType>(combineEltTy))
            if (intTy.getWidth() == 32)
              combineOk = true;
      }
      if (!combineOk) {
        if (combineName == "arith.addf" || combineName == "arith.addi") {
          // It's a supported combine op shape but wrong dtype (e.g. f16).
          red.emitOpError("reduce dtype must be f32 or i32 in Session L3");
          reduceOk = false;
          return;
        }
        red.emitOpError("reduce combine requires Session L3c (future) — got ")
            << combineName;
        reduceOk = false;
        return;
      }
      // 4) dynamic reduce extent — static shape required. Rank-aware to avoid
      // out-of-bounds isDynamicDim(1) on rank-1 inputs.
      bool dynExtent = rtt.isDynamicDim(0);
      if (rtt.getRank() == 2)
        dynExtent = dynExtent || rtt.isDynamicDim(1);
      if (dynExtent) {
        red.emitOpError("dynamic reduce extent unsupported in L3");
        reduceOk = false;
        return;
      }
      // 5) Per-row size check (L3 budget chunked-reduce).
      //
      // The 32 KiB threadgroup-memory budget is enforced ON THE STAGING
      // BUFFER (`metal.threadgroup_alloca`). For in-budget tiles
      // (`M*N*sizeof(T) ≤ 32 KiB`), `ReduceLowering` emits the L3a
      // single-pass body (one alloca sized to `M*N`). For over-budget
      // tiles, `ReduceLowering` emits the chunked body: one alloca sized
      // to `M*chunk_size` (where `chunk_size = floor(32 KiB / (M*sizeof(T)))`),
      // reused across `N_chunks = ceil(N/chunk_size)` unrolled passes via
      // inter-chunk barriers. See
      // `.omc/specs/deep-interview-leet-triton-l3-budget-chunked-reduce.md`.
      //
      // The only remaining pre-pass reject is when a single row already
      // exceeds the budget (`M*sizeof(T) > 32 KiB` ⇒ `chunk_size = 0`);
      // chunking can't help and a per-thread fan-in is out of envelope.
      int64_t elemBytes = 0;
      if (combineEltTy.isF32()) elemBytes = 4;
      else if (mlir::isa<mlir::IntegerType>(combineEltTy)) elemBytes = 4;
      int64_t perRowBytes = rtt.getDimSize(0) * elemBytes;
      // L3a-tileloop carve-out: when the reduce axis is fully per-thread-
      // owned (`threadsPerCTA[axis_dim] == 1`), the new branch in
      // `ReduceLowering` emits a register-level chain and uses ZERO
      // threadgroup memory, so the per-row budget reject doesn't apply.
      // See spec AC.B1 and the per-thread branch comment in
      // `ReduceLowering`.
      bool perThreadOwned = false;
      if (auto srcBlocked = mlir::dyn_cast_or_null<
              mlir::triton::gpu::BlockedEncodingAttr>(rtt.getEncoding())) {
        auto tpW = srcBlocked.getThreadsPerWarp();
        auto wpC = srcBlocked.getWarpsPerCTA();
        if (tpW.size() == 2 && wpC.size() == 2 &&
            tpW[1] * wpC[1] == 1) {
          perThreadOwned = true;
        }
      }
      // Wall 14: rank-1 reduce uses the per-thread serial pre-reduce +
      // tpb-sized butterfly scratch (`metal.threadgroup_alloca : !metal.memref
      // <256 x f32>`), NEVER the perRowBytes-sized staging buffer. The 32 KiB
      // budget is irrelevant for rank-1; only rank-2 (which uses the staging
      // alloca) needs the chunking gate. Without this exemption, the rank-1
      // BLOCK > 8192 path is rejected here even though `lowerRank1Reduce` can
      // handle it via bounded unroll.
      bool isRank1 = (rtt.getRank() == 1);
      if (!isRank1 && !perThreadOwned && perRowBytes > 32 * 1024) {
        red.emitOpError(
            "per-row size exceeds 32 KiB; chunking impossible");
        reduceOk = false;
        return;
      }
      // L3a (Phase C) / L3 budget ship axis=1 f32/i32 reduce lowering via
      // the `ReduceLowering` OpConversionPattern below. In-budget tiles hit
      // the single-pass branch; over-budget tiles hit the chunked branch.
    });
    if (!reduceOk) {
      signalPassFailure();
      return;
    }

    // Matmul track session 3a: rewrite tt.dot chains to metal.simdgroup_*
    // BEFORE the main scalar conversion. Pre-pass leaves !tt.ptr operands
    // bridged via unrealized_conversion_cast; FuncOpLowering + unwrapToMemref
    // resolve them later. See
    // `.omc/specs/deep-interview-metal-matmul-session3-tt-dot-wiring.md`.
    preprocessDotChains(moduleOp);

    // AC4 v6: dead-code eliminate the C-side pointer-arithmetic chain that
    // fed the original `tt.store` after the matcher rewrote it into
    // explicit-origin `metal.simdgroup_store` ops. The dead chain includes
    // `tensor<64x64x!tt.ptr<f32>>` and large `tensor<64x64xi32>` broadcast
    // values whose presence would otherwise drive `findTileInfo` to pick a
    // huge elements-per-thread tile (e.g. 32), wrapping the simdgroup body
    // in `scf.for(0..32)` and exploding the emitted MSL beyond Apple's
    // compiler resource limits. Iterate to fixed-point so multi-step
    // chains (broadcast → expand_dims → make_range) all collapse.
    {
      bool changed = true;
      while (changed) {
        changed = false;
        llvm::SmallVector<mlir::Operation *> deadOps;
        moduleOp.walk([&](mlir::Operation *op) {
          if (mlir::isa<mlir::triton::FuncOp, mlir::ModuleOp>(op))
            return;
          if (op->getNumResults() == 0) return;
          if (!op->use_empty()) return;
          if (!mlir::isOpTriviallyDead(op)) return;
          // Skip region-bearing ops (e.g. `tt.reduce`, `tt.scan`, `scf.for`).
          // Their bodies are inspected by `isOpTriviallyDead` for side-effect-
          // freeness, but the conversion patterns (`ReduceLowering` etc.) run
          // AFTER this DCE block — reaping the unused tt.reduce here means
          // the conversion pattern never sees it. AC4-v6's actual target is
          // the no-region chain (`arith.constant dense<>`, `tt.splat`,
          // `tt.broadcast`, `tt.addptr`, `tt.expand_dims`) — none of which
          // carry regions — so this skip is safe for the original concern.
          // See `.omc/specs/deep-interview-metal-deferred-followups.md` AC5–AC7
          // and the L3a reduce fixtures in
          // `test/Dialect/Metal/convert-tritongpu-to-metal/reduce_*.mlir`.
          if (op->getNumRegions() > 0) return;
          deadOps.push_back(op);
        });
        for (auto *op : deadOps) {
          op->erase();
          changed = true;
        }
      }
    }

    // L1d2c Phase B pre-conversion sentinel emission. For every tt.func with
    // a masked tt.store, hoist one `metal.threadgroup_alloca<tpb × T>` per
    // element type to function entry; MaskedStoreLowering consumes it via
    // `scratchMap`. See
    // `.omc/specs/deep-interview-leet-triton-l1d2c-phase-b-fix.md`.
    MaskedStoreScratchMap scratchMap;
    preprocessMaskedStoreSentinels(moduleOp, scratchMap);

    // Share one (inbuf, scanbuf) threadgroup pair across all the scans of a
    // function instead of one pair each — 16 unrolled cumsums otherwise ask for
    // 128 KB against a 32 KB budget. Must run pre-conversion so the allocas land
    // in the original function's entry block and dominate every scan.
    ScanBufPool scanBufPool;
    preprocessScanBuffers(moduleOp, scanBufPool);

    // Inc 2.5 rank-2: stage loop-carried [M,N] reduce tiles in threadgroup
    // memory. Must run pre-conversion — the SCF structural conversion rebuilds
    // the loop with per-thread scalar iter_args and detaches the original body
    // block, so the loop structure is only legible from here.
    LoopCarriedTileMap loopCarriedTiles;
    preprocessLoopCarriedReduceTiles(moduleOp, loopCarriedTiles);

    // Chained reduces: each rank-2 axis=1 reduce registers its rowBuf here as
    // it lowers, so a LATER reduce whose cone reads the earlier result gets it
    // at its own row instead of through the producer's per-thread scalar.
    llvm::DenseMap<mlir::Value, mlir::Value> reduceRowBufs;
    g_reduceRowBufs = &reduceRowBufs;

    TritonGPUToMetalTypeConverter typeConverter(ctx);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<MetalDialect>();
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addIllegalDialect<mlir::triton::TritonDialect>();
    target.addIllegalDialect<mlir::triton::gpu::TritonGPUDialect>();
    // arith and func ops are legal only on scalar types (post-conversion).
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect>(
        [&](mlir::Operation *op) {
          return llvm::all_of(op->getResultTypes(), [](mlir::Type t) {
            return !mlir::isa<mlir::RankedTensorType>(t);
          });
        });
    // Session L4 / L4b: math.sqrt / math.erf / math.exp / math.log /
    // math.rsqrt on f32 are illegal; the `Math{Sqrt,Erf,Exp,Log,Rsqrt}Lowering`
    // patterns replace them with `metal.unary_exp`. Other math ops / non-f32
    // element types remain legal so they fall through unchanged (e.g. for any
    // post-pass that expects them, or for the verifier to reject them later).
    target.addDynamicallyLegalDialect<mlir::math::MathDialect>(
        [&](mlir::Operation *op) {
          if (!mlir::isa<mlir::math::SqrtOp, mlir::math::ErfOp,
                         mlir::math::ExpOp, mlir::math::LogOp,
                         mlir::math::RsqrtOp, mlir::math::SinOp,
                         mlir::math::CosOp>(op))
            return true;
          mlir::Type ty = op->getResultTypes().front();
          if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(ty))
            ty = rt.getElementType();
          return !ty.isF32();
        });
    target.addLegalDialect<mlir::func::FuncDialect>();
    // SCF ops (scf.for / scf.if / scf.yield) are legal ONLY when their
    // iter-arg / result / block-arg types are already scalar (i.e. after the
    // tensor->scalar conversion). The dynamic legality + structural rewrites
    // are supplied by `populateSCFStructuralTypeConversionsAndLegality` added
    // to `patterns` below. This converts a user-written `scf.for` carrying a
    // tensor accumulator (e.g. the 3D-conv reduction loop) into one carrying
    // the per-tile-iteration scalar — folding it into the per-thread scalar
    // model + the FuncOpLowering tile loop. Without it the tensor-typed loop
    // survives, the framework bridges it with `unrealized_conversion_cast
    // tensor<->scalar`, and the MSL emitter hits `llvm_unreachable` (see
    // ModuleTranslation::translateValue Default).
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FuncOpLowering, ReturnOpLowering, GetProgramIdLowering,
                 GetNumProgramsLowering,
                 MakeRangeLowering, SplatLowering, ConvertLayoutLowering,
                 AddPtrLowering,
                 ArithConstantDenseLowering, ExpandDimsLowering,
                 BroadcastLowering, ReshapeLowering, TransLowering,
                 LoadLowering, ScalarLoadLowering, MaskedLoadLowering, StoreLowering,
                 AtomicRmwLowering,
                 ArithMuliLowering, ArithAddILowering,
                 ArithAddFLowering, ArithCmpILowering, ArithCmpFLowering,
                 ArithAndILowering,
                 ArithSubILowering, ArithSubFLowering,
                 ArithDivSILowering, ArithDivFLowering, ArithRemSILowering,
                 ArithShRSILowering, ArithShLILowering, ArithOrILowering,
                 ArithXOrILowering, ArithDivUILowering, ArithRemUILowering,
                 ArithShRUILowering, ArithSelectLowering,
                 ArithMulFLowering, ArithSIToFPLowering, ArithNegFLowering,
                 ArithMaxFLowering<mlir::arith::MaxNumFOp>,
                 ArithMaxFLowering<mlir::arith::MaximumFOp>,
                 ArithIntMinMaxLowering<mlir::arith::MaxSIOp>,
                 ArithIntMinMaxLowering<mlir::arith::MinSIOp>,
                 ArithExtFLowering, ArithTruncFLowering,
                 ArithIntCastLowering<mlir::arith::ExtUIOp>,
                 ArithIntCastLowering<mlir::arith::ExtSIOp>,
                 ArithIntCastLowering<mlir::arith::TruncIOp>,
                 MathSinLowering, MathCosLowering,
                 MathSqrtLowering, MathErfLowering,
                 MathExpLowering, MathLogLowering, MathRsqrtLowering>(
                 typeConverter, ctx);
    // ReduceLowering needs the (reduce op)->staged tile mapping that
    // `preprocessLoopCarriedReduceTiles` built above.
    patterns.add<ReduceLowering>(typeConverter, ctx, &loopCarriedTiles,
                                 &reduceRowBufs);
    // MaskedStoreLowering needs the (func, elem-type)→scratch mapping
    // populated by `preprocessMaskedStoreSentinels` above. See L1d2c Phase B.
    patterns.add<MaskedStoreLowering>(typeConverter, ctx, &scratchMap);

    // W-C scan: ScanLowering fills a threadgroup scanbuf + registers the result
    // placeholder in `scanBufMap`, consulted by the rich cone evaluator (g_scan-
    // Buffers) during the consuming reduce. Pass-lifetime; cleared after.
    llvm::DenseMap<mlir::Value, mlir::Value> scanBufMap;
    g_scanBuffers = &scanBufMap;
    patterns.add<ScanLowering>(typeConverter, ctx, &scanBufMap, &scanBufPool);

    // Structural type conversion for user-written control flow: rewrites the
    // iter-arg / result / block-arg / yield types of `scf.for` / `scf.if`
    // through the tensor->scalar TypeConverter and installs the matching
    // dynamic legality on `target`. A reduction loop accumulating a tensor
    // (the 3D-conv `for i/j/k: acc += ...`) thereby carries the per-thread
    // scalar accumulator, nested inside the FuncOpLowering tile loop, which the
    // MSL emitter already supports (ModuleTranslation Wall-15 single-scalar
    // iter_arg path).
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

    auto conversionResult =
        mlir::applyFullConversion(moduleOp, target, std::move(patterns));
    g_scanBuffers = nullptr; // scanBufMap goes out of scope below
    g_reduceRowBufs = nullptr;
    if (mlir::failed(conversionResult)) {
      signalPassFailure();
      return;
    }

    // Safety guard against silent miscompiles: the MSL emitter supports an
    // `scf.for` carrying scalar (f32/i32) iter_args — one (Wall-15 single
    // reduction accumulator) or several (multi-accumulator reduce, K-way ILP;
    // metal-multiacc-reduce-plan.md) — or a single `simdgroup_matrix`
    // accumulator (W2a runtime-K matmul; the loop emitted by
    // `tryRuntimeKLoopCanonicalDot`). A loop carrying anything else would be
    // emitted with its accumulation dropped, so reject that with a clean
    // diagnostic. (The canonical pointer-advance matmul loops carry pointer
    // iter_args but are replaced by the matmul lowering before this guard, so
    // they never reach here.)
    {
      bool loopOk = true;
      moduleOp.walk([&](mlir::scf::ForOp forOp) {
        for (mlir::Value iterArg : forOp.getRegionIterArgs()) {
          mlir::Type t = iterArg.getType();
          if (!(t.isF32() || t.isInteger(32) || t.isInteger(1) ||
                mlir::isa<mlir::triton::metal::MetalSimdgroupMatrixType>(t))) {
            forOp.emitOpError("Metal backend: scf.for iter_arg must be a scalar "
                              "f32/i32/i1 accumulator or simdgroup_matrix after "
                              "conversion");
            loopOk = false;
            break;
          }
        }
      });
      if (!loopOk) {
        signalPassFailure();
        return;
      }
    }

    // Post-conversion cleanup: the MSL emitter walks each statement via
    // a TypeSwitch with a `llvm_unreachable` default. Residual dead
    // `arith.constant` / `arith.muli` / `arith.addi` ops left over from
    // our offset arithmetic (which `LoadLowering`/`StoreLowering`
    // bypass by using `metal.thread_id` directly) would trip that
    // unreachable. Walk all metal.kernel bodies and erase such dead
    // arith ops (must process in reverse so def-after-use ordering
    // works).
    // Iterate to fixed-point: erasing a use chain may free upstream ops.
    moduleOp.walk([&](KernelOp kernel) {
      bool changed = true;
      while (changed) {
        changed = false;
        llvm::SmallVector<mlir::Operation *> toErase;
        kernel.getBodyRegion().walk([&](mlir::Operation *op) {
          // Pure metal-dialect helpers (thread_id, threadgroup_id) and
          // unrealized_conversion_cast can also become dead when downstream
          // patterns (e.g. LoadLowering) bypass their result. Erase them so
          // the MSL emitter doesn't print stray `id.x;` / `tgid.x;` lines.
          if (!mlir::isa<mlir::arith::ConstantOp, mlir::arith::MulIOp,
                         mlir::arith::AddIOp, mlir::arith::SubIOp,
                         mlir::arith::CmpIOp, mlir::arith::CmpFOp,
                         mlir::arith::AndIOp, mlir::arith::DivSIOp,
                         mlir::arith::RemSIOp,
                         mlir::arith::ShRSIOp, mlir::arith::ShLIOp,
                         mlir::arith::OrIOp, mlir::arith::XOrIOp,
                         mlir::arith::DivUIOp, mlir::arith::RemUIOp,
                         mlir::arith::ShRUIOp, mlir::arith::SelectOp,
                         mlir::triton::metal::ThreadIdOp,
                         mlir::triton::metal::ThreadgroupIdOp,
                         mlir::UnrealizedConversionCastOp>(op))
            return;
          if (!op->use_empty())
            return;
          toErase.push_back(op);
        });
        for (auto *op : llvm::reverse(toErase)) {
          op->erase();
          changed = true;
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertTritonGPUToMetalPass() {
  return std::make_unique<ConvertTritonGPUToMetalPass>();
}

} // namespace metal
} // namespace triton
} // namespace mlir
