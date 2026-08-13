//===--- ModuleTranslation.h ------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_MODULETRANSLATION_H
#define METAL_MODULETRANSLATION_H

#include "Dialect/Metal/IR/MetalOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include <llvm/Support/FileSystem.h>
#include <map>

namespace mlir {
class Location;
class ModuleOp;
class Operation;

namespace triton { namespace metal {
class KernelOp;

class ModuleTranslation {
public:
  static llvm::LogicalResult translateModule(mlir::ModuleOp m,
                                             raw_ostream &output);

private:
  ModuleTranslation(Operation *module, raw_ostream &output)
      : _metalModule(module), _output(output){};
  mlir::triton::metal::ModuleOp _metalModule;
  std::map<mlir::Operation *, unsigned> _alloca;
  std::map<void *, size_t> _buffers;
  // Maps an `scf.if` op (with a single result) to the index of the temp
  // variable holding its result. The result is pre-declared before the `if`
  // and assigned inside the then/else regions by `scf.yield`.
  std::map<mlir::Operation *, unsigned> _scfIfTemp;
  // Maps an `scf.for` op to the temp index of its induction variable.
  // The induction var BlockArgument is registered in `_buffers` so
  // `translateVarName(iv)` prints `v<idx>`.
  std::map<mlir::Operation *, unsigned> _scfForIv;
  // Wall 15: maps an `scf.for` op carrying ONE f32 iter_arg to the temp
  // index of its emitted MSL accumulator. The init is emitted as
  // `float v<idx> = init;` before the C-style `for`; the matching
  // `scf.yield` emits `v<idx> = yielded;` inside the body. The region
  // iter-arg BlockArgument is registered in `_buffers` mapped to this
  // idx so reads inside the body resolve to `v<idx>`. Empty for ForOps
  // with zero iter_args (matmul / vector_add tile loops) — those keep
  // the existing IV-only emission. See
  // .omc/plans/tutorial02-wall15-iter-args-translator-consensus.md AC1.
  std::map<mlir::Operation *, unsigned> _scfForIterArg;
  // Multi-accumulator reduce (K-way ILP, metal-multiacc-reduce-plan.md): maps
  // an `scf.for` carrying N>=2 scalar (f32/i32) iter_args to their emitted MSL
  // temp indices, in order. Each region iter-arg BlockArgument and each loop
  // result is registered in `_buffers` mapped to its temp so reads/uses resolve
  // to `v<idx_i>`; the matching `scf.yield` writes all N back. Single-iter_arg
  // loops keep using `_scfForIterArg` above (byte-identical emission).
  std::map<mlir::Operation *, llvm::SmallVector<unsigned, 8>>
      _scfForIterArgsMulti;
  // L1d2b inline-barrier contract: maps an op whose result has been
  // force-materialized as a named MSL let-binding to that temp's index.
  // Currently populated for `metal.tg_load_indexed` at statement-walk
  // time (see `translate(mlir::Region&)`); `translateValue` then renders
  // subsequent uses as `v<idx>` instead of re-inlining the producing
  // expression. This prevents the Apple Metal compiler bug where an
  // inlined `metal.tg_load_indexed` re-evaluated inside an
  // `scf.if(mask)` body (after the trailing `threadgroup_barrier`)
  // causes dropped stores from higher-warp lanes and a warp-0 race.
  // The boundary set is `{metal.barrier, metal.tg_load_indexed,
  // metal.tg_store_indexed}`; only `tg_load_indexed` produces a value
  // and so is the only entry that needs an SSA→let-binding map. The
  // other two are statement-only and naturally form ordering points
  // because the emitter walks ops in IR order without statement-level
  // re-ordering. See `.omc/specs/deep-interview-leet-triton-l1d2b-...md`.
  std::map<mlir::Operation *, unsigned> _letBound;
  // AC4 v6: emit ONE threadgroup buffer per kernel that all
  // SimdgroupLoadDeviceStagedOp calls reuse, instead of one buffer per
  // call. With 64×64 multi-tile multi-warp kernels each warp issues
  // O(mTiles/warpsM × nTiles/warpsN × K_TILES × 2) staged loads (≥128);
  // per-call buffers exceed Apple's threadgroup memory budget and crash
  // the Metal compiler (XPC_ERROR_CONNECTION_INTERRUPTED). Each staged
  // load brackets its coop-load with `threadgroup_barrier` so the shared
  // buffer can be safely reused. Tracked here so the kernel-body walker
  // emits the declaration exactly once.
  bool _sharedStageBufferDeclared = false;
  bool _fstoreScratchDeclared = false;
  // Set by a translator that has emitted an `op.emitError()` and cannot produce
  // correct MSL. `translateModule` turns it into a `failure()` so the compile
  // aborts instead of handing back a silently-wrong kernel. Without this the
  // only signal is a diagnostic on stderr, which nothing checks.
  bool _emitFailed = false;
  unsigned _varCount = 0;
  bool inWhileCondition = false;
  int _curIndent = 0;
  raw_ostream &_output;

  void indent();
  void printDelim();
  void translateVarName(mlir::Value memref);

  void translateKernels();
  void translateKernel(mlir::triton::metal::KernelOp op);

  bool isStatementPrintable(Operation *opInst);
  void translateStatement(Operation *opInst);
  void translate(mlir::triton::metal::AllocaOp op);
  void translate(mlir::triton::metal::ThreadgroupAllocaOp op);
  void translate(mlir::triton::metal::BarrierOp op);
  void translate(mlir::triton::metal::TgStoreIndexedOp op);
  void translate(mlir::triton::metal::StoreOp op);
  void translate(mlir::triton::metal::ThreadgroupPrefixSumOp op);
  void translate(mlir::triton::metal::ThreadgroupSegmentedPrefixSumOp op);
  void translate(mlir::triton::metal::ThreadgroupAffinePrefixScanOp op);
  void translate(mlir::triton::metal::AtomicRmwOp op);
  void translate(mlir::triton::metal::IfOp op);
  void translate(mlir::triton::metal::WhileOp op);
  void translate(mlir::triton::metal::MatmulOp op);
  void emitScalarMatmul_(mlir::triton::metal::MatmulOp op);
  void emitMmaMatmul_(mlir::triton::metal::MatmulOp op);
  void translate(mlir::triton::metal::GemvOp op);
  void translate(mlir::triton::metal::QmvOp op);
  void translate(mlir::triton::metal::QmmOp op);
  void translate(mlir::triton::metal::Int4WeightOnlyMatmulOp op);
  void translate(mlir::triton::metal::Int8QuantizedMatmulOp op);
  void translate(mlir::triton::metal::LinearAttentionPreprocessOp op);
  void translate(mlir::triton::metal::LinearAttentionApplyOp op);
  void translate(mlir::triton::metal::ReduceOp op);
  void translate(mlir::triton::metal::ArgmaxOp op);
  void translate(mlir::triton::metal::SoftmaxOp op);
  void translate(mlir::triton::metal::LogsumexpOp op);
  void translate(mlir::triton::metal::SdpaOp op);
  void translate(mlir::triton::metal::FusedAttentionOp op);
  // The two bodies behind `metal.fused_attention`, mirroring the dialect's
  // existing `MatmulKind::Scalar|Mma` split: a simdgroup body for tiles whose
  // working set fits threadgroup memory, and a per-key scalar body that is the
  // correctness floor and always applies. `translate` picks between them.
  void emitFusedAttentionScalar_(mlir::triton::metal::FusedAttentionOp op);
  void emitFusedAttentionMma_(mlir::triton::metal::FusedAttentionOp op);
  // Binds a fused-attention region's block args to MSL temps: the pinned
  // prefix from `fixedInits`, then one per `score_params` buffer. Both regions
  // take the same params in the same order and the verifier pins their types,
  // so one walk serves both. Returns false (and sets _emitFailed) if a param
  // does not resolve to a kernel buffer.
  bool bindFusedRegionArgs_(mlir::triton::metal::FusedAttentionOp op,
                            mlir::Block &body,
                            llvm::ArrayRef<std::string> fixedInits,
                            llvm::StringRef ind);
  // Emits the op's score-transform region inline as a run of MSL let-bindings
  // and returns the name holding the transformed score. The four expressions
  // are what the region's pinned block args bind to. Emitted inside the
  // innermost (per-key) loop, so the bindings are scoped per iteration; with
  // several key phases it is emitted once per phase, each emission rebinding
  // every name before its uses.
  std::string emitScoreRegion_(mlir::triton::metal::FusedAttentionOp op,
                               llvm::StringRef scoreExpr,
                               llvm::StringRef rowExpr, llvm::StringRef keyExpr,
                               llvm::StringRef phaseExpr, llvm::StringRef ind);
  // Emits the op's key-bounds region for one phase and returns the two names
  // holding the half-open key range `[start, end)` that phase sweeps.
  std::pair<std::string, std::string>
  emitKeyBoundsRegion_(mlir::triton::metal::FusedAttentionOp op,
                       llvm::StringRef blkExpr, llvm::StringRef phaseExpr,
                       llvm::StringRef mExpr, llvm::StringRef nExpr,
                       llvm::StringRef ind);
  void emitCausal_(mlir::triton::metal::SdpaOp op);
  void emitBoolMask_(mlir::triton::metal::SdpaOp op);
  void emitFloatMask_(mlir::triton::metal::SdpaOp op);
  void emitSinks_(mlir::triton::metal::SdpaOp op);
  void emitNonCausal_(mlir::triton::metal::SdpaOp op);
  void translate(mlir::triton::metal::RmsNormOp op);
  void translate(mlir::triton::metal::ReturnOp op);
  void translate(mlir::scf::IfOp op);
  void translate(mlir::scf::ForOp op);
  void translate(mlir::scf::YieldOp op);
  void translate(mlir::Region &region);

  void translateValue(Operation *opInst);
  // Wall 15: dispatch helper for operands that may be either an op-result
  // (defining op exists → translateValue) or a BlockArgument such as a
  // region iter-arg or induction variable (no defining op → translateVarName
  // via the `_buffers` mapping). Callsites that may consume iter-args MUST
  // route through this helper to avoid null-deref in translateValue.
  void translateValueOrVarName(mlir::Value v);
  void translate(mlir::triton::metal::ConstantOp op);
  void translate(mlir::triton::metal::GetElementOp op);
  void translate(mlir::triton::metal::TgLoadIndexedOp op);
  void translate(mlir::triton::metal::ThreadIdOp op);
  void translate(mlir::triton::metal::ThreadgroupIdOp op);
  void translate(mlir::triton::metal::ThreadgroupsPerGridOp op);
  void translate(mlir::triton::metal::CastOp op);
  void translate(mlir::triton::metal::BitcastOp op);
  void translate(mlir::triton::metal::UnaryExpOp op);
  void translate(mlir::triton::metal::BinaryExpOp op);
  void translate(mlir::triton::metal::YieldWhileOp op);

  // SIMD-group matrix translators. Emit the modern Metal 17.5 surface
  // (`simdgroup_load` / `simdgroup_multiply_accumulate` / `simdgroup_store`).
  // The legacy `_matrix`-suffixed names are rejected by the current MSL
  // compiler — see comment block in ModuleTranslation.cpp.
  void translate(mlir::triton::metal::SimdgroupIndexOp op);
  void translate(mlir::triton::metal::SimdgroupMatrixZeroOp op);
  void translate(mlir::triton::metal::SimdgroupLoadDeviceStagedOp op);
  void translate(mlir::triton::metal::SimdgroupLoadDeviceStagedMaskedOp op);
  void translate(mlir::triton::metal::SimdgroupLoadOp op);
  void translate(mlir::triton::metal::SimdgroupMultiplyAccumulateOp op);
  void translate(mlir::triton::metal::SimdgroupStoreOp op);
  void translate(mlir::triton::metal::SimdgroupFusedStoreOp op);
};

} } // end namespace metal, triton
} // end namespace mlir

#endif // METAL_MODULETRANSLATION_H
