//===--- ModuleTranslation.h ------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_MODULETRANSLATION_H
#define METAL_MODULETRANSLATION_H

#include "Dialect/Metal/IR/MetalOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
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
  void translate(mlir::triton::metal::IfOp op);
  void translate(mlir::triton::metal::WhileOp op);
  void translate(mlir::triton::metal::MatmulOp op);
  void emitScalarMatmul_(mlir::triton::metal::MatmulOp op);
  void emitMmaMatmul_(mlir::triton::metal::MatmulOp op);
  void translate(mlir::triton::metal::GemvOp op);
  void translate(mlir::triton::metal::QmvOp op);
  void translate(mlir::triton::metal::QmmOp op);
  void translate(mlir::triton::metal::ReduceOp op);
  void translate(mlir::triton::metal::ArgmaxOp op);
  void translate(mlir::triton::metal::SoftmaxOp op);
  void translate(mlir::triton::metal::LogsumexpOp op);
  void translate(mlir::triton::metal::SdpaOp op);
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
  void translate(mlir::triton::metal::ConstantOp op);
  void translate(mlir::triton::metal::GetElementOp op);
  void translate(mlir::triton::metal::TgLoadIndexedOp op);
  void translate(mlir::triton::metal::ThreadIdOp op);
  void translate(mlir::triton::metal::ThreadgroupIdOp op);
  void translate(mlir::triton::metal::CastOp op);
  void translate(mlir::triton::metal::UnaryExpOp op);
  void translate(mlir::triton::metal::BinaryExpOp op);
  void translate(mlir::triton::metal::YieldWhileOp op);

  // SIMD-group matrix scaffolding (matmul track session 2). The stubs emit
  // real MSL function-call syntax (simdgroup_load_matrix /
  // simdgroup_matrix_multiply_accumulate / simdgroup_store_matrix) so a
  // future tt.dot lowering's MSL is compilable; no conversion pattern uses
  // these ops yet.
  void translate(mlir::triton::metal::SimdgroupLoadOp op);
  void translate(mlir::triton::metal::SimdgroupMultiplyAccumulateOp op);
  void translate(mlir::triton::metal::SimdgroupStoreOp op);
};

} } // end namespace metal, triton
} // end namespace mlir

#endif // METAL_MODULETRANSLATION_H
