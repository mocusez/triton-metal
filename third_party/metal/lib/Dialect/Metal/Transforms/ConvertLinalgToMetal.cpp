//===--- ConvertLinalgToMetal.cpp -------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Dialect/Metal/Transforms/LinalgToMetal.h"
#include "Dialect/Metal/Transforms/MetalPasses.h"
#include "Dialect/Metal/IR/MetalDialect.h"
#include "Dialect/Metal/IR/MetalOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::metal {

#define GEN_PASS_DEF_CONVERTLINALGTOMETAL
#include "Dialect/Metal/Transforms/MetalPasses.h.inc"

namespace {

/// Returns true iff `divOp` is a `linalg.generic` whose body is exactly
/// `arith.divf %v, %const` (numerator-first) followed by `linalg.yield`,
/// where %const is the constant value `expectedR` and writes back to the
/// reduce's output buffer. This is the canonical "reduce-mean" detector.
static bool isCanonicalMeanDivide(mlir::linalg::GenericOp divOp,
                                  mlir::Value reduceOutBuffer,
                                  int64_t expectedR) {
  if (divOp.getNumDpsInputs() != 1 || divOp.getNumDpsInits() != 1)
    return false;
  if (divOp.getDpsInputOperand(0)->get() != reduceOutBuffer)
    return false;
  if (divOp.getDpsInitOperand(0)->get() != reduceOutBuffer)
    return false;
  // All iterators parallel; identity indexing.
  for (auto it : divOp.getIteratorTypesArray())
    if (it != mlir::utils::IteratorType::parallel)
      return false;
  for (auto m : divOp.getIndexingMapsArray())
    if (!m.isIdentity())
      return false;
  mlir::Region &body = divOp.getRegion();
  if (!body.hasOneBlock())
    return false;
  mlir::Block &block = body.front();
  if (block.getNumArguments() != 2)
    return false;
  auto it = block.begin(), end = block.end();
  if (it == end)
    return false;
  mlir::Operation &arithOp = *it++;
  if (it == end)
    return false;
  mlir::Operation &yieldOp = *it++;
  if (it != end)
    return false;
  if (!llvm::isa<mlir::linalg::YieldOp>(yieldOp))
    return false;
  auto divf = llvm::dyn_cast<mlir::arith::DivFOp>(arithOp);
  if (!divf)
    return false;
  if (yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0) != divf.getResult())
    return false;
  // Numerator-first form: arith.divf %v, %const where %v is block arg 0
  // and %const matches expectedR.
  if (divf.getLhs() != block.getArgument(0))
    return false;
  mlir::Attribute constAttr;
  if (!mlir::matchPattern(divf.getRhs(),
                          mlir::m_Constant(&constAttr)))
    return false;
  if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(constAttr)) {
    double v = fa.getValueAsDouble();
    return v == static_cast<double>(expectedR);
  }
  return false;
}

struct ConvertLinalgToMetal
    : public impl::ConvertLinalgToMetalBase<ConvertLinalgToMetal> {
  using impl::ConvertLinalgToMetalBase<
      ConvertLinalgToMetal>::ConvertLinalgToMetalBase;

  // Adjacent divide-by-r generic ops that were paired with a reduce-sum
  // and reclassified as mean. Erased after applyPartialConversion succeeds.
  llvm::SmallVector<mlir::linalg::GenericOp> meanDivides;

  // Pre-validates every linalg op in scope against the pass's input
  // contract (static shape, supported element type, innermost-axis,
  // recognized combiner, kernel-context for matmul/gemv). Emits
  // user-visible diagnostics for every violation so that
  // `--verify-diagnostics` fixtures can pin specific messages.
  bool preValidate(mlir::ModuleOp module) {
    bool ok = true;

    module.walk([&](mlir::linalg::SoftmaxOp op) {
      auto inMref = llvm::dyn_cast<mlir::MemRefType>(op.getInput().getType());
      auto outMref =
          llvm::dyn_cast<mlir::MemRefType>(op.getOutput().getType());
      if (!inMref || !outMref) {
        op.emitOpError() << "convert-linalg-to-metal: softmax operands must "
                            "be memrefs";
        ok = false;
        return;
      }
      if (!inMref.hasStaticShape() || !outMref.hasStaticShape()) {
        op.emitOpError() << "convert-linalg-to-metal: dynamic shapes are not "
                            "supported";
        ok = false;
        return;
      }
      auto et = inMref.getElementType();
      if (!(et.isF32() || et.isF16() || et.isBF16())) {
        op.emitOpError() << "convert-linalg-to-metal: unsupported element "
                            "type for softmax";
        ok = false;
        return;
      }
      int64_t rank = inMref.getRank();
      if (rank < 1 || op.getDimension() != rank - 1) {
        op.emitOpError() << "convert-linalg-to-metal: only innermost-axis "
                            "softmax is supported";
        ok = false;
        return;
      }
    });

    module.walk([&](mlir::linalg::ReduceOp op) {
      if (op.getInputs().size() != 1 || op.getInits().size() != 1) {
        op.emitOpError() << "convert-linalg-to-metal: linalg.reduce must "
                            "have exactly one input and one init";
        ok = false;
        return;
      }
      auto inMref =
          llvm::dyn_cast<mlir::MemRefType>(op.getInputs()[0].getType());
      auto outMref =
          llvm::dyn_cast<mlir::MemRefType>(op.getInits()[0].getType());
      if (!inMref || !outMref) {
        op.emitOpError() << "convert-linalg-to-metal: reduce operands must "
                            "be memrefs";
        ok = false;
        return;
      }
      if (!inMref.hasStaticShape() || !outMref.hasStaticShape()) {
        op.emitOpError() << "convert-linalg-to-metal: dynamic shapes are not "
                            "supported";
        ok = false;
        return;
      }
      auto et = inMref.getElementType();
      if (!(et.isF32() || et.isF16() || et.isBF16())) {
        op.emitOpError() << "convert-linalg-to-metal: unsupported element "
                            "type for reduce";
        ok = false;
        return;
      }
      int64_t rank = inMref.getRank();
      auto dims = op.getDimensions();
      if (dims.size() != 1 || dims[0] != rank - 1) {
        op.emitOpError() << "convert-linalg-to-metal: only innermost-axis "
                            "reduce is supported";
        ok = false;
        return;
      }

      // Combiner shape: exactly one arith op (addf or maximumf) followed by
      // linalg.yield consuming its result; commutative operand ordering on
      // the two block args is accepted.
      mlir::Region &combiner = op.getCombiner();
      bool combinerOk = false;
      bool isAddf = false;
      if (combiner.hasOneBlock()) {
        mlir::Block &block = combiner.front();
        if (block.getNumArguments() == 2) {
          auto it = block.begin(), end = block.end();
          if (it != end) {
            mlir::Operation &arithOp = *it++;
            if (it != end) {
              mlir::Operation &yieldOp = *it++;
              if (it == end &&
                  llvm::isa<mlir::linalg::YieldOp>(yieldOp) &&
                  yieldOp.getNumOperands() == 1 &&
                  yieldOp.getOperand(0) == arithOp.getResult(0) &&
                  arithOp.getNumOperands() == 2) {
                mlir::Value a0 = block.getArgument(0);
                mlir::Value a1 = block.getArgument(1);
                bool operandsAreBlockArgs =
                    (arithOp.getOperand(0) == a0 &&
                     arithOp.getOperand(1) == a1) ||
                    (arithOp.getOperand(0) == a1 &&
                     arithOp.getOperand(1) == a0);
                if (operandsAreBlockArgs &&
                    (llvm::isa<mlir::arith::AddFOp>(arithOp) ||
                     llvm::isa<mlir::arith::MaximumFOp>(arithOp))) {
                  combinerOk = true;
                  isAddf = llvm::isa<mlir::arith::AddFOp>(arithOp);
                }
              }
            }
          }
        }
      }
      if (!combinerOk) {
        op.emitOpError() << "convert-linalg-to-metal: linalg.reduce combiner "
                            "must be a single arith.addf or arith.maximumf";
        ok = false;
        return;
      }

      // Mean detection: only for addf combiners. Dedupe users of the
      // reduce output buffer (a single op consuming it as both ins and
      // outs counts once); accept exactly one unique non-reduce user that
      // is the canonical divide-by-r generic.
      if (isAddf) {
        int64_t r = inMref.getShape()[rank - 1];
        mlir::Value outBuf = op.getInits()[0];
        llvm::SmallPtrSet<mlir::Operation *, 4> uniqueUsers;
        for (auto user : outBuf.getUsers()) {
          if (user == op.getOperation())
            continue;
          uniqueUsers.insert(user);
        }
        if (uniqueUsers.size() == 1) {
          if (auto gen = llvm::dyn_cast<mlir::linalg::GenericOp>(
                  *uniqueUsers.begin())) {
            if (isCanonicalMeanDivide(gen, outBuf, r)) {
              op->setAttr("convert_linalg_to_metal.mean",
                          mlir::UnitAttr::get(op.getContext()));
              meanDivides.push_back(gen);
            }
          }
        }
      }
    });

    // Matmul validation: kernel-context, static shape, dtype, M/N>=2.
    module.walk([&](mlir::linalg::MatmulOp op) {
      if (!op->getParentOfType<triton::metal::KernelOp>()) {
        op.emitOpError() << "convert-linalg-to-metal: linalg.matmul outside "
                            "metal.kernel — run "
                            "--convert-funcs-to-metal-kernels first (add "
                            "`attributes {metal.kernel}` to the enclosing "
                            "func.func if needed)";
        ok = false;
        return;
      }
      auto inputs = op.getDpsInputs();
      auto inits = op.getDpsInits();
      if (inputs.size() != 2 || inits.size() != 1) {
        op.emitOpError() << "convert-linalg-to-metal: linalg.matmul must "
                            "have exactly two inputs and one init";
        ok = false;
        return;
      }
      auto a = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      auto b = llvm::dyn_cast<mlir::MemRefType>(inputs[1].getType());
      auto c = llvm::dyn_cast<mlir::MemRefType>(inits[0].getType());
      if (!a || !b || !c) {
        op.emitOpError() << "convert-linalg-to-metal: matmul operands must "
                            "be memrefs";
        ok = false;
        return;
      }
      if (!a.hasStaticShape() || !b.hasStaticShape() || !c.hasStaticShape()) {
        op.emitOpError() << "convert-linalg-to-metal: dynamic shapes are "
                            "not supported";
        ok = false;
        return;
      }
      auto et = a.getElementType();
      if (!(et.isF32() || et.isF16() || et.isBF16())) {
        op.emitOpError() << "convert-linalg-to-metal: unsupported element "
                            "type for matmul";
        ok = false;
        return;
      }
      if (b.getElementType() != et || c.getElementType() != et) {
        op.emitOpError() << "convert-linalg-to-metal: matmul operands must "
                            "share element type";
        ok = false;
        return;
      }
      if (a.getRank() != 2 || b.getRank() != 2 || c.getRank() != 2) {
        op.emitOpError() << "convert-linalg-to-metal: matmul operands must "
                            "be rank-2 memrefs";
        ok = false;
        return;
      }
      int64_t M = a.getShape()[0], N = b.getShape()[1];
      if (M == 1 || N == 1) {
        op.emitOpError() << "convert-linalg-to-metal: use linalg.matvec for "
                            "M==1 or N==1";
        ok = false;
        return;
      }
    });

    // Matvec validation: kernel-context, static shape, dtype.
    module.walk([&](mlir::linalg::MatvecOp op) {
      if (!op->getParentOfType<triton::metal::KernelOp>()) {
        op.emitOpError() << "convert-linalg-to-metal: linalg.matvec outside "
                            "metal.kernel — run "
                            "--convert-funcs-to-metal-kernels first (add "
                            "`attributes {metal.kernel}` to the enclosing "
                            "func.func if needed)";
        ok = false;
        return;
      }
      auto inputs = op.getDpsInputs();
      auto inits = op.getDpsInits();
      if (inputs.size() != 2 || inits.size() != 1) {
        op.emitOpError() << "convert-linalg-to-metal: linalg.matvec must "
                            "have exactly two inputs and one init";
        ok = false;
        return;
      }
      auto A = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
      auto x = llvm::dyn_cast<mlir::MemRefType>(inputs[1].getType());
      auto y = llvm::dyn_cast<mlir::MemRefType>(inits[0].getType());
      if (!A || !x || !y) {
        op.emitOpError() << "convert-linalg-to-metal: matvec operands must "
                            "be memrefs";
        ok = false;
        return;
      }
      if (!A.hasStaticShape() || !x.hasStaticShape() || !y.hasStaticShape()) {
        op.emitOpError() << "convert-linalg-to-metal: dynamic shapes are "
                            "not supported";
        ok = false;
        return;
      }
      auto et = A.getElementType();
      if (!(et.isF32() || et.isF16() || et.isBF16())) {
        op.emitOpError() << "convert-linalg-to-metal: unsupported element "
                            "type for matvec";
        ok = false;
        return;
      }
      if (x.getElementType() != et || y.getElementType() != et) {
        op.emitOpError() << "convert-linalg-to-metal: matvec operands must "
                            "share element type";
        ok = false;
        return;
      }
      if (A.getRank() != 2 || x.getRank() != 1 || y.getRank() != 1) {
        op.emitOpError() << "convert-linalg-to-metal: matvec operand ranks "
                            "must be A:rank-2, x:rank-1, y:rank-1";
        ok = false;
        return;
      }
    });

    return ok;
  }


  void runOnOperation() final {
    auto *ctx = &getContext();

    if (!preValidate(getOperation())) {
      signalPassFailure();
      return;
    }

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<triton::metal::MetalDialect>();
    // Other dialects remain legal — this is a partial conversion.
    target.addLegalDialect<mlir::func::FuncDialect,
                           mlir::memref::MemRefDialect,
                           mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                           mlir::linalg::LinalgDialect>();
    // The four supported linalg ops are illegal.
    target.addIllegalOp<mlir::linalg::SoftmaxOp, mlir::linalg::ReduceOp,
                        mlir::linalg::MatmulOp, mlir::linalg::MatvecOp>();

    mlir::TypeConverter typeConverter;
    populateLinalgToMetalTypeConversions(typeConverter);

    mlir::RewritePatternSet patterns(ctx);
    populateLinalgToMetalConversionPatterns(typeConverter, patterns);

    // Convert func.func signatures and func.return operand types so
    // memref<...> block arguments become !metal.memref<...> without
    // leaving unresolved materialization casts.
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return typeConverter.isLegal(op.getOperandTypes());
        });
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();

    // Erase the divide-by-r generic ops whose paired reduce was lowered
    // as metal.reduce kind=mean. Done post-conversion so we don't disturb
    // the dialect-conversion driver's value-mapping for the reduce output.
    for (auto div : meanDivides)
      div.erase();
    meanDivides.clear();

    // Erase any unrealized_conversion_cast ops left over with no users
    // (typically the casts inserted by convert-funcs-to-metal-kernels that
    // were superseded by patterns consuming the kernel block args directly).
    llvm::SmallVector<mlir::UnrealizedConversionCastOp> deadCasts;
    getOperation().walk([&](mlir::UnrealizedConversionCastOp castOp) {
      bool allDead = true;
      for (auto r : castOp.getResults())
        if (!r.use_empty()) {
          allDead = false;
          break;
        }
      if (allDead)
        deadCasts.push_back(castOp);
    });
    for (auto c : deadCasts)
      c.erase();
  }
};

} // namespace
} // namespace mlir::triton::metal
