//===--- LinalgToMetal.cpp ---------------------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Dialect/Metal/Transforms/LinalgToMetal.h"
#include "Dialect/Metal/IR/MetalOps.h"
#include "Dialect/Metal/IR/MetalTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::triton::metal {

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

static bool isSupportedElementType(mlir::Type t) {
  return t.isF32() || t.isF16() || t.isBF16();
}

// Public helper declared in LinalgToMetal.h. Shared with the wrapping pass.
std::optional<unsigned> safeFlatSize(llvm::ArrayRef<int64_t> shape) {
  uint64_t acc = 1;
  for (int64_t d : shape) {
    if (d < 0)
      return std::nullopt;
    uint64_t ud = static_cast<uint64_t>(d);
    if (ud != 0 && acc > std::numeric_limits<uint64_t>::max() / ud)
      return std::nullopt;
    acc *= ud;
  }
  if (acc > std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return static_cast<unsigned>(acc);
}

void populateLinalgToMetalTypeConversions(mlir::TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](MetalMemRefType t) -> std::optional<mlir::Type> { return t; });

  // Returning std::nullopt signals failure to the conversion framework so
  // unsupported memrefs (dynamic shape, wrong element type, overflowing
  // flat size) are rejected without producing wrong-sized metal memrefs.
  typeConverter.addConversion(
      [](mlir::MemRefType t) -> std::optional<mlir::Type> {
        if (!t.hasStaticShape())
          return std::nullopt;
        auto et = t.getElementType();
        if (!isSupportedElementType(et))
          return std::nullopt;
        auto flat = safeFlatSize(t.getShape());
        if (!flat)
          return std::nullopt;
        return MetalMemRefType::get(t.getContext(), et, *flat);
      });

  // Allow scalar types of interest to pass through unchanged so that
  // attribute-only operations remain legal.
  typeConverter.addConversion([](mlir::FloatType t) { return t; });
  typeConverter.addConversion([](mlir::IntegerType t) { return t; });
  typeConverter.addConversion([](mlir::IndexType t) { return t; });
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Lightweight precondition check used by the patterns. Pre-validation in
/// the pass driver already produced human-readable diagnostics for any
/// violation, so on failure the pattern silently bails via
/// `notifyMatchFailure` from its caller.
bool isStaticFpMemRef(mlir::MemRefType mref) {
  return mref.hasStaticShape() &&
         isSupportedElementType(mref.getElementType());
}

//===----------------------------------------------------------------------===//
// linalg.softmax -> metal.softmax
//===----------------------------------------------------------------------===//

struct ConvertLinalgSoftmax
    : public mlir::OpConversionPattern<mlir::linalg::SoftmaxOp> {
  using mlir::OpConversionPattern<mlir::linalg::SoftmaxOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::SoftmaxOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // The pass driver pre-validates softmax inputs and emits user-visible
    // diagnostics; here we only silently bail if the precondition is
    // somehow violated by a non-driver caller of the pattern set.
    auto inMref = llvm::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto outMref = llvm::dyn_cast<mlir::MemRefType>(op.getOutput().getType());
    if (!inMref || !outMref || !isStaticFpMemRef(inMref) ||
        !isStaticFpMemRef(outMref))
      return rewriter.notifyMatchFailure(op, "softmax preconditions unmet");

    int64_t rank = inMref.getRank();
    if (rank < 1 || op.getDimension() != rank - 1)
      return rewriter.notifyMatchFailure(op, "softmax axis precondition unmet");

    auto inShape = inMref.getShape();
    auto rFlat = safeFlatSize({inShape[rank - 1]});
    auto mOuterFlat = safeFlatSize(inShape.drop_back(1));
    if (!rFlat || !mOuterFlat)
      return rewriter.notifyMatchFailure(op, "softmax size overflow");
    unsigned r = *rFlat;
    unsigned mOuter = *mOuterFlat;

    mlir::Value newIn = adaptor.getInput();
    mlir::Value newOut = adaptor.getOutput();

    triton::metal::SoftmaxOp::create(rewriter, 
        op.getLoc(), newIn, newOut,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(mOuter)),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(r)));

    // In memref-form linalg.softmax has zero SSA results (the destination is
    // an `outs` buffer); in tensor-form it has one result that aliases the
    // output. Branch on the op's result count to do the right thing.
    if (op.getNumResults() == 0)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, newOut);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// linalg.reduce -> metal.reduce (sum/max)
//===----------------------------------------------------------------------===//

/// Inspects the combiner region of a linalg.reduce and classifies it as
/// `sum`, `max`, or unsupported. Returns std::nullopt on unsupported shape.
std::optional<MetalReduceKind>
classifyReduceCombiner(mlir::linalg::ReduceOp op) {
  mlir::Region &combiner = op.getCombiner();
  if (!combiner.hasOneBlock())
    return std::nullopt;

  mlir::Block &block = combiner.front();
  if (block.getNumArguments() != 2)
    return std::nullopt;

  auto opsBegin = block.begin();
  auto opsEnd = block.end();
  if (opsBegin == opsEnd)
    return std::nullopt;
  mlir::Operation &arithOp = *opsBegin;
  ++opsBegin;
  if (opsBegin == opsEnd)
    return std::nullopt;
  mlir::Operation &yieldOp = *opsBegin;
  ++opsBegin;
  if (opsBegin != opsEnd)
    return std::nullopt;

  if (!llvm::isa<mlir::linalg::YieldOp>(yieldOp))
    return std::nullopt;
  if (arithOp.getNumResults() != 1 || arithOp.getNumOperands() != 2)
    return std::nullopt;
  if (yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0) != arithOp.getResult(0))
    return std::nullopt;
  mlir::Value a0 = block.getArgument(0);
  mlir::Value a1 = block.getArgument(1);
  bool operandsAreBlockArgs =
      (arithOp.getOperand(0) == a0 && arithOp.getOperand(1) == a1) ||
      (arithOp.getOperand(0) == a1 && arithOp.getOperand(1) == a0);
  if (!operandsAreBlockArgs)
    return std::nullopt;

  if (llvm::isa<mlir::arith::AddFOp>(arithOp))
    return MetalReduceKind::sum;
  if (llvm::isa<mlir::arith::MaximumFOp>(arithOp))
    return MetalReduceKind::max;
  return std::nullopt;
}

struct ConvertLinalgReduce
    : public mlir::OpConversionPattern<mlir::linalg::ReduceOp> {
  using mlir::OpConversionPattern<mlir::linalg::ReduceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::ReduceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Pre-validation in the pass driver guarantees the preconditions below;
    // patterns just bail silently if invoked from a non-driver context.
    if (op.getInputs().size() != 1 || op.getInits().size() != 1)
      return rewriter.notifyMatchFailure(op, "reduce arity precondition unmet");

    auto inMref =
        llvm::dyn_cast<mlir::MemRefType>(op.getInputs()[0].getType());
    auto outMref =
        llvm::dyn_cast<mlir::MemRefType>(op.getInits()[0].getType());
    if (!inMref || !outMref || !isStaticFpMemRef(inMref) ||
        !isStaticFpMemRef(outMref))
      return rewriter.notifyMatchFailure(op, "reduce preconditions unmet");

    int64_t rank = inMref.getRank();
    auto dims = op.getDimensions();
    if (dims.size() != 1 || dims[0] != rank - 1)
      return rewriter.notifyMatchFailure(op, "reduce axis precondition unmet");

    auto kindOpt = classifyReduceCombiner(op);
    if (!kindOpt)
      return rewriter.notifyMatchFailure(op, "reduce combiner unrecognized");

    // Mean override: ConvertLinalgToMetal::preValidate tags reduces whose
    // adjacent divide-by-r matched the canonical mean shape with the
    // `convert_linalg_to_metal.mean` unit attribute. Honor it here so
    // the converted op carries kind=mean instead of sum.
    MetalReduceKind kind = *kindOpt;
    if (op->hasAttr("convert_linalg_to_metal.mean"))
      kind = MetalReduceKind::mean;

    auto inShape = inMref.getShape();
    auto rFlat = safeFlatSize({inShape[rank - 1]});
    auto mOuterFlat = safeFlatSize(inShape.drop_back(1));
    if (!rFlat || !mOuterFlat)
      return rewriter.notifyMatchFailure(op, "reduce size overflow");
    unsigned r = *rFlat;
    unsigned mOuter = *mOuterFlat;

    auto kindAttr = MetalReduceKindAttr::get(rewriter.getContext(), kind);
    mlir::Value newIn = adaptor.getInputs()[0];
    mlir::Value newOut = adaptor.getInits()[0];

    triton::metal::ReduceOp::create(rewriter, 
        op.getLoc(), newIn, newOut, kindAttr,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(mOuter)),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(r)));

    if (op.getNumResults() == 0)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, newOut);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// linalg.matmul -> metal.matmul (kind=Scalar)
//===----------------------------------------------------------------------===//

struct ConvertLinalgMatmul
    : public mlir::OpConversionPattern<mlir::linalg::MatmulOp> {
  using mlir::OpConversionPattern<mlir::linalg::MatmulOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatmulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Preconditions guaranteed by preValidate in the pass driver; silently
    // bail otherwise.
    auto inputs = op.getDpsInputs();
    auto inits = op.getDpsInits();
    if (inputs.size() != 2 || inits.size() != 1)
      return rewriter.notifyMatchFailure(op, "matmul arity precondition unmet");
    auto a = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
    auto b = llvm::dyn_cast<mlir::MemRefType>(inputs[1].getType());
    auto c = llvm::dyn_cast<mlir::MemRefType>(inits[0].getType());
    if (!a || !b || !c || !isStaticFpMemRef(a) || !isStaticFpMemRef(b) ||
        !isStaticFpMemRef(c))
      return rewriter.notifyMatchFailure(op, "matmul preconditions unmet");
    if (a.getRank() != 2 || b.getRank() != 2 || c.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "matmul rank precondition unmet");
    int64_t M = a.getShape()[0];
    int64_t K = a.getShape()[1];
    int64_t N = b.getShape()[1];
    if (M == 1 || N == 1)
      return rewriter.notifyMatchFailure(op, "matmul M==1/N==1 rejected");
    if (a.getElementType() != b.getElementType() ||
        a.getElementType() != c.getElementType())
      return rewriter.notifyMatchFailure(op, "matmul mixed dtypes");

    auto kindAttr = triton::metal::MatmulKindAttr::get(rewriter.getContext(),
                                               triton::metal::MatmulKind::Scalar);
    triton::metal::MatmulOp::create(rewriter, 
        op.getLoc(), adaptor.getInputs()[0], adaptor.getInputs()[1],
        adaptor.getOutputs()[0],
        rewriter.getI64IntegerAttr(M),
        rewriter.getI64IntegerAttr(N),
        rewriter.getI64IntegerAttr(K),
        kindAttr);

    if (op.getNumResults() == 0)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, adaptor.getOutputs()[0]);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// linalg.matvec -> metal.gemv
//===----------------------------------------------------------------------===//

struct ConvertLinalgMatvec
    : public mlir::OpConversionPattern<mlir::linalg::MatvecOp> {
  using mlir::OpConversionPattern<mlir::linalg::MatvecOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatvecOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto inputs = op.getDpsInputs();
    auto inits = op.getDpsInits();
    if (inputs.size() != 2 || inits.size() != 1)
      return rewriter.notifyMatchFailure(op, "matvec arity precondition unmet");
    auto A = llvm::dyn_cast<mlir::MemRefType>(inputs[0].getType());
    auto x = llvm::dyn_cast<mlir::MemRefType>(inputs[1].getType());
    auto y = llvm::dyn_cast<mlir::MemRefType>(inits[0].getType());
    if (!A || !x || !y || !isStaticFpMemRef(A) || !isStaticFpMemRef(x) ||
        !isStaticFpMemRef(y))
      return rewriter.notifyMatchFailure(op, "matvec preconditions unmet");
    if (A.getRank() != 2 || x.getRank() != 1 || y.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "matvec rank precondition unmet");
    int64_t M = A.getShape()[0];
    int64_t K = A.getShape()[1];
    if (A.getElementType() != x.getElementType() ||
        A.getElementType() != y.getElementType())
      return rewriter.notifyMatchFailure(op, "matvec mixed dtypes");

    triton::metal::GemvOp::create(rewriter, 
        op.getLoc(), adaptor.getInputs()[0], adaptor.getInputs()[1],
        adaptor.getOutputs()[0],
        rewriter.getI64IntegerAttr(M),
        rewriter.getI64IntegerAttr(K));

    if (op.getNumResults() == 0)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, adaptor.getOutputs()[0]);
    return mlir::success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void populateLinalgToMetalConversionPatterns(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertLinalgSoftmax, ConvertLinalgReduce, ConvertLinalgMatmul,
               ConvertLinalgMatvec>(typeConverter, patterns.getContext());
}

} // namespace mlir::triton::metal
