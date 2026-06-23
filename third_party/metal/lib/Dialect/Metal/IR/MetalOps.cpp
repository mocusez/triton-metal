//===--- MetalOps.cpp - Metal dialect ops ---------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Dialect/Metal/IR/MetalOps.h"
#include "Dialect/Metal/IR/MetalDialect.h"
#include "Dialect/Metal/IR/MetalOpsEnums.cpp.inc"
#include "Dialect/Metal/IR/MetalQuantizedHelpers.h"
#include "Dialect/Metal/IR/MetalTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir::triton::metal;

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

void ModuleOp::build(OpBuilder &builder, OperationState &result) {
  ensureTerminator(*result.addRegion(), builder, result.location);
}

void ModuleOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printRegion(getRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

mlir::ParseResult ModuleOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}))
    return mlir::failure();
  ModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

void KernelOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     llvm::SmallVectorImpl<Type> &buffers,
                     llvm::SmallVectorImpl<bool> &isAddressSpaceDevice) {
  result.addAttribute("name", builder.getStringAttr(name));
  result.addAttribute("address_space_device",
                      builder.getBoolArrayAttr(isAddressSpaceDevice));
  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  auto block = builder.createBlock(bodyRegion);
  for (auto type : buffers) {
    auto memref = MetalMemRefType::get(builder.getContext(), type, 0);
    block->addArguments(memref, result.location);
  }
}

mlir::Block &KernelOp::getEntryBlock() { return getRegion().front(); }

llvm::LogicalResult KernelOp::verify() {
  auto index = -1;
  for (auto it : llvm::enumerate(getBuffers())) {
    auto memRef =
        llvm::dyn_cast<mlir::triton::metal::MetalMemRefType>(it.value().getType());
    if (!memRef) {
      index = it.index();
      break;
    }

    auto type = memRef.getType();
    if (type.isF16() || type.isF32() || type.isBF16() || type.isIndex())
      continue;
    if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type)) {
      switch (intTy.getWidth()) {
      case 1:
      case 8:
      case 16:
      case 32:
      case 64:
        continue;
      }
    }

    index = it.index();
    break;
  }
  if (index != -1)
    return emitOpError() << "type #" << index << " must be compatible type";
  else
    return mlir::success();
}

mlir::Value KernelOp::getBuffer(uint32_t index) {
  return getBodyRegion().getBlocks().begin()->getArgument(index);
}

mlir::MutableArrayRef<mlir::BlockArgument> KernelOp::getBuffers() {
  return getBodyRegion().getBlocks().begin()->getArguments();
}

void KernelOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getName();
  printer << " address_space_device ";
  printer.printAttribute(getAddressSpaceDevice());
  printer << " ";
  printer.printRegion(getRegion(),
                      /*printEntryBlockArgs=*/true,
                      /*printBlockTerminators=*/true);
}

mlir::ParseResult KernelOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  llvm::StringRef name;
  mlir::Region *body = result.addRegion();
  mlir::Attribute value;
  if (parser.parseKeyword(&name) ||
      parser.parseKeyword("address_space_device") ||
      parser.parseAttribute(value, "address_space_device", result.attributes) ||
      parser.parseRegion(*body, {}))
    return mlir::failure();

  result.addAttribute("name", parser.getBuilder().getStringAttr(name));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                       TypedAttr attr) {
  ConstantOp::build(builder, state, attr.getType(), attr);
}

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult AllocaOp::verify() {
  if (llvm::dyn_cast<MetalMemRefType>(getResult().getType()).getSize() == 0)
    return emitOpError() << "memRef size cannot be 0";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ThreadgroupAllocaOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ThreadgroupAllocaOp::verify() {
  auto memRef = llvm::dyn_cast<MetalMemRefType>(getResult().getType());
  if (!memRef)
    return emitOpError() << "result type must be !metal.memref";
  if (memRef.getSize() == 0)
    return emitOpError() << "threadgroup memref size cannot be 0";
  auto elemTy = memRef.getType();
  // Restrict to the same scalar set the rest of the dialect uses (matches
  // KernelOp::verify). f32/i32 are the primary L3 reduce element types.
  bool ok = elemTy.isF16() || elemTy.isF32() || elemTy.isBF16();
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(elemTy)) {
    switch (intTy.getWidth()) {
    case 1:
    case 8:
    case 16:
    case 32:
    case 64:
      ok = true;
      break;
    }
  }
  if (!ok)
    return emitOpError() << "unsupported threadgroup element type " << elemTy;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TgStoreIndexedOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult TgStoreIndexedOp::verify() {
  auto memRef = llvm::dyn_cast<MetalMemRefType>(getBuffer().getType());
  if (!memRef)
    return emitOpError() << "buffer must be a !metal.memref";
  if (memRef.getType() != getValue().getType())
    return emitOpError() << "value type (" << getValue().getType()
                         << ") must match buffer element type ("
                         << memRef.getType() << ")";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TgLoadIndexedOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult TgLoadIndexedOp::verify() {
  auto memRef = llvm::dyn_cast<MetalMemRefType>(getBuffer().getType());
  if (!memRef)
    return emitOpError() << "buffer must be a !metal.memref";
  if (memRef.getType() != getResult().getType())
    return emitOpError() << "result type (" << getResult().getType()
                         << ") must match buffer element type ("
                         << memRef.getType() << ")";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Check Index
//===----------------------------------------------------------------------===//

static llvm::LogicalResult
checkIndex(mlir::Operation *op, MetalMemRefType memRef, mlir::Value index) {
  if (auto constantOp = index.getDefiningOp<ConstantOp>()) {
    auto attr = llvm::dyn_cast<mlir::IntegerAttr>(constantOp.getValue());
    uint64_t index = attr.getUInt();
    uint32_t size = memRef.getSize();
    if (size > 0 && index >= size)
      return op->emitOpError()
             << "index " << index << " is past the end of the memRef "
             << "(which contains " << size << " elements)";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult StoreOp::verify() {
  auto memRef = llvm::dyn_cast<MetalMemRefType>(getMemref().getType());
  auto valueType = getValue().getType();
  auto memRefType = memRef.getType();
  if (memRefType != valueType)
    return emitOpError() << "requires value's type (" << valueType
                         << ") to match memref type (" << memRefType << ")";
  return checkIndex(*this, memRef, getIndex());
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

void GetElementOp::build(OpBuilder &builder, OperationState &result,
                         Value memref, Value index) {
  result.addOperands(memref);
  result.addOperands(index);
  auto type = llvm::cast<MetalMemRefType>(memref.getType()).getType();
  result.types.push_back(type);
};

llvm::LogicalResult GetElementOp::verify() {
  auto memRef = llvm::dyn_cast<MetalMemRefType>(getMemref().getType());
  auto resultType = getResult().getType();
  auto memRefType = memRef.getType();
  if (memRefType != resultType)
    return emitOpError() << "requires memref type (" << memRefType
                         << ") to match return type (" << resultType << ")";

  return checkIndex(*this, memRef, getIndex());
}

//===----------------------------------------------------------------------===//
// ThreadIdOp
//===----------------------------------------------------------------------===//

void ThreadIdOp::build(OpBuilder &builder, OperationState &result,
                       StringRef dimension) {
  result.addAttribute("dimension", builder.getStringAttr(dimension));
  result.addTypes(builder.getIntegerType(32, false));
};

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ThreadIdOp::verify() {
  auto dim = getDimension();
  if (dim != "x" && dim != "y" && dim != "z")
    return emitOpError() << "requires dimension to be `x` or `y` or `z`, "
                         << "found `" << dim << "`";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ThreadgroupIdOp
//===----------------------------------------------------------------------===//

void ThreadgroupIdOp::build(OpBuilder &builder, OperationState &result,
                            StringRef dimension) {
  result.addAttribute("dimension", builder.getStringAttr(dimension));
  result.addTypes(builder.getIntegerType(32, false));
};

llvm::LogicalResult ThreadgroupIdOp::verify() {
  auto dim = getDimension();
  if (dim != "x" && dim != "y" && dim != "z")
    return emitOpError() << "requires dimension to be `x` or `y` or `z`, "
                         << "found `" << dim << "`";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ThreadgroupsPerGridOp
//===----------------------------------------------------------------------===//

void ThreadgroupsPerGridOp::build(OpBuilder &builder, OperationState &result,
                                  StringRef dimension) {
  result.addAttribute("dimension", builder.getStringAttr(dimension));
  result.addTypes(builder.getIntegerType(32, false));
};

llvm::LogicalResult ThreadgroupsPerGridOp::verify() {
  auto dim = getDimension();
  if (dim != "x" && dim != "y" && dim != "z")
    return emitOpError() << "requires dimension to be `x` or `y` or `z`, "
                         << "found `" << dim << "`";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnaryExpOp
//===----------------------------------------------------------------------===//

void UnaryExpOp::build(OpBuilder &builder, OperationState &result,
                       UnaryExpOperator unaryOperator, Value argument) {
  result.addTypes(argument.getType());
  auto attr = builder.getI64IntegerAttr(static_cast<int64_t>(unaryOperator));
  result.addAttribute("unaryOperator", attr);
  result.addOperands(argument);
}

llvm::LogicalResult UnaryExpOp::verify() {
  auto argType = getArgument().getType();
  auto resultType = getResult().getType();
  if (argType != resultType)
    return emitOpError() << "result type mismatch";

  using OP = mlir::triton::metal::UnaryExpOperator;
  switch (getUnaryOperator()) {
  case OP::notOp:
    if (!argType.isInteger(1))
      return emitOpError() << "argument type must be i1";
    break;
  case OP::minusOp:
    if (argType.isInteger(1) ||
        (!argType.isSignedInteger() && !argType.isF16() && !argType.isF32() &&
         !argType.isBF16()))
      return emitOpError() << "argument type must be signed integer or float";
    break;
  case OP::expOp:
  case OP::sqrtOp:
  case OP::erfOp:
  case OP::logOp:
  case OP::rsqrtOp:
  case OP::sinOp:
  case OP::cosOp:
    if (!argType.isF32())
      return emitOpError() << "argument type must be f32";
    break;
  }
  return mlir::success();
}

mlir::OpFoldResult UnaryExpOp::fold(FoldAdaptor adaptor) {
  auto constant =
      dyn_cast<mlir::triton::metal::ConstantOp>(getArgument().getDefiningOp());
  if (!constant)
    return nullptr;

  mlir::OpBuilder builder{constant.getContext()};

  switch (getUnaryOperator()) {
  case mlir::triton::metal::UnaryExpOperator::notOp: {
    auto value =
        llvm::dyn_cast<mlir::BoolAttr>(constant.getValueAttr()).getValue();
    return builder.getBoolAttr(!value);
  }
  case mlir::triton::metal::UnaryExpOperator::minusOp: {
    auto attr = constant.getValueAttr();
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
      return builder.getIntegerAttr(attr.getType(), -intAttr.getSInt());
    }
    if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
      return builder.getFloatAttr(attr.getType(),
                                  -floatAttr.getValueAsDouble());
    }
    return nullptr;
  }
  case mlir::triton::metal::UnaryExpOperator::expOp:
  case mlir::triton::metal::UnaryExpOperator::sqrtOp:
  case mlir::triton::metal::UnaryExpOperator::erfOp:
  case mlir::triton::metal::UnaryExpOperator::logOp:
  case mlir::triton::metal::UnaryExpOperator::rsqrtOp:
  case mlir::triton::metal::UnaryExpOperator::sinOp:
  case mlir::triton::metal::UnaryExpOperator::cosOp:
    return nullptr;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// BinaryExpOp
//===----------------------------------------------------------------------===//

void BinaryExpOp::build(OpBuilder &builder, OperationState &result,
                        BinaryExpOperator binaryOperator, Value lhs,
                        Value rhs) {
  using OP = mlir::triton::metal::BinaryExpOperator;
  switch (binaryOperator) {
  case OP::addOp:
  case OP::subOp:
  case OP::mulOp:
  case OP::divOp:
  case OP::remOp:
  case OP::maxOp:
    result.addTypes(lhs.getType());
    break;
  case OP::eqOp:
  case OP::neOp:
  case OP::ltOp:
  case OP::leOp:
  case OP::gtOp:
  case OP::geOp:
  case OP::andOp:
  case OP::orOp:
    result.addTypes(builder.getI1Type());
    break;
  }

  auto attr = builder.getI64IntegerAttr(static_cast<int64_t>(binaryOperator));
  result.addAttribute("binaryOperator", attr);
  result.addOperands(lhs);
  result.addOperands(rhs);
}

llvm::LogicalResult BinaryExpOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto resultType = getResult().getType();
  if (lhsType != rhsType)
    return emitOpError() << "arguments type mismatch";

  using OP = mlir::triton::metal::BinaryExpOperator;
  switch (getBinaryOperator()) {
  case OP::addOp:
  case OP::subOp:
  case OP::mulOp:
  case OP::divOp:
  case OP::remOp:
  case OP::maxOp:
    if (lhsType != resultType)
      return emitOpError() << "result type mismatch";
    break;
  case OP::eqOp:
  case OP::neOp:
  case OP::ltOp:
  case OP::leOp:
  case OP::gtOp:
  case OP::geOp:
  case OP::andOp:
  case OP::orOp:
    if (!resultType.isInteger(1))
      return emitOpError() << "result type mismatch";
    break;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult MatmulOp::verify() {
  auto lhsRef = llvm::dyn_cast<MetalMemRefType>(getLhs().getType());
  auto rhsRef = llvm::dyn_cast<MetalMemRefType>(getRhs().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!lhsRef || !rhsRef || !outRef)
    return emitOpError() << "metal.matmul operands must be metal memrefs";

  auto checkFloat = [&](Type t, const char *name) -> llvm::LogicalResult {
    if (!(t.isF16() || t.isF32() || t.isBF16()))
      return emitOpError() << "metal.matmul requires f16/f32/bf16 element type, "
                           << name << " has " << t;
    return mlir::success();
  };
  if (failed(checkFloat(lhsRef.getType(), "lhs")))
    return mlir::failure();
  if (failed(checkFloat(rhsRef.getType(), "rhs")))
    return mlir::failure();
  if (failed(checkFloat(outRef.getType(), "out")))
    return mlir::failure();
  // All three operands must share the same element type.
  auto lhsElemTy = lhsRef.getType();
  if (rhsRef.getType() != lhsElemTy || outRef.getType() != lhsElemTy)
    return emitOpError() << "metal.matmul operands must share element type; "
                         << "got lhs=" << lhsElemTy << ", rhs=" << rhsRef.getType()
                         << ", out=" << outRef.getType();

  int64_t m = getM(), n = getN(), k = getK();
  if (m <= 0 || n <= 0 || k <= 0)
    return emitOpError() << "metal.matmul requires positive M, N, K (got M="
                         << m << ", N=" << n << ", K=" << k << ")";
  auto kindAttr = getKindAttr();
  auto kind = kindAttr ? kindAttr.getValue() : ::mlir::triton::metal::MatmulKind::Scalar;
  switch (kind) {
  case ::mlir::triton::metal::MatmulKind::Scalar:
    if (m == 1 || n == 1) {
      auto err = emitOpError()
          << "metal.matmul rejects degenerate M==1 or N==1 in Scalar mode; "
             "GEMV is reserved for a later stage";
      err.attachNote(getLoc())
          << "consider kind = ::Mma with M and N padded to multiples of 8";
      return err;
    }
    break;
  case ::mlir::triton::metal::MatmulKind::Mma:
    if (m % 8 != 0 || n % 8 != 0 || k % 8 != 0)
      return emitOpError()
             << "M, N, and K must be multiples of 8 in ::Mma mode (got M=" << m
             << ", N=" << n << ", K=" << k << "); pad input dimensions";
    break;
  }

  auto checkStaticSize =
      [&](MetalMemRefType ref, int64_t expected,
          const char *name) -> llvm::LogicalResult {
    auto size = ref.getSize();
    if (size > 0 && static_cast<int64_t>(size) != expected)
      return emitOpError() << name << " static size " << size
                           << " disagrees with expected " << expected;
    return mlir::success();
  };
  if (failed(checkStaticSize(lhsRef, m * k, "lhs (M*K)")))
    return mlir::failure();
  if (failed(checkStaticSize(rhsRef, k * n, "rhs (K*N)")))
    return mlir::failure();
  if (failed(checkStaticSize(outRef, m * n, "out (M*N)")))
    return mlir::failure();

  auto outBlockArg = llvm::dyn_cast<mlir::BlockArgument>(getOut());
  if (!outBlockArg)
    return emitOpError()
           << "metal.matmul `out` must be a kernel buffer block argument";
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError() << "metal.matmul must be inside a metal.kernel";
  if (outBlockArg.getOwner() != &kernel.getEntryBlock())
    return emitOpError()
           << "metal.matmul `out` must be an entry-block buffer of the "
              "enclosing metal.kernel";
  unsigned outIdx = outBlockArg.getArgNumber();
  auto deviceAttr = kernel.getAddressSpaceDevice();
  if (outIdx >= deviceAttr.size())
    return emitOpError()
           << "out buffer index out of address_space_device bounds";
  auto outIsDevice =
      llvm::cast<mlir::BoolAttr>(deviceAttr[outIdx]).getValue();
  if (!outIsDevice)
    return emitOpError()
           << "metal.matmul `out` must be a device-address-space "
              "(writable) kernel buffer; got a constant buffer";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GemvOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult GemvOp::verify() {
  auto lhsRef = llvm::dyn_cast<MetalMemRefType>(getLhs().getType());
  auto rhsRef = llvm::dyn_cast<MetalMemRefType>(getRhs().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!lhsRef || !rhsRef || !outRef)
    return emitOpError() << "metal.gemv operands must be metal memrefs";

  auto elemTy = lhsRef.getType();
  if (!(elemTy.isF16() || elemTy.isF32() || elemTy.isBF16()))
    return emitOpError() << "metal.gemv requires f16/f32/bf16 element type, "
                         << "lhs has " << elemTy;
  if (rhsRef.getType() != elemTy || outRef.getType() != elemTy)
    return emitOpError() << "metal.gemv operands must share element type; "
                         << "got lhs=" << elemTy << ", rhs=" << rhsRef.getType()
                         << ", out=" << outRef.getType();

  int64_t m = getM(), k = getK();
  if (m <= 0 || k <= 0)
    return emitOpError() << "metal.gemv requires positive M, K (got M=" << m
                         << ", K=" << k << ")";

  auto checkStaticSize =
      [&](MetalMemRefType ref, int64_t expected,
          const char *name) -> llvm::LogicalResult {
    auto size = ref.getSize();
    if (size > 0 && static_cast<int64_t>(size) != expected)
      return emitOpError() << name << " static size " << size
                           << " disagrees with expected " << expected;
    return mlir::success();
  };
  if (failed(checkStaticSize(lhsRef, m * k, "lhs (M*K)")))
    return mlir::failure();
  if (failed(checkStaticSize(rhsRef, k, "rhs (K)")))
    return mlir::failure();
  if (failed(checkStaticSize(outRef, m, "out (M)")))
    return mlir::failure();

  auto outBlockArg = llvm::dyn_cast<mlir::BlockArgument>(getOut());
  if (!outBlockArg)
    return emitOpError()
           << "metal.gemv `out` must be a kernel buffer block argument";
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError() << "metal.gemv must be inside a metal.kernel";
  if (outBlockArg.getOwner() != &kernel.getEntryBlock())
    return emitOpError()
           << "metal.gemv `out` must be an entry-block buffer of the "
              "enclosing metal.kernel";
  unsigned outIdx = outBlockArg.getArgNumber();
  auto deviceAttr = kernel.getAddressSpaceDevice();
  if (outIdx >= deviceAttr.size())
    return emitOpError()
           << "out buffer index out of address_space_device bounds";
  if (!llvm::cast<mlir::BoolAttr>(deviceAttr[outIdx]).getValue())
    return emitOpError()
           << "metal.gemv `out` must be a device-address-space "
              "(writable) kernel buffer; got a constant buffer";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Quantized helpers (shared with ModuleTranslation via MetalQuantizedHelpers.h)
//===----------------------------------------------------------------------===//

// Shared verifier core for both QmvOp and QmmOp. Diagnostics emitted via `op`.
static llvm::LogicalResult
verifyQuantizedCommon(mlir::Operation *op, MetalMemRefType wqRef,
                      MetalMemRefType scalesRef, MetalMemRefType biasesRef,
                      MetalMemRefType xRef, MetalMemRefType outRef, int64_t bits,
                      int64_t groupSize, int64_t k, int64_t numRows,
                      int64_t xLen, int64_t outLen) {
  if (!isValidBits(bits))
    return op->emitOpError() << "bits must be in {2,3,4,5,6,8}, got " << bits;
  if (k <= 0)
    return op->emitOpError() << "K must be positive, got " << k;
  if (!isValidGroupSize(bits, groupSize, k))
    return op->emitOpError()
           << "group_size " << groupSize << " is invalid for bits=" << bits
           << ", K=" << k << " (require >=32, power-of-two, "
           << "group_size % pack_factor == 0, K % group_size == 0)";
  // Activation/scales/biases/out share float dtype.
  auto sharedTy = scalesRef.getType();
  if (!(sharedTy.isF16() || sharedTy.isF32() || sharedTy.isBF16()))
    return op->emitOpError()
           << "scales/biases/x/out element type must be f16/f32/bf16, got "
           << sharedTy;
  if (biasesRef.getType() != sharedTy || xRef.getType() != sharedTy ||
      outRef.getType() != sharedTy)
    return op->emitOpError() << "scales, biases, x, out must share dtype";
  // Wq element type is bits-dependent.
  if (!isWqElementTypeFor(wqRef.getType(), bits))
    return op->emitOpError()
           << "wq element type for bits=" << bits << " must be "
           << wqElementTypeName(bits) << ", got " << wqRef.getType();
  // Static-size checks when available.
  int64_t pf = packFactor(bits);
  int64_t bpp = bytesPerPack(bits);
  int64_t numGroups = numRows * (k / groupSize);
  int64_t wqExpected =
      (bits == 2 || bits == 4 || bits == 8) ? (numRows * (k / pf))
                                            : (numRows * (k / pf) * bpp);
  auto checkSize = [&](MetalMemRefType ref, int64_t expected,
                       const char *name) -> llvm::LogicalResult {
    auto size = ref.getSize();
    if (size > 0 && static_cast<int64_t>(size) != expected)
      return op->emitOpError() << name << " static size " << size
                               << " disagrees with expected " << expected;
    return mlir::success();
  };
  if (failed(checkSize(wqRef, wqExpected, "wq")))
    return mlir::failure();
  if (failed(checkSize(scalesRef, numGroups, "scales")))
    return mlir::failure();
  if (failed(checkSize(biasesRef, numGroups, "biases")))
    return mlir::failure();
  if (failed(checkSize(xRef, xLen, "x")))
    return mlir::failure();
  if (failed(checkSize(outRef, outLen, "out")))
    return mlir::failure();
  return mlir::success();
}

static llvm::LogicalResult verifyQuantizedOutBuffer(mlir::Operation *op,
                                                    mlir::Value outVal) {
  auto outBlockArg = llvm::dyn_cast<mlir::BlockArgument>(outVal);
  if (!outBlockArg)
    return op->emitOpError() << "`out` must be a kernel buffer block argument";
  auto kernel = op->getParentOfType<KernelOp>();
  if (!kernel)
    return op->emitOpError() << "must be inside a metal.kernel";
  if (outBlockArg.getOwner() != &kernel.getEntryBlock())
    return op->emitOpError() << "`out` must be an entry-block buffer";
  unsigned idx = outBlockArg.getArgNumber();
  auto deviceAttr = kernel.getAddressSpaceDevice();
  if (idx >= deviceAttr.size())
    return op->emitOpError()
           << "out buffer index out of address_space_device bounds";
  if (!llvm::cast<mlir::BoolAttr>(deviceAttr[idx]).getValue())
    return op->emitOpError()
           << "`out` must be a device-address-space (writable) kernel buffer";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// QmvOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult QmvOp::verify() {
  auto wqRef = llvm::dyn_cast<MetalMemRefType>(getWq().getType());
  auto scalesRef = llvm::dyn_cast<MetalMemRefType>(getScales().getType());
  auto biasesRef = llvm::dyn_cast<MetalMemRefType>(getBiases().getType());
  auto xRef = llvm::dyn_cast<MetalMemRefType>(getX().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!wqRef || !scalesRef || !biasesRef || !xRef || !outRef)
    return emitOpError() << "metal.qmv operands must be metal memrefs";
  int64_t m = getM(), k = getK();
  if (m <= 0) return emitOpError() << "M must be positive, got " << m;
  if (failed(verifyQuantizedCommon(*this, wqRef, scalesRef, biasesRef, xRef,
                                    outRef, getBits(), getGroupSize(), k, m,
                                    k, m)))
    return mlir::failure();
  return verifyQuantizedOutBuffer(*this, getOut());
}

//===----------------------------------------------------------------------===//
// QmmOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult QmmOp::verify() {
  auto wqRef = llvm::dyn_cast<MetalMemRefType>(getWq().getType());
  auto scalesRef = llvm::dyn_cast<MetalMemRefType>(getScales().getType());
  auto biasesRef = llvm::dyn_cast<MetalMemRefType>(getBiases().getType());
  auto xRef = llvm::dyn_cast<MetalMemRefType>(getX().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!wqRef || !scalesRef || !biasesRef || !xRef || !outRef)
    return emitOpError() << "metal.qmm operands must be metal memrefs";
  int64_t m = getM(), n = getN(), k = getK();
  if (m <= 0 || n <= 0)
    return emitOpError() << "M and N must be positive, got M=" << m
                         << ", N=" << n;
  // qmm: Wq describes [N, K] weight, scales/biases [N, K/group_size],
  // x is [M, K] activation, out is [M, N]. numRows = N for the helper.
  if (failed(verifyQuantizedCommon(*this, wqRef, scalesRef, biasesRef, xRef,
                                    outRef, getBits(), getGroupSize(), k, n,
                                    m * k, m * n)))
    return mlir::failure();
  return verifyQuantizedOutBuffer(*this, getOut());
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ReduceOp::verify() {
  auto inRef = llvm::dyn_cast<MetalMemRefType>(getIn().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!inRef || !outRef)
    return emitOpError() << "metal.reduce operands must be metal memrefs";

  auto elemTy = inRef.getType();
  if (!(elemTy.isF16() || elemTy.isF32() || elemTy.isBF16()))
    return emitOpError() << "metal.reduce requires f32/f16/bf16 element type, "
                         << "in has " << elemTy;
  if (outRef.getType() != elemTy)
    return emitOpError() << "metal.reduce operands must share element type; "
                         << "got in=" << elemTy << ", out=" << outRef.getType();

  int64_t mOuter = getMOuter();
  int64_t r = getR();
  if (r < 1)
    return emitOpError() << "r must be >= 1";
  if (mOuter < 1)
    return emitOpError() << "m_outer must be >= 1";

  auto inSize = inRef.getSize();
  if (inSize > 0 && static_cast<int64_t>(inSize) != mOuter * r)
    return emitOpError() << "input flat size must equal m_outer * r";
  auto outSize = outRef.getSize();
  if (outSize > 0 && static_cast<int64_t>(outSize) != mOuter)
    return emitOpError() << "output flat size must equal m_outer";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArgmaxOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ArgmaxOp::verify() {
  auto inRef = llvm::dyn_cast<MetalMemRefType>(getIn().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!inRef || !outRef)
    return emitOpError() << "metal.argmax operands must be metal memrefs";

  auto inElemTy = inRef.getType();
  if (!(inElemTy.isF16() || inElemTy.isF32() || inElemTy.isBF16()))
    return emitOpError()
           << "metal.argmax requires f32/f16/bf16 input element type, in has "
           << inElemTy;

  auto outElemTy = outRef.getType();
  auto outIntTy = llvm::dyn_cast<mlir::IntegerType>(outElemTy);
  if (!outIntTy || !outIntTy.isSignless() || outIntTy.getWidth() != 32)
    return emitOpError() << "metal.argmax output must be i32, got "
                         << outElemTy;

  int64_t mOuter = getMOuter();
  int64_t r = getR();
  if (r < 1)
    return emitOpError() << "r must be >= 1";
  if (mOuter < 1)
    return emitOpError() << "m_outer must be >= 1";

  auto inSize = inRef.getSize();
  if (inSize > 0 && static_cast<int64_t>(inSize) != mOuter * r)
    return emitOpError() << "input flat size must equal m_outer * r";
  auto outSize = outRef.getSize();
  if (outSize > 0 && static_cast<int64_t>(outSize) != mOuter)
    return emitOpError() << "output flat size must equal m_outer";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult SoftmaxOp::verify() {
  auto inRef = llvm::dyn_cast<MetalMemRefType>(getIn().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!inRef || !outRef)
    return emitOpError() << "metal.softmax operands must be metal memrefs";

  auto inElemTy = inRef.getType();
  if (!(inElemTy.isF16() || inElemTy.isF32() || inElemTy.isBF16()))
    return emitOpError()
           << "metal.softmax requires f32/f16/bf16 element type, in has "
           << inElemTy;

  auto outElemTy = outRef.getType();
  if (outElemTy != inElemTy)
    return emitOpError()
           << "softmax input and output must share element type; got in="
           << inElemTy << ", out=" << outElemTy;

  int64_t mOuter = getMOuter();
  int64_t r = getR();
  if (r < 1)
    return emitOpError() << "r must be >= 1";
  if (mOuter < 1)
    return emitOpError() << "m_outer must be >= 1";

  auto inSize = inRef.getSize();
  if (inSize > 0 && static_cast<int64_t>(inSize) != mOuter * r)
    return emitOpError() << "input flat size must equal m_outer * r";
  auto outSize = outRef.getSize();
  if (outSize > 0 && static_cast<int64_t>(outSize) != mOuter * r)
    return emitOpError() << "softmax output shape must equal input shape";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LogsumexpOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult LogsumexpOp::verify() {
  auto inRef = llvm::dyn_cast<MetalMemRefType>(getIn().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!inRef || !outRef)
    return emitOpError() << "metal.logsumexp operands must be metal memrefs";

  auto inElemTy = inRef.getType();
  if (!(inElemTy.isF16() || inElemTy.isF32() || inElemTy.isBF16()))
    return emitOpError()
           << "metal.logsumexp requires f32/f16/bf16 element type, in has "
           << inElemTy;

  auto outElemTy = outRef.getType();
  if (outElemTy != inElemTy)
    return emitOpError()
           << "logsumexp input and output must share element type; got in="
           << inElemTy << ", out=" << outElemTy;

  int64_t mOuter = getMOuter();
  int64_t r = getR();
  if (r < 1)
    return emitOpError() << "r must be >= 1";
  if (mOuter < 1)
    return emitOpError() << "m_outer must be >= 1";

  auto inSize = inRef.getSize();
  if (inSize > 0 && static_cast<int64_t>(inSize) != mOuter * r)
    return emitOpError() << "input flat size must equal m_outer * r";
  auto outSize = outRef.getSize();
  if (outSize > 0 && static_cast<int64_t>(outSize) != mOuter)
    return emitOpError() << "logsumexp output flat size must equal m_outer";

  return mlir::success();
}

llvm::LogicalResult SdpaOp::verify() {
  // Rule 1: all four operands must be MetalMemRefType.
  auto qRef = llvm::dyn_cast<MetalMemRefType>(getQ().getType());
  auto kRef = llvm::dyn_cast<MetalMemRefType>(getK().getType());
  auto vRef = llvm::dyn_cast<MetalMemRefType>(getV().getType());
  auto outRef = llvm::dyn_cast<MetalMemRefType>(getOut().getType());
  if (!qRef || !kRef || !vRef || !outRef)
    return emitOpError() << "metal.sdpa operands must be metal memrefs";

  // Rule 2: T ∈ {f16, f32, bf16} and all four share T.
  auto elemTy = qRef.getType();
  if (!(elemTy.isF16() || elemTy.isF32() || elemTy.isBF16()))
    return emitOpError()
           << "metal.sdpa q element type must be f16/f32/bf16; got " << elemTy;
  if (kRef.getType() != elemTy || vRef.getType() != elemTy ||
      outRef.getType() != elemTy)
    return emitOpError()
           << "metal.sdpa operands must share element type; q=" << elemTy
           << ", k=" << kRef.getType() << ", v=" << vRef.getType()
           << ", out=" << outRef.getType();

  // Rule 3: d ∈ {64, 128}.
  int64_t d = getD();
  if (d != 64 && d != 128)
    return emitOpError()
           << "metal.sdpa head dim d must be in {64, 128}; got " << d;

  // Rule 4: n ∈ {1, 8}.
  int64_t n = getN();
  if (n != 1 && n != 8)
    return emitOpError()
           << "metal.sdpa query-row count n must be in {1, 8}; got " << n;

  // Rule 5: k_len > 0.
  int64_t kLen = getKLen();
  if (kLen <= 0)
    return emitOpError() << "metal.sdpa k_len must be > 0";

  // Rule 14: k_len must be a multiple of 8 (MMA tile constraint introduced in Stage 8).
  if (kLen % 8 != 0)
    return emitOpError() << "metal.sdpa k_len must be a multiple of 8 (MMA tile constraint); got " << kLen;

  // R6 (replaced): mode ↔ has_causal consistency.
  auto modeVal = getMode();
  bool hasCausalVal = getHasCausal();
  bool expectedCausal = (modeVal == ::mlir::triton::metal::MaskSinkMode::Causal ||
                         modeVal == ::mlir::triton::metal::MaskSinkMode::Sinks);
  if (hasCausalVal != expectedCausal) {
    return emitOpError() << "metal.sdpa mode/has_causal mismatch: mode "
                         << ::mlir::triton::metal::stringifyMaskSinkMode(modeVal)
                         << " requires has_causal="
                         << (expectedCausal ? "true" : "false");
  }

  // R6b: scale must be a finite f32 value.
  auto scaleVal = getScale();
  if (!scaleVal.isFinite())
    return emitOpError() << "metal.sdpa scale must be a finite f32 value";

  // Rule 7: static-size shape checks.
  auto checkSize = [&](MetalMemRefType ref, int64_t expected,
                       llvm::StringRef name,
                       llvm::StringRef formula) -> llvm::LogicalResult {
    auto size = ref.getSize();
    if (size > 0 && static_cast<int64_t>(size) != expected)
      return emitOpError() << "metal.sdpa " << name
                           << " size mismatch: expected " << formula << " = "
                           << expected << ", got " << size;
    return mlir::success();
  };
  if (failed(checkSize(qRef, n * d, "q", "n*d")))
    return mlir::failure();
  if (failed(checkSize(kRef, kLen * d, "k", "k_len*d")))
    return mlir::failure();
  if (failed(checkSize(vRef, kLen * d, "v", "k_len*d")))
    return mlir::failure();
  if (failed(checkSize(outRef, n * d, "out", "n*d")))
    return mlir::failure();

  // Rule 8: `out` must be a device-address-space block argument of enclosing
  // metal.kernel.
  auto outBlockArg = llvm::dyn_cast<mlir::BlockArgument>(getOut());
  if (!outBlockArg)
    return emitOpError()
           << "metal.sdpa `out` must be a kernel buffer block argument";
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError() << "metal.sdpa must be inside a metal.kernel";
  if (outBlockArg.getOwner() != &kernel.getEntryBlock())
    return emitOpError()
           << "metal.sdpa `out` must be an entry-block buffer of the "
              "enclosing metal.kernel";
  unsigned outIdx = outBlockArg.getArgNumber();
  auto deviceAttr = kernel.getAddressSpaceDevice();
  if (outIdx >= deviceAttr.size())
    return emitOpError()
           << "out buffer index out of address_space_device bounds";
  if (!llvm::cast<mlir::BoolAttr>(deviceAttr[outIdx]).getValue())
    return emitOpError()
           << "metal.sdpa `out` must be a device-address-space "
              "(writable) kernel buffer; got a constant buffer";

  // Rule 9: q, k, v must each be a kernel block argument or alloca result.
  for (auto [val, nm] : llvm::zip_equal(
           llvm::SmallVector<mlir::Value, 3>{getQ(), getK(), getV()},
           llvm::SmallVector<llvm::StringRef, 3>{"q", "k", "v"})) {
    auto ba = llvm::dyn_cast<mlir::BlockArgument>(val);
    bool isAlloca = val.getDefiningOp() != nullptr &&
                    llvm::isa<mlir::triton::metal::AllocaOp>(val.getDefiningOp());
    if (!ba && !isAlloca)
      return emitOpError() << "metal.sdpa `" << nm
                           << "` must be a kernel block argument or alloca result";
  }

  // R13: extras source-kind — each extras operand must be a kernel block argument
  // or an alloca result, mirroring Rule 9 for q/k/v.
  for (auto [idx, val] : llvm::enumerate(getExtras())) {
    auto ba = llvm::dyn_cast<mlir::BlockArgument>(val);
    bool isAlloca = val.getDefiningOp() != nullptr &&
                    llvm::isa<mlir::triton::metal::AllocaOp>(val.getDefiningOp());
    if (!ba && !isAlloca)
      return emitOpError() << "metal.sdpa extras[" << idx
                           << "] must be a kernel block argument or alloca result";
  }

  // R10: extras count per mode.
  size_t expectedExtras =
      (modeVal == ::mlir::triton::metal::MaskSinkMode::Causal ||
       modeVal == ::mlir::triton::metal::MaskSinkMode::NonCausal)
          ? 0
          : 1;
  if (getExtras().size() != expectedExtras) {
    return emitOpError() << "metal.sdpa mode "
                         << ::mlir::triton::metal::stringifyMaskSinkMode(modeVal)
                         << " expects " << expectedExtras
                         << " extra operands, got " << getExtras().size();
  }

  // R11 & R12: extras dtype and size per mode.
  if (expectedExtras == 1) {
    auto extraRef =
        llvm::dyn_cast<::mlir::triton::metal::MetalMemRefType>(getExtras()[0].getType());
    if (!extraRef)
      return emitOpError() << "metal.sdpa extras[0] must be metal.memref";

    // R11: dtype check.
    auto extraElem = extraRef.getType();
    ::mlir::Type expectedElem;
    llvm::StringRef expectedName;
    switch (modeVal) {
    case ::mlir::triton::metal::MaskSinkMode::BoolMask:
      expectedElem = ::mlir::IntegerType::get(getContext(), 1);
      expectedName = "i1";
      break;
    case ::mlir::triton::metal::MaskSinkMode::FloatMask:
      expectedElem = elemTy;
      expectedName = "q.elementType";
      break;
    case ::mlir::triton::metal::MaskSinkMode::Sinks:
      expectedElem = ::mlir::Float32Type::get(getContext());
      expectedName = "f32";
      break;
    default:
      expectedElem = nullptr;
    }
    if (expectedElem && extraElem != expectedElem) {
      return emitOpError() << "metal.sdpa "
                           << ::mlir::triton::metal::stringifyMaskSinkMode(modeVal)
                           << " extra dtype mismatch: expected " << expectedName
                           << ", got " << extraElem;
    }

    // R12: size check.
    size_t extraSize = extraRef.getSize();
    size_t expectedSize;
    llvm::StringRef expectedFormula;
    switch (modeVal) {
    case ::mlir::triton::metal::MaskSinkMode::BoolMask:
    case ::mlir::triton::metal::MaskSinkMode::FloatMask:
      expectedSize = static_cast<size_t>(getN() * getKLen());
      expectedFormula = "n*k_len";
      break;
    case ::mlir::triton::metal::MaskSinkMode::Sinks:
      expectedSize = static_cast<size_t>(getN());
      expectedFormula = "n";
      break;
    default:
      llvm_unreachable("unhandled MaskSinkMode in R12 size check");
    }
    if (extraSize != 0 && extraSize != expectedSize) {
      return emitOpError() << "metal.sdpa "
                           << ::mlir::triton::metal::stringifyMaskSinkMode(modeVal)
                           << " extra size mismatch: expected " << expectedFormula
                           << " = " << expectedSize << ", got " << extraSize;
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RmsNormOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult RmsNormOp::verify() {
  auto xRef = llvm::dyn_cast<MetalMemRefType>(getX().getType());
  auto gRef = llvm::dyn_cast<MetalMemRefType>(getGamma().getType());
  auto oRef = llvm::dyn_cast<MetalMemRefType>(getO().getType());
  if (!xRef || !gRef || !oRef)
    return emitOpError() << "metal.rms_norm operands must be metal memrefs";

  // R-N4: x, gamma, o must share element type.
  auto xTy = xRef.getType();
  if (gRef.getType() != xTy || oRef.getType() != xTy)
    return emitOpError() << "operands must share element type; x=" << xTy
                         << ", gamma=" << gRef.getType()
                         << ", o=" << oRef.getType();

  // R-N5: element type must be in {f32, f16, bf16}.
  if (!(xTy.isF32() || xTy.isF16() || xTy.isBF16()))
    return emitOpError() << "requires f32/f16/bf16 element type; got " << xTy;

  // R-N3: n >= 1.
  int64_t n = getN();
  if (n < 1)
    return emitOpError() << "n must be >= 1; got " << n;

  // R-N1: d must be a positive multiple of 32 (single-warp simd_sum lane count).
  int64_t d = getD();
  if (d <= 0 || d % 32 != 0)
    return emitOpError() << "d must be a multiple of 32; got " << d;

  // R-N2: d ∈ {64, 128, 256} (Stage-9 tested envelope).
  if (d != 64 && d != 128 && d != 256)
    return emitOpError() << "d must be one of {64, 128, 256}; got " << d;

  // R-N1b: eps must be finite.
  auto epsVal = getEps();
  if (!epsVal.isFinite())
    return emitOpError() << "eps must be a finite f32 value";

  // Size cross-checks: x flat = n*d, gamma flat = d, o flat = n*d.
  auto xSize = xRef.getSize();
  if (xSize > 0 && static_cast<int64_t>(xSize) != n * d)
    return emitOpError() << "x flat size must equal n*d = " << (n * d)
                         << "; got " << xSize;
  auto gSize = gRef.getSize();
  if (gSize > 0 && static_cast<int64_t>(gSize) != d)
    return emitOpError() << "gamma flat size must equal d = " << d
                         << "; got " << gSize;
  auto oSize = oRef.getSize();
  if (oSize > 0 && static_cast<int64_t>(oSize) != n * d)
    return emitOpError() << "o flat size must equal n*d = " << (n * d)
                         << "; got " << oSize;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult PrintOp::verify() {
  // R-P3: must not appear inside a metal.kernel body (host-only op).
  if ((*this)->getParentOfType<mlir::triton::metal::KernelOp>())
    return emitOpError()
           << "metal.print is host-only; not legal inside metal.kernel";

  auto memTy = llvm::dyn_cast<mlir::MemRefType>(getMem().getType());
  if (!memTy)
    return emitOpError() << "metal.print operand must be a standard memref";

  // R-P2: rank-1 memref only.
  if (memTy.getRank() != 1)
    return emitOpError() << "metal.print operand must be rank-1 memref";

  // R-P1: element type must be in {f32, f16, bf16}.
  auto elt = memTy.getElementType();
  if (!(elt.isF32() || elt.isF16() || elt.isBF16()))
    return emitOpError() << "metal.print requires f32/f16/bf16 element type";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

auto IfOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, mlir::Value cond,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> thenBuilder,
    function_ref<void(OpBuilder &, mlir::Location)> elseBuilder) -> void {
  result.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);

  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (elseBuilder) {
    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
  }
}

mlir::ParseResult IfOp::parse(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::UnresolvedOperand condition;
  mlir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(condition) ||
      parser.resolveOperand(condition, i1Type, result.operands))
    return mlir::failure();

  if (parser.parseRegion(*thenRegion, {}))
    return mlir::failure();

  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, {}))
      return mlir::failure();
  }

  return mlir::success();
}

void IfOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getCondition() << " ";
  printer.printRegion(getThenRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  auto &elseRegion = this->getElseRegion();
  if (!elseRegion.empty()) {
    printer << " else ";
    printer.printRegion(elseRegion,
                        /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

auto WhileOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> conditionBuilder,
    function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder) -> void {

  OpBuilder::InsertionGuard guard(builder);

  Region *conditionRegion = result.addRegion();
  builder.createBlock(conditionRegion);
  conditionBuilder(builder, result.location);

  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  bodyBuilder(builder, result.location);
}

llvm::LogicalResult WhileOp::verify() {
  auto &region = getConditionRegion();

  for (auto it = region.op_begin(); it != region.op_end(); it++) {
    if (!llvm::isa<ConstantOp, StoreOp, GetElementOp, ThreadIdOp, CastOp,
                   UnaryExpOp, BinaryExpOp, YieldWhileOp>(*it))
      return emitOpError() << it->getName()
                           << " op not allowed in the condition region";
  }

  return mlir::success();
}

mlir::ParseResult WhileOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *conditionRegion = result.addRegion();
  mlir::Region *bodyRegion = result.addRegion();
  if (parser.parseKeyword("condition") ||
      parser.parseRegion(*conditionRegion, {}) ||
      parser.parseKeyword("loop") ||
      parser.parseRegion(*bodyRegion, {}))
    return mlir::failure();

  return mlir::success();
}

void WhileOp::print(mlir::OpAsmPrinter &printer) {
  printer << " condition ";
  printer.printRegion(getConditionRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  auto &bodyRegion = this->getBodyRegion();
  printer << " loop ";
  printer.printRegion(bodyRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// Runtime - Device
//===----------------------------------------------------------------------===//

void DeviceMakeDefaultOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getIndexType());
};

void DeviceMakeCommandQueueOp::build(OpBuilder &builder, OperationState &result,
                                     Value device) {
  result.addOperands(device);
  result.addTypes(builder.getIndexType());
};

void DeviceMakeBufferOp::build(OpBuilder &builder, OperationState &result,
                               Value device, Value isStorageModeManaged,
                               Value count, Value sizeType) {
  result.addOperands(device);
  result.addOperands(isStorageModeManaged);
  result.addOperands(count);
  result.addOperands(sizeType);
  result.addTypes(builder.getIndexType());
};

//===----------------------------------------------------------------------===//
// Runtime - Buffer
//===----------------------------------------------------------------------===//

void BufferGetContentsOp::build(OpBuilder &builder, OperationState &result,
                                Value device, Type elementType) {
  result.addOperands(device);
  auto memRefType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType);
  result.addTypes(memRefType);
};

llvm::LogicalResult BufferGetContentsOp::verify() {
  auto elementType =
      llvm::cast<mlir::MemRefType>(getResult().getType()).getElementType();
  if (isa<mlir::IntegerType>(elementType) || elementType.isF16() ||
      elementType.isF32() || elementType.isBF16())
    return mlir::success();

  return emitOpError() << "the buffer has an incompatible type";
}

//===----------------------------------------------------------------------===//
// Runtime - CommandQueue
//===----------------------------------------------------------------------===//

void CommandQueueMakeCommandBufferOp::build(OpBuilder &builder,
                                            OperationState &result,
                                            Value commandQueue,
                                            StringRef functionName, Value width,
                                            Value height, Value depth) {
  result.addAttribute("functionName", builder.getStringAttr(functionName));
  result.addOperands(commandQueue);
  result.addOperands(width);
  result.addOperands(height);
  result.addOperands(depth);
  result.addTypes(builder.getIndexType());
};

void CommandQueueMakeCommandBufferOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getFunctionName() << " ";
  printer << getCommandQueue() << ", ";
  printer << getWidth() << ", ";
  printer << getHeight() << ", ";
  printer << getDepth();
  printer << ": (" << getOperandTypes() << ") -> ";
  printer.printType(getResult().getType());
}

mlir::ParseResult
CommandQueueMakeCommandBufferOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  llvm::StringRef functionName;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SmallVector<mlir::Type, 4> operandTypes;
  if (parser.parseKeyword(&functionName) ||
      parser.parseOperandList(operands, 4) || parser.parseColon() ||
      parser.parseLParen() || parser.parseTypeList(operandTypes) ||
      parser.parseRParen() || parser.parseArrowTypeList(result.types))
    return mlir::failure();

  result.addAttribute("functionName",
                      parser.getBuilder().getStringAttr(functionName));

  return parser.resolveOperands(operands, operandTypes,
                                parser.getCurrentLocation(), result.operands);
}

//===----------------------------------------------------------------------===//
// SimdgroupLoadDeviceStagedOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult SimdgroupLoadDeviceStagedOp::verify() {
  auto widx = getWarpIndex();
  if (widx.size() > 1)
    return emitOpError()
           << "warp_index must be empty (single-warp / bit-identical) or "
              "exactly 1 value (per-warp staged-load); got "
           << widx.size();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SimdgroupStoreOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult SimdgroupStoreOp::verify() {
  auto extents = getPartialExtents();
  if (!extents.empty() && extents.size() != 2)
    return emitOpError()
           << "partial_extents must be empty (full 8x8 store) or exactly 2 "
              "values [m_extent, n_extent]; got "
           << extents.size();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen's op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/Metal/IR/MetalOps.cpp.inc"
