//===--- MetalToLLVM.cpp --------------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Dialect/Metal/Transforms/MetalToLLVM.h"
#include "Dialect/Metal/IR/MetalOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace LLVM;

static SymbolRefAttr insertFunction(ConversionPatternRewriter &rewriter,
                                    ModuleOp module,
                                    LLVM::LLVMFunctionType llvmFnType,
                                    llvm::StringRef functionName) {
  auto *context = module.getContext();
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), functionName, llvmFnType);
  return SymbolRefAttr::get(context, functionName);
}

static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type =
        LLVM::LLVMArrayType::get(builder.getIntegerType(8), value.size());
    global = LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value));
  }

  Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
  Value cst0 = LLVM::ConstantOp::create(builder, 
      loc, builder.getI64Type(),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return LLVM::GEPOp::create(builder, 
      loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

static void rewriteOp(Operation *op, ArrayRef<Value> operands,
                      ConversionPatternRewriter &rewriter,
                      llvm::StringRef functionName,
                      std::optional<Type> resultType, ArrayRef<Type> params) {
  auto loc = op->getLoc();
  auto module = op->getParentOfType<ModuleOp>();
  auto *context = module.getContext();

  SymbolRefAttr callee;
  if (module.lookupSymbol<LLVMFuncOp>(functionName))
    callee = SymbolRefAttr::get(context, functionName);
  else {
    if (resultType.has_value()) {
      auto llvmFnType =
          LLVM::LLVMFunctionType::get(resultType.value(), params, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    } else {
      auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
      auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, params, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    }
  }
  if (resultType.has_value()) {
    auto call = func::CallOp::create(rewriter, loc, callee, resultType.value(),
                                              operands);
    rewriter.replaceOp(op, call.getResult(0));
  } else {
    func::CallOp::create(rewriter, loc, callee, TypeRange{}, operands);
    rewriter.eraseOp(op);
  }
}

class ModuleOpLowering : public ConversionPattern {
public:
  explicit ModuleOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::triton::metal::ModuleOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

class ReleaseOpLowering : public ConversionPattern {
public:
  explicit ReleaseOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::triton::metal::ReleaseOp::getOperationName(), 1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalRelease", std::nullopt, {i64Ty});
    return success();
  }
};

class DeviceMakeDefaultOpLowering : public ConversionPattern {
public:
  explicit DeviceMakeDefaultOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::triton::metal::DeviceMakeDefaultOp::getOperationName(),
                          1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalDeviceMakeDefault", i64Ty, {});
    return success();
  }
};

class DeviceMakeCommandQueueOpLowering : public ConversionPattern {
public:
  explicit DeviceMakeCommandQueueOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::triton::metal::DeviceMakeCommandQueueOp::getOperationName(), 1,
            context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalDeviceMakeCommandQueue", i64Ty,
              {i64Ty});
    return success();
  }
};

class DeviceMakeBufferOpLowering : public ConversionPattern {
public:
  explicit DeviceMakeBufferOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::triton::metal::DeviceMakeBufferOp::getOperationName(),
                          1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    auto i1Ty = rewriter.getI1Type();
    rewriteOp(op, operands, rewriter, "_MetalDeviceMakeBuffer", i64Ty,
              {i64Ty, i1Ty, i64Ty, i64Ty});
    return success();
  }
};

class BufferGetContentsOpLowering : public ConversionPattern {
public:
  explicit BufferGetContentsOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::triton::metal::BufferGetContentsOp::getOperationName(),
                          1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto *context = module.getContext();
    auto functionName = "_MetalBufferGetContents";

    auto voidTy = LLVM::LLVMVoidType::get(context);
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, 1);
    auto structTy = LLVM::LLVMStructType::getLiteral(
        context, {ptrTy, ptrTy, i64Ty, arrayTy, arrayTy});

    SymbolRefAttr callee;
    if (module.lookupSymbol<LLVMFuncOp>(functionName))
      callee = SymbolRefAttr::get(context, functionName);
    else {
      ArrayRef<Type> types = {i64Ty, ptrTy};
      auto llvmFnType = LLVM::LLVMFunctionType::get(voidTy, types, false);
      callee = insertFunction(rewriter, module, llvmFnType, functionName);
    }

    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
    auto alloca =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, structTy, one, 0);

    ArrayRef<Value> newOperand = {operands[0], alloca};
    func::CallOp::create(rewriter, loc, callee, TypeRange{}, newOperand);
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, structTy, alloca);
    return success();
  }
};

class CommandQueueMakeCommandBufferOpLowering : public ConversionPattern {
public:
  explicit CommandQueueMakeCommandBufferOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::triton::metal::CommandQueueMakeCommandBufferOp::getOperationName(), 1,
            context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = rewriter.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto lib =
        getOrCreateGlobalString(loc, rewriter, "metallib",
                                StringRef("./default.metallib\0", 19), module);

    auto createOp = cast<mlir::triton::metal::CommandQueueMakeCommandBufferOp>(op);
    // The runtime reads the kernel name as a C string via NSUTF8StringEncoding,
    // so the global must include a trailing null terminator. Build a
    // null-terminated copy backed by a SmallString that outlives this call.
    llvm::SmallString<32> kernelName(createOp.getFunctionName());
    kernelName.push_back('\0');
    auto kernel = getOrCreateGlobalString(
        loc, rewriter, createOp.getFunctionName(),
        StringRef(kernelName.data(), kernelName.size()), module);

    ArrayRef<Value> newOperands = {operands[0], lib,         kernel,
                                   operands[1], operands[2], operands[3]};

    rewriteOp(op, newOperands, rewriter, "_MetalCommandQueueMakeCommandBuffer",
              i64Ty, {i64Ty, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty});
    return success();
  }
};

class CommandBufferAddBufferOpLowering : public ConversionPattern {
public:
  explicit CommandBufferAddBufferOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::triton::metal::CommandBufferAddBufferOp::getOperationName(), 1,
            context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferAddBuffer",
              std::nullopt, {i64Ty, i64Ty, i64Ty});
    return success();
  }
};

class CommandBufferCommitOpLowering : public ConversionPattern {
public:
  explicit CommandBufferCommitOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::triton::metal::CommandBufferCommitOp::getOperationName(), 1,
            context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferCommit", std::nullopt,
              {i64Ty});
    return success();
  }
};

class CommandBufferWaitUntilCompletedOpLowering : public ConversionPattern {
public:
  explicit CommandBufferWaitUntilCompletedOpLowering(MLIRContext *context)
      : ConversionPattern(
            mlir::triton::metal::CommandBufferWaitUntilCompletedOp::getOperationName(),
            1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto i64Ty = rewriter.getI64Type();
    rewriteOp(op, operands, rewriter, "_MetalCommandBufferWaitUntilCompleted",
              std::nullopt, {i64Ty});
    return success();
  }
};

// Lowering for `metal.print`. The memref operand has already been converted
// by populateFinalizeMemRefToLLVMConversionPatterns (registered alongside
// our patterns in ConvertMetalToLLVM.cpp), so adaptor.getMem() is the LLVM
// memref descriptor struct {ptr allocated, ptr aligned, i64 offset, ...}.
// We extract the `aligned` pointer (field 1) and pass it plus the i64 count
// to one of _MetalPrintF32/F16/BF16 selected by the original element type.
//
// This pattern derives from ConvertOpToLLVMPattern so that the
// LLVMTypeConverter is threaded through and the framework performs operand
// type conversion (memref<?xT> -> !llvm.struct<...>) before matchAndRewrite.
class PrintOpLowering : public ConvertOpToLLVMPattern<mlir::triton::metal::PrintOp> {
public:
  using ConvertOpToLLVMPattern<mlir::triton::metal::PrintOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::triton::metal::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    // Element type comes from the ORIGINAL op, not the converted operand.
    auto memTy = llvm::cast<mlir::MemRefType>(op.getMem().getType());
    auto elt = memTy.getElementType();

    llvm::StringRef fnName;
    if (elt.isF32())
      fnName = "_MetalPrintF32";
    else if (elt.isF16())
      fnName = "_MetalPrintF16";
    else if (elt.isBF16())
      fnName = "_MetalPrintBF16";
    else
      llvm_unreachable("PrintOp::verify guarantees f32/f16/bf16 element type");

    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    Value alignedPtr = LLVM::ExtractValueOp::create(rewriter, 
        loc, adaptor.getMem(), ArrayRef<int64_t>{1});

    SmallVector<Value, 2> callOperands = {alignedPtr, adaptor.getN()};
    rewriteOp(op, callOperands, rewriter, fnName, std::nullopt,
              {ptrTy, i64Ty});
    return success();
  }
};

} // end namespace

void mlir::triton::metal::populateMetalToLLVMConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx,
    LLVMTypeConverter &typeConverter) {
  patterns.insert<
      ModuleOpLowering, ReleaseOpLowering, DeviceMakeDefaultOpLowering,
      DeviceMakeCommandQueueOpLowering, DeviceMakeBufferOpLowering,
      BufferGetContentsOpLowering, CommandQueueMakeCommandBufferOpLowering,
      CommandBufferAddBufferOpLowering, CommandBufferCommitOpLowering,
      CommandBufferWaitUntilCompletedOpLowering>(ctx);
  patterns.insert<PrintOpLowering>(typeConverter);
}
