//===--- ConvertFuncsToMetalKernels.cpp -------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Dialect/Metal/Transforms/LinalgToMetal.h"
#include "Dialect/Metal/Transforms/MetalPasses.h"
#include "Dialect/Metal/IR/MetalDialect.h"
#include "Dialect/Metal/IR/MetalOps.h"
#include "Dialect/Metal/IR/MetalTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::metal {

#define GEN_PASS_DEF_CONVERTFUNCSTOMETALKERNELS
#include "Dialect/Metal/Transforms/MetalPasses.h.inc"

namespace {

static bool isSupportedElementType(mlir::Type t) {
  return t.isF32() || t.isF16() || t.isBF16();
}

struct ConvertFuncsToMetalKernels
    : public impl::ConvertFuncsToMetalKernelsBase<ConvertFuncsToMetalKernels> {
  using impl::ConvertFuncsToMetalKernelsBase<
      ConvertFuncsToMetalKernels>::ConvertFuncsToMetalKernelsBase;

  // Validates the func meets v2's wrapping contract and emits clear
  // diagnostics for any violation. Returns true if the func should be
  // skipped (i.e. produced an error and we should signalPassFailure later).
  bool validateAnnotatedFunc(mlir::func::FuncOp f) {
    bool failed = false;
    if (!f.getBody().hasOneBlock()) {
      f.emitOpError() << "convert-funcs-to-metal-kernels: multi-block func "
                         "bodies are not supported";
      failed = true;
    }
    for (auto arg : f.getArguments()) {
      auto mref = llvm::dyn_cast<mlir::MemRefType>(arg.getType());
      if (!mref) {
        f.emitOpError() << "convert-funcs-to-metal-kernels: arguments must "
                           "be memrefs";
        failed = true;
        continue;
      }
      if (!mref.hasStaticShape()) {
        f.emitOpError() << "convert-funcs-to-metal-kernels: dynamic shapes "
                           "are not supported";
        failed = true;
        continue;
      }
      if (!isSupportedElementType(mref.getElementType())) {
        f.emitOpError() << "convert-funcs-to-metal-kernels: unsupported "
                           "element type";
        failed = true;
      }
    }
    return failed;
  }

  void runOnOperation() final {
    auto module = getOperation();
    auto *ctx = &getContext();

    llvm::SmallVector<mlir::func::FuncOp> toWrap;
    module.walk([&](mlir::func::FuncOp f) {
      if (f->hasAttr("metal.kernel"))
        toWrap.push_back(f);
    });

    // Pre-validate everything first so all diagnostics fire even if multiple
    // funcs are bad in one module.
    bool anyError = false;
    for (auto f : toWrap)
      if (validateAnnotatedFunc(f))
        anyError = true;
    if (anyError)
      return signalPassFailure();

    for (auto f : toWrap) {
      mlir::OpBuilder builder(f);
      auto loc = f.getLoc();
      auto *ctx = &getContext();

      // Build a fresh metal.module that will host the kernel.
      auto metalModule = triton::metal::ModuleOp::create(builder, loc);

      // Compute per-arg flat sizes via the shared safeFlatSize helper so
      // the kernel block-arg types match what convert-linalg-to-metal's
      // TypeConverter produces (lets unrealized_conversion_cast bridges
      // fold cleanly) AND so an overflowing static shape is rejected with
      // a diagnostic instead of silently wrapping to a wrong size.
      llvm::SmallVector<mlir::Type> kernelArgTypes;
      llvm::SmallVector<bool> isDevice;
      bool sizeFail = false;
      for (auto arg : f.getArguments()) {
        auto mref = llvm::cast<mlir::MemRefType>(arg.getType());
        auto flatOpt = safeFlatSize(mref.getShape());
        if (!flatOpt) {
          f.emitOpError() << "convert-funcs-to-metal-kernels: arg flat size "
                             "overflows MetalMemRefType capacity";
          sizeFail = true;
          break;
        }
        kernelArgTypes.push_back(
            MetalMemRefType::get(ctx, mref.getElementType(), *flatOpt));
        isDevice.push_back(true);
      }
      if (sizeFail) {
        signalPassFailure();
        return;
      }

      builder.setInsertionPointToStart(
          &metalModule.getBodyRegion().front());
      // KernelOp::build sets size=0 on every arg; bypass it and construct
      // the kernel with the typed args directly.
      mlir::OperationState state(loc, triton::metal::KernelOp::getOperationName());
      state.addAttribute("name", builder.getStringAttr(f.getName()));
      state.addAttribute("address_space_device",
                         builder.getBoolArrayAttr(isDevice));
      mlir::Region *bodyRegion = state.addRegion();
      auto *kernelBlock = new mlir::Block();
      bodyRegion->push_back(kernelBlock);
      for (auto t : kernelArgTypes)
        kernelBlock->addArgument(t, loc);
      auto kernel =
          llvm::cast<triton::metal::KernelOp>(builder.create(state));
      mlir::Block &kernelEntry = kernel.getEntryBlock();
      builder.setInsertionPointToStart(&kernelEntry);

      // Insert unrealized_conversion_cast per arg so the body can still
      // reference standard memrefs until convert-linalg-to-metal folds the
      // casts away.
      llvm::SmallVector<mlir::Value> compatArgs;
      for (auto [origArg, newArg] :
           llvm::zip(f.getArguments(), kernelEntry.getArguments())) {
        auto castOp =
            mlir::UnrealizedConversionCastOp::create(builder, 
                loc, origArg.getType(), newArg);
        compatArgs.push_back(castOp.getResult(0));
      }

      // Clone the func body into the kernel body, remapping arg uses
      // through the compat casts. Skip the original func.return; the
      // metal.kernel has its own ImplicitTerminator<metal.return>.
      mlir::IRMapping mapping;
      for (auto [origArg, compat] :
           llvm::zip(f.getArguments(), compatArgs))
        mapping.map(origArg, compat);

      mlir::Block &origBlock = f.getBody().front();
      for (auto &op : origBlock.without_terminator()) {
        builder.clone(op, mapping);
      }

      // metal.kernel's region uses ImplicitTerminator<metal.return>; insert
      // the terminator explicitly since we copied body ops without it.
      triton::metal::ReturnOp::create(builder, loc);

      f.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton::metal
