//===--- ModuleTranslation.cpp -----------------------------------------------//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Target/Metal/ModuleTranslation.h"
#include "Dialect/Metal/IR/MetalOps.h"
#include "Dialect/Metal/IR/MetalQuantizedHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::triton::metal;

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT()                                                               \
  Indent level_(_curIndent);                                                   \
  indent();

// `xcrun metal` rejects `-inf` / `inf` / `nan` (the textual output of
// `APFloat::convertToDouble()` for non-finite floats), so FloatAttr emission
// sites must route through this helper. Finite values fall through to the
// existing `<<` formatter to preserve precision contracts elsewhere in the TU.
static void emitFloatLiteral(llvm::raw_ostream &os, mlir::FloatAttr attr) {
  // MSL permits an implicit float->half conversion but NOT float->bfloat, so a
  // bare `0.0` literal cannot be assigned/yielded where a `bfloat` is expected
  // (e.g. a masked load's `other=0.0`). Wrap bf16 constants in `bfloat(...)`;
  // f16 (implicit-ok) and f32 stay bare literals.
  const bool isBf16 = attr.getType().isBF16();
  if (isBf16)
    os << "bfloat(";
  const llvm::APFloat &v = attr.getValue();
  if (v.isNaN())
    os << "NAN";
  else if (v.isInfinity())
    os << (v.isNegative() ? "-INFINITY" : "INFINITY");
  else
    os << attr.getValueAsDouble();
  if (isBf16)
    os << ")";
}

// MSL is a C++14 dialect, so the emitted entry-point name must not collide
// with a reserved keyword. A Triton `@triton.jit` function may legitimately be
// named `kernel`, `device`, `vertex`, `float`, ... — Python only forbids its
// OWN keywords as function names, and none of those overlap with the C++/MSL
// words below. Emitting `kernel void kernel(...)` is a hard MSL syntax error
// (observed compiling leet-triton/medium-monte_carlo_integration.py through
// torch.mps.compile_shader). We mangle ONLY on collision (prefix `triton_`) so
// every non-colliding kernel keeps its exact original name — both the
// launcher's `metadata["name"]` (compiler.py make_msl re-greps `kernel void
// <name>` from this text) and its `getattr(lib, name)` (driver.py load_binary)
// re-derive from the name emitted here, so sanitizing at this single site keeps
// all three in lockstep.
static std::string sanitizeKernelName(llvm::StringRef name) {
  static const llvm::StringSet<> kReserved = {
      // C++14 keywords (MSL Specification §1.4.3). Python keywords that also
      // appear here (e.g. `if`, `return`, `class`) can never be a JIT function
      // name, but harmless to list; the rest (`auto`, `float`, `int`, `static`,
      // `new`, `switch`, `union`, `template`, `default`, ...) ARE valid Python
      // identifiers a kernel could be named.
      "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
      "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class",
      "compl", "const", "constexpr", "const_cast", "continue", "decltype",
      "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
      "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
      "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
      "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
      "protected", "public", "register", "reinterpret_cast", "return", "short",
      "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
      "switch", "template", "this", "thread_local", "throw", "true", "try",
      "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual",
      "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
      // MSL function / address-space / attribute qualifiers (§4.x) and common
      // typedefs a function name could shadow.
      "kernel", "vertex", "fragment", "device", "constant", "threadgroup",
      "threadgroup_imageblock", "thread", "ray_data", "object_data", "packed",
      "sample", "sampler", "uniform", "patch", "half", "uchar", "ushort",
      "uint", "ulong", "size_t", "ptrdiff_t",
  };
  if (name.empty() || kReserved.count(name))
    return "triton_" + name.str();
  return name.str();
}

static llvm::StringRef typeToString(mlir::Type type) {
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type))
    switch (intTy.getWidth()) {
    case 1:
      return "bool";
    case 8:
      return intTy.isSigned() ? "int8_t" : "uint8_t";
    case 16:
      return intTy.isSigned() ? "int16_t" : "uint16_t";
    case 32:
      // Signless i32 is the MLIR-side bridge for argmax indices.
      // MSL emits 'int' for it; signed -> int32_t, unsigned -> uint32_t.
      if (intTy.isSignless())
        return "int";
      return intTy.isSigned() ? "int32_t" : "uint32_t";
    case 64:
      return intTy.isSigned() ? "int64_t" : "uint64_t";
    default:
      llvm_unreachable("wrong type");
    }
  if (type.isF16())
    return "half";
  if (type.isF32())
    return "float";
  if (type.isBF16())
    return "bfloat";
  llvm_unreachable("wrong type");
}

void ModuleTranslation::indent() {
  for (int i = 0; i < _curIndent; i++)
    _output << "  ";
}

llvm::LogicalResult ModuleTranslation::translateModule(mlir::ModuleOp m,
                                                       raw_ostream &output) {
  bool emittedPreamble = false;
  for (auto module : m.getOps<mlir::triton::metal::ModuleOp>()) {
    if (!emittedPreamble) {
      output << "#include <metal_stdlib>\n";
      output << "#include <metal_math>\n";
      output << "#include <metal_atomic>\n";
      // Stage-7: include simdgroup_matrix header only when an `::Mma` matmul is
      // present in any kernel; preserves Stage-1 ::Scalar byte-identity SHA.
      // Stage-8: also include when any `metal.sdpa` op is present — all 5 mode
      // helpers now emit `simdgroup_matrix` MMA tiles.
      bool hasMma = false;
      m.walk([&](mlir::triton::metal::MatmulOp matmulOp) {
        if (matmulOp.getKind() == ::mlir::triton::metal::MatmulKind::Mma) {
          hasMma = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (!hasMma) {
        m.walk([&](mlir::triton::metal::SdpaOp) {
          hasMma = true;
          return mlir::WalkResult::interrupt();
        });
      }
      if (hasMma)
        output << "#include <metal_simdgroup_matrix>\n";
      output << "using namespace metal;\n\n";
      // Session L4: MSL stdlib does not ship `erf`, so we emit a polynomial
      // approximation (Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7)
      // when any kernel in the module uses `metal.unary_exp ..., erfOp`. The
      // `metal.unary_exp` switch emits `__triton_erff(<arg>)`.
      bool hasErf = false;
      m.walk([&](mlir::triton::metal::UnaryExpOp uop) {
        if (uop.getUnaryOperator() ==
            mlir::triton::metal::UnaryExpOperator::erfOp) {
          hasErf = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (hasErf) {
        output << "static inline float __triton_erff(float x) {\n"
                  "  const float a1 =  0.254829592f;\n"
                  "  const float a2 = -0.284496736f;\n"
                  "  const float a3 =  1.421413741f;\n"
                  "  const float a4 = -1.453152027f;\n"
                  "  const float a5 =  1.061405429f;\n"
                  "  const float p  =  0.3275911f;\n"
                  "  float sign = (x < 0.0f) ? -1.0f : 1.0f;\n"
                  "  float ax = metal::fabs(x);\n"
                  "  float t = 1.0f / (1.0f + p * ax);\n"
                  "  float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2)"
                  " * t + a1) * t * metal::precise::exp(-ax * ax);\n"
                  "  return sign * y;\n"
                  "}\n\n";
      }
      emittedPreamble = true;
    }
    ModuleTranslation translator{module, output};
    translator.translateKernels();
    output.flush();
    // A translator that hit an unemittable op has already called emitError();
    // surface it as a real failure so ttgir_to_msl throws instead of returning
    // half-formed MSL. See the `_emitFailed` comment in ModuleTranslation.h.
    if (translator._emitFailed)
      return mlir::failure();
  }
  return mlir::success();
}

void ModuleTranslation::translateVarName(mlir::Value memref) {
  // A value explicitly materialized as an MSL temp resolves by its buffer name
  // first — covers scf.for iter_args / results (W2a runtime-K matmul maps the
  // simdgroup_matrix loop result into `_buffers`) as well as block args. The
  // simdgroup op results live in `_alloca` (not `_buffers`), so this does not
  // shadow the isa-dispatch below.
  if (auto it = _buffers.find(memref.getAsOpaquePointer());
      it != _buffers.end()) {
    _output << "v" << it->second;
    return;
  }
  // Walk through unrealized_conversion_cast wrappers (e.g. the
  // !tt.ptr → !metal.memref bridge inserted by the matmul track's
  // tt.dot pre-pass; post-conversion this becomes an identity
  // !metal.memref → !metal.memref cast that persists).
  while (auto cast = memref.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1) break;
    memref = cast.getInputs()[0];
  }
  auto opInst = memref.getDefiningOp();
  if (opInst == nullptr) {
    _output << "v" << _buffers[memref.getAsOpaquePointer()];
  } else if (isa<mlir::triton::metal::SimdgroupIndexOp>(opInst)) {
    // AC4 v6: SimdgroupIndexOp uses the kernel parameter `sgid`
    // (declared via [[simdgroup_index_in_threadgroup]] in translateKernel).
    _output << "sgid";
  } else if (isa<mlir::triton::metal::AllocaOp,
                  mlir::triton::metal::ThreadgroupAllocaOp,
                  mlir::triton::metal::SimdgroupMatrixZeroOp,
                  mlir::triton::metal::SimdgroupLoadDeviceStagedOp,
            mlir::triton::metal::SimdgroupLoadDeviceStagedMaskedOp,
                  mlir::triton::metal::SimdgroupLoadOp,
                  mlir::triton::metal::SimdgroupMultiplyAccumulateOp>(opInst)) {
    _output << "v" << _alloca[opInst];
  } else {
    llvm_unreachable("llvm_unreachable");
  }
}

void ModuleTranslation::translateKernels() {
  for (auto &op : _metalModule.getOps()) {
    if (auto kernelOp = dyn_cast<mlir::triton::metal::KernelOp>(op)) {
      _varCount = 0;
      _buffers = {};
      translateKernel(kernelOp);
      _output << "\n\n";
    } else if (isa<mlir::triton::metal::ConstantOp, mlir::triton::metal::ModuleEndOp>(op)) {
      // do nothing
    } else {
      llvm_unreachable("unexpected operation");
    }
  }
}

void ModuleTranslation::translateKernel(mlir::triton::metal::KernelOp op) {
  _output << "kernel void " << sanitizeKernelName(op.getName()) << "(\n";
  for (auto tuple : llvm::zip(op.getBuffers(), op.getAddressSpaceDevice())) {
    auto buffer = std::get<0>(tuple);
    auto memRef = llvm::cast<mlir::triton::metal::MetalMemRefType>(buffer.getType());
    auto stringType = typeToString(memRef.getType());

    auto isDevice = llvm::cast<BoolAttr>(std::get<1>(tuple)).getValue();
    _output << (isDevice ? "  device " : "  constant ");
    _output << stringType << " *v" << _varCount << " [[buffer(" << _varCount
            << ")]],\n";
    _varCount++;
  }
  _output << "  uint3 id [[thread_position_in_grid]]";
  // Conditionally add the threadgroup-position parameter only when the
  // kernel body references it (via metal.threadgroup_id). This keeps
  // existing single-program fixtures' MSL signatures unchanged. See
  // `.omc/specs/deep-interview-metal-pid-lowering.md`.
  // metal.flash_attention needs the threadgroup position (program id) AND the
  // LOCAL thread index: the single-warp guard must key off
  // thread_position_in_threadgroup, since thread_position_in_grid is global and
  // would mis-identify the 2nd query block's warp (Phase-0 finding).
  bool usesFlashAttention = false;
  op.walk([&](mlir::triton::metal::FlashAttentionOp) {
    usesFlashAttention = true;
    return mlir::WalkResult::interrupt();
  });
  bool usesThreadgroupId = usesFlashAttention;
  op.walk([&](mlir::triton::metal::ThreadgroupIdOp) {
    usesThreadgroupId = true;
    return mlir::WalkResult::interrupt();
  });
  if (usesThreadgroupId)
    _output << ",\n  uint3 tgid [[threadgroup_position_in_grid]]";
  if (usesFlashAttention)
    _output << ",\n  uint3 ltid [[thread_position_in_threadgroup]]";
  // Conditionally add the threadgroups-per-grid parameter only when the
  // kernel body references it (via metal.threadgroups_per_grid). Mirrors the
  // usesThreadgroupId walk above; keeps existing single-program fixtures'
  // MSL signatures unchanged. See
  // `.omc/plans/tutorial02-fused-softmax-fix-consensus.md`.
  bool usesThreadgroupsPerGrid = false;
  op.walk([&](mlir::triton::metal::ThreadgroupsPerGridOp) {
    usesThreadgroupsPerGrid = true;
    return mlir::WalkResult::interrupt();
  });
  if (usesThreadgroupsPerGrid)
    _output << ",\n  uint3 tgpg [[threadgroups_per_grid]]";
  // AC4 v6: conditionally add the SIMD-group index parameter when any
  // SimdgroupIndexOp is referenced in the body. `simdgroup_index_in_threadgroup`
  // is an MSL parameter attribute (cf. MSL Spec §5.8) — it cannot be referenced
  // as a free-standing expression. Single-warp kernels skip this parameter
  // so legacy MSL signatures stay unchanged.
  bool usesSimdgroupIndex = false;
  op.walk([&](mlir::triton::metal::SimdgroupIndexOp) {
    usesSimdgroupIndex = true;
    return mlir::WalkResult::interrupt();
  });
  if (usesSimdgroupIndex)
    _output << ",\n  uint sgid [[simdgroup_index_in_threadgroup]]";
  _output << ")\n";

  auto firstBlock = op.getBodyRegion().getBlocks().begin();
  for (auto const &it : llvm::enumerate(firstBlock->getArguments()))
    _buffers[it.value().getAsOpaquePointer()] = it.index();

  translate(op.getBodyRegion());
}

void ModuleTranslation::printDelim() {
  if (inWhileCondition) {
    _output << ",";
  } else {
    _output << ";";
  }
}

bool ModuleTranslation::isStatementPrintable(Operation *opInst) {
  auto printable = false;
  llvm::TypeSwitch<Operation *>(opInst)
      .Case<mlir::triton::metal::AllocaOp,
            mlir::triton::metal::ThreadgroupAllocaOp,
            mlir::triton::metal::BarrierOp,
            mlir::triton::metal::TgStoreIndexedOp,
            mlir::triton::metal::StoreOp, mlir::triton::metal::AtomicRmwOp,
            mlir::triton::metal::ThreadgroupPrefixSumOp,
            mlir::triton::metal::IfOp,
            mlir::triton::metal::WhileOp, mlir::triton::metal::MatmulOp, mlir::triton::metal::GemvOp,
            mlir::triton::metal::QmvOp, mlir::triton::metal::QmmOp, mlir::triton::metal::ReduceOp,
            mlir::triton::metal::ArgmaxOp, mlir::triton::metal::SoftmaxOp,
            mlir::triton::metal::LogsumexpOp, mlir::triton::metal::SdpaOp,
            mlir::triton::metal::FlashAttentionOp,
            mlir::triton::metal::RmsNormOp, mlir::triton::metal::ReturnOp,
            mlir::triton::metal::SimdgroupMatrixZeroOp,
            mlir::triton::metal::SimdgroupLoadDeviceStagedOp,
            mlir::triton::metal::SimdgroupLoadDeviceStagedMaskedOp,
            mlir::triton::metal::SimdgroupLoadOp,
            mlir::triton::metal::SimdgroupMultiplyAccumulateOp,
            mlir::triton::metal::SimdgroupStoreOp,
            mlir::triton::metal::SimdgroupFusedStoreOp,
            mlir::scf::IfOp, mlir::scf::ForOp>(
          [&](auto &op) { printable = true; })
      .Case<mlir::triton::metal::YieldWhileOp, mlir::triton::metal::YieldOp>([&](auto &op) {
        // do nothing
        printable = false;
      })
      .Case<mlir::scf::YieldOp>([&](auto &op) {
        // Yielding into a result-bearing `scf.if` produces assignment
        // statements to the temp vars pre-declared before the `if`. Yields in
        // void `scf.if` are no-ops.
        printable = false;
        if (auto parentIf =
                llvm::dyn_cast_or_null<mlir::scf::IfOp>(op->getParentOp())) {
          if (parentIf.getNumResults() > 0)
            printable = true;
        }
      })
      .Default([&](Operation *) {
        if (opInst->use_empty()) {
          printable = true;
        }
      });
  return printable;
}

void ModuleTranslation::translateStatement(Operation *opInst) {
  llvm::TypeSwitch<Operation *>(opInst)
      .Case<mlir::triton::metal::AllocaOp,
            mlir::triton::metal::ThreadgroupAllocaOp,
            mlir::triton::metal::BarrierOp,
            mlir::triton::metal::TgStoreIndexedOp,
            mlir::triton::metal::StoreOp, mlir::triton::metal::AtomicRmwOp,
            mlir::triton::metal::ThreadgroupPrefixSumOp,
            mlir::triton::metal::IfOp,
            mlir::triton::metal::WhileOp, mlir::triton::metal::MatmulOp, mlir::triton::metal::GemvOp,
            mlir::triton::metal::QmvOp, mlir::triton::metal::QmmOp, mlir::triton::metal::ReduceOp,
            mlir::triton::metal::ArgmaxOp, mlir::triton::metal::SoftmaxOp,
            mlir::triton::metal::LogsumexpOp, mlir::triton::metal::SdpaOp,
            mlir::triton::metal::FlashAttentionOp,
            mlir::triton::metal::RmsNormOp, mlir::triton::metal::ReturnOp,
            mlir::triton::metal::SimdgroupMatrixZeroOp,
            mlir::triton::metal::SimdgroupLoadDeviceStagedOp,
            mlir::triton::metal::SimdgroupLoadDeviceStagedMaskedOp,
            mlir::triton::metal::SimdgroupLoadOp,
            mlir::triton::metal::SimdgroupMultiplyAccumulateOp,
            mlir::triton::metal::SimdgroupStoreOp,
            mlir::triton::metal::SimdgroupFusedStoreOp,
            mlir::triton::metal::SimdgroupIndexOp>(
          [&](auto &op) { translate(op); })
      .Case<mlir::scf::IfOp>([&](auto &op) { translate(op); })
      .Case<mlir::scf::ForOp>([&](auto &op) { translate(op); })
      .Case<mlir::scf::YieldOp>([&](auto &op) { translate(op); })
      .Case<mlir::triton::metal::YieldWhileOp, mlir::triton::metal::YieldOp>([&](auto &op) {
        // do nothing;
      })
      .Default([&](Operation *) {
        if (opInst->use_empty()) {
          translateValue(opInst);
          printDelim();
        }
      });
}

void ModuleTranslation::translate(mlir::triton::metal::AllocaOp op) {
  auto memRef = llvm::cast<MetalMemRefType>(op.getResult().getType());
  auto stringType = typeToString(memRef.getType());
  _output << stringType << " v" << _varCount << "[" << memRef.getSize() << "]";
  _alloca[op] = _varCount++;
  _output << ";";
}

void ModuleTranslation::translate(mlir::triton::metal::ThreadgroupAllocaOp op) {
  auto memRef = llvm::cast<MetalMemRefType>(op.getResult().getType());
  auto stringType = typeToString(memRef.getType());
  // Emit a declaration in the kernel body with the `threadgroup` address-space
  // qualifier so all threads in the same threadgroup share the buffer. The
  // surrounding indent/level is established by the caller. See
  // `.omc/specs/deep-interview-leet-triton-l3-reduce-axis-2d.md` AC.I5.
  _output << "threadgroup " << stringType << " v" << _varCount << "["
          << memRef.getSize() << "]";
  _alloca[op] = _varCount++;
  _output << ";";
}

void ModuleTranslation::translate(mlir::triton::metal::BarrierOp op) {
  (void)op;
  _output << "threadgroup_barrier(mem_flags::mem_threadgroup);";
}

void ModuleTranslation::translate(mlir::triton::metal::TgStoreIndexedOp op) {
  // Emit `<bufVar>[<idx>] = <val>;` for an indexed store into a threadgroup
  // buffer. Phase A infrastructure for §3.5 staged-transpose (L1d2). See
  // `.omc/specs/deep-interview-leet-triton-l1d-phase-a-staged-transpose-infra.md`.
  translateVarName(op.getBuffer());
  _output << "[";
  translateValue(op.getIndex().getDefiningOp());
  _output << "] = ";
  translateValue(op.getValue().getDefiningOp());
  printDelim();
}

void ModuleTranslation::translate(mlir::triton::metal::TgLoadIndexedOp op) {
  // Emit `<bufVar>[<idx>]` as a value expression (called from translateValue).
  // Phase A infrastructure for §3.5 staged-transpose (L1d2). See
  // `.omc/specs/deep-interview-leet-triton-l1d-phase-a-staged-transpose-infra.md`.
  translateVarName(op.getBuffer());
  _output << "[";
  translateValue(op.getIndex().getDefiningOp());
  _output << "]";
}

void ModuleTranslation::translate(mlir::triton::metal::StoreOp op) {
  translateVarName(op.getMemref());
  _output << "[";
  translateValue(op.getIndex().getDefiningOp());
  _output << "] = ";
  translateValue(op.getValue().getDefiningOp());
  printDelim();
}

void ModuleTranslation::translate(
    mlir::triton::metal::ThreadgroupPrefixSumOp op) {
  // Inclusive prefix-sum (cumsum) of inbuf -> outbuf over `block` elements,
  // tiled over E = block/tpb cyclic iv-blocks (pos = iv*tpb + tid). Per iv-block
  // an in-place double-barriered Hillis-Steele inclusive scan runs across the
  // tpb threads (its window is outbuf[base + 0..tpb), contiguous), then a
  // running `carry` (sum of prior iv-blocks) is added. Validated bit-close to
  // torch.cumsum (scan_spike.py). Barriers are OUTSIDE any per-thread branch so
  // all threads reach them (caller guarantees uniform control flow).
  const int64_t BLOCK = op.getBlock();
  const int64_t TPB = op.getTpb();
  const int64_t E = BLOCK / TPB;
  auto bufName = [&](mlir::Value m) -> std::string {
    while (auto cast = m.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        break;
      m = cast.getInputs()[0];
    }
    if (auto def = m.getDefiningOp())
      if (auto it = _alloca.find(def); it != _alloca.end())
        return "v" + std::to_string(it->second);
    auto it = _buffers.find(m.getAsOpaquePointer());
    return "v" + std::to_string(it != _buffers.end() ? it->second : 0);
  };
  const std::string IN = bufName(op.getInbuf());
  const std::string OUT = bufName(op.getOutbuf());
  auto S = [](int64_t x) { return std::to_string(x); };
  // The accumulator/temporaries must carry the BUFFER's element type, not a
  // hardcoded float: an integer cumsum (`tl.cumsum` over i32, the per-digit
  // rank in a radix sort) stages through a ui32 threadgroup buffer, and a float
  // carry would silently round every partial sum past 2^24.
  auto bufElemTy = [](mlir::Value m) -> mlir::Type {
    return llvm::cast<MetalMemRefType>(m.getType()).getType();
  };
  const llvm::StringRef ELEM = typeToString(bufElemTy(op.getOutbuf()));
  const std::string ZERO =
      llvm::isa<mlir::FloatType>(bufElemTy(op.getOutbuf())) ? "0.0f" : "0";
  auto &os = _output;
  os << "\n  // ---- metal.threadgroup_prefix_sum (cumsum) ----\n";
  os << "  {\n";
  os << "    uint _ps_tid = id.x - tgid.x * " << S(TPB) << "u;\n";
  os << "    " << ELEM << " _ps_carry = " << ZERO << ";\n";
  os << "    for (uint _ps_k = 0u; _ps_k < " << S(E) << "u; ++_ps_k) {\n";
  os << "      uint _ps_base = _ps_k * " << S(TPB) << "u;\n";
  os << "      " << OUT << "[_ps_base + _ps_tid] = " << IN
     << "[_ps_base + _ps_tid];\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      for (uint _ps_off = 1u; _ps_off < " << S(TPB)
     << "u; _ps_off <<= 1) {\n";
  os << "        " << ELEM << " _ps_add = (_ps_tid >= _ps_off) ? " << OUT
     << "[_ps_base + _ps_tid - _ps_off] : " << ZERO << ";\n";
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "        " << OUT << "[_ps_base + _ps_tid] += _ps_add;\n";
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      }\n";
  os << "      " << ELEM << " _ps_total = " << OUT << "[_ps_base + "
     << S(TPB - 1) << "u];\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      " << OUT << "[_ps_base + _ps_tid] += _ps_carry;\n";
  os << "      _ps_carry += _ps_total;\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "    }\n";
  os << "  }\n";
}

void ModuleTranslation::translate(mlir::triton::metal::AtomicRmwOp op) {
  // Atomic fadd into device memory with relaxed ordering. Reinterpret the
  // element address `&buf[idx]` (a `device float*`) as a `device atomic_float*`;
  // valid for relaxed atomics in the `device` address space on Apple GPUs. The
  // returned old value is discarded — the conversion guarantees the result is
  // unused (AtomicRmwLowering rejects atomics whose old value is consumed).
  _output << "atomic_fetch_add_explicit((device atomic_float*)&";
  translateVarName(op.getMemref());
  _output << "[";
  translateValueOrVarName(op.getIndex());
  _output << "], ";
  translateValueOrVarName(op.getValue());
  _output << ", memory_order_relaxed)";
  printDelim();
}

//===----------------------------------------------------------------------===//
// SIMD-group matrix ops (matmul track). Emit the modern Metal 17.5 SIMD-group
// matrix surface: `simdgroup_load` / `simdgroup_multiply_accumulate` /
// `simdgroup_store`. The legacy names `simdgroup_load_matrix`,
// `simdgroup_store_matrix`, and `simdgroup_matrix_multiply_accumulate` are
// ALL rejected by the current MSL compiler — do not emit them.
//
// Emitter is origin-aware: when both `originRow` and `originCol` are literal
// `0` constants (the staging path in `rewriteSingleDot` etc. arranges this),
// emit the 3-arg form `simdgroup_load(dst, ptr, stride)` matching the
// already-shipping flash-attention precedent at line 1212+. Otherwise emit
// the 5-arg fallback `simdgroup_load(dst, ptr, stride, ulong2(col,row), false)`.
//===----------------------------------------------------------------------===//

static void emitSimdgroupMatrixType(
    llvm::raw_ostream &os, mlir::triton::metal::MetalSimdgroupMatrixType ty) {
  os << "simdgroup_" << typeToString(ty.getElem()) << ty.getRows() << "x"
     << ty.getCols();
}

// Returns true iff `v` is defined by an arith.constant or metal.constant
// with integer value 0.
static bool isLiteralZero(mlir::Value v) {
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return intAttr.getValue().isZero();
  }
  if (auto cst = v.getDefiningOp<mlir::triton::metal::ConstantOp>()) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return intAttr.getValue().isZero();
  }
  return false;
}

void ModuleTranslation::translate(
    mlir::triton::metal::SimdgroupIndexOp op) {
  // No statement emission. `simdgroup_index_in_threadgroup` is an MSL
  // *parameter attribute* (kernel input qualifier), not a free-standing
  // built-in expression. translateKernel injects the parameter
  // `uint sgid [[simdgroup_index_in_threadgroup]]` when the kernel body
  // references SimdgroupIndexOp; uses resolve to `sgid` via the
  // translateValue/translateVarName cases below. See Metal Shading
  // Language Specification §5.8 (Attributes for Kernel Function Input).
}

void ModuleTranslation::translate(
    mlir::triton::metal::SimdgroupMatrixZeroOp op) {
  // Emit `simdgroup_<dtype><rows>x<cols> vN(0.0f);` — Apple's matmul +
  // flash-attention samples (and the FA emitter at line ~1381 of this
  // file) initialize the accumulator this way. Empirical: initializing
  // the accumulator via `simdgroup_load` from a pre-zeroed device buffer
  // instead causes chained `simdgroup_multiply_accumulate` calls to drop
  // contributions at specific output columns on Apple GPU family 9 /
  // Metal 17.5 (probe10/probe5/probe6 in iter-8 diagnostics).
  auto resTy = llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
      op.getResult().getType());
  emitSimdgroupMatrixType(_output, resTy);
  unsigned id = _varCount;
  _output << " v" << id << "(0.0f)";
  _alloca[op] = _varCount++;
  printDelim();
}

void ModuleTranslation::translate(
    mlir::triton::metal::SimdgroupLoadDeviceStagedOp op) {
  // Emits the FA-style staged load: cooperative device->threadgroup copy,
  // simdgroup_barrier, then simdgroup_load from the threadgroup buffer.
  // Apple's matmul + FA samples ALL chain `simdgroup_multiply_accumulate`
  // only on values loaded from `threadgroup T*`. Direct chained sgmma on
  // `const device T*`-loaded values silently drops output-column
  // contributions on Apple GPU family 9 / Metal 17.5 (iter-8 root cause).
  //
  // AC4 v6: two branches.
  //   Branch A (`warp_index` empty): legacy single-warp/bit-identical
  //   `_stage_<id>[elems]` shared across all warps. Used by single-tile
  //   (m=n=1) emission and any pre-AC4 caller.
  //   Branch B (`warp_index` size==1): per-warp slice
  //   `_stage_<id>[<num_warps>][elems]` indexed by `[v<widx>]` so each
  //   warp owns a disjoint stage buffer. `<num_warps>` is baked in at
  //   translate time via `triton::gpu::lookupNumWarps(op)`.
  auto resTy = llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
      op.getResult().getType());
  unsigned id = _varCount;
  unsigned elems = resTy.getRows() * resTy.getCols();
  auto warpIdx = op.getWarpIndex();
  const bool perWarp = !warpIdx.empty();

  // AC4 v6: entry barrier — ensure the previous use of `_stage_shared` is
  // complete on all warps before this load overwrites it. The shared
  // buffer is declared at kernel-body top (see `translate(Region&)`); each
  // staged load brackets its coop-load with `threadgroup_barrier` so the
  // buffer can be safely reused 100+ times within a kernel.
  _output << "threadgroup_barrier(mem_flags::mem_threadgroup)";
  printDelim();
  _output << "\n";
  indent();
  _output << "for (uint c = (id.x & 31u); c < " << elems << "u; c += 32u) {";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint si = c / " << resTy.getCols() << "u";
    printDelim();
    _output << "\n";
    indent();
    _output << "uint sj = c % " << resTy.getCols() << "u";
    printDelim();
    _output << "\n";
    indent();
    // W1 transposed-B: write the gathered element to the swapped staging slot
    // `sj*cols + si` instead of `c == si*cols + sj`, so the subsequent
    // `simdgroup_load` reads the transpose of the [origin_row, origin_col]
    // device tile (used for `tl.dot(a, tl.trans(b))`). The device address is
    // unchanged; only the threadgroup-buffer destination is transposed.
    const bool transposed = op.getTransposed();
    if (perWarp) {
      // Use translateVarName so SimdgroupIndexOp (and any
      // UnrealizedConversionCast wrapper) resolves to `sgid`.
      _output << "_stage_shared[";
      translateVarName(warpIdx[0]);
      _output << "][";
    } else {
      _output << "_stage_shared[";
    }
    if (transposed)
      _output << "sj * " << resTy.getCols() << "u + si";
    else
      _output << "c";
    _output << "] = ";
    translateVarName(op.getMemref());
    _output << "[(";
    translateValue(op.getOriginRow().getDefiningOp());
    _output << " + si) * ";
    translateValue(op.getStride().getDefiningOp());
    _output << " + ";
    translateValue(op.getOriginCol().getDefiningOp());
    _output << " + sj]";
    printDelim();
  }
  _output << "\n";
  indent();
  _output << "}\n";
  indent();
  _output << "threadgroup_barrier(mem_flags::mem_threadgroup)";
  printDelim();
  _output << "\n";
  indent();
  emitSimdgroupMatrixType(_output, resTy);
  _output << " v" << id;
  printDelim();
  _output << "\n";
  indent();
  if (perWarp) {
    _output << "simdgroup_load(v" << id << ", &_stage_shared[";
    translateVarName(warpIdx[0]);
    _output << "][0], " << resTy.getCols() << ")";
  } else {
    _output << "simdgroup_load(v" << id << ", &_stage_shared[0], "
            << resTy.getCols() << ")";
  }
  _alloca[op] = _varCount++;
  printDelim();
}

void ModuleTranslation::translate(
    mlir::triton::metal::SimdgroupLoadDeviceStagedMaskedOp op) {
  // Masked variant: same coop-load into `_stage_shared`, but each lane writes
  // `(gi < row_extent && gj < col_extent) ? memref[...] : 0` (K-tail + edges).
  auto resTy = llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
      op.getResult().getType());
  unsigned id = _varCount;
  unsigned elems = resTy.getRows() * resTy.getCols();
  const bool transposed = op.getTransposed();
  const bool perWarp = !op.getWarpIndex().empty();
  auto warpDim = [&]() {
    if (perWarp) {
      _output << "[";
      translateVarName(op.getWarpIndex()[0]);
      _output << "]";
    }
  };

  _output << "threadgroup_barrier(mem_flags::mem_threadgroup)";
  printDelim();
  _output << "\n";
  indent();
  _output << "for (uint c = (id.x & 31u); c < " << elems << "u; c += 32u) {";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint si = c / " << resTy.getCols() << "u";
    printDelim();
    _output << "\n";
    indent();
    _output << "uint sj = c % " << resTy.getCols() << "u";
    printDelim();
    _output << "\n";
    indent();
    _output << "uint gi = ";
    translateValue(op.getOriginRow().getDefiningOp());
    _output << " + si";
    printDelim();
    _output << "\n";
    indent();
    _output << "uint gj = ";
    translateValue(op.getOriginCol().getDefiningOp());
    _output << " + sj";
    printDelim();
    _output << "\n";
    indent();
    _output << "_stage_shared";
    warpDim();
    _output << "[";
    if (transposed)
      _output << "sj * " << resTy.getCols() << "u + si";
    else
      _output << "c";
    _output << "] = (gi < ";
    translateValueOrVarName(op.getRowExtent());
    _output << " && gj < ";
    translateValueOrVarName(op.getColExtent());
    _output << ") ? ";
    translateVarName(op.getMemref());
    _output << "[gi * ";
    translateValue(op.getStride().getDefiningOp());
    _output << " + gj] : 0.0f";
    printDelim();
  }
  _output << "\n";
  indent();
  _output << "}\n";
  indent();
  _output << "threadgroup_barrier(mem_flags::mem_threadgroup)";
  printDelim();
  _output << "\n";
  indent();
  emitSimdgroupMatrixType(_output, resTy);
  _output << " v" << id;
  printDelim();
  _output << "\n";
  indent();
  _output << "simdgroup_load(v" << id << ", &_stage_shared";
  warpDim();
  _output << "[0], " << resTy.getCols() << ")";
  _alloca[op] = _varCount++;
  printDelim();
}

void ModuleTranslation::translate(mlir::triton::metal::SimdgroupLoadOp op) {
  auto resTy = llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
      op.getResult().getType());
  // Declare-then-call form so the destination has a stable C++ name for the
  // out-parameter call.
  emitSimdgroupMatrixType(_output, resTy);
  unsigned id = _varCount;
  _output << " v" << id;
  printDelim();
  _output << "\n";
  indent();
  _output << "simdgroup_load(v" << id << ", ";
  // iter-8: emit pointer-arithmetic for the (origin_row, origin_col) offset
  // instead of the ulong2 origin parameter. Empirical: `simdgroup_load(...,
  // ulong2(col_off, 0), false)` with non-zero col_off and `device float *`
  // src produces wrong results on Apple GPU family 9 / Metal 17.5 (probe10:
  // chained K-loop sgmma accumulator loses contributions at specific
  // output columns even when the load pattern is in-bounds). Folding the
  // offset into `&ptr[row*stride + col]` bypasses the ulong2 origin
  // entirely and the chained-accumulator pattern matches Apple's own
  // matmul sample (https://developer.apple.com/documentation/metal/optimizing_compute_shaders_with_simdgroup_matrix).
  if (isLiteralZero(op.getOriginRow()) && isLiteralZero(op.getOriginCol())) {
    translateVarName(op.getMemref());
  } else {
    _output << "&(";
    translateVarName(op.getMemref());
    _output << "[";
    translateValue(op.getOriginRow().getDefiningOp());
    _output << " * ";
    translateValue(op.getStride().getDefiningOp());
    _output << " + ";
    translateValue(op.getOriginCol().getDefiningOp());
    _output << "])";
  }
  _output << ", ";
  translateValue(op.getStride().getDefiningOp());
  _output << ")";
  _alloca[op] = _varCount++;
  printDelim();
}

void ModuleTranslation::translate(
    mlir::triton::metal::SimdgroupMultiplyAccumulateOp op) {
  auto resTy = llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
      op.getResult().getType());
  // iter-8 (Apple accumulator pattern): when the C operand has exactly one
  // use (this sgmma) AND has a defining op with a stored variable name,
  // reuse the C operand's MSL variable as the destination instead of
  // declaring a fresh one. Emits `simdgroup_multiply_accumulate(vC, A, B,
  // vC)` — the in-place form Apple's matmul sample uses
  // (https://developer.apple.com/documentation/metal/optimizing_compute_shaders_with_simdgroup_matrix).
  // Empirical: emitting `simdgroup_float8x8 vNew; sgmma(vNew, A, B, vC)`
  // for a chained K-loop accumulator silently drops some output-column
  // contributions on Apple GPU family 9 / Metal 17.5 (probe10: only some
  // columns receive iter-1's contribution).
  mlir::Operation *cDef = op.getC().getDefiningOp();
  bool reuseC = cDef && op.getC().hasOneUse() && _alloca.count(cDef);
  unsigned id;
  if (reuseC) {
    id = _alloca[cDef];
  } else if (!cDef && _buffers.count(op.getC().getAsOpaquePointer())) {
    // W2a runtime-K loop: the C operand is the scf.for accumulator iter_arg
    // (a BlockArgument mapped to an MSL temp by translate(scf::ForOp)). Reuse
    // that temp so the chain accumulates in place —
    // `simdgroup_multiply_accumulate(vAcc, A, B, vAcc)` — which is the
    // Apple-family-9 correct form (a fresh destination silently drops
    // output-column contributions; see the SimdgroupMatrixZeroOp comment).
    reuseC = true;
    id = _buffers[op.getC().getAsOpaquePointer()];
  } else {
    emitSimdgroupMatrixType(_output, resTy);
    id = _varCount++;
    _output << " v" << id;
    printDelim();
    _output << "\n";
    indent();
  }
  _output << "simdgroup_multiply_accumulate(v" << id << ", ";
  translateVarName(op.getA());
  _output << ", ";
  translateVarName(op.getB());
  _output << ", ";
  translateVarName(op.getC());
  _output << ")";
  _alloca[op] = id;
  printDelim();
}

void ModuleTranslation::translate(mlir::triton::metal::SimdgroupStoreOp op) {
  // AC2 partial-tile path: when the optional `m_extent`/`n_extent` UI32
  // operands are present, emit a two-stage epilogue mirroring
  // SimdgroupLoadDeviceStagedOp:
  //   1. simdgroup_store into per-threadgroup scratch
  //   2. threadgroup_barrier(mem_flags::mem_threadgroup)
  //   3. coop-loop that scalar-copies only the lanes whose `(gi, gj)`
  //      global coordinates satisfy `gi < m_extent && gj < n_extent` back
  //      to the destination memref.
  // Apple's simdgroup_store has no per-lane mask; this is the documented
  // masked-tail pattern from Apple's matmul samples.
  auto partialExtents = op.getPartialExtents();
  if (partialExtents.size() == 2) {
    mlir::Value mExtentVal = partialExtents[0];
    mlir::Value nExtentVal = partialExtents[1];
    auto matTy = llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
        op.getMatrix().getType());
    unsigned id = _varCount++;
    unsigned elems = matTy.getRows() * matTy.getCols();

    _output << "threadgroup " << typeToString(matTy.getElem())
            << " _scratch_" << id << "[" << elems << "]";
    printDelim();
    _output << "\n";
    indent();
    _output << "simdgroup_store(";
    translateVarName(op.getMatrix());
    _output << ", &_scratch_" << id << "[0], " << matTy.getCols() << ")";
    printDelim();
    _output << "\n";
    indent();
    _output << "threadgroup_barrier(mem_flags::mem_threadgroup)";
    printDelim();
    _output << "\n";
    indent();
    _output << "for (uint c = (id.x & 31u); c < " << elems << "u; c += 32u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "uint si = c / " << matTy.getCols() << "u";
      printDelim();
      _output << "\n";
      indent();
      _output << "uint sj = c % " << matTy.getCols() << "u";
      printDelim();
      _output << "\n";
      indent();
      _output << "uint gi = ";
      translateValue(op.getOriginRow().getDefiningOp());
      _output << " + si";
      printDelim();
      _output << "\n";
      indent();
      _output << "uint gj = ";
      translateValue(op.getOriginCol().getDefiningOp());
      _output << " + sj";
      printDelim();
      _output << "\n";
      indent();
      _output << "if (gi < ";
      translateValue(mExtentVal.getDefiningOp());
      _output << " && gj < ";
      translateValue(nExtentVal.getDefiningOp());
      _output << ") {";
      {
        INDENT();
        _output << "\n";
        indent();
        translateVarName(op.getMemref());
        _output << "[gi * ";
        translateValue(op.getStride().getDefiningOp());
        _output << " + gj] = _scratch_" << id << "[c]";
        printDelim();
      }
      _output << "\n";
      indent();
      _output << "}";
    }
    _output << "\n";
    indent();
    _output << "}";
    return;
  }

  _output << "simdgroup_store(";
  translateVarName(op.getMatrix());
  _output << ", ";
  // iter-8: same pointer-arithmetic emission as SimdgroupLoadOp.
  if (isLiteralZero(op.getOriginRow()) && isLiteralZero(op.getOriginCol())) {
    translateVarName(op.getMemref());
  } else {
    _output << "&(";
    translateVarName(op.getMemref());
    _output << "[";
    translateValue(op.getOriginRow().getDefiningOp());
    _output << " * ";
    translateValue(op.getStride().getDefiningOp());
    _output << " + ";
    translateValue(op.getOriginCol().getDefiningOp());
    _output << "])";
  }
  _output << ", ";
  translateValue(op.getStride().getDefiningOp());
  _output << ")";
  printDelim();
}

void ModuleTranslation::translate(
    mlir::triton::metal::SimdgroupFusedStoreOp op) {
  // W2c fused epilogue: store `base + scale * delta`. Both matrices are staged
  // to per-threadgroup scratch, then a coop-loop writes
  // `scratch_base[c] + scale * scratch_delta[c]` back to the destination
  // (optionally masked by partial_extents = [m_extent, n_extent]).
  auto matTy = llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
      op.getBase().getType());
  unsigned elems = matTy.getRows() * matTy.getCols();
  auto partialExtents = op.getPartialExtents();
  const bool masked = partialExtents.size() == 2;
  // Multi-warp: each warp writes its own output tiles, so it uses its own
  // `[num_warps]` slice of the shared _fstore scratch (indexed by sgid).
  const bool perWarp = !op.getWarpIndex().empty();
  auto warpDim = [&]() {
    if (perWarp) {
      _output << "[";
      translateVarName(op.getWarpIndex()[0]);
      _output << "]";
    }
  };

  // The `_fstore_base`/`_fstore_delta` scratch is declared once at kernel top
  // and reused across fused stores; entry barrier waits for the previous
  // store's coop-read before overwriting.
  _output << "threadgroup_barrier(mem_flags::mem_threadgroup)";
  printDelim();
  _output << "\n";
  indent();
  _output << "simdgroup_store(";
  translateVarName(op.getBase());
  _output << ", &_fstore_base";
  warpDim();
  _output << "[0], " << matTy.getCols() << ")";
  printDelim();
  _output << "\n";
  indent();
  _output << "simdgroup_store(";
  translateVarName(op.getDelta());
  _output << ", &_fstore_delta";
  warpDim();
  _output << "[0], " << matTy.getCols() << ")";
  printDelim();
  _output << "\n";
  indent();
  _output << "threadgroup_barrier(mem_flags::mem_threadgroup)";
  printDelim();
  _output << "\n";
  indent();
  _output << "for (uint c = (id.x & 31u); c < " << elems << "u; c += 32u) {";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint si = c / " << matTy.getCols() << "u";
    printDelim();
    _output << "\n";
    indent();
    _output << "uint sj = c % " << matTy.getCols() << "u";
    printDelim();
    _output << "\n";
    indent();
    _output << "uint gi = ";
    translateValue(op.getOriginRow().getDefiningOp());
    _output << " + si";
    printDelim();
    _output << "\n";
    indent();
    _output << "uint gj = ";
    translateValue(op.getOriginCol().getDefiningOp());
    _output << " + sj";
    printDelim();
    _output << "\n";
    indent();
    if (masked) {
      _output << "if (gi < ";
      translateValue(partialExtents[0].getDefiningOp());
      _output << " && gj < ";
      translateValue(partialExtents[1].getDefiningOp());
      _output << ") {";
      INDENT();
      _output << "\n";
      indent();
    }
    translateVarName(op.getMemref());
    _output << "[gi * ";
    translateValue(op.getStride().getDefiningOp());
    _output << " + gj] = _fstore_base";
    warpDim();
    _output << "[c] + ";
    translateValueOrVarName(op.getScale());
    _output << " * _fstore_delta";
    warpDim();
    _output << "[c]";
    printDelim();
    if (masked) {
      _output << "\n";
      indent();
      _output << "}";
    }
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translate(IfOp op) {
  _output << "if (";
  translateValue(op.getCondition().getDefiningOp());
  _output << ") ";

  translate(op.getThenRegion());

  auto &elseRegion = op.getElseRegion();
  if (elseRegion.getBlocks().size()) {
    _output << " else ";
    translate(elseRegion);
  }
}

void ModuleTranslation::translate(WhileOp op) {
  _output << "while (";

  auto &conditionRegion = op.getConditionRegion();
  {
    inWhileCondition = true;
    for (auto &op : conditionRegion.getOps()) {
      translateStatement(&op);
      if (isStatementPrintable(&op))
        _output << " ";
    }
    inWhileCondition = false;
  }
  auto conditionOp =
      dyn_cast<YieldWhileOp>(conditionRegion.back().getTerminator());
  translateValue(conditionOp);
  _output << ") ";

  auto &bodyRegion = op.getBodyRegion();
  translate(bodyRegion);
}

void ModuleTranslation::translate(ReturnOp op) { _output << "return;"; }

void ModuleTranslation::translate(mlir::scf::IfOp op) {
  // Pre-declare a temp var per result so the assignment inside then/else
  // (emitted by translate(scf.yield)) has somewhere to land. Single-result
  // scf.if is supported; multi-result would need a tuple of temps — left to
  // future work since the masked vector_add path only uses the 1-result form.
  for (auto res : op.getResults()) {
    auto idx = _varCount++;
    _scfIfTemp[op.getOperation()] = idx;
    _output << typeToString(res.getType()) << " v" << idx << ";\n";
    indent();
  }
  _output << "if (";
  // The condition may be a block arg (e.g. an i1 scf.for iter_arg like
  // speculative decoding's `accepted_all`), so resolve by name when it has no
  // defining op rather than crashing on a null Operation.
  translateValueOrVarName(op.getCondition());
  _output << ") ";
  translate(op.getThenRegion());
  auto &elseRegion = op.getElseRegion();
  if (!elseRegion.empty() && !elseRegion.front().empty()) {
    _output << " else ";
    translate(elseRegion);
  }
}

void ModuleTranslation::translate(mlir::scf::ForOp op) {
  // Emit `for (int v_iv = lb; v_iv < ub; v_iv += step) { body }`. The
  // induction var BlockArgument is registered in `_buffers` so any
  // downstream translateVarName / translateValue on it resolves to the
  // emitted temp name. Trip-count bounds are translated via the existing
  // arith.constant path; non-constant bounds fall through to the
  // generic Operation translator.
  //
  // Wall 15: when the scf.for carries exactly one f32 scalar iter_arg
  // (the rank-1 reduce accumulator), emit `float v<accIdx> = init;` BEFORE
  // the C-style for line; map the region iter-arg BlockArgument into
  // `_buffers` so reads inside the body resolve to v<accIdx>; and record
  // the mapping in `_scfForIterArg` so the matching scf.yield can emit
  // `v<accIdx> = yielded;`. Multi-iter_arg or non-f32 iter_arg falls
  // through to the existing IV-only emission unchanged. See
  // .omc/plans/tutorial02-wall15-iter-args-translator-consensus.md AC1/AC2.
  auto singleIterArgTy = op.getNumRegionIterArgs() == 1
                             ? op.getRegionIterArgs()[0].getType()
                             : mlir::Type();
  if (singleIterArgTy &&
      (singleIterArgTy.isF32() || singleIterArgTy.isInteger(32) ||
       singleIterArgTy.isInteger(1))) {
    unsigned accIdx = _varCount++;
    _scfForIterArg[op.getOperation()] = accIdx;
    auto iterArg = op.getRegionIterArgs()[0];
    _buffers[iterArg.getAsOpaquePointer()] = accIdx;
    // The loop's final result (op.getResult(0)) is the same MSL temp once
    // the loop exits — register it so downstream `translateVarName` /
    // `translateValueOrVarName` on the scf.for result renders `v<accIdx>`.
    _buffers[op.getResult(0).getAsOpaquePointer()] = accIdx;
    // Emit the accumulator with its real element type: `float` for f32
    // reduces, `int32_t` / `uint32_t` for the i32 max/sum rank-1 tile-loop
    // accumulator (BLOCK > threads_per_block).
    _output << typeToString(iterArg.getType()) << " v" << accIdx << " = ";
    if (auto initOp = op.getInitArgs()[0].getDefiningOp())
      translateValue(initOp);
    else
      translateVarName(op.getInitArgs()[0]);
    _output << ";\n";
    indent();
  } else if (op.getNumRegionIterArgs() >= 1 &&
             llvm::all_of(op.getRegionIterArgs(), [](mlir::Value v) {
               return mlir::isa<mlir::triton::metal::MetalSimdgroupMatrixType>(
                   v.getType());
             })) {
    // W2a/W2b runtime-K matmul accumulator(s) — one or several. Each iter_arg
    // init op (simdgroup_matrix_zero / simdgroup_load) is emitted before the
    // loop as its own MSL temp; reuse THAT temp as the loop-carried accumulator
    // so the in-place `simdgroup_multiply_accumulate` chain writes directly
    // into it. No new declarations, and the matching scf.yield is intentionally
    // a no-op (not recorded in the iter-arg maps) since the accumulators update
    // in place.
    for (unsigned i = 0; i < op.getNumRegionIterArgs(); ++i) {
      auto iterArg = op.getRegionIterArgs()[i];
      if (auto *initDef = op.getInitArgs()[i].getDefiningOp();
          initDef && _alloca.count(initDef)) {
        unsigned accIdx = _alloca[initDef];
        _buffers[iterArg.getAsOpaquePointer()] = accIdx;
        _buffers[op.getResult(i).getAsOpaquePointer()] = accIdx;
      }
    }
  } else if (op.getNumRegionIterArgs() >= 2 &&
             llvm::all_of(op.getRegionIterArgs(), [](mlir::Value v) {
               return v.getType().isF32() || v.getType().isInteger(32) ||
                      v.getType().isInteger(1);
             })) {
    // Multi-accumulator reduce loop (K-way ILP): declare one MSL temp per
    // iter_arg, map each region iter-arg and each loop result to its temp, and
    // record them so the matching scf.yield writes all K back. Mirrors the
    // single-iter_arg path above, N times. See metal-multiacc-reduce-plan.md.
    llvm::SmallVector<unsigned, 8> idxs;
    for (unsigned i = 0; i < op.getNumRegionIterArgs(); ++i) {
      unsigned accIdx = _varCount++;
      idxs.push_back(accIdx);
      auto iterArg = op.getRegionIterArgs()[i];
      _buffers[iterArg.getAsOpaquePointer()] = accIdx;
      _buffers[op.getResult(i).getAsOpaquePointer()] = accIdx;
      _output << typeToString(iterArg.getType()) << " v" << accIdx << " = ";
      if (auto initOp = op.getInitArgs()[i].getDefiningOp())
        translateValue(initOp);
      else
        translateVarName(op.getInitArgs()[i]);
      _output << ";\n";
      indent();
    }
    _scfForIterArgsMulti[op.getOperation()] = idxs;
  }

  unsigned idx = _varCount++;
  _scfForIv[op.getOperation()] = idx;
  auto iv = op.getInductionVar();
  _buffers[iv.getAsOpaquePointer()] = idx;

  _output << "for (int v" << idx << " = ";
  if (auto lbOp = op.getLowerBound().getDefiningOp())
    translateValue(lbOp);
  else
    translateVarName(op.getLowerBound());
  _output << "; v" << idx << " < ";
  if (auto ubOp = op.getUpperBound().getDefiningOp())
    translateValue(ubOp);
  else
    translateVarName(op.getUpperBound());
  _output << "; v" << idx << " += ";
  if (auto stepOp = op.getStep().getDefiningOp())
    translateValue(stepOp);
  else
    translateVarName(op.getStep());
  _output << ") ";
  translate(op.getRegion());
}

void ModuleTranslation::translate(mlir::scf::YieldOp op) {
  // scf::IfOp parent: existing single-result assignment path.
  if (auto parentIf =
          llvm::dyn_cast_or_null<mlir::scf::IfOp>(op->getParentOp())) {
    if (parentIf.getNumResults() == 0)
      return; // no-op for void scf.if
    auto it = _scfIfTemp.find(parentIf.getOperation());
    assert(it != _scfIfTemp.end() &&
           "scf.yield: parent scf.if has no temp var pre-declared");
    _output << "v" << it->second << " = ";
    translateValueOrVarName(op.getOperand(0));
    _output << ";";
    return;
  }
  // Wall 15: scf::ForOp parent with one f32 iter_arg — assign the yielded
  // value back to the iter_arg temp registered in _scfForIterArg. Loops
  // with zero iter_args (matmul / vector_add tile loops) are no-ops here
  // and stay byte-identical to the pre-Wall-15 emission.
  if (auto parentFor =
          llvm::dyn_cast_or_null<mlir::scf::ForOp>(op->getParentOp())) {
    // Multi-accumulator reduce: write all K iter_args back (`v_i = yielded_i;`).
    auto mit = _scfForIterArgsMulti.find(parentFor.getOperation());
    if (mit != _scfForIterArgsMulti.end()) {
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        _output << "v" << mit->second[i] << " = ";
        translateValueOrVarName(op.getOperand(i));
        _output << "; ";
      }
      return;
    }
    if (op.getNumOperands() != 1)
      return; // zero iter_args or (non-multi) unsupported: not supported, no-op
    auto it = _scfForIterArg.find(parentFor.getOperation());
    if (it == _scfForIterArg.end())
      return; // not a single-f32-iter_arg loop
    _output << "v" << it->second << " = ";
    translateValueOrVarName(op.getOperand(0));
    _output << ";";
    return;
  }
}

// Emit MSL that loads one packed weight value q (an unsigned integer in
// [0, 2^bits)) given an output row index `rowExpr`, a K-axis offset `kkExpr`,
// the K dimension, the bits value, and the Wq buffer name (already
// translated). The emitter writes into _output via the caller's surrounding
// statement context. For bits in {2,4,8} we read a uint32 word; for bits in
// {3,5,6} we assemble bytes from the byte-stream layout into a uint64.
//
// The emitted expression resolves to `uint q;` set on the line preceding the
// caller's use; thus this helper emits a STATEMENT.
static void emitUnpackStatement(llvm::raw_ostream &out,
                                const std::string &wqName,
                                const std::string &rowExpr,
                                const std::string &kkExpr, int64_t k,
                                int64_t bits) {
  int64_t pf = packFactor(bits);
  int64_t mask = (1 << bits) - 1;
  if (bits == 2 || bits == 4 || bits == 8) {
    // uint32 word = wq[row * (K/pf) + kk/pf];
    out << "uint __qword = " << wqName << "[((" << rowExpr << ") * (" << k
        << " / " << pf << ")) + ((" << kkExpr << ") / " << pf << ")];\n";
    out << "      uint q = (__qword >> (((" << kkExpr << ") % " << pf << ") * "
        << bits << ")) & " << mask << ";";
  } else {
    int64_t bpp = bytesPerPack(bits);
    // byte stream: bytes_per_row = (K/pf) * bpp = K*bits/8 (integer).
    // base = wq + row*(K/pf)*bpp + (kk/pf)*bpp
    out << "uint __qbase = ((" << rowExpr << ") * (" << k << " / " << pf
        << ") * " << bpp << ") + (((" << kkExpr << ") / " << pf << ") * "
        << bpp << ");\n";
    out << "      ulong __qword = 0;\n";
    for (int64_t i = 0; i < bpp; ++i) {
      out << "      __qword |= ((ulong)" << wqName << "[__qbase + " << i
          << "]) << " << (i * 8) << ";\n";
    }
    out << "      uint q = (uint)((__qword >> (((" << kkExpr << ") % " << pf
        << ") * " << bits << ")) & " << mask << ");";
  }
}

void ModuleTranslation::translate(mlir::triton::metal::QmvOp op) {
  auto m = op.getM();
  auto k = op.getK();
  auto bits = op.getBits();
  auto gs = op.getGroupSize();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto accTy = typeToString(elemTy);

  // Get var names as strings via a temporary stream — translateVarName uses
  // _output directly, so capture into a small buffer.
  std::string wqName, scalesName, biasesName, xName, outName;
  {
    llvm::raw_string_ostream s(wqName);
    auto saved = &_output;
    // No way to redirect _output cleanly; use auxiliary helper:
    (void)saved;
  }
  // Simpler: re-implement var-name retrieval here using the same lookup that
  // translateVarName uses (memref maps via _buffers when block argument, via
  // _alloca otherwise).
  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  wqName = nameOf(op.getWq());
  scalesName = nameOf(op.getScales());
  biasesName = nameOf(op.getBiases());
  xName = nameOf(op.getX());
  outName = nameOf(op.getOut());

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint row = id.x;\n";
    indent();
    _output << "if (row < " << m << ") {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << accTy << " acc = (" << accTy << ")(0);\n";
      indent();
      _output << "for (uint kk = 0; kk < " << k << "; ++kk) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint __g = kk / " << gs << ";\n";
        indent();
        _output << accTy << " __scale = " << scalesName << "[(row * (" << k
                << " / " << gs << ")) + __g];\n";
        indent();
        _output << accTy << " __bias  = " << biasesName << "[(row * (" << k
                << " / " << gs << ")) + __g];\n";
        indent();
        emitUnpackStatement(_output, wqName, "row", "kk", k, bits);
        _output << "\n";
        indent();
        _output << accTy << " __w = ((" << accTy
                << ")q * __scale) + __bias;\n";
        indent();
        _output << "acc = acc + (__w * " << xName << "[kk]);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << outName << "[row] = acc;";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translate(mlir::triton::metal::QmmOp op) {
  auto m = op.getM();
  auto n = op.getN();
  auto k = op.getK();
  auto bits = op.getBits();
  auto gs = op.getGroupSize();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto accTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string wqName = nameOf(op.getWq());
  std::string scalesName = nameOf(op.getScales());
  std::string biasesName = nameOf(op.getBiases());
  std::string xName = nameOf(op.getX());
  std::string outName = nameOf(op.getOut());

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint row = id.y;\n";
    indent();
    _output << "uint col = id.x;\n";
    indent();
    _output << "if ((row < " << m << ") && (col < " << n << ")) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << accTy << " acc = (" << accTy << ")(0);\n";
      indent();
      _output << "for (uint kk = 0; kk < " << k << "; ++kk) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint __g = kk / " << gs << ";\n";
        indent();
        _output << accTy << " __scale = " << scalesName << "[(col * (" << k
                << " / " << gs << ")) + __g];\n";
        indent();
        _output << accTy << " __bias  = " << biasesName << "[(col * (" << k
                << " / " << gs << ")) + __g];\n";
        indent();
        emitUnpackStatement(_output, wqName, "col", "kk", k, bits);
        _output << "\n";
        indent();
        _output << accTy << " __w = ((" << accTy
                << ")q * __scale) + __bias;\n";
        indent();
        _output << "acc = acc + (__w * " << xName << "[(row * " << k
                << ") + kk]);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << outName << "[(row * " << n << ") + col] = acc;";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translate(mlir::triton::metal::ReduceOp op) {
  auto mOuter = op.getMOuter();
  auto r = op.getR();
  auto kind = op.getKind();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string inName = nameOf(op.getIn());
  std::string outName = nameOf(op.getOut());

  bool isMax = (kind == mlir::triton::metal::MetalReduceKind::max);
  bool isMean = (kind == mlir::triton::metal::MetalReduceKind::mean);
  const char *initVal = isMax ? "-INFINITY" : "0.0f";
  const char *simdOp = isMax ? "simd_max" : "simd_sum";
  const char *combineFmt = isMax ? "acc = max(acc, v)" : "acc = acc + v";

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "if (gid < " << mOuter << ") {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "float acc = " << initVal << ";\n";
      indent();
      _output << "uint base = gid * " << r << ";\n";
      indent();
      _output << "for (uint i = lid; i < " << r << "; i += 32u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "float v = float(" << inName << "[base + i]);\n";
        indent();
        _output << combineFmt << ";";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "acc = " << simdOp << "(acc);\n";
      indent();
      _output << "if (lid == 0u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        if (isMean) {
          _output << outName << "[gid] = " << outTy << "(acc * (1.0f / float("
                  << r << ")));";
        } else {
          _output << outName << "[gid] = " << outTy << "(acc);";
        }
      }
      _output << "\n";
      indent();
      _output << "}";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// MLIR side: signless i32 output; MSL emits 'int' (32-bit); IndexValPair::index
// is uint32_t inside the kernel, cast to (int) at the store site.
void ModuleTranslation::translate(mlir::triton::metal::ArgmaxOp op) {
  auto mOuter = op.getMOuter();
  auto r = op.getR();
  auto inElemTy = llvm::cast<MetalMemRefType>(op.getIn().getType()).getType();
  auto valTy = typeToString(inElemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string inName = nameOf(op.getIn());
  std::string outName = nameOf(op.getOut());

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "struct IndexValPair { uint32_t index; " << valTy << " val; };\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "if (gid < " << mOuter << ") {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "IndexValPair best;\n";
      indent();
      _output << "best.index = 0u;\n";
      indent();
      _output << "best.val = " << valTy << "(-INFINITY);\n";
      indent();
      _output << "uint base = gid * " << r << ";\n";
      indent();
      _output << "for (uint i = lid; i < " << r << "; i += 32u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << valTy << " v = " << inName << "[base + i];\n";
        indent();
        _output << "if (v > best.val || (v == best.val && i < best.index)) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "best.val = v;\n";
          indent();
          _output << "best.index = i;";
        }
        _output << "\n";
        indent();
        _output << "}";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "for (uint offset = 16u; offset > 0u; offset >>= 1) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "IndexValPair current;\n";
        indent();
        if (inElemTy.isBF16()) {
          // simd_shuffle_down does not support bfloat directly; bitcast
          // through ushort and back to preserve T-typed val.
          _output << "current.val = as_type<bfloat>(simd_shuffle_down("
                     "as_type<ushort>(best.val), offset));\n";
        } else {
          _output << "current.val = simd_shuffle_down(best.val, offset);\n";
        }
        indent();
        _output << "current.index = simd_shuffle_down(best.index, offset);\n";
        indent();
        _output << "if (best.val < current.val || (best.val == current.val && "
                   "best.index > current.index)) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "best = current;";
        }
        _output << "\n";
        indent();
        _output << "}";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "if (lid == 0u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << outName << "[gid] = (int)best.index;";
      }
      _output << "\n";
      indent();
      _output << "}";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Emit MSL for the shared softmax/logsumexp max-and-sum body.
// After this returns, the emitted MSL will have in scope:
//   - int base = gid * r;
//   - float maxval;            (per-lane row-max, post-simd_max)
//   - float normalizer;        (1.0f / simd_sum(...) for softmax;
//                               raw simd_sum for logsumexp before log())
// The caller emits the op-specific write (softmax: per-element divide-write;
// logsumexp: lane-0-only log+maxval store).
//
// The emitted body uses a per-thread strided loop (`for (int i = lid; i < r;
// i += 32)`) — no fixed-size `local_max[]` array (this is a deliberate
// Stage-4 specialization for a single SIMD group per row).
//
// `tElem` is the input dtype string (e.g. "float", "half", "bfloat").
// `inName` is the already-translated input buffer name (e.g. "v0").
// `r` is the row length (static).
//
// Shared by translate(SoftmaxOp) and (in US-4) translate(LogsumexpOp).
static void emitMaxSumBody(llvm::raw_ostream &os, int curIndent,
                           llvm::StringRef tElem, llvm::StringRef inName,
                           int64_t r) {
  auto pad = [&](int extra) {
    for (int i = 0; i < curIndent + extra; ++i)
      os << "  ";
  };
  (void)tElem; // tElem unused — body always accumulates in float (AccT=float).

  // Caller emitted an `indent()` before calling — first line starts in-place.
  os << "int base = (int)gid * " << r << ";\n";
  pad(0);
  os << "float maxval = -INFINITY;\n";
  pad(0);
  os << "for (int i = lid; i < " << r << "; i += 32) {\n";
  pad(1);
  os << "maxval = max(maxval, float(" << inName << "[base + i]));\n";
  pad(0);
  os << "}\n";
  pad(0);
  os << "maxval = simd_max(maxval);\n";
  pad(0);
  os << "float normalizer = 0.0f;\n";
  pad(0);
  os << "for (int i = lid; i < " << r << "; i += 32) {\n";
  pad(1);
  os << "normalizer += fast::exp(float(" << inName << "[base + i]) - maxval);\n";
  pad(0);
  os << "}\n";
  pad(0);
  os << "normalizer = simd_sum(normalizer);";
}

void ModuleTranslation::translate(mlir::triton::metal::FlashAttentionOp op) {
  // Emits the Phase-0-validated flash-attention body: two matmuls on simdgroup
  // hardware (Q/K^T/V/P tiles staged in threadgroup, read via simdgroup_load),
  // the online softmax + O accumulator + running max/sum in the threadgroup
  // scalar domain. Runs on ONE warp (guard `_fa_active = ltid.x < 32`); idle
  // warps under num_warps>1 still reach every threadgroup_barrier (barriers are
  // OUTSIDE the guard). See metal-flash-attention-phase0-spike.py.
  const int64_t BM = op.getBm(), BN = op.getBn(), BD = op.getBd();
  const int64_t mT = BM / 8, nT = BN / 8, dT = BD / 8;
  const int64_t SZ_Q = BM * BD, SZ_KTV = BD * BN, SZ_S = BM * BN;

  auto bufName = [&](mlir::Value m) -> std::string {
    // Walk conversion casts to the kernel buffer. A pointer arg resolves
    // directly to its memref block-arg; a scalar arg (n/d_model/h) is
    // materialized post-conversion as get_element(buffer[0]) — follow that to
    // the underlying buffer memref so it resolves to v<i> (read as v<i>[0]),
    // not v0. Without this, _buffers[...] misses and operator[] yields 0.
    for (;;) {
      while (auto cast = m.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (cast.getInputs().size() != 1)
          break;
        m = cast.getInputs()[0];
      }
      if (auto ge = m.getDefiningOp<mlir::triton::metal::GetElementOp>()) {
        m = ge.getMemref();
        continue;
      }
      break;
    }
    auto it = _buffers.find(m.getAsOpaquePointer());
    if (it == _buffers.end()) {
      // NEVER fall back to buffer 0. An operand that does not resolve to a
      // kernel buffer means the matcher bound a computed value (or a constant)
      // where a kernel argument was required; emitting `v0[0]` for it produces
      // a kernel that reads its own Q pointer as an integer and silently
      // computes garbage. Fail the translation instead. The matcher-side gate
      // is tryFlashAttentionLoop step (5a); this is the backstop.
      // See metal-sliding-window-attention-plan.md §1b.
      op.emitError() << "metal.flash_attention: operand does not resolve to a "
                        "kernel buffer (matcher bound a non-kernel-arg value); "
                        "refusing to emit";
      _emitFailed = true;
      return "<unresolved>";
    }
    return "v" + std::to_string(it->second);
  };
  const std::string Q = bufName(op.getQ()), K = bufName(op.getK()),
                    V = bufName(op.getV()), O = bufName(op.getOut());
  const std::string M = bufName(op.getM()) + "[0]";
  const std::string N = bufName(op.getN()) + "[0]";
  const std::string DM = bufName(op.getDModel()) + "[0]";
  // Optional: absent `h` means no head split (d_head == d_model, column offset
  // 0); absent `window` means full attention.
  const bool hasHeads = op.getH() != nullptr;
  const bool hasWindow = op.getWindow() != nullptr;
  const std::string H = hasHeads ? bufName(op.getH()) + "[0]" : std::string();
  const std::string W =
      hasWindow ? bufName(op.getWindow()) + "[0]" : std::string();
  auto S = [](int64_t x) { return std::to_string(x); };
  // Band predicate for a (query row, key) pair. Signed on purpose: the window
  // arrives through a `device uint32_t*` buffer, and comparing the unsigned
  // difference of two uints would turn any negative offset into a huge positive
  // one — i.e. silently widen the band to everything.
  auto inWin = [&](const char *row, const std::string &key) {
    return hasWindow ? "abs((int)" + std::string(row) + " - (int)(" + key +
                           ")) <= _fa_win"
                     : std::string("true");
  };

  auto &os = _output;
  os << "\n  // ---- metal.flash_attention (online softmax, simdgroup dots) ----\n";
  os << "  threadgroup float _fa_qbuf[" << S(SZ_Q) << "];\n";
  os << "  threadgroup float _fa_ktbuf[" << S(SZ_KTV) << "];\n";
  os << "  threadgroup float _fa_vbuf[" << S(SZ_KTV) << "];\n";
  os << "  threadgroup float _fa_sbuf[" << S(SZ_S) << "];\n";
  os << "  threadgroup float _fa_pbuf[" << S(SZ_S) << "];\n";
  os << "  threadgroup float _fa_obuf[" << S(SZ_Q) << "];\n";
  os << "  threadgroup float _fa_otbuf[" << S(SZ_Q) << "];\n";
  os << "  threadgroup float _fa_rmax[" << S(BM) << "];\n";
  os << "  threadgroup float _fa_rsum[" << S(BM) << "];\n";
  os << "  {\n";
  os << "  uint _fa_lane = ltid.x & 31u;\n";
  os << "  bool _fa_active = ltid.x < 32u;\n";
  os << "  uint _fa_M = " << M << ";\n";
  os << "  uint _fa_N = " << N << ";\n";
  os << "  uint _fa_dm = " << DM << ";\n";
  if (hasHeads) {
    os << "  uint _fa_dhead = " << DM << " / " << H << ";\n";
    os << "  uint _fa_coloff = tgid.y * _fa_dhead;\n";
  } else {
    os << "  uint _fa_dhead = _fa_dm;\n";
    os << "  uint _fa_coloff = 0u;\n";
  }
  if (hasWindow)
    os << "  int _fa_win = (int)" << W << ";\n";
  os << "  uint _fa_rowoff = tgid.x * " << S(BM) << "u;\n";
  os << "  float _fa_scale = 1.0f / sqrt((float)_fa_dhead);\n";
  // load Q + zero-init accumulator/state
  os << "  if (_fa_active) {\n";
  os << "    for (uint c = _fa_lane; c < " << S(SZ_Q) << "u; c += 32u) {\n";
  os << "      uint q = c / " << S(BD) << "u; uint d = c % " << S(BD) << "u; uint row = _fa_rowoff + q;\n";
  os << "      _fa_qbuf[c] = (row < _fa_M && d < _fa_dhead) ? " << Q << "[row * _fa_dm + _fa_coloff + d] : 0.0f;\n";
  os << "      _fa_obuf[c] = 0.0f;\n";
  os << "    }\n";
  os << "    if (_fa_lane < " << S(BM) << "u) { _fa_rmax[_fa_lane] = -INFINITY; _fa_rsum[_fa_lane] = 0.0f; }\n";
  os << "  }\n";
  os << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  // main loop over key blocks
  os << "  for (uint kb = 0; kb < _fa_N; kb += " << S(BN) << "u) {\n";
  os << "    if (_fa_active) {\n";
  os << "      for (uint c = _fa_lane; c < " << S(SZ_KTV) << "u; c += 32u) {\n";
  os << "        uint d = c / " << S(BN) << "u; uint key = c % " << S(BN) << "u; uint kk = kb + key;\n";
  os << "        _fa_ktbuf[c] = (kk < _fa_N && d < _fa_dhead) ? " << K << "[kk * _fa_dm + _fa_coloff + d] : 0.0f;\n";
  os << "      }\n";
  os << "      for (uint c = _fa_lane; c < " << S(SZ_KTV) << "u; c += 32u) {\n";
  os << "        uint key = c / " << S(BD) << "u; uint d = c % " << S(BD) << "u; uint kk = kb + key;\n";
  os << "        _fa_vbuf[c] = (kk < _fa_N && d < _fa_dhead) ? " << V << "[kk * _fa_dm + _fa_coloff + d] : 0.0f;\n";
  os << "      }\n";
  os << "    }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  // Dot A: S = Q @ K^T
  os << "    if (_fa_active) {\n";
  os << "      for (uint mi = 0; mi < " << S(mT) << "u; ++mi)\n";
  os << "      for (uint ni = 0; ni < " << S(nT) << "u; ++ni) {\n";
  os << "        simdgroup_float8x8 acc(0.0f);\n";
  os << "        for (uint ki = 0; ki < " << S(dT) << "u; ++ki) {\n";
  os << "          simdgroup_float8x8 a, b;\n";
  os << "          simdgroup_load(a, &_fa_qbuf[(mi*8u)*" << S(BD) << "u + ki*8u], " << S(BD) << ");\n";
  os << "          simdgroup_load(b, &_fa_ktbuf[(ki*8u)*" << S(BN) << "u + ni*8u], " << S(BN) << ");\n";
  os << "          simdgroup_multiply_accumulate(acc, a, b, acc);\n";
  os << "        }\n";
  os << "        simdgroup_store(acc, &_fa_sbuf[(mi*8u)*" << S(BN) << "u + ni*8u], " << S(BN) << ");\n";
  os << "      }\n";
  os << "    }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  // online softmax (one query row per lane)
  os << "    if (_fa_active) {\n";
  os << "      uint q = _fa_lane; uint row = _fa_rowoff + q;\n";
  // `q` runs over all 32 lanes but every per-row buffer below is sized by BM
  // (_fa_rmax/_fa_rsum: BM floats; _fa_pbuf: BM*BN; _fa_obuf: BM*BD). BM == 32
  // happens to be in bounds; BM < 32 writes past the end of every one of them.
  // Guard on `q < BM` FIRST — `row < _fa_M` does not imply it. The Q-load /
  // rmax-init block above is already lane-guarded; this mirrors it.
  os << "      if (q < " << S(BM) << "u && row < _fa_M) {\n";
  os << "        float m_cur = -INFINITY;\n";
  os << "        for (uint kk = 0; kk < " << S(BN) << "u; ++kk)\n";
  os << "          if (kb + kk < _fa_N && (" << inWin("row", "kb + kk")
     << ")) m_cur = max(m_cur, _fa_sbuf[q*" << S(BN) << "u + kk] * _fa_scale);\n";
  os << "        float m_old = _fa_rmax[q];\n";
  os << "        float m_new = max(m_old, m_cur);\n";
  // With a window, a whole key block can fall outside the band, leaving
  // m_cur == -inf; if m_old is also -inf (every earlier block was outside too)
  // then exp(-inf - -inf) is exp(NaN) = NaN and it poisons the row's running
  // sum and accumulator for good. Nothing to rescale in that case, so use 1.
  // Unreachable without a window (row < M && kb < N implies kk == 0 is in
  // range, so m_cur is finite) — but the guard is free and correct either way.
  os << "        float scaler = (m_old == m_new) ? 1.0f : exp(m_old - m_new);\n";
  os << "        float denom = 0.0f;\n";
  os << "        for (uint kk = 0; kk < " << S(BN) << "u; ++kk) {\n";
  os << "          float p = (kb + kk < _fa_N && (" << inWin("row", "kb + kk")
     << ")) ? exp(_fa_sbuf[q*" << S(BN) << "u + kk]*_fa_scale - m_new) : 0.0f;\n";
  os << "          _fa_pbuf[q*" << S(BN) << "u + kk] = p; denom += p;\n";
  os << "        }\n";
  os << "        _fa_rsum[q] = _fa_rsum[q]*scaler + denom;\n";
  os << "        _fa_rmax[q] = m_new;\n";
  os << "        for (uint d = 0; d < " << S(BD) << "u; ++d) _fa_obuf[q*" << S(BD) << "u + d] *= scaler;\n";
  os << "      } else if (q < " << S(BM) << "u) {\n";
  os << "        for (uint kk = 0; kk < " << S(BN) << "u; ++kk) _fa_pbuf[q*" << S(BN) << "u + kk] = 0.0f;\n";
  os << "      }\n";
  os << "    }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  // Dot B: O_tile = P @ V
  os << "    if (_fa_active) {\n";
  os << "      for (uint mi = 0; mi < " << S(mT) << "u; ++mi)\n";
  os << "      for (uint di = 0; di < " << S(dT) << "u; ++di) {\n";
  os << "        simdgroup_float8x8 acc(0.0f);\n";
  os << "        for (uint ki = 0; ki < " << S(nT) << "u; ++ki) {\n";
  os << "          simdgroup_float8x8 a, b;\n";
  os << "          simdgroup_load(a, &_fa_pbuf[(mi*8u)*" << S(BN) << "u + ki*8u], " << S(BN) << ");\n";
  os << "          simdgroup_load(b, &_fa_vbuf[(ki*8u)*" << S(BD) << "u + di*8u], " << S(BD) << ");\n";
  os << "          simdgroup_multiply_accumulate(acc, a, b, acc);\n";
  os << "        }\n";
  os << "        simdgroup_store(acc, &_fa_otbuf[(mi*8u)*" << S(BD) << "u + di*8u], " << S(BD) << ");\n";
  os << "      }\n";
  os << "    }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "    if (_fa_active) { for (uint c = _fa_lane; c < " << S(SZ_Q) << "u; c += 32u) _fa_obuf[c] += _fa_otbuf[c]; }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "  }\n";
  // epilogue: O = obuf / run_sum, masked store
  os << "  if (_fa_active) {\n";
  os << "    uint q = _fa_lane; uint row = _fa_rowoff + q;\n";
  os << "    if (q < " << S(BM) << "u && row < _fa_M) {\n"; // see BM<32 note above
  os << "      float denom = _fa_rsum[q];\n";
  os << "      float inv = (denom != 0.0f) ? (1.0f / denom) : 0.0f;\n";
  os << "      for (uint d = 0; d < _fa_dhead; ++d)\n";  // d_head <= BD; skip padded cols
  os << "        " << O << "[row * _fa_dm + _fa_coloff + d] = _fa_obuf[q*" << S(BD) << "u + d] * inv;\n";
  os << "    }\n";
  os << "  }\n";
  os << "  }";
}

void ModuleTranslation::translate(mlir::triton::metal::SoftmaxOp op) {
  auto mOuter = op.getMOuter();
  auto r = op.getR();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string inName = nameOf(op.getIn());
  std::string outName = nameOf(op.getOut());

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "if (gid < " << mOuter << ") {";
    {
      INDENT();
      _output << "\n";
      indent();
      emitMaxSumBody(_output, _curIndent, outTy, inName, r);
      _output << "\n";
      indent();
      _output << "normalizer = 1.0f / normalizer;\n";
      indent();
      _output << "for (int i = lid; i < " << r << "; i += 32) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << outName << "[base + i] = " << outTy
                << "(fast::exp(float(" << inName
                << "[base + i]) - maxval) * normalizer);";
      }
      _output << "\n";
      indent();
      _output << "}";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translate(mlir::triton::metal::LogsumexpOp op) {
  auto mOuter = op.getMOuter();
  auto r = op.getR();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string inName = nameOf(op.getIn());
  std::string outName = nameOf(op.getOut());

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "if (gid < " << mOuter << ") {";
    {
      INDENT();
      _output << "\n";
      indent();
      emitMaxSumBody(_output, _curIndent, outTy, inName, r);
      _output << "\n";
      indent();
      _output << "if (lid == 0) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << outName << "[gid] = isinf(maxval) ? " << outTy
                << "(maxval) : " << outTy
                << "(log(normalizer) + maxval);";
      }
      _output << "\n";
      indent();
      _output << "}";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translate(mlir::triton::metal::GemvOp op) {
  auto m = op.getM();
  auto k = op.getK();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto accTy = typeToString(elemTy);
  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint row = id.x;\n";
    indent();
    _output << "if (row < " << m << ") {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << accTy << " acc = (" << accTy << ")(0);\n";
      indent();
      _output << "for (uint kk = 0; kk < " << k << "; ++kk) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "acc = acc + (";
        translateVarName(op.getLhs());
        _output << "[(row * " << k << ") + kk] * ";
        translateVarName(op.getRhs());
        _output << "[kk]);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      translateVarName(op.getOut());
      _output << "[row] = acc;";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Stage-6 dispatch: route `metal.sdpa` to a per-mode emitter helper based on
// `op.getMode()`. The Causal arm preserves Stage-5 byte-identity.
void ModuleTranslation::translate(mlir::triton::metal::SdpaOp op) {
  switch (op.getMode()) {
  case ::mlir::triton::metal::MaskSinkMode::Causal:
    emitCausal_(op);
    break;
  case ::mlir::triton::metal::MaskSinkMode::BoolMask:
    emitBoolMask_(op);
    break;
  case ::mlir::triton::metal::MaskSinkMode::FloatMask:
    emitFloatMask_(op);
    break;
  case ::mlir::triton::metal::MaskSinkMode::Sinks:
    emitSinks_(op);
    break;
  case ::mlir::triton::metal::MaskSinkMode::NonCausal:
    emitNonCausal_(op);
    break;
  default:
    op.emitError() << "metal.sdpa unrecognized mode integer "
                   << static_cast<int>(op.getMode())
                   << "; cannot emit MSL";
    return;
  }
}

// Stage-8 SDPA MMA-tile emitter helpers.
// - AccT = float invariant: all accumulators (Q·Kᵀ score, O accumulator) are
//   `simdgroup_matrix<float, 8, 8>` regardless of T ∈ {f16, f32, bf16}.
// - 32-wide SIMD group; one SIMD group covers the single 8×8 MMA M-tile per
//   dispatch. N=1 is internally zero-extended to 8 query rows (only row 0 is
//   stored at the end); N=8 stores all 8 rows.
// - K-outer-softmax-inner with running-max + per-fragment alpha rescale.
// - `o_acc` is a FRAGMENT ARRAY of size D/8 (D-2 fix); per-fragment rescale
//   via `rescale_scratch` happens BEFORE each k_tile's softmax·V MMA (D-1 fix).
void ModuleTranslation::emitCausal_(mlir::triton::metal::SdpaOp op) {
  auto n = op.getN();
  auto d = op.getD();
  auto kLen = op.getKLen();
  float scale = op.getScale().convertToFloat();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string qName = nameOf(op.getQ());
  std::string kName = nameOf(op.getK());
  std::string vName = nameOf(op.getV());
  std::string outName = nameOf(op.getOut());

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "uint simd_id = id.x / 32u;\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "if (gid == 0u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "alignas(16) threadgroup float q_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float k_t_tile[" << d << "][8];\n";
      indent();
      _output << "alignas(16) threadgroup float v_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float score_tile[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float alphas_tile[8];\n";
      indent();
      _output << "alignas(16) threadgroup float rescale_scratch[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float maxval[8];\n";
      indent();
      _output << "alignas(16) threadgroup float sumexp[8];\n";
      indent();
      _output << "if (lid < 8u) { maxval[lid] = -INFINITY; sumexp[lid] = 0.0f; }\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> q_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> k_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> v_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> s_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> o_acc[" << (d / 8) << "];\n";
      indent();
      _output << "for (uint t = 0; t < " << (d / 8)
              << "u; ++t) { o_acc[t] = simdgroup_matrix<float, 8, 8>(0.0f); }\n";
      indent();
      _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint i = c / " << d << "u;\n";
        indent();
        _output << "uint j = c % " << d << "u;\n";
        indent();
        if (n == 1) {
          _output << "q_tile[i][j] = (i == 0u) ? float(" << qName
                  << "[j]) : 0.0f;";
        } else {
          _output << "q_tile[i][j] = float(" << qName << "[i * " << d
                  << "u + j]);";
        }
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "for (uint k_tile = 0; k_tile < " << (kLen / 8)
              << "u; ++k_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "for (uint c = lid; c < " << (d * 8) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / 8u;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "k_t_tile[i][j] = float(" << kName
                  << "[(k_tile * 8u + j) * " << d << "u + i]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << "v_tile[i][j] = float(" << vName
                  << "[(k_tile * 8u + i) * " << d << "u + j]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_matrix<float, 8, 8> s_acc(0.0f);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(q_frag, &q_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_load(k_frag, &k_t_tile[d_tile * 8u][0], 8);\n";
          indent();
          _output << "simdgroup_multiply_accumulate(s_acc, q_frag, k_frag, s_acc);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "simdgroup_store(s_acc, &score_tile[0][0], 8);\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint c = lid; c < 64u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c >> 3;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "uint kk = k_tile * 8u + j;\n";
          indent();
          _output << "float score = score_tile[i][j] * " << scale << "f;\n";
          indent();
          _output << "if (kk <= (" << kLen << "u - " << n
                  << "u + i)) { } else { score = -INFINITY; }\n";
          indent();
          _output << "score_tile[i][j] = score;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "if (lid < 8u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = lid;\n";
          indent();
          _output << "float row_max = -INFINITY;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) { row_max = max(row_max, score_tile[i][j]); }\n";
          indent();
          _output << "float m_new = max(maxval[i], row_max);\n";
          indent();
          _output << "float alpha = fast::exp(maxval[i] - m_new);\n";
          indent();
          _output << "alphas_tile[i] = alpha;\n";
          indent();
          _output << "sumexp[i] = sumexp[i] * alpha;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "float w = fast::exp(score_tile[i][j] - m_new);\n";
            indent();
            _output << "sumexp[i] += w;\n";
            indent();
            _output << "score_tile[i][j] = w;";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "maxval[i] = m_new;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_store(o_acc[d_tile], &rescale_scratch[0][0], 8);\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "for (uint c = lid; c < 64u; c += 32u) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "uint i = c >> 3;\n";
            indent();
            _output << "uint j = c & 7u;\n";
            indent();
            _output << "rescale_scratch[i][j] = rescale_scratch[i][j] * alphas_tile[i];";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "simdgroup_load(o_acc[d_tile], &rescale_scratch[0][0], 8);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_load(s_frag, &score_tile[0][0], 8);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(v_frag, &v_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_multiply_accumulate(o_acc[d_tile], s_frag, v_frag, o_acc[d_tile]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "alignas(16) threadgroup float o_tile[8][" << d << "];\n";
      indent();
      _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
              << "u; ++d_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "simdgroup_store(o_acc[d_tile], &o_tile[0][d_tile * 8u], "
                << d << ");";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      if (n == 1) {
        _output << "for (uint j = lid; j < " << d << "u; j += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << outName << "[j] = " << outTy
                  << "(o_tile[0][j] / sumexp[0]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      } else {
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << outName << "[i * " << d << "u + j] = " << outTy
                  << "(o_tile[i][j] / sumexp[i]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      }
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Stage-8 BoolMask: per-key predicate from extras[0] bool buffer [n, k_len].
void ModuleTranslation::emitBoolMask_(mlir::triton::metal::SdpaOp op) {
  auto n = op.getN();
  auto d = op.getD();
  auto kLen = op.getKLen();
  float scale = op.getScale().convertToFloat();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string qName = nameOf(op.getQ());
  std::string kName = nameOf(op.getK());
  std::string vName = nameOf(op.getV());
  std::string outName = nameOf(op.getOut());
  std::string maskName = nameOf(op.getExtras()[0]);

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "uint simd_id = id.x / 32u;\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "if (gid == 0u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "alignas(16) threadgroup float q_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float k_t_tile[" << d << "][8];\n";
      indent();
      _output << "alignas(16) threadgroup float v_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float score_tile[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float alphas_tile[8];\n";
      indent();
      _output << "alignas(16) threadgroup float rescale_scratch[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float maxval[8];\n";
      indent();
      _output << "alignas(16) threadgroup float sumexp[8];\n";
      indent();
      _output << "if (lid < 8u) { maxval[lid] = -INFINITY; sumexp[lid] = 0.0f; }\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> q_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> k_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> v_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> s_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> o_acc[" << (d / 8) << "];\n";
      indent();
      _output << "for (uint t = 0; t < " << (d / 8)
              << "u; ++t) { o_acc[t] = simdgroup_matrix<float, 8, 8>(0.0f); }\n";
      indent();
      _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint i = c / " << d << "u;\n";
        indent();
        _output << "uint j = c % " << d << "u;\n";
        indent();
        if (n == 1) {
          _output << "q_tile[i][j] = (i == 0u) ? float(" << qName
                  << "[j]) : 0.0f;";
        } else {
          _output << "q_tile[i][j] = float(" << qName << "[i * " << d
                  << "u + j]);";
        }
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "for (uint k_tile = 0; k_tile < " << (kLen / 8)
              << "u; ++k_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "for (uint c = lid; c < " << (d * 8) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / 8u;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "k_t_tile[i][j] = float(" << kName
                  << "[(k_tile * 8u + j) * " << d << "u + i]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << "v_tile[i][j] = float(" << vName
                  << "[(k_tile * 8u + i) * " << d << "u + j]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_matrix<float, 8, 8> s_acc(0.0f);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(q_frag, &q_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_load(k_frag, &k_t_tile[d_tile * 8u][0], 8);\n";
          indent();
          _output << "simdgroup_multiply_accumulate(s_acc, q_frag, k_frag, s_acc);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "simdgroup_store(s_acc, &score_tile[0][0], 8);\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint c = lid; c < 64u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c >> 3;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "uint kk = k_tile * 8u + j;\n";
          indent();
          _output << "float score = score_tile[i][j] * " << scale << "f;\n";
          indent();
          _output << "bool mask_val = bool(" << maskName << "[i * " << kLen
                  << "u + kk]);\n";
          indent();
          _output << "if (!mask_val) { score = -INFINITY; }\n";
          indent();
          _output << "score_tile[i][j] = score;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "if (lid < 8u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = lid;\n";
          indent();
          _output << "float row_max = -INFINITY;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) { row_max = max(row_max, score_tile[i][j]); }\n";
          indent();
          _output << "float m_new = max(maxval[i], row_max);\n";
          indent();
          _output << "float alpha = fast::exp(maxval[i] - m_new);\n";
          indent();
          _output << "alphas_tile[i] = alpha;\n";
          indent();
          _output << "sumexp[i] = sumexp[i] * alpha;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "float w = fast::exp(score_tile[i][j] - m_new);\n";
            indent();
            _output << "sumexp[i] += w;\n";
            indent();
            _output << "score_tile[i][j] = w;";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "maxval[i] = m_new;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_store(o_acc[d_tile], &rescale_scratch[0][0], 8);\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "for (uint c = lid; c < 64u; c += 32u) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "uint i = c >> 3;\n";
            indent();
            _output << "uint j = c & 7u;\n";
            indent();
            _output << "rescale_scratch[i][j] = rescale_scratch[i][j] * alphas_tile[i];";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "simdgroup_load(o_acc[d_tile], &rescale_scratch[0][0], 8);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_load(s_frag, &score_tile[0][0], 8);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(v_frag, &v_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_multiply_accumulate(o_acc[d_tile], s_frag, v_frag, o_acc[d_tile]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "alignas(16) threadgroup float o_tile[8][" << d << "];\n";
      indent();
      _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
              << "u; ++d_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "simdgroup_store(o_acc[d_tile], &o_tile[0][d_tile * 8u], "
                << d << ");";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      if (n == 1) {
        _output << "for (uint j = lid; j < " << d << "u; j += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << outName << "[j] = " << outTy
                  << "(o_tile[0][j] / sumexp[0]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      } else {
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << outName << "[i * " << d << "u + j] = " << outTy
                  << "(o_tile[i][j] / sumexp[i]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      }
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Stage-6 FloatMask: additive bias loaded from extras[0] of shape [n, k_len]
// applied AFTER simd_sum(score) and BEFORE the running-max update.
// All keys participate (use_key = true).
void ModuleTranslation::emitFloatMask_(mlir::triton::metal::SdpaOp op) {
  auto n = op.getN();
  auto d = op.getD();
  auto kLen = op.getKLen();
  float scale = op.getScale().convertToFloat();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string qName = nameOf(op.getQ());
  std::string kName = nameOf(op.getK());
  std::string vName = nameOf(op.getV());
  std::string outName = nameOf(op.getOut());
  std::string maskName = nameOf(op.getExtras()[0]);

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "uint simd_id = id.x / 32u;\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "if (gid == 0u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "alignas(16) threadgroup float q_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float k_t_tile[" << d << "][8];\n";
      indent();
      _output << "alignas(16) threadgroup float v_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float score_tile[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float alphas_tile[8];\n";
      indent();
      _output << "alignas(16) threadgroup float rescale_scratch[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float maxval[8];\n";
      indent();
      _output << "alignas(16) threadgroup float sumexp[8];\n";
      indent();
      _output << "if (lid < 8u) { maxval[lid] = -INFINITY; sumexp[lid] = 0.0f; }\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> q_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> k_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> v_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> s_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> o_acc[" << (d / 8) << "];\n";
      indent();
      _output << "for (uint t = 0; t < " << (d / 8)
              << "u; ++t) { o_acc[t] = simdgroup_matrix<float, 8, 8>(0.0f); }\n";
      indent();
      _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint i = c / " << d << "u;\n";
        indent();
        _output << "uint j = c % " << d << "u;\n";
        indent();
        if (n == 1) {
          _output << "q_tile[i][j] = (i == 0u) ? float(" << qName
                  << "[j]) : 0.0f;";
        } else {
          _output << "q_tile[i][j] = float(" << qName << "[i * " << d
                  << "u + j]);";
        }
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "for (uint k_tile = 0; k_tile < " << (kLen / 8)
              << "u; ++k_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "for (uint c = lid; c < " << (d * 8) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / 8u;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "k_t_tile[i][j] = float(" << kName
                  << "[(k_tile * 8u + j) * " << d << "u + i]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << "v_tile[i][j] = float(" << vName
                  << "[(k_tile * 8u + i) * " << d << "u + j]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_matrix<float, 8, 8> s_acc(0.0f);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(q_frag, &q_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_load(k_frag, &k_t_tile[d_tile * 8u][0], 8);\n";
          indent();
          _output << "simdgroup_multiply_accumulate(s_acc, q_frag, k_frag, s_acc);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "simdgroup_store(s_acc, &score_tile[0][0], 8);\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint c = lid; c < 64u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c >> 3;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "uint kk = k_tile * 8u + j;\n";
          indent();
          _output << "float score = score_tile[i][j] * " << scale << "f;\n";
          indent();
          _output << "score += float(" << maskName << "[i * " << kLen
                  << "u + kk]);\n";
          indent();
          _output << "score_tile[i][j] = score;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "if (lid < 8u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = lid;\n";
          indent();
          _output << "float row_max = -INFINITY;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) { row_max = max(row_max, score_tile[i][j]); }\n";
          indent();
          _output << "float m_new = max(maxval[i], row_max);\n";
          indent();
          _output << "float alpha = fast::exp(maxval[i] - m_new);\n";
          indent();
          _output << "alphas_tile[i] = alpha;\n";
          indent();
          _output << "sumexp[i] = sumexp[i] * alpha;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "float w = fast::exp(score_tile[i][j] - m_new);\n";
            indent();
            _output << "sumexp[i] += w;\n";
            indent();
            _output << "score_tile[i][j] = w;";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "maxval[i] = m_new;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_store(o_acc[d_tile], &rescale_scratch[0][0], 8);\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "for (uint c = lid; c < 64u; c += 32u) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "uint i = c >> 3;\n";
            indent();
            _output << "uint j = c & 7u;\n";
            indent();
            _output << "rescale_scratch[i][j] = rescale_scratch[i][j] * alphas_tile[i];";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "simdgroup_load(o_acc[d_tile], &rescale_scratch[0][0], 8);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_load(s_frag, &score_tile[0][0], 8);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(v_frag, &v_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_multiply_accumulate(o_acc[d_tile], s_frag, v_frag, o_acc[d_tile]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "alignas(16) threadgroup float o_tile[8][" << d << "];\n";
      indent();
      _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
              << "u; ++d_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "simdgroup_store(o_acc[d_tile], &o_tile[0][d_tile * 8u], "
                << d << ");";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      if (n == 1) {
        _output << "for (uint j = lid; j < " << d << "u; j += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << outName << "[j] = " << outTy
                  << "(o_tile[0][j] / sumexp[0]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      } else {
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << outName << "[i * " << d << "u + j] = " << outTy
                  << "(o_tile[i][j] / sumexp[i]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      }
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Stage-6 Sinks: causal predicate retained; AFTER the KV loop, the per-row
// sink value (extras[0][gid]) contributes to sumexp (normalizer-only,
// matching MLX sdpa_vector.h:144-148).
void ModuleTranslation::emitSinks_(mlir::triton::metal::SdpaOp op) {
  auto n = op.getN();
  auto d = op.getD();
  auto kLen = op.getKLen();
  float scale = op.getScale().convertToFloat();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string qName = nameOf(op.getQ());
  std::string kName = nameOf(op.getK());
  std::string vName = nameOf(op.getV());
  std::string outName = nameOf(op.getOut());
  std::string sinkName = nameOf(op.getExtras()[0]);

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "uint simd_id = id.x / 32u;\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "if (gid == 0u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "alignas(16) threadgroup float q_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float k_t_tile[" << d << "][8];\n";
      indent();
      _output << "alignas(16) threadgroup float v_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float score_tile[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float alphas_tile[8];\n";
      indent();
      _output << "alignas(16) threadgroup float rescale_scratch[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float maxval[8];\n";
      indent();
      _output << "alignas(16) threadgroup float sumexp[8];\n";
      indent();
      _output << "if (lid < 8u) { maxval[lid] = -INFINITY; sumexp[lid] = 0.0f; }\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> q_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> k_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> v_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> s_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> o_acc[" << (d / 8) << "];\n";
      indent();
      _output << "for (uint t = 0; t < " << (d / 8)
              << "u; ++t) { o_acc[t] = simdgroup_matrix<float, 8, 8>(0.0f); }\n";
      indent();
      _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint i = c / " << d << "u;\n";
        indent();
        _output << "uint j = c % " << d << "u;\n";
        indent();
        if (n == 1) {
          _output << "q_tile[i][j] = (i == 0u) ? float(" << qName
                  << "[j]) : 0.0f;";
        } else {
          _output << "q_tile[i][j] = float(" << qName << "[i * " << d
                  << "u + j]);";
        }
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "for (uint k_tile = 0; k_tile < " << (kLen / 8)
              << "u; ++k_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "for (uint c = lid; c < " << (d * 8) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / 8u;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "k_t_tile[i][j] = float(" << kName
                  << "[(k_tile * 8u + j) * " << d << "u + i]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << "v_tile[i][j] = float(" << vName
                  << "[(k_tile * 8u + i) * " << d << "u + j]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_matrix<float, 8, 8> s_acc(0.0f);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(q_frag, &q_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_load(k_frag, &k_t_tile[d_tile * 8u][0], 8);\n";
          indent();
          _output << "simdgroup_multiply_accumulate(s_acc, q_frag, k_frag, s_acc);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "simdgroup_store(s_acc, &score_tile[0][0], 8);\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint c = lid; c < 64u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c >> 3;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "uint kk = k_tile * 8u + j;\n";
          indent();
          _output << "float score = score_tile[i][j] * " << scale << "f;\n";
          indent();
          _output << "if (kk <= (" << kLen << "u - " << n
                  << "u + i)) { } else { score = -INFINITY; }\n";
          indent();
          _output << "score_tile[i][j] = score;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "if (lid < 8u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = lid;\n";
          indent();
          _output << "float row_max = -INFINITY;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) { row_max = max(row_max, score_tile[i][j]); }\n";
          indent();
          _output << "float m_new = max(maxval[i], row_max);\n";
          indent();
          _output << "float alpha = fast::exp(maxval[i] - m_new);\n";
          indent();
          _output << "alphas_tile[i] = alpha;\n";
          indent();
          _output << "sumexp[i] = sumexp[i] * alpha;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "float w = fast::exp(score_tile[i][j] - m_new);\n";
            indent();
            _output << "sumexp[i] += w;\n";
            indent();
            _output << "score_tile[i][j] = w;";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "maxval[i] = m_new;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_store(o_acc[d_tile], &rescale_scratch[0][0], 8);\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "for (uint c = lid; c < 64u; c += 32u) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "uint i = c >> 3;\n";
            indent();
            _output << "uint j = c & 7u;\n";
            indent();
            _output << "rescale_scratch[i][j] = rescale_scratch[i][j] * alphas_tile[i];";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "simdgroup_load(o_acc[d_tile], &rescale_scratch[0][0], 8);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_load(s_frag, &score_tile[0][0], 8);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(v_frag, &v_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_multiply_accumulate(o_acc[d_tile], s_frag, v_frag, o_acc[d_tile]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "if (lid < 8u) { sumexp[lid] += fast::exp(float(" << sinkName
              << "[lid]) - maxval[lid]); }\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "alignas(16) threadgroup float o_tile[8][" << d << "];\n";
      indent();
      _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
              << "u; ++d_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "simdgroup_store(o_acc[d_tile], &o_tile[0][d_tile * 8u], "
                << d << ");";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      if (n == 1) {
        _output << "for (uint j = lid; j < " << d << "u; j += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << outName << "[j] = " << outTy
                  << "(o_tile[0][j] / sumexp[0]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      } else {
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << outName << "[i * " << d << "u + j] = " << outTy
                  << "(o_tile[i][j] / sumexp[i]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      }
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Stage-6 NonCausal: causal predicate replaced with `use_key = true`; no
// extras consumed.
void ModuleTranslation::emitNonCausal_(mlir::triton::metal::SdpaOp op) {
  auto n = op.getN();
  auto d = op.getD();
  auto kLen = op.getKLen();
  float scale = op.getScale().convertToFloat();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string qName = nameOf(op.getQ());
  std::string kName = nameOf(op.getK());
  std::string vName = nameOf(op.getV());
  std::string outName = nameOf(op.getOut());

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "uint simd_id = id.x / 32u;\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "if (gid == 0u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "alignas(16) threadgroup float q_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float k_t_tile[" << d << "][8];\n";
      indent();
      _output << "alignas(16) threadgroup float v_tile[8][" << d << "];\n";
      indent();
      _output << "alignas(16) threadgroup float score_tile[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float alphas_tile[8];\n";
      indent();
      _output << "alignas(16) threadgroup float rescale_scratch[8][8];\n";
      indent();
      _output << "alignas(16) threadgroup float maxval[8];\n";
      indent();
      _output << "alignas(16) threadgroup float sumexp[8];\n";
      indent();
      _output << "if (lid < 8u) { maxval[lid] = -INFINITY; sumexp[lid] = 0.0f; }\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> q_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> k_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> v_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> s_frag;\n";
      indent();
      _output << "simdgroup_matrix<float, 8, 8> o_acc[" << (d / 8) << "];\n";
      indent();
      _output << "for (uint t = 0; t < " << (d / 8)
              << "u; ++t) { o_acc[t] = simdgroup_matrix<float, 8, 8>(0.0f); }\n";
      indent();
      _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint i = c / " << d << "u;\n";
        indent();
        _output << "uint j = c % " << d << "u;\n";
        indent();
        if (n == 1) {
          _output << "q_tile[i][j] = (i == 0u) ? float(" << qName
                  << "[j]) : 0.0f;";
        } else {
          _output << "q_tile[i][j] = float(" << qName << "[i * " << d
                  << "u + j]);";
        }
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "for (uint k_tile = 0; k_tile < " << (kLen / 8)
              << "u; ++k_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "for (uint c = lid; c < " << (d * 8) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / 8u;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "k_t_tile[i][j] = float(" << kName
                  << "[(k_tile * 8u + j) * " << d << "u + i]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << "v_tile[i][j] = float(" << vName
                  << "[(k_tile * 8u + i) * " << d << "u + j]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_matrix<float, 8, 8> s_acc(0.0f);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(q_frag, &q_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_load(k_frag, &k_t_tile[d_tile * 8u][0], 8);\n";
          indent();
          _output << "simdgroup_multiply_accumulate(s_acc, q_frag, k_frag, s_acc);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "simdgroup_store(s_acc, &score_tile[0][0], 8);\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint c = lid; c < 64u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c >> 3;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "uint kk = k_tile * 8u + j;\n";
          indent();
          _output << "float score = score_tile[i][j] * " << scale << "f;\n";
          indent();
          _output << "score_tile[i][j] = score;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "if (lid < 8u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = lid;\n";
          indent();
          _output << "float row_max = -INFINITY;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) { row_max = max(row_max, score_tile[i][j]); }\n";
          indent();
          _output << "float m_new = max(maxval[i], row_max);\n";
          indent();
          _output << "float alpha = fast::exp(maxval[i] - m_new);\n";
          indent();
          _output << "alphas_tile[i] = alpha;\n";
          indent();
          _output << "sumexp[i] = sumexp[i] * alpha;\n";
          indent();
          _output << "for (uint j = 0u; j < 8u; ++j) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "float w = fast::exp(score_tile[i][j] - m_new);\n";
            indent();
            _output << "sumexp[i] += w;\n";
            indent();
            _output << "score_tile[i][j] = w;";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "maxval[i] = m_new;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_store(o_acc[d_tile], &rescale_scratch[0][0], 8);\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "for (uint c = lid; c < 64u; c += 32u) {";
          {
            INDENT();
            _output << "\n";
            indent();
            _output << "uint i = c >> 3;\n";
            indent();
            _output << "uint j = c & 7u;\n";
            indent();
            _output << "rescale_scratch[i][j] = rescale_scratch[i][j] * alphas_tile[i];";
          }
          _output << "\n";
          indent();
          _output << "}\n";
          indent();
          _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
          indent();
          _output << "simdgroup_load(o_acc[d_tile], &rescale_scratch[0][0], 8);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_load(s_frag, &score_tile[0][0], 8);\n";
        indent();
        _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
                << "u; ++d_tile) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "simdgroup_load(v_frag, &v_tile[0][d_tile * 8u], " << d
                  << ");\n";
          indent();
          _output << "simdgroup_multiply_accumulate(o_acc[d_tile], s_frag, v_frag, o_acc[d_tile]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "alignas(16) threadgroup float o_tile[8][" << d << "];\n";
      indent();
      _output << "for (uint d_tile = 0; d_tile < " << (d / 8)
              << "u; ++d_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "simdgroup_store(o_acc[d_tile], &o_tile[0][d_tile * 8u], "
                << d << ");";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      if (n == 1) {
        _output << "for (uint j = lid; j < " << d << "u; j += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << outName << "[j] = " << outTy
                  << "(o_tile[0][j] / sumexp[0]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      } else {
        _output << "for (uint c = lid; c < " << (8 * d) << "u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c / " << d << "u;\n";
          indent();
          _output << "uint j = c % " << d << "u;\n";
          indent();
          _output << outName << "[i * " << d << "u + j] = " << outTy
                  << "(o_tile[i][j] / sumexp[i]);";
        }
        _output << "\n";
        indent();
        _output << "}";
      }
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Stage-9 Llama-style RMS normalization.
//
// y[i,c] = x[i,c] * gamma[c] * rsqrt(mean_c(x[i,c]^2) + eps).
//
// Threadgroup layout: flat (32*n, 1, 1); `gid = id.x / 32u` is the row,
// `lid = id.x & 31u` is the lane within the row's warp. simd_sum natively
// broadcasts the scalar reduction to all 32 lanes, so every lane redundantly
// computes `rms_inv` (no explicit simd_broadcast, no lane-0 gate). The write
// loop mirrors translate(SoftmaxOp) — each lane stores its own D/32 outputs.
// (Contrast translate(ReduceOp), which DOES gate stores under lid==0 because
// reductions produce one scalar per row; RMSNorm writes D elements per row.)
//
// AccT=float invariant: x is promoted to float BEFORE the square (bf16 safety),
// the sum-of-squares, eps add, and rsqrt are all in f32; only the final store
// is cast back to T.
void ModuleTranslation::translate(mlir::triton::metal::RmsNormOp op) {
  auto n = op.getN();
  auto d = op.getD();
  llvm::APFloat epsAP = op.getEps();
  float eps = epsAP.convertToFloat();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getO().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string xName = nameOf(op.getX());
  std::string gName = nameOf(op.getGamma());
  std::string oName = nameOf(op.getO());

  int64_t dDiv32 = d / 32;

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint gid = id.x / 32u;\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "if (gid < " << n << "u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "uint base = gid * " << d << "u;\n";
      indent();
      _output << "float acc = 0.0f;\n";
      indent();
      _output << "for (uint k = 0u; k < " << dDiv32 << "u; ++k) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint c = lid + k * 32u;\n";
        indent();
        _output << "float xv = float(" << xName << "[base + c]);\n";
        indent();
        _output << "acc += xv * xv;";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "acc = simd_sum(acc);\n";
      indent();
      _output << "float rms_inv = rsqrt(acc / float(" << d << ") + "
              << llvm::format("%.6ef", eps) << ");\n";
      indent();
      _output << "for (uint k = 0u; k < " << dDiv32 << "u; ++k) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint c = lid + k * 32u;\n";
        indent();
        _output << "float xv = float(" << xName << "[base + c]);\n";
        indent();
        _output << "float gv = float(" << gName << "[c]);\n";
        indent();
        _output << oName << "[base + c] = " << outTy
                << "(xv * rms_inv * gv);";
      }
      _output << "\n";
      indent();
      _output << "}";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translate(mlir::triton::metal::MatmulOp op) {
  switch (op.getKind()) {
  case ::mlir::triton::metal::MatmulKind::Scalar:
    emitScalarMatmul_(op);
    break;
  case ::mlir::triton::metal::MatmulKind::Mma:
    emitMmaMatmul_(op);
    break;
  default:
    op.emitError() << "metal.matmul unrecognized kind "
                   << static_cast<int>(op.getKind());
    return;
  }
}

void ModuleTranslation::emitScalarMatmul_(mlir::triton::metal::MatmulOp op) {
  auto m = op.getM();
  auto n = op.getN();
  auto k = op.getK();
  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint row = id.y;\n";
    indent();
    _output << "uint col = id.x;\n";
    indent();
    _output << "if ((row < " << m << ") && (col < " << n << ")) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "float acc = 0.0f;\n";
      indent();
      _output << "for (uint kk = 0; kk < " << k << "; ++kk) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "acc = acc + (";
        translateVarName(op.getLhs());
        _output << "[(row * " << k << ") + kk] * ";
        translateVarName(op.getRhs());
        _output << "[(kk * " << n << ") + col]);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      translateVarName(op.getOut());
      _output << "[(row * " << n << ") + col] = acc;";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Emit MSL for `metal.matmul` with `kind = ::Mma` using Apple `simdgroup_matrix`
// intrinsics (Option A per Stage-7 US-1 probe). Per-lane partition mirrors
// Stage-5/6 SDPA `uint lid = id.x & 31u;` idiom (ModuleTranslation.cpp:923-925).
// AccT = float invariant: accumulator is always `simdgroup_matrix<float, 8, 8>`
// regardless of operand element type T ∈ {f16, f32, bf16}.
void ModuleTranslation::emitMmaMatmul_(mlir::triton::metal::MatmulOp op) {
  auto m = op.getM();
  auto n = op.getN();
  auto k = op.getK();
  auto elemTy = llvm::cast<MetalMemRefType>(op.getOut().getType()).getType();
  auto outTy = typeToString(elemTy);

  auto nameOf = [&](mlir::Value memref) -> std::string {
    std::string buf;
    llvm::raw_string_ostream s(buf);
    auto opInst = memref.getDefiningOp();
    if (opInst == nullptr) {
      s << "v" << _buffers[memref.getAsOpaquePointer()];
    } else if (llvm::isa<mlir::triton::metal::AllocaOp>(opInst)) {
      s << "v" << _alloca[opInst];
    }
    return buf;
  };
  std::string lhsName = nameOf(op.getLhs());
  std::string rhsName = nameOf(op.getRhs());
  std::string outName = nameOf(op.getOut());

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "uint lid = id.x & 31u;\n";
    indent();
    _output << "alignas(16) threadgroup float lhs_tile[8][8];\n";
    indent();
    _output << "alignas(16) threadgroup float rhs_tile[8][8];\n";
    indent();
    _output << "simdgroup_matrix<float, 8, 8> acc(0.0f);\n";
    indent();
    _output << "simdgroup_matrix<float, 8, 8> lhs_frag;\n";
    indent();
    _output << "simdgroup_matrix<float, 8, 8> rhs_frag;\n";
    indent();
    _output << "uint simd_id = id.x / 32u;\n";
    indent();
    _output << "uint m_tile = simd_id / " << (n / 8) << "u;\n";
    indent();
    _output << "uint n_tile = simd_id % " << (n / 8) << "u;\n";
    indent();
    _output << "if ((m_tile * 8 < " << m << ") && (n_tile * 8 < " << n << ")) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "for (uint k_tile = 0; k_tile < " << (k / 8) << "; ++k_tile) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "for (uint c = lid; c < 64u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c >> 3;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "lhs_tile[i][j] = float(" << lhsName
                  << "[((m_tile * 8 + i) * " << k << ") + (k_tile * 8 + j)]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "for (uint c = lid; c < 64u; c += 32u) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uint i = c >> 3;\n";
          indent();
          _output << "uint j = c & 7u;\n";
          indent();
          _output << "rhs_tile[i][j] = float(" << rhsName
                  << "[((k_tile * 8 + i) * " << n << ") + (n_tile * 8 + j)]);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        indent();
        _output << "simdgroup_load(lhs_frag, &lhs_tile[0][0], 8);\n";
        indent();
        _output << "simdgroup_load(rhs_frag, &rhs_tile[0][0], 8);\n";
        indent();
        _output << "simdgroup_multiply_accumulate(acc, lhs_frag, rhs_frag, acc);\n";
        indent();
        _output << "threadgroup_barrier(mem_flags::mem_threadgroup);";
      }
      _output << "\n";
      indent();
      _output << "}\n";
      indent();
      _output << "alignas(16) threadgroup float out_tile[8][8];\n";
      indent();
      _output << "simdgroup_store(acc, &out_tile[0][0], 8);\n";
      indent();
      _output << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      indent();
      _output << "for (uint c = lid; c < 64u; c += 32u) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint i = c >> 3;\n";
        indent();
        _output << "uint j = c & 7u;\n";
        indent();
        _output << outName << "[((m_tile * 8 + i) * " << n
                << ") + (n_tile * 8 + j)] = " << outTy << "(out_tile[i][j]);";
      }
      _output << "\n";
      indent();
      _output << "}";
    }
    _output << "\n";
    indent();
    _output << "}";
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translate(mlir::Region &region) {
  _output << "{";
  {
    INDENT();
    // AC4 v6: emit a single shared threadgroup stage buffer at the top
    // of the kernel body when any SimdgroupLoadDeviceStagedOp exists in
    // the kernel. All staged loads reuse this buffer (each is bracketed
    // by `threadgroup_barrier` for safe reuse). Without sharing, a
    // 64×64 multi-tile kernel issues ≥128 staged loads, each with its
    // own [num_warps][64] buffer (~1 KiB), exceeding Apple's threadgroup
    // memory budget and crashing the Metal compiler with
    // XPC_ERROR_CONNECTION_INTERRUPTED. Bit-identity for legacy 8×8
    // single-tile kernels is preserved up to the buffer name (the
    // shared buffer holds the same 64-element working set).
    auto kernelOp =
        mlir::dyn_cast<mlir::triton::metal::KernelOp>(region.getParentOp());
    if (kernelOp && !_sharedStageBufferDeclared) {
      mlir::triton::metal::SimdgroupLoadDeviceStagedOp firstStaged;
      kernelOp.walk([&](mlir::triton::metal::SimdgroupLoadDeviceStagedOp op) {
        firstStaged = op;
        return mlir::WalkResult::interrupt();
      });
      // A kernel may carry only MASKED staged loads (no unmasked); both share
      // the single `_stage_shared` buffer, and either may be per-warp.
      mlir::triton::metal::SimdgroupLoadDeviceStagedMaskedOp firstMasked;
      if (!firstStaged) {
        kernelOp.walk(
            [&](mlir::triton::metal::SimdgroupLoadDeviceStagedMaskedOp op) {
              firstMasked = op;
              return mlir::WalkResult::interrupt();
            });
      }
      if (firstStaged || firstMasked) {
        auto resTy = firstStaged
                         ? llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
                               firstStaged.getResult().getType())
                         : llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
                               firstMasked.getResult().getType());
        unsigned elems = resTy.getRows() * resTy.getCols();
        bool perWarp = firstStaged ? !firstStaged.getWarpIndex().empty()
                                   : !firstMasked.getWarpIndex().empty();
        _output << "\n";
        indent();
        if (perWarp) {
          int numWarps = firstStaged
                             ? mlir::triton::gpu::lookupNumWarps(firstStaged)
                             : mlir::triton::gpu::lookupNumWarps(firstMasked);
          _output << "threadgroup " << typeToString(resTy.getElem())
                  << " _stage_shared[" << numWarps << "][" << elems << "]";
        } else {
          _output << "threadgroup " << typeToString(resTy.getElem())
                  << " _stage_shared[" << elems << "]";
        }
        printDelim();
        _sharedStageBufferDeclared = true;
      }
      // Fused-store scratch, shared across all simdgroup_fused_store ops (they
      // run sequentially with barriers) so threadgroup memory does not scale
      // with the number of output tiles.
      if (!_fstoreScratchDeclared) {
        mlir::triton::metal::SimdgroupFusedStoreOp firstFused;
        kernelOp.walk([&](mlir::triton::metal::SimdgroupFusedStoreOp op) {
          firstFused = op;
          return mlir::WalkResult::interrupt();
        });
        if (firstFused) {
          auto ft = llvm::cast<mlir::triton::metal::MetalSimdgroupMatrixType>(
              firstFused.getBase().getType());
          unsigned felems = ft.getRows() * ft.getCols();
          bool fpw = !firstFused.getWarpIndex().empty();
          auto fty = typeToString(ft.getElem());
          for (const char *nm : {"_fstore_base", "_fstore_delta"}) {
            _output << "\n";
            indent();
            _output << "threadgroup " << fty << " " << nm;
            if (fpw)
              _output << "[" << mlir::triton::gpu::lookupNumWarps(firstFused) << "]";
            _output << "[" << felems << "]";
            printDelim();
          }
          _fstoreScratchDeclared = true;
        }
      }
    }
    for (auto &op : region.getOps()) {
      // L1d2b inline-barrier contract — boundary set:
      //   { metal.barrier, metal.tg_load_indexed, metal.tg_store_indexed }.
      //
      // Without this hook, `translateValue(metal.tg_load_indexed)` is invoked
      // lazily from the consumer's emission path (e.g. inside the
      // `MaskedStoreLowering`-emitted `scf.if(mask){ store … }` body), which
      // re-evaluates `<buf>[<idx>]` AFTER the trailing `threadgroup_barrier`
      // in execution order on every masked-true lane. Apple's Metal shading
      // compiler miscompiles that configuration (drops stores from non-zero
      // warps and races warp 0). The fix: force-materialize `tg_load_indexed`
      // results as named MSL let-bindings at their IR position — before any
      // subsequent `scf.if` block — and route uses through `_letBound` so
      // they render as `v<N>` instead of re-inlining the load expression.
      //
      // The other two boundary ops (`metal.barrier`, `metal.tg_store_indexed`)
      // produce no result, so there is nothing to let-bind for them; they are
      // already emitted as standalone statements (see `isStatementPrintable`
      // / `translateStatement`) and naturally act as ordering points because
      // this walk emits statements in IR order without re-ordering.
      if (auto tgLoad =
              llvm::dyn_cast<mlir::triton::metal::TgLoadIndexedOp>(&op)) {
        if (_letBound.find(&op) == _letBound.end()) {
          _output << "\n";
          indent();
          unsigned idx = _varCount++;
          _output << typeToString(tgLoad.getResult().getType()) << " v" << idx
                  << " = ";
          translate(tgLoad);
          _output << ";";
          _letBound[&op] = idx;
          continue;
        }
      }
      // General CSE: materialize any MULTI-USE inlineable value op as a named
      // let-binding `T v<N> = <expr>;` at its IR position, so the expression is
      // emitted ONCE and every use renders as `v<N>`. Without this, the emitter
      // re-inlines a value's expression at every use; a deep reuse chain (adder:
      // embedding->RMSNorm->RoPE->attn->MLP->argmax, each value used 2-3x) then
      // expands the MSL exponentially (~37 MB, single 36-MB line). Constants are
      // cheap leaves (no duplication blow-up) so they stay inlined. Emission is
      // in IR order, so ordering (e.g. relative to barriers) is preserved.
      // `metal.materialize` forces the let-binding even for a SINGLE-use value.
      // Set by ScanLowering on a scan-result placeholder that reads a
      // threadgroup buffer which a later scan (or the next trip of the tile
      // loop) will overwrite: inlining such a read at its use site moves it
      // PAST the overwrite and silently returns the wrong scan's data. Binding
      // it here pins the read to the scan's own position in IR order.
      if (op.getNumResults() == 1 && !op.getResult(0).use_empty() &&
          (!op.getResult(0).hasOneUse() ||
           op.hasAttr("metal.materialize")) &&
          _letBound.find(&op) == _letBound.end() &&
          // Allowlist of pure inlineable value ops translateValue emits as a
          // standalone expression. Restricting to these keeps CSE safe (some
          // other value ops assume they're inlined at a use). Covers adder's
          // deep arith/math reuse chain that drives the exponential blow-up.
          llvm::isa<mlir::arith::AddFOp, mlir::arith::SubFOp,
                    mlir::arith::MulFOp, mlir::arith::DivFOp,
                    mlir::arith::MaximumFOp, mlir::arith::MaxNumFOp,
                    mlir::arith::AddIOp, mlir::arith::SubIOp,
                    mlir::arith::MulIOp, mlir::arith::CmpIOp,
                    mlir::arith::CmpFOp, mlir::arith::SelectOp,
                    mlir::arith::SIToFPOp, mlir::math::ExpOp,
                    mlir::math::SqrtOp, mlir::math::LogOp, mlir::math::SinOp,
                    mlir::math::CosOp, mlir::math::ErfOp, mlir::math::RsqrtOp,
                    mlir::triton::metal::BinaryExpOp,
                    mlir::triton::metal::UnaryExpOp,
                    mlir::triton::metal::GetElementOp>(&op)) {
        _output << "\n";
        indent();
        unsigned idx = _varCount++;
        _output << typeToString(op.getResult(0).getType()) << " v" << idx
                << " = ";
        translateValue(&op);
        _output << ";";
        _letBound[&op] = idx;
        continue;
      }
      if (isStatementPrintable(&op)) {
        _output << "\n";
        indent();
      }
      translateStatement(&op);
    }
  }
  _output << "\n";
  indent();
  _output << "}";
}

// Wall 15: dispatch helper — for operands that may be either an op-result
// (use `translateValue(definingOp)`) OR a BlockArgument like a region
// iter-arg / induction variable (use `translateVarName(value)` which
// renders `v<idx>` from `_buffers`). The pre-Wall-15 op translators
// (BinaryExpOp etc.) called `translateValue(value.getDefiningOp())`
// unconditionally and crashed on BlockArgument operands; the re-rolled
// rank-1 reduce body's combine step has the iter-arg BlockArgument as
// LHS, so callsites consuming iter-args MUST route through this helper.
void ModuleTranslation::translateValueOrVarName(mlir::Value v) {
  // A value explicitly materialized as an MSL temp resolves by its own buffer
  // name first. This is required for a specific result of a MULTI-result op
  // (e.g. one accumulator of a multi-accumulator reduce scf.for): translating
  // its defining op as an expression can't pick out which result, and would hit
  // the translateValue default (UNREACHABLE). Single-result ops in `_buffers`
  // resolve to the same name they got via translateValue, so this is a no-op
  // for the pre-existing single-iter_arg path.
  auto it = _buffers.find(v.getAsOpaquePointer());
  if (it != _buffers.end()) {
    _output << "v" << it->second;
    return;
  }
  if (auto *defOp = v.getDefiningOp())
    translateValue(defOp);
  else
    translateVarName(v);
}

void ModuleTranslation::translateValue(Operation *opInst) {
  // L1d2b inline-barrier contract: if this op's result has been
  // force-materialized as an MSL let-binding (see `translate(Region&)`),
  // render the use as the let-binding name rather than re-inlining its
  // value expression. This is what keeps the `tg_load_indexed` expression
  // OUT of any subsequent `scf.if(mask){ … }` body emitted by
  // `MaskedStoreLowering` and prevents the downstream Apple-MSL miscompile.
  {
    auto it = _letBound.find(opInst);
    if (it != _letBound.end()) {
      _output << "v" << it->second;
      return;
    }
  }
  // Wall 15: if this op's result has been mapped into `_buffers` (e.g. an
  // `scf.for` carrying a single f32 iter_arg, whose result is the loop's
  // final-value MSL temp), render the use as the buffer name. This keeps
  // downstream consumers (`metal.store`, `metal.binary_exp`, etc.) from
  // re-translating the scf.for op as an expression.
  if (opInst->getNumResults() == 1) {
    auto it = _buffers.find(opInst->getResult(0).getAsOpaquePointer());
    if (it != _buffers.end()) {
      _output << "v" << it->second;
      return;
    }
  }
  llvm::TypeSwitch<Operation *>(opInst)
      .Case<mlir::triton::metal::ConstantOp, mlir::triton::metal::GetElementOp,
            mlir::triton::metal::TgLoadIndexedOp,
            mlir::triton::metal::ThreadIdOp,
            mlir::triton::metal::ThreadgroupIdOp,
            mlir::triton::metal::ThreadgroupsPerGridOp,
            mlir::triton::metal::CastOp,
            mlir::triton::metal::UnaryExpOp, mlir::triton::metal::BinaryExpOp,
            mlir::triton::metal::YieldWhileOp>([&](auto &op) { translate(op); })
      .Case<mlir::arith::CmpIOp>([&](mlir::arith::CmpIOp op) {
        // Emit `(lhs <pred> rhs)` matching arith.cmpi semantics. Only the
        // predicates needed by the masked vector-add path are wired; extend
        // as new fixtures land.
        using P = mlir::arith::CmpIPredicate;
        const char *opStr = nullptr;
        switch (op.getPredicate()) {
        case P::eq:  opStr = " == "; break;
        case P::ne:  opStr = " != "; break;
        case P::slt: opStr = " < ";  break;
        case P::sle: opStr = " <= "; break;
        case P::sgt: opStr = " > ";  break;
        case P::sge: opStr = " >= "; break;
        case P::ult: opStr = " < ";  break;
        case P::ule: opStr = " <= "; break;
        case P::ugt: opStr = " > ";  break;
        case P::uge: opStr = " >= "; break;
        }
        // Operands may be block arguments (e.g. an scf.for induction var fed
        // straight into a comparison by the computed-cone reduce evaluator),
        // so route through translateValueOrVarName, not translateValue.
        _output << "(";
        translateValueOrVarName(op.getLhs());
        _output << opStr;
        translateValueOrVarName(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::CmpFOp>([&](mlir::arith::CmpFOp op) {
        // Emit `(lhs <pred> rhs)` matching arith.cmpf semantics. Only the
        // ordered predicates needed by the leaky_relu fixture are wired;
        // extend as new fixtures land.
        using P = mlir::arith::CmpFPredicate;
        const char *opStr = nullptr;
        switch (op.getPredicate()) {
        // Ordered preds: MSL `>`/`<`/`==` on `float` return false on NaN,
        // matching MLIR's ordered semantics.
        case P::OEQ: opStr = " == "; break;
        case P::OGT: opStr = " > ";  break;
        case P::OGE: opStr = " >= "; break;
        case P::OLT: opStr = " < ";  break;
        case P::OLE: opStr = " <= "; break;
        case P::ONE: opStr = " != "; break;
        // Unordered not-equal: MSL `a != b` is true when either operand is NaN
        // or they differ, matching `une` semantics (Triton lowers `x != y` to
        // this). The `tl.atomic`-guarded subarray-sum kernels use `sum != 0`.
        case P::UNE: opStr = " != "; break;
        // Other unordered / non-ordered predicates: deferred. Listed explicitly
        // so the diagnostic names the predicate that hit.
        case P::UEQ:
          llvm_unreachable(
              "arith.cmpf UEQ (unordered ==) not yet supported on Metal");
        case P::UGT:
          llvm_unreachable(
              "arith.cmpf UGT (unordered >) not yet supported on Metal");
        case P::UGE:
          llvm_unreachable(
              "arith.cmpf UGE (unordered >=) not yet supported on Metal");
        case P::ULT:
          llvm_unreachable(
              "arith.cmpf ULT (unordered <) not yet supported on Metal");
        case P::ULE:
          llvm_unreachable(
              "arith.cmpf ULE (unordered <=) not yet supported on Metal");
        case P::ORD:
          llvm_unreachable(
              "arith.cmpf ORD (ordered, neither is NaN) not yet supported on Metal");
        case P::UNO:
          llvm_unreachable(
              "arith.cmpf UNO (unordered, at least one NaN) not yet supported on Metal");
        case P::AlwaysTrue:
          llvm_unreachable(
              "arith.cmpf AlwaysTrue not yet supported on Metal");
        case P::AlwaysFalse:
          llvm_unreachable(
              "arith.cmpf AlwaysFalse not yet supported on Metal");
        }
        _output << "(";
        translateValueOrVarName(op.getLhs());
        _output << opStr;
        translateValueOrVarName(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp op) {
        // arith.constant (signless integer or float) survives in the masked
        // path as the upper-bound N of the comparison. Emit as a literal.
        if (auto v = llvm::dyn_cast<IntegerAttr>(op.getValue()))
          _output << v.getValue();
        else if (auto v = llvm::dyn_cast<FloatAttr>(op.getValue()))
          emitFloatLiteral(_output, v);
        else
          llvm_unreachable("Unexpected arith.constant attribute kind");
      })
      .Case<mlir::UnrealizedConversionCastOp>(
          [&](mlir::UnrealizedConversionCastOp op) {
            // The masked path uses this cast as a ui32→i32 reinterpret so
            // arith.cmpi (signless) can consume metal.thread_id (ui32). MSL
            // treats uint and int interchangeably in this expression context;
            // forward the source value. The BLOCK_SIZE>threads slice also
            // casts an scf.for induction var (BlockArgument) — handle that
            // by routing through translateVarName.
            assert(op.getInputs().size() == 1 &&
                   "unrealized_conversion_cast: expected single-input form");
            auto in = op.getInputs()[0];
            if (auto inOp = in.getDefiningOp())
              translateValue(inOp);
            else
              translateVarName(in);
          })
      .Case<mlir::arith::MulIOp>([&](mlir::arith::MulIOp op) {
        // Used by BLOCK_SIZE>threads idx arithmetic (e.g. `tid * E`).
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " * ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::AddIOp>([&](mlir::arith::AddIOp op) {
        // Used by BLOCK_SIZE>threads idx arithmetic (e.g. `tid + iv * T`).
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " + ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::SubIOp>([&](mlir::arith::SubIOp op) {
        // Used by 2D MakeRangeLowering for the global->local tid mapping:
        // `lid.x = id.x - tgid.x * threadsPerCTA`.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " - ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp op) {
        // Scalar float add surviving to translation (the conversion pass only
        // folds TENSOR float arith into metal.binary_exp). MSL `+` on float.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " + ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp op) {
        // Scalar float subtract, e.g. `1 - p` in tutorial-04 _dropout (the
        // scalar operand of a metal.binary_exp divOp). MSL `-` on float.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " - ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp op) {
        // Scalar float multiply surviving to translation. MSL `*` on float.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " * ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::DivFOp>([&](mlir::arith::DivFOp op) {
        // Scalar float divide surviving to translation (tensor divf is folded
        // to metal.binary_exp divOp by the conversion pass). MSL `/` on float.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " / ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::NegFOp>([&](mlir::arith::NegFOp op) {
        // Unary float negate. MSL prefix `-`.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(-";
        emit(op.getOperand());
        _output << ")";
      })
      .Case<mlir::arith::DivSIOp>([&](mlir::arith::DivSIOp op) {
        // Used by 2D MakeRangeLowering: `row = idx / BLOCK_N`. MSL integer
        // divsion follows C semantics, which matches arith.divsi for the
        // non-negative thread indices we emit here.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " / ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::RemSIOp>([&](mlir::arith::RemSIOp op) {
        // Used by 2D MakeRangeLowering: `col = idx % BLOCK_N`.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " % ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::AndIOp>([&](mlir::arith::AndIOp op) {
        // Used by 2D mask reduction: `(row<M) & (col<N)`. MSL `&` on bools
        // is short-circuit-free bitwise AND, which matches arith.andi.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " & ";
        emit(op.getRhs());
        _output << ")";
      })
      // L2: Elementwise integer arith emitter cases. Each mirrors the SubI
      // shape — emit `(lhs <op> rhs)` with the MSL built-in operator.
      // Signed/unsigned semantics are carried by operand dtype.
      // See `.omc/specs/deep-interview-leet-triton-l2-int-arith-broad.md`.
      .Case<mlir::arith::ShRSIOp>([&](mlir::arith::ShRSIOp op) {
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " >> ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::ShLIOp>([&](mlir::arith::ShLIOp op) {
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " << ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::OrIOp>([&](mlir::arith::OrIOp op) {
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " | ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::XOrIOp>([&](mlir::arith::XOrIOp op) {
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " ^ ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::DivUIOp>([&](mlir::arith::DivUIOp op) {
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " / ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::RemUIOp>([&](mlir::arith::RemUIOp op) {
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " % ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::MinSIOp, mlir::arith::MinUIOp, mlir::arith::MaxSIOp,
            mlir::arith::MaxUIOp>([&](Operation *op) {
        // MSL `min`/`max` (used by e.g. tl.swizzle2d's group-size clamp). Cast
        // both operands to the op's signedness so the overload isn't ambiguous
        // when one operand is `tgid.x` (uint) and the other signed arithmetic.
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        bool isMin =
            mlir::isa<mlir::arith::MinSIOp, mlir::arith::MinUIOp>(op);
        bool isSigned =
            mlir::isa<mlir::arith::MinSIOp, mlir::arith::MaxSIOp>(op);
        const char *cast = isSigned ? "(int)(" : "(uint)(";
        _output << (isMin ? "min(" : "max(") << cast;
        emit(op->getOperand(0));
        _output << "), " << cast;
        emit(op->getOperand(1));
        _output << "))";
      })
      .Case<mlir::arith::ShRUIOp>([&](mlir::arith::ShRUIOp op) {
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getLhs());
        _output << " >> ";
        emit(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::SIToFPOp, mlir::arith::UIToFPOp,
            mlir::arith::FPToSIOp, mlir::arith::FPToUIOp, mlir::arith::ExtFOp,
            mlir::arith::TruncFOp, mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
            mlir::arith::TruncIOp>([&](auto op) {
        // Scalar numeric conversion surviving to translation, e.g.
        // `idx.to(tl.float32)` -> arith.sitofp (tensor conversions are folded
        // into metal ops by the conversion pass; only scalar ones reach here).
        // MSL spells every numeric conversion as a constructor cast `T(x)`,
        // mirroring metal.cast's emission. float->int uses C truncation toward
        // zero, matching arith.fptosi/fptoui; int->float and width changes are
        // exact for the value ranges Triton emits here. The generic lambda is
        // instantiated per op type, so .getIn()/.getType() resolve concretely.
        // C-style cast `(T)(x)`, NOT functional `T(x)`: the latter is a variable
        // DECLARATION when x is a bare identifier (C++ most-vexing-parse), so a
        // dead conversion emitted as a statement (e.g. a reduce that re-derives
        // its cone leaves the original `x.to(f32)` extf use-empty) would become
        // `float(v3);` == `float v3;` — a redefinition. The C-style form is a
        // valid no-op as a statement and identical as an expression.
        _output << "(" << typeToString(op.getType()) << ")(";
        translateValueOrVarName(op.getIn());
        _output << ")";
      })
      .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp op) {
        auto emit = [&](mlir::Value v) {
          // Route through translateValueOrVarName so a specific result of a
          // MULTI-result op (e.g. one field of a 2-iter_arg scf.for like
          // speculative decoding's `chosen_k`) resolves via _buffers instead of
          // hitting the translateValue default on the whole op.
          translateValueOrVarName(v);
        };
        _output << "(";
        emit(op.getCondition());
        _output << " ? ";
        emit(op.getTrueValue());
        _output << " : ";
        emit(op.getFalseValue());
        _output << ")";
      })
      .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp op) {
        auto it = _scfIfTemp.find(op.getOperation());
        assert(it != _scfIfTemp.end() &&
               "scf.if result referenced before pre-declaration");
        _output << "v" << it->second;
      })
      .Case<mlir::triton::metal::SimdgroupIndexOp>(
          [&](mlir::triton::metal::SimdgroupIndexOp op) {
            // AC4 v6: SimdgroupIndexOp resolves to the kernel parameter
            // `sgid` (declared via [[simdgroup_index_in_threadgroup]] in
            // translateKernel). It has no separate statement form.
            _output << "sgid";
          })
      .Default([&](Operation *o) {
        llvm::errs() << "[metal] translateValue: unexpected op " << o->getName()
                     << "\n";
        llvm_unreachable("Unexpected operation");
      });
}

void ModuleTranslation::translate(mlir::triton::metal::ConstantOp op) {
  if (auto v = llvm::dyn_cast<BoolAttr>(op.getValue()))
    _output << (v.getValue() ? "true" : "false");
  else if (auto v = llvm::dyn_cast<IntegerAttr>(op.getValue()))
    _output << v.getValue();
  else if (auto v = llvm::dyn_cast<FloatAttr>(op.getValue()))
    emitFloatLiteral(_output, v);
  else
    llvm_unreachable("Unexpected constant");
}

void ModuleTranslation::translate(mlir::triton::metal::GetElementOp op) {
  translateVarName(op.getMemref());
  _output << "[";
  translateValue(op.getIndex().getDefiningOp());
  _output << "]";
}

void ModuleTranslation::translate(mlir::triton::metal::ThreadIdOp op) {
  _output << "id." << op.getDimension();
}

void ModuleTranslation::translate(mlir::triton::metal::ThreadgroupIdOp op) {
  _output << "tgid." << op.getDimension();
}

void ModuleTranslation::translate(mlir::triton::metal::ThreadgroupsPerGridOp op) {
  _output << "tgpg." << op.getDimension();
}

void ModuleTranslation::translate(mlir::triton::metal::CastOp op) {
  _output << typeToString(op.getType());
  _output << "(";
  translateValue(op.getArgument().getDefiningOp());
  _output << ")";
}

void ModuleTranslation::translate(mlir::triton::metal::UnaryExpOp op) {
  auto translateArgument = [&] {
    translateValue(op.getArgument().getDefiningOp());
  };

  using OP = mlir::triton::metal::UnaryExpOperator;
  switch (op.getUnaryOperator()) {
  case OP::minusOp:
    _output << "(-";
    translateArgument();
    _output << ")";
    break;
  case OP::notOp:
    _output << "(!";
    translateArgument();
    _output << ")";
    break;
  case OP::expOp:
    _output << "metal::precise::exp(";
    translateArgument();
    _output << ")";
    break;
  case OP::sqrtOp:
    _output << "metal::precise::sqrt(";
    translateArgument();
    _output << ")";
    break;
  case OP::erfOp:
    // MSL stdlib has no `metal::erf`; we emit a call to `__triton_erff`,
    // a polynomial approximation defined in the module preamble when any
    // kernel uses the `erfOp` case (see `translateModule`).
    _output << "__triton_erff(";
    translateArgument();
    _output << ")";
    break;
  case OP::logOp:
    _output << "metal::precise::log(";
    translateArgument();
    _output << ")";
    break;
  case OP::rsqrtOp:
    _output << "metal::precise::rsqrt(";
    translateArgument();
    _output << ")";
    break;
  case OP::sinOp:
    _output << "metal::precise::sin(";
    translateArgument();
    _output << ")";
    break;
  case OP::cosOp:
    _output << "metal::precise::cos(";
    translateArgument();
    _output << ")";
    break;
  }
}

void ModuleTranslation::translate(mlir::triton::metal::BinaryExpOp op) {
  using OP = mlir::triton::metal::BinaryExpOperator;
  // maxOp lowers to MSL's `max(a, b)` function call (rank-1 reduce spec
  // `.omc/specs/deep-interview-metal-rank1-reduce.md`). All other ops
  // lower to an infix C operator.
  if (op.getBinaryOperator() == OP::maxOp ||
      op.getBinaryOperator() == OP::minOp) {
    // Emit `max(T(lhs), T(rhs))` / `min(...)` where T is the result element
    // type. The explicit cast forces a SIGNED comparison for i32 signed-min/max
    // even when an operand is rendered from ui32 storage (the ui32->si32 MLIR
    // bridge is a no-op in MSL). For f32 this is `max/min(float(x), float(y))`.
    const char *fn = op.getBinaryOperator() == OP::minOp ? "min" : "max";
    auto ty = typeToString(op.getResult().getType());
    _output << fn << "(" << ty << "(";
    translateValueOrVarName(op.getLhs());
    _output << "), " << ty << "(";
    translateValueOrVarName(op.getRhs());
    _output << "))";
    return;
  }

  _output << "(";
  translateValueOrVarName(op.getLhs());
  _output << ") ";

  switch (op.getBinaryOperator()) {
  case OP::addOp:
    _output << "+";
    break;
  case OP::subOp:
    _output << "-";
    break;
  case OP::mulOp:
    _output << "*";
    break;
  case OP::divOp:
    _output << "/";
    break;
  case OP::remOp:
    _output << "%";
    break;
  case OP::eqOp:
    _output << "==";
    break;
  case OP::neOp:
    _output << "!=";
    break;
  case OP::ltOp:
    _output << "<";
    break;
  case OP::leOp:
    _output << "<=";
    break;
  case OP::gtOp:
    _output << ">";
    break;
  case OP::geOp:
    _output << ">=";
    break;
  case OP::andOp:
    _output << "&&";
    break;
  case OP::orOp:
    _output << "||";
    break;
  case OP::maxOp:
  case OP::minOp:
    llvm_unreachable("min/maxOp handled above via function-call form");
  }

  _output << " (";
  translateValueOrVarName(op.getRhs());
  _output << ")";
}

void ModuleTranslation::translate(mlir::triton::metal::YieldWhileOp op) {
  translateValue(op.getCondition().getDefiningOp());
}
