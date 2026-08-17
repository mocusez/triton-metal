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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
      // Triton uses signless i8 for torch.int8 buffers. Only an explicitly
      // unsigned MLIR integer denotes uint8; treating signless as unsigned
      // turns negative quantized K/V values into 128..255 before sitofp.
      return intTy.isUnsigned() ? "uint8_t" : "int8_t";
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

// MSL takes signed-vs-unsigned semantics from the C TYPE of the operand
// expression, but in MLIR the signedness lives in the OP: slt vs ult, divsi vs
// divui, remsi vs remui, shrsi vs shrui, sitofp vs uitofp, extsi vs extui.
// Triton spells both `tl.int32` and `tl.uint32` as a signless i32 and lets the
// op carry the distinction, and `metalStorageElementType` declares a signless
// device buffer as `uint32_t`. Because a load is inlined at its use site, an
// uncast operand silently turns every SIGNED operation on loaded data into an
// unsigned one — `x < 0` never fires, `x.to(tl.float32)` yields 2^32, `//`,
// `%` and `>>` are all wrong, and `tl.sort` orders negatives last.
//
// So an op that cares must spell its own signedness on the operand. Returns the
// C-style cast prefix to wrap the operand in, or "" when signedness cannot
// change the result (i1 is `bool`; non-integers carry no signedness).
// arith.minsi/maxsi have always done this inline — this is the same rule,
// hoisted so every signedness-sensitive op can apply it.
static llvm::StringRef signednessCast(mlir::Type type, bool wantSigned) {
  // An elementwise op can still be TENSOR-typed at this point — the emitter
  // renders it once per element, inlining the operand's buffer read — so the
  // signedness that matters is the element's. Without this, `math.absi` on a
  // tile emitted `abs(v0[i])` on a `uint32_t` buffer, which is the identity.
  if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(type))
    type = shaped.getElementType();
  auto intTy = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!intTy)
    return "";
  switch (intTy.getWidth()) {
  case 8:
    return wantSigned ? "(int8_t)" : "(uint8_t)";
  case 16:
    return wantSigned ? "(int16_t)" : "(uint16_t)";
  case 32:
    return wantSigned ? "(int32_t)" : "(uint32_t)";
  case 64:
    return wantSigned ? "(int64_t)" : "(uint64_t)";
  default:
    // i1 renders as `bool`, where signed and unsigned agree.
    return "";
  }
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
      // MSL has no atomic half. `tl.atomic_add` on an f16/bf16 buffer becomes a
      // compare-exchange loop over the 32-bit word that CONTAINS the element:
      // load the pair, add into this element's lane, swap the pair back, retry
      // if another thread got there first. The word is 4-byte aligned because
      // the buffer base is and the index is masked to an even element.
      bool hasHalfAtomic = false;
      m.walk([&](mlir::triton::metal::AtomicRmwOp aop) {
        if (aop.getKind() == mlir::triton::metal::AtomicRmwKind::Add &&
            aop.getValue().getType().isF16()) {
          hasHalfAtomic = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (hasHalfAtomic) {
        for (llvm::StringRef ty : {"half"}) {
          output << "static inline " << ty << " __triton_atomic_add_"
                 << (ty == "half" ? "half" : "bfloat") << "(device " << ty
                 << "* base, uint idx, " << ty << " v) {\n"
                    "  device atomic_uint* word = (device atomic_uint*)(base + "
                    "(idx & ~1u));\n"
                    "  uint lane = idx & 1u;\n"
                    "  uint old = atomic_load_explicit(word, "
                    "memory_order_relaxed);\n"
                    "  uint desired;\n"
                    "  "
                 << ty << " prev;\n"
                    "  do {\n"
                    "    "
                 << ty << "2 pair = as_type<" << ty << "2>(old);\n"
                    "    prev = pair[lane];\n"
                    "    pair[lane] = prev + v;\n"
                    "    desired = as_type<uint>(pair);\n"
                    "  } while (!atomic_compare_exchange_weak_explicit("
                    "word, &old, desired, memory_order_relaxed, "
                    "memory_order_relaxed));\n"
                    "  return prev;\n"
                    "}\n\n";
        }
      }
      // MSL has no 8-bit float type, so `tl.float8e4nv` / `tl.float8e5` casts
      // run in software. Round-to-nearest-even is applied ONCE straight off
      // the f32 bit pattern: rounding through f16 first double-rounds, and 119
      // of 60032 fuzz values then landed a ulp away from `torch.Tensor.to`.
      bool hasFp8Convert = false;
      m.walk([&](mlir::triton::metal::Fp8ConvertOp) {
        hasFp8Convert = true;
        return mlir::WalkResult::interrupt();
      });
      if (hasFp8Convert) {
        output <<
            "static inline float __triton_fp8_to_f32(uchar b, uint mant_bits,\n"
            "                                        uint exp_bits, uint bias) {\n"
            "  uint max_exp = (1u << exp_bits) - 1u;\n"
            "  uint s = (uint)(b & 0x80) << 24;\n"
            "  uint e = ((uint)b >> mant_bits) & max_exp;\n"
            "  uint m = (uint)b & ((1u << mant_bits) - 1u);\n"
            "  if (e == 0u) {\n"
            "    if (m == 0u) return as_type<float>(s);\n"
            "    float v = (float)m * exp2((float)(1 - (int)bias - (int)mant_bits));\n"
            "    return as_type<float>(as_type<uint>(v) | s);\n"
            "  }\n"
            "  if (e == max_exp) {\n"
            "    if (exp_bits == 5u)\n"
            "      return as_type<float>(s | 0x7F800000u | (m << (23u - mant_bits)));\n"
            "    if (m == ((1u << mant_bits) - 1u)) return as_type<float>(s | 0x7FC00000u);\n"
            "  }\n"
            "  return as_type<float>(s | ((e + 127u - bias) << 23) |\n"
            "                        (m << (23u - mant_bits)));\n"
            "}\n\n"
            "static inline uchar __triton_f32_to_fp8(float x, uint mant_bits,\n"
            "                                        uint exp_bits, uint bias,\n"
            "                                        bool has_inf) {\n"
            "  uint xb = as_type<uint>(x);\n"
            "  uchar sign = (uchar)((xb >> 24) & 0x80u);\n"
            "  uint max_exp = (1u << exp_bits) - 1u;\n"
            "  uchar nan_pat = (uchar)(sign | (max_exp << mant_bits) |\n"
            "                          ((1u << mant_bits) - 1u));\n"
            "  if (isnan(x)) return nan_pat;\n"
            "  float ax = fabs(x);\n"
            "  if (isinf(ax))\n"
            "    return has_inf ? (uchar)(sign | (max_exp << mant_bits)) : nan_pat;\n"
            "  uint u = as_type<uint>(ax);\n"
            "  int e = (int)((u >> 23) & 0xFFu);\n"
            "  uint m = u & 0x7FFFFFu;\n"
            "  int min_normal_e = 127 - (int)bias + 1;\n"
            "  int shift; int biased;\n"
            "  if (e >= min_normal_e) {\n"
            "    shift = 23 - (int)mant_bits;\n"
            "    biased = e - (127 - (int)bias);\n"
            "  } else {\n"
            "    shift = 23 - (int)mant_bits + (min_normal_e - e);\n"
            "    biased = 0;\n"
            "    m |= 0x800000u;\n"
            "  }\n"
            "  if (shift > 31) return sign;\n"
            "  uint low = m & ((1u << shift) - 1u);\n"
            "  uint half_ = 1u << (shift - 1);\n"
            "  uint q = m >> shift;\n"
            "  if (low > half_ || (low == half_ && (q & 1u))) q += 1u;\n"
            "  if (q >= (1u << mant_bits)) { q -= (1u << mant_bits); biased += 1; }\n"
            "  if (biased > (int)max_exp ||\n"
            "      (biased == (int)max_exp && (has_inf || q >= ((1u << mant_bits) - 1u))))\n"
            "    return has_inf ? (uchar)(sign | (max_exp << mant_bits)) : nan_pat;\n"
            "  return (uchar)(sign | ((uint)biased << mant_bits) | q);\n"
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
  // The debug buffer is a trailing parameter, present only in kernels that
  // print or assert, so every other kernel's signature is unchanged. The
  // launcher binds it at exactly this slot — the count of real arguments —
  // which both sides derive from the same compile.
  bool usesDebugRecords = false;
  op.walk([&](mlir::triton::metal::DebugRecordOp) {
    usesDebugRecords = true;
    return mlir::WalkResult::interrupt();
  });
  if (usesDebugRecords) {
    _output << "  device uint32_t *__triton_dbg [[buffer(" << _varCount
            << ")]],\n";
    _varCount++;
  }
  _output << "  uint3 id [[thread_position_in_grid]]";
  // Conditionally add the threadgroup-position parameter only when the
  // kernel body references it (via metal.threadgroup_id). This keeps
  // existing single-program fixtures' MSL signatures unchanged. See
  // `.omc/specs/deep-interview-metal-pid-lowering.md`.
  // metal.fused_attention needs the threadgroup position (program id) AND the
  // LOCAL thread index: the threadgroup position selects the query block and,
  // with a head split, the head, while the single-warp guard must key off
  // thread_position_in_threadgroup -- thread_position_in_grid is global and
  // would mis-identify the 2nd query block's warp (Phase-0 finding).
  bool usesFusedAttention = false;
  op.walk([&](mlir::triton::metal::FusedAttentionOp) {
    usesFusedAttention = true;
    return mlir::WalkResult::interrupt();
  });
  bool usesInt4WeightOnlyMatmul = false;
  op.walk([&](mlir::triton::metal::Int4WeightOnlyMatmulOp) {
    usesInt4WeightOnlyMatmul = true;
    return mlir::WalkResult::interrupt();
  });
  bool usesInt8QuantizedMatmul = false;
  op.walk([&](mlir::triton::metal::Int8QuantizedMatmulOp) {
    usesInt8QuantizedMatmul = true;
    return mlir::WalkResult::interrupt();
  });
  bool usesLinearAttention = false;
  op.walk([&](mlir::triton::metal::LinearAttentionPreprocessOp) {
    usesLinearAttention = true;
    return mlir::WalkResult::interrupt();
  });
  op.walk([&](mlir::triton::metal::LinearAttentionApplyOp) {
    usesLinearAttention = true;
    return mlir::WalkResult::interrupt();
  });
  bool usesThreadgroupId = usesFusedAttention || usesInt4WeightOnlyMatmul ||
                           usesInt8QuantizedMatmul || usesLinearAttention;
  op.walk([&](mlir::triton::metal::ThreadgroupIdOp) {
    usesThreadgroupId = true;
    return mlir::WalkResult::interrupt();
  });
  if (usesThreadgroupId)
    _output << ",\n  uint3 tgid [[threadgroup_position_in_grid]]";
  if (usesFusedAttention || usesInt4WeightOnlyMatmul ||
      usesInt8QuantizedMatmul || usesLinearAttention)
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
            mlir::triton::metal::StoreOp, mlir::triton::metal::AtomicCasOp,
            mlir::triton::metal::AtomicRmwOp,
            mlir::triton::metal::ThreadgroupPrefixSumOp,
            mlir::triton::metal::ThreadgroupSegmentedPrefixSumOp,
            mlir::triton::metal::ThreadgroupAffinePrefixScanOp,
            mlir::triton::metal::IfOp,
            mlir::triton::metal::WhileOp, mlir::triton::metal::MatmulOp, mlir::triton::metal::GemvOp,
            mlir::triton::metal::QmvOp, mlir::triton::metal::QmmOp,
            mlir::triton::metal::Int4WeightOnlyMatmulOp,
            mlir::triton::metal::Int8QuantizedMatmulOp,
            mlir::triton::metal::LinearAttentionPreprocessOp,
            mlir::triton::metal::LinearAttentionApplyOp,
            mlir::triton::metal::ReduceOp,
            mlir::triton::metal::ArgmaxOp, mlir::triton::metal::SoftmaxOp,
            mlir::triton::metal::LogsumexpOp, mlir::triton::metal::SdpaOp,
            mlir::triton::metal::FusedAttentionOp,
            mlir::triton::metal::RmsNormOp, mlir::triton::metal::ReturnOp,
            mlir::triton::metal::SimdgroupMatrixZeroOp,
            mlir::triton::metal::SimdgroupLoadDeviceStagedOp,
            mlir::triton::metal::SimdgroupLoadDeviceStagedMaskedOp,
            mlir::triton::metal::SimdgroupLoadOp,
            mlir::triton::metal::SimdgroupMultiplyAccumulateOp,
            mlir::triton::metal::SimdgroupStoreOp,
            mlir::triton::metal::SimdgroupFusedStoreOp,
            mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp,
            mlir::triton::metal::DebugRecordOp,
            mlir::triton::metal::Fp8ConvertOp,
            mlir::scf::ConditionOp>(
          [&](auto &op) { printable = true; })
      .Case<mlir::triton::metal::YieldWhileOp, mlir::triton::metal::YieldOp>([&](auto &op) {
        // do nothing
        printable = false;
      })
      .Case<mlir::scf::YieldOp>([&](auto &op) {
        // Yielding into a result-bearing `scf.if` produces assignment
        // statements to the temp vars pre-declared before the `if`. Yields in
        // void `scf.if` are no-ops. A yield into `scf.while` always writes the
        // carried temps back.
        printable = false;
        if (auto parentIf =
                llvm::dyn_cast_or_null<mlir::scf::IfOp>(op->getParentOp())) {
          if (parentIf.getNumResults() > 0)
            printable = true;
        }
        if (llvm::isa_and_nonnull<mlir::scf::WhileOp>(op->getParentOp()))
          printable = true;
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
            mlir::triton::metal::StoreOp, mlir::triton::metal::AtomicCasOp,
            mlir::triton::metal::AtomicRmwOp,
            mlir::triton::metal::ThreadgroupPrefixSumOp,
            mlir::triton::metal::ThreadgroupSegmentedPrefixSumOp,
            mlir::triton::metal::ThreadgroupAffinePrefixScanOp,
            mlir::triton::metal::IfOp,
            mlir::triton::metal::WhileOp, mlir::triton::metal::MatmulOp, mlir::triton::metal::GemvOp,
            mlir::triton::metal::QmvOp, mlir::triton::metal::QmmOp,
            mlir::triton::metal::Int4WeightOnlyMatmulOp,
            mlir::triton::metal::Int8QuantizedMatmulOp,
            mlir::triton::metal::LinearAttentionPreprocessOp,
            mlir::triton::metal::LinearAttentionApplyOp,
            mlir::triton::metal::ReduceOp,
            mlir::triton::metal::ArgmaxOp, mlir::triton::metal::SoftmaxOp,
            mlir::triton::metal::LogsumexpOp, mlir::triton::metal::SdpaOp,
            mlir::triton::metal::FusedAttentionOp,
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
      .Case<mlir::triton::metal::DebugRecordOp>([&](auto &op) { translate(op); })
      .Case<mlir::scf::WhileOp>([&](auto &op) { translate(op); })
      .Case<mlir::scf::ConditionOp>([&](auto &op) { translate(op); })
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
  translateValueOrVarName(op.getIndex());
  _output << "] = ";
  // A user scf.for may return the online-softmax state that is stored after
  // the loop. Its individual results are mapped to accumulator temporaries;
  // translating the defining scf.for as a value cannot select a result.
  translateValueOrVarName(op.getValue());
  printDelim();
}

void ModuleTranslation::translate(mlir::triton::metal::AtomicCasOp op) {
  // Metal compare-exchange mutates the expected argument to the observed old
  // value. Initialize that temp from Triton's cmp operand and expose the same
  // temp as the op result.
  auto valueInt = llvm::cast<mlir::IntegerType>(op.getValue().getType());
  llvm::StringRef atomicTy = valueInt.isUnsigned() ? "atomic_uint"
                                                   : "atomic_int";
  unsigned expectedIdx = _varCount++;
  if (!op.getResult().use_empty())
    _buffers[op.getResult().getAsOpaquePointer()] = expectedIdx;

  _output << typeToString(op.getResult().getType()) << " v" << expectedIdx
          << " = ";
  translateValueOrVarName(op.getCmp());
  _output << ";\n";
  indent();
  // MSL exposes only weak compare-exchange. Retry a spurious failure while
  // the observed value still equals Triton's compare operand; stop after a
  // successful exchange or a real mismatch. This recovers strong CAS
  // semantics while retaining the expected temporary as Triton's old value.
  _output << "while (!atomic_compare_exchange_weak_explicit((device "
          << atomicTy << "*)&";
  translateVarName(op.getMemref());
  _output << "[";
  translateValueOrVarName(op.getIndex());
  _output << "], &v" << expectedIdx << ", ";
  translateValueOrVarName(op.getValue());
  _output << ", memory_order_relaxed, memory_order_relaxed) && v"
          << expectedIdx << " == ";
  translateValueOrVarName(op.getCmp());
  _output << ") {}";
  printDelim();
}

void ModuleTranslation::translate(
    mlir::triton::metal::ThreadgroupPrefixSumOp op) {
  // Inclusive prefix scan of inbuf -> outbuf over `block` elements, tiled over
  // E = block/tpb cyclic iv-blocks (pos = iv*tpb + tid). Per iv-block an
  // in-place double-barriered Hillis-Steele inclusive scan runs across the tpb
  // threads (its window is outbuf[base + 0..tpb), contiguous), then a running
  // `carry` (the combine of all prior iv-blocks) is folded in. Validated
  // bit-close to torch.cumsum (scan_spike.py). Barriers are OUTSIDE any
  // per-thread branch so all threads reach them (caller guarantees uniform
  // control flow).
  const int64_t BLOCK = op.getBlock();
  const int64_t TPB = op.getTpb();
  const int64_t E = BLOCK / TPB;
  const bool REVERSE = op->hasAttr("reverse");
  // Monoid: absent `combine` means add, so every cumsum caller predating the
  // attribute keeps its exact emission. Only add/mul are meaningful here — both
  // are commutative and associative, which is what lets the Hillis-Steele step
  // fold neighbours in whatever order the threads reach them.
  auto combineKind = op.getCombine().value_or(
      mlir::triton::metal::BinaryExpOperator::addOp);
  if (combineKind != mlir::triton::metal::BinaryExpOperator::addOp &&
      combineKind != mlir::triton::metal::BinaryExpOperator::mulOp) {
    op.emitError("metal.threadgroup_prefix_sum supports only the add and mul "
                 "combines");
    _emitFailed = true;
    return;
  }
  const bool IS_MUL =
      combineKind == mlir::triton::metal::BinaryExpOperator::mulOp;
  const llvm::StringRef FOLD = IS_MUL ? "*=" : "+=";
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
  const bool ELEM_IS_FLOAT =
      llvm::isa<mlir::FloatType>(bufElemTy(op.getOutbuf()));
  // Identity of the selected monoid, in the buffer's own element type.
  const std::string UNIT =
      IS_MUL ? (ELEM_IS_FLOAT ? "1.0f" : "1") : (ELEM_IS_FLOAT ? "0.0f" : "0");
  auto &os = _output;
  os << "\n  // ---- metal.threadgroup_prefix_sum ("
     << (IS_MUL ? "cumprod" : "cumsum") << ") ----\n";
  os << "  {\n";
  os << "    uint _ps_tid = id.x - tgid.x * " << S(TPB) << "u;\n";
  os << "    " << ELEM << " _ps_carry = " << UNIT << ";\n";
  os << "    for (uint _ps_k = 0u; _ps_k < " << S(E) << "u; ++_ps_k) {\n";
  if (REVERSE) {
    os << "      uint _ps_base = " << S(BLOCK)
       << "u - (_ps_k + 1u) * " << S(TPB) << "u;\n";
    os << "      uint _ps_orig = _ps_base + " << S(TPB - 1)
       << "u - _ps_tid;\n";
    // Cross-thread read: `_ps_orig` is the slot thread `tpb-1-tid` filled, so
    // the fill (or the previous chunk's write-back) must be visible first. The
    // forward path reads only the slot this thread wrote itself and needs no
    // such barrier.
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os << "      " << OUT << "[_ps_base + _ps_tid] = " << IN
       << "[_ps_orig];\n";
  } else {
    os << "      uint _ps_base = _ps_k * " << S(TPB) << "u;\n";
    os << "      " << OUT << "[_ps_base + _ps_tid] = " << IN
       << "[_ps_base + _ps_tid];\n";
  }
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      for (uint _ps_off = 1u; _ps_off < " << S(TPB)
     << "u; _ps_off <<= 1) {\n";
  os << "        " << ELEM << " _ps_add = (_ps_tid >= _ps_off) ? " << OUT
     << "[_ps_base + _ps_tid - _ps_off] : " << UNIT << ";\n";
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "        " << OUT << "[_ps_base + _ps_tid] " << FOLD << " _ps_add;\n";
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      }\n";
  os << "      " << ELEM << " _ps_total = " << OUT << "[_ps_base + "
     << S(TPB - 1) << "u];\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      " << OUT << "[_ps_base + _ps_tid] " << FOLD << " _ps_carry;\n";
  os << "      _ps_carry " << FOLD << " _ps_total;\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  if (REVERSE) {
    os << "      " << ELEM << " _ps_result = " << OUT
       << "[_ps_base + _ps_tid];\n";
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os << "      " << OUT << "[_ps_orig] = _ps_result;\n";
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  }
  os << "    }\n";
  os << "  }\n";
}

void ModuleTranslation::translate(
    mlir::triton::metal::ThreadgroupSegmentedPrefixSumOp op) {
  const int64_t BLOCK = op.getBlock();
  const int64_t TPB = op.getTpb();
  if (BLOCK <= 0 || TPB <= 0 || (BLOCK % TPB) != 0) {
    op.emitError("metal.threadgroup_segmented_prefix_sum requires positive "
                 "block/tpb with block divisible by tpb");
    _emitFailed = true;
    return;
  }

  auto valueTy = llvm::cast<MetalMemRefType>(op.getValues().getType());
  auto flagTy = llvm::cast<MetalMemRefType>(op.getFlags().getType());
  if (!valueTy.getType().isF32()) {
    op.emitError("metal.threadgroup_segmented_prefix_sum values buffer must "
                 "have f32 element type");
    _emitFailed = true;
    return;
  }
  auto flagIntTy = llvm::dyn_cast<mlir::IntegerType>(flagTy.getType());
  if (!flagIntTy || flagIntTy.getWidth() != 1) {
    op.emitError("metal.threadgroup_segmented_prefix_sum flags buffer must "
                 "have i1 element type");
    _emitFailed = true;
    return;
  }
  if (valueTy.getSize() < BLOCK || flagTy.getSize() < BLOCK) {
    op.emitError("metal.threadgroup_segmented_prefix_sum buffers must have at "
                 "least block elements");
    _emitFailed = true;
    return;
  }

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
  const std::string VALUES = bufName(op.getValues());
  const std::string FLAGS = bufName(op.getFlags());
  auto S = [](int64_t x) { return std::to_string(x); };
  const int64_t E = BLOCK / TPB;
  auto &os = _output;

  os << "\n  // ---- metal.threadgroup_segmented_prefix_sum ----\n";
  os << "  {\n";
  os << "    uint _sgps_tid = id.x - tgid.x * " << S(TPB) << "u;\n";
  os << "    float _sgps_carry_v = 0.0f;\n";
  os << "    bool _sgps_carry_f = false;\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "    for (uint _sgps_k = 0u; _sgps_k < " << S(E)
     << "u; ++_sgps_k) {\n";
  os << "      uint _sgps_base = _sgps_k * " << S(TPB) << "u;\n";
  os << "      for (uint _sgps_off = 1u; _sgps_off < " << S(TPB)
     << "u; _sgps_off <<= 1) {\n";
  os << "        float _sgps_cur_v = " << VALUES
     << "[_sgps_base + _sgps_tid];\n";
  os << "        bool _sgps_cur_f = " << FLAGS
     << "[_sgps_base + _sgps_tid];\n";
  os << "        float _sgps_prev_v = 0.0f;\n";
  os << "        bool _sgps_prev_f = false;\n";
  os << "        if (_sgps_tid >= _sgps_off) {\n";
  os << "          _sgps_prev_v = " << VALUES
     << "[_sgps_base + _sgps_tid - _sgps_off];\n";
  os << "          _sgps_prev_f = " << FLAGS
     << "[_sgps_base + _sgps_tid - _sgps_off];\n";
  os << "        }\n";
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "        if (_sgps_tid >= _sgps_off) {\n";
  os << "          " << VALUES
     << "[_sgps_base + _sgps_tid] = _sgps_cur_f ? _sgps_cur_v : "
        "(_sgps_prev_v + _sgps_cur_v);\n";
  os << "          " << FLAGS
     << "[_sgps_base + _sgps_tid] = _sgps_prev_f || _sgps_cur_f;\n";
  os << "        }\n";
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      }\n";
  os << "      float _sgps_total_v = " << VALUES << "[_sgps_base + "
     << S(TPB - 1) << "u];\n";
  os << "      bool _sgps_total_f = " << FLAGS << "[_sgps_base + "
     << S(TPB - 1) << "u];\n";
  os << "      bool _sgps_local_f = " << FLAGS
     << "[_sgps_base + _sgps_tid];\n";
  os << "      float _sgps_local_v = " << VALUES
     << "[_sgps_base + _sgps_tid];\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      " << VALUES
     << "[_sgps_base + _sgps_tid] = _sgps_local_f ? _sgps_local_v : "
        "(_sgps_carry_v + _sgps_local_v);\n";
  os << "      " << FLAGS
     << "[_sgps_base + _sgps_tid] = _sgps_carry_f || _sgps_local_f;\n";
  os << "      _sgps_carry_v = _sgps_total_f ? _sgps_total_v : "
        "(_sgps_carry_v + _sgps_total_v);\n";
  os << "      _sgps_carry_f = _sgps_carry_f || _sgps_total_f;\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "    }\n";
  os << "  }\n";
}

void ModuleTranslation::translate(
    mlir::triton::metal::ThreadgroupAffinePrefixScanOp op) {
  const int64_t BLOCK = op.getBlock();
  const int64_t TPB = op.getTpb();
  if (BLOCK <= 0 || TPB <= 0 || (BLOCK % TPB) != 0) {
    op.emitError("metal.threadgroup_affine_prefix_scan requires positive "
                 "block/tpb with block divisible by tpb");
    _emitFailed = true;
    return;
  }

  auto aTy = llvm::cast<MetalMemRefType>(op.getA().getType());
  auto xTy = llvm::cast<MetalMemRefType>(op.getX().getType());
  if (!aTy.getType().isF32() || !xTy.getType().isF32()) {
    op.emitError("metal.threadgroup_affine_prefix_scan buffers must have f32 "
                 "element type");
    _emitFailed = true;
    return;
  }
  if (aTy.getSize() < BLOCK || xTy.getSize() < BLOCK) {
    op.emitError("metal.threadgroup_affine_prefix_scan buffers must have at "
                 "least block elements");
    _emitFailed = true;
    return;
  }

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
  const std::string A = bufName(op.getA());
  const std::string X = bufName(op.getX());
  auto S = [](int64_t value) { return std::to_string(value); };
  const int64_t E = BLOCK / TPB;
  const bool REVERSE = op->hasAttr("reverse");
  auto &os = _output;

  os << "\n  // ---- metal.threadgroup_affine_prefix_scan"
     << (REVERSE ? " (reverse)" : "") << " ----\n";
  os << "  {\n";
  os << "    uint _aps_tid = id.x - tgid.x * " << S(TPB) << "u;\n";
  os << "    float _aps_carry_a = 1.0f;\n";
  os << "    float _aps_carry_x = 0.0f;\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "    for (uint _aps_k = 0u; _aps_k < " << S(E) << "u; ++_aps_k) {\n";
  if (REVERSE) {
    // Visit chunks from the last one down, and mirror each chunk IN PLACE so
    // the forward template below runs verbatim over the reversed sequence. The
    // affine combine is not commutative, so the element order — not just the
    // carry order — has to be the reversed one. Both halves of the swap are a
    // cross-thread read, hence the barrier on either side.
    os << "      uint _aps_base = " << S(BLOCK) << "u - (_aps_k + 1u) * "
       << S(TPB) << "u;\n";
    os << "      uint _aps_orig = _aps_base + " << S(TPB - 1)
       << "u - _aps_tid;\n";
    os << "      float _aps_mir_a = " << A << "[_aps_orig];\n";
    os << "      float _aps_mir_x = " << X << "[_aps_orig];\n";
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os << "      " << A << "[_aps_base + _aps_tid] = _aps_mir_a;\n";
    os << "      " << X << "[_aps_base + _aps_tid] = _aps_mir_x;\n";
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else {
    os << "      uint _aps_base = _aps_k * " << S(TPB) << "u;\n";
  }
  os << "      for (uint _aps_off = 1u; _aps_off < " << S(TPB)
     << "u; _aps_off <<= 1) {\n";
  os << "        float _aps_cur_a = " << A << "[_aps_base + _aps_tid];\n";
  os << "        float _aps_cur_x = " << X << "[_aps_base + _aps_tid];\n";
  os << "        float _aps_prev_a = 1.0f;\n";
  os << "        float _aps_prev_x = 0.0f;\n";
  os << "        if (_aps_tid >= _aps_off) {\n";
  os << "          _aps_prev_a = " << A
     << "[_aps_base + _aps_tid - _aps_off];\n";
  os << "          _aps_prev_x = " << X
     << "[_aps_base + _aps_tid - _aps_off];\n";
  os << "        }\n";
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "        if (_aps_tid >= _aps_off) {\n";
  os << "          " << A
     << "[_aps_base + _aps_tid] = _aps_prev_a * _aps_cur_a;\n";
  os << "          " << X
     << "[_aps_base + _aps_tid] = _aps_cur_a * _aps_prev_x + "
        "_aps_cur_x;\n";
  os << "        }\n";
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      }\n";
  os << "      float _aps_total_a = " << A << "[_aps_base + " << S(TPB - 1)
     << "u];\n";
  os << "      float _aps_total_x = " << X << "[_aps_base + " << S(TPB - 1)
     << "u];\n";
  os << "      float _aps_local_a = " << A << "[_aps_base + _aps_tid];\n";
  os << "      float _aps_local_x = " << X << "[_aps_base + _aps_tid];\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "      " << A
     << "[_aps_base + _aps_tid] = _aps_carry_a * _aps_local_a;\n";
  os << "      " << X
     << "[_aps_base + _aps_tid] = _aps_local_a * _aps_carry_x + "
        "_aps_local_x;\n";
  os << "      _aps_carry_x = _aps_total_a * _aps_carry_x + "
        "_aps_total_x;\n";
  os << "      _aps_carry_a *= _aps_total_a;\n";
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  if (REVERSE) {
    // Mirror the finished chunk back, so results land on their ORIGINAL logical
    // positions — the conversion's per-element placeholder reads by position.
    os << "      float _aps_out_a = " << A << "[_aps_base + _aps_tid];\n";
    os << "      float _aps_out_x = " << X << "[_aps_base + _aps_tid];\n";
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os << "      " << A << "[_aps_orig] = _aps_out_a;\n";
    os << "      " << X << "[_aps_orig] = _aps_out_x;\n";
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  }
  os << "    }\n";
  os << "  }\n";
}


// One fixed-size record appended to the trailing debug buffer.
//
// Word 0 of the buffer is the count; each record is six words:
//   [kind | dtype << 8, msg id, program id, lane, value bits, reserved]
// The slot comes from an atomic increment, so a full buffer DROPS records
// instead of overwriting — the host compares the count it reads against the
// cap and says how many it lost. `metal.debug_record`'s optional predicate is
// what makes an assert record only on failure.
void ModuleTranslation::translate(mlir::triton::metal::DebugRecordOp op) {
  const unsigned cap = 1024;
  if (op.getPred()) {
    _output << "if (";
    translateValueOrVarName(op.getPred());
    _output << ") {\n";
    indent();
  }
  unsigned slot = _varCount++;
  _output << "uint v" << slot
          << " = atomic_fetch_add_explicit((device atomic_uint*)"
             "&__triton_dbg[0], 1u, memory_order_relaxed);\n";
  indent();
  _output << "if (v" << slot << " < " << cap << "u) {\n";
  indent();
  unsigned base = _varCount++;
  _output << "uint v" << base << " = 1u + v" << slot << " * 6u;\n";
  indent();
  const unsigned header =
      static_cast<unsigned>(op.getKind()) |
      (static_cast<unsigned>(op.getDtypeTag()) << 8);
  _output << "__triton_dbg[v" << base << " + 0u] = " << header << "u;\n";
  indent();
  _output << "__triton_dbg[v" << base << " + 1u] = "
          << static_cast<unsigned>(op.getMsgId()) << "u;\n";
  indent();
  _output << "__triton_dbg[v" << base << " + 2u] = tgid.x;\n";
  indent();
  _output << "__triton_dbg[v" << base << " + 3u] = ";
  if (mlir::Value lane = op.getLane()) {
    // The ELEMENT this record belongs to, which is the threadgroup-local index
    // — `id.x` is global and would label every program's records with the
    // wrong element.
    _output << "uint32_t(";
    translateValueOrVarName(lane);
    _output << ")";
  } else {
    _output << "id.x";
  }
  _output << ";\n";
  indent();
  _output << "__triton_dbg[v" << base << " + 4u] = ";
  if (mlir::Value value = op.getValue()) {
    mlir::Type ty = value.getType();
    if (ty.isF32()) {
      _output << "as_type<uint32_t>(";
      translateValueOrVarName(value);
      _output << ")";
    } else if (ty.isF16() || ty.isBF16()) {
      _output << "uint32_t(as_type<ushort>(";
      translateValueOrVarName(value);
      _output << "))";
    } else {
      _output << "uint32_t(";
      translateValueOrVarName(value);
      _output << ")";
    }
  } else {
    _output << "0u";
  }
  _output << ";\n";
  indent();
  _output << "__triton_dbg[v" << base << " + 5u] = 0u;\n";
  indent();
  _output << "}\n";
  if (op.getPred()) {
    indent();
    _output << "}\n";
  }
  indent();
}

void ModuleTranslation::translate(mlir::triton::metal::AtomicRmwOp op) {
  // Atomic RMW into device memory with relaxed ordering. Reinterpret the
  // element address `&buf[idx]` as the matching `device atomic_*` pointer;
  // valid for relaxed atomics in the `device` address space on Apple GPUs.
  // 32-bit fetch operations materialize their old-value result when consumed;
  // this is required before a result-bearing scf.if can yield it. MSL's ulong
  // min/max functions return void, so the Metal op verifier requires their
  // result to be unused.
  //
  // Add picks its atomic type from the value. The signed/unsigned integer
  // kinds deliberately permit a 32-bit f32 buffer: this is the exact
  // ordered-bit expansion Triton's frontend emits for f32 atomic min/max.
  const bool resultUsed = !op.getResult().use_empty();
  auto valueInt = llvm::dyn_cast<mlir::IntegerType>(op.getValue().getType());
  const bool is64 = valueInt && valueInt.getWidth() == 64;

  // Half add has no atomic instruction; it goes through the preamble's
  // compare-exchange helper, which takes the BASE pointer and the element index
  // (it has to mask the index down to the containing 32-bit word itself).
  if (op.getKind() == mlir::triton::metal::AtomicRmwKind::Add &&
      (op.getValue().getType().isF16() || op.getValue().getType().isBF16())) {
    if (resultUsed) {
      unsigned idx = _varCount++;
      _buffers[op.getResult().getAsOpaquePointer()] = idx;
      _output << typeToString(op.getResult().getType()) << " v" << idx << " = ";
    }
    _output << (op.getValue().getType().isF16() ? "__triton_atomic_add_half("
                                                : "__triton_atomic_add_bfloat(");
    translateVarName(op.getMemref());
    _output << ", ";
    translateValueOrVarName(op.getIndex());
    _output << ", ";
    translateValueOrVarName(op.getValue());
    _output << ")";
    printDelim();
    return;
  }
  const char *function = "atomic_fetch_add_explicit";
  llvm::StringRef atomicTy = "atomic_float";
  switch (op.getKind()) {
  case mlir::triton::metal::AtomicRmwKind::Add: {
    if (valueInt)
      atomicTy = valueInt.isUnsigned() ? "atomic_uint" : "atomic_int";
    break;
  }
  case mlir::triton::metal::AtomicRmwKind::Max:
    function = "atomic_fetch_max_explicit";
    atomicTy = "atomic_int";
    break;
  case mlir::triton::metal::AtomicRmwKind::Min:
    function = "atomic_fetch_min_explicit";
    atomicTy = "atomic_int";
    break;
  case mlir::triton::metal::AtomicRmwKind::UMin:
    function = is64 ? "atomic_min_explicit" : "atomic_fetch_min_explicit";
    atomicTy = is64 ? "atomic_ulong" : "atomic_uint";
    break;
  case mlir::triton::metal::AtomicRmwKind::UMax:
    function = is64 ? "atomic_max_explicit" : "atomic_fetch_max_explicit";
    atomicTy = is64 ? "atomic_ulong" : "atomic_uint";
    break;
  // Bitwise and exchange take the value in its own type — no ordered-bit
  // reinterpretation, so no `as_type` wrapper. 32-bit only; the pass rejects
  // wider payloads because MSL has no atomic_ulong overload for them.
  case mlir::triton::metal::AtomicRmwKind::And:
    function = "atomic_fetch_and_explicit";
    atomicTy = valueInt && valueInt.isUnsigned() ? "atomic_uint" : "atomic_int";
    break;
  case mlir::triton::metal::AtomicRmwKind::Or:
    function = "atomic_fetch_or_explicit";
    atomicTy = valueInt && valueInt.isUnsigned() ? "atomic_uint" : "atomic_int";
    break;
  case mlir::triton::metal::AtomicRmwKind::Xor:
    function = "atomic_fetch_xor_explicit";
    atomicTy = valueInt && valueInt.isUnsigned() ? "atomic_uint" : "atomic_int";
    break;
  case mlir::triton::metal::AtomicRmwKind::Xchg:
    function = "atomic_exchange_explicit";
    if (valueInt)
      atomicTy = valueInt.isUnsigned() ? "atomic_uint" : "atomic_int";
    break;
  }
  if (resultUsed) {
    unsigned idx = _varCount++;
    _buffers[op.getResult().getAsOpaquePointer()] = idx;
    _output << typeToString(op.getResult().getType()) << " v" << idx << " = ";
  }
  _output << function << "((device " << atomicTy << "*)&";
  translateVarName(op.getMemref());
  _output << "[";
  translateValueOrVarName(op.getIndex());
  _output << "], ";
  // Only the min/max kinds reinterpret their payload: they implement Triton's
  // ordered-bit f32 expansion, where the value arrives as the bit pattern to
  // compare. Add, the bitwise kinds and exchange pass their value through.
  bool casted = true;
  if (op.getKind() == mlir::triton::metal::AtomicRmwKind::Max ||
      op.getKind() == mlir::triton::metal::AtomicRmwKind::Min)
    _output << "as_type<int32_t>(";
  else if (op.getKind() == mlir::triton::metal::AtomicRmwKind::UMin ||
           op.getKind() == mlir::triton::metal::AtomicRmwKind::UMax)
    _output << (is64 ? "as_type<ulong>(" : "as_type<uint32_t>(");
  else
    casted = false;
  translateValueOrVarName(op.getValue());
  if (casted)
    _output << ")";
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
      (singleIterArgTy.isF32() || singleIterArgTy.isF16() ||
       singleIterArgTy.isBF16() || singleIterArgTy.isInteger(32) ||
       singleIterArgTy.isInteger(64) || singleIterArgTy.isInteger(1))) {
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
               return v.getType().isF32() || v.getType().isF16() ||
                      v.getType().isBF16() || v.getType().isInteger(32) ||
                      v.getType().isInteger(64) || v.getType().isInteger(1);
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

// A Python `while` in a @triton.jit kernel lowers to
//
//   %r:N = scf.while (%a0 = %init0, ...) {
//            %c = <predicate over %a0..>
//            scf.condition(%c) %a0, ...        // forwards the block args
//          } do {
//          ^bb0(%b0, ...):
//            ...body...
//            scf.yield %next0, ...
//          }
//
// MSL has no multi-value loop carry, so each carried value becomes a temp
// declared before the loop and the whole thing emits as
//
//   T v0 = init0; ...
//   bool vC;
//   while (true) {
//     { ...before ops...  vC = <predicate>; }
//     if (!vC) break;
//     { ...after ops...   v0 = next0; ... }
//   }
//
// The two regions keep their own C scopes (that is just what `translate(Region)`
// emits) which is why `scf.condition` MUST forward the before-region block
// arguments verbatim: anything else would forward a value whose MSL temp is
// scoped inside the before block and therefore dead by the time the after block
// reads it. Every kernel shape Triton emits satisfies this; anything else is
// rejected loudly rather than mistranslated.
void ModuleTranslation::translate(mlir::scf::WhileOp op) {
  auto &beforeRegion = op.getBefore();
  auto &afterRegion = op.getAfter();
  auto condOp =
      mlir::dyn_cast<mlir::scf::ConditionOp>(beforeRegion.front().getTerminator());
  if (!condOp) {
    llvm::errs() << "[metal] scf.while: before region is not terminated by "
                    "scf.condition\n";
    llvm_unreachable("Unexpected operation");
  }
  bool forwardsBlockArgs =
      condOp.getArgs().size() == beforeRegion.getNumArguments();
  for (auto [i, forwarded] : llvm::enumerate(condOp.getArgs()))
    if (forwarded != beforeRegion.getArgument(i))
      forwardsBlockArgs = false;
  if (!forwardsBlockArgs) {
    llvm::errs() << "[metal] scf.while: scf.condition must forward the "
                    "before-region block arguments unchanged\n";
    llvm_unreachable("Unexpected operation");
  }

  // One MSL temp per carried value; before-arg, after-arg and loop result all
  // alias it.
  llvm::SmallVector<unsigned, 8> carried;
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    mlir::Value init = op.getOperand(i);
    unsigned idx = _varCount++;
    carried.push_back(idx);
    _buffers[beforeRegion.getArgument(i).getAsOpaquePointer()] = idx;
    _buffers[afterRegion.getArgument(i).getAsOpaquePointer()] = idx;
    _buffers[op.getResult(i).getAsOpaquePointer()] = idx;
    _output << typeToString(init.getType()) << " v" << idx << " = ";
    if (auto *initDef = init.getDefiningOp())
      translateValue(initDef);
    else
      translateVarName(init);
    _output << ";\n";
    indent();
  }
  _scfWhileCarried[op.getOperation()] = carried;

  unsigned condIdx = _varCount++;
  _scfWhileCond[op.getOperation()] = condIdx;
  _output << "bool v" << condIdx << " = false;\n";
  indent();
  _output << "while (true) ";
  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    translate(beforeRegion);
    _output << "\n";
    indent();
    _output << "if (!v" << condIdx << ") break;";
    _output << "\n";
    indent();
    translate(afterRegion);
  }
  _output << "\n";
  indent();
  _output << "}";
}

void ModuleTranslation::translate(mlir::scf::ConditionOp op) {
  auto parent = mlir::cast<mlir::scf::WhileOp>(op->getParentOp());
  auto it = _scfWhileCond.find(parent.getOperation());
  assert(it != _scfWhileCond.end() &&
         "scf.condition: parent scf.while has no predicate temp");
  _output << "v" << it->second << " = ";
  translateValueOrVarName(op.getCondition());
  _output << ";";
}

void ModuleTranslation::translate(mlir::scf::YieldOp op) {
  // scf::WhileOp parent: write every yielded value back to its carried temp.
  if (auto parentWhile =
          llvm::dyn_cast_or_null<mlir::scf::WhileOp>(op->getParentOp())) {
    auto it = _scfWhileCarried.find(parentWhile.getOperation());
    assert(it != _scfWhileCarried.end() &&
           "scf.yield: parent scf.while has no carried temps");
    for (auto [i, yielded] : llvm::enumerate(op.getOperands())) {
      if (i)
        _output << "\n", indent();
      _output << "v" << it->second[i] << " = ";
      translateValueOrVarName(yielded);
      _output << ";";
    }
    return;
  }
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
      // scf.yield updates all loop-carried values simultaneously. Materialize
      // every RHS before assigning any accumulator: online softmax yields
      // `(new_max, old_sum * exp(old_max - new_max) + block_sum)`, so assigning
      // new_max first would make the second expression observe the new value
      // and incorrectly turn its rescale factor into exp(0).
      llvm::SmallVector<unsigned, 8> rhsIdxs;
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        unsigned rhsIdx = _varCount++;
        rhsIdxs.push_back(rhsIdx);
        _output << typeToString(op.getOperand(i).getType()) << " v" << rhsIdx
                << " = ";
        translateValueOrVarName(op.getOperand(i));
        _output << "; ";
      }
      for (unsigned i = 0; i < rhsIdxs.size(); ++i)
        _output << "v" << mit->second[i] << " = v" << rhsIdxs[i] << "; ";
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

bool ModuleTranslation::bindFusedRegionArgs_(
    mlir::triton::metal::FusedAttentionOp op, mlir::Block &body,
    llvm::ArrayRef<std::string> fixedInits, llvm::StringRef ind) {
  auto &os = _output;
  // Bind each block arg to an MSL temp and register it in `_buffers`, which is
  // what `translateVarName` consults — so every use inside the region renders
  // as `v<idx>` with no special-casing in the value translators.
  auto bind = [&](mlir::Value arg, llvm::StringRef init) {
    unsigned idx = _varCount++;
    os << ind << typeToString(arg.getType()) << " v" << idx << " = " << init
       << ";\n";
    _buffers[arg.getAsOpaquePointer()] = idx;
  };
  for (auto [i, init] : llvm::enumerate(fixedInits))
    bind(body.getArgument(i), init);
  for (auto [i, p] : llvm::enumerate(op.getScoreParams())) {
    // `bufName`-equivalent walk; the operand is a kernel buffer, and the op
    // verifier already pinned the block-arg type to its element type.
    mlir::Value m = p;
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
      op.emitError() << "metal.fused_attention: score_param " << i
                     << " does not resolve to a kernel buffer (matcher bound a "
                        "non-kernel-arg value); refusing to emit";
      _emitFailed = true;
      return false;
    }
    bind(body.getArgument(fixedInits.size() + i),
         "v" + std::to_string(it->second) + "[0]");
  }
  return true;
}

// Both regions are pure scalar DAGs (masking is `arith.select`, not control
// flow), so IR order is a valid emission order below.

std::string ModuleTranslation::emitScoreRegion_(
    mlir::triton::metal::FusedAttentionOp op, llvm::StringRef scoreExpr,
    llvm::StringRef rowExpr, llvm::StringRef keyExpr,
    llvm::StringRef phaseExpr, llvm::StringRef ind) {
  mlir::Block &body = op.getScore().front();
  auto &os = _output;
  // Each key phase re-emits the region. Clear the previous pass's let-bindings
  // first: `translateValue` consults `_letBound`, so a stale entry makes this
  // phase's code reference the PREVIOUS phase's block-scoped temp -- which
  // compiles to "use of undeclared identifier" at best, and to reading another
  // phase's value at worst.
  for (mlir::Operation &o : body)
    _letBound.erase(&o);
  if (!bindFusedRegionArgs_(op, body,
                            {scoreExpr.str(), rowExpr.str(), keyExpr.str(),
                             phaseExpr.str()},
                            ind))
    return "0.0f";

  for (mlir::Operation &o : body) {
    if (mlir::isa<mlir::triton::metal::ScoreYieldOp>(o))
      continue;
    if (o.getNumResults() != 1) {
      op.emitError() << "metal.fused_attention: score region op '"
                     << o.getName() << "' does not produce exactly one result";
      _emitFailed = true;
      return "0.0f";
    }
    unsigned idx = _varCount++;
    os << ind << typeToString(o.getResult(0).getType()) << " v" << idx << " = ";
    translateValue(&o);
    os << ";\n";
    _letBound[&o] = idx;
  }

  auto yield = mlir::cast<mlir::triton::metal::ScoreYieldOp>(body.getTerminator());
  unsigned outIdx = _varCount++;
  os << ind << "float v" << outIdx << " = ";
  translateValueOrVarName(yield.getValue());
  os << ";\n";
  return "v" + std::to_string(outIdx);
}

std::pair<std::string, std::string> ModuleTranslation::emitKeyBoundsRegion_(
    mlir::triton::metal::FusedAttentionOp op, llvm::StringRef blkExpr,
    llvm::StringRef phaseExpr, llvm::StringRef mExpr, llvm::StringRef nExpr,
    llvm::StringRef ind) {
  mlir::Block &body = op.getKeyBounds().front();
  auto &os = _output;
  // Each key phase re-emits the region. Clear the previous pass's let-bindings
  // first: `translateValue` consults `_letBound`, so a stale entry makes this
  // phase's code reference the PREVIOUS phase's block-scoped temp -- which
  // compiles to "use of undeclared identifier" at best, and to reading another
  // phase's value at worst.
  for (mlir::Operation &o : body)
    _letBound.erase(&o);
  if (!bindFusedRegionArgs_(op, body,
                            {blkExpr.str(), phaseExpr.str(), mExpr.str(),
                             nExpr.str()},
                            ind))
    return {"0", "0"};

  for (mlir::Operation &o : body) {
    if (mlir::isa<mlir::triton::metal::KeyBoundsYieldOp>(o))
      continue;
    if (o.getNumResults() != 1) {
      op.emitError() << "metal.fused_attention: key_bounds region op '"
                     << o.getName() << "' does not produce exactly one result";
      _emitFailed = true;
      return {"0", "0"};
    }
    unsigned idx = _varCount++;
    os << ind << typeToString(o.getResult(0).getType()) << " v" << idx << " = ";
    translateValue(&o);
    os << ";\n";
    _letBound[&o] = idx;
  }

  auto yield = mlir::cast<mlir::triton::metal::KeyBoundsYieldOp>(
      body.getTerminator());
  // Floored at 0 on the way out: the region reproduces the source's own bound
  // arithmetic, which for a sliding window goes NEGATIVE
  // (`pid*bm - window + 1`) before its `max(..., num_sinks)`, and the loop
  // counter is unsigned, so a negative start would wrap to a colossal trip
  // count. There is deliberately no upper clamp against `n`: each phase's range
  // already carries the `key < X` bound of that phase's OWN K load, which is
  // what decides the keys the source reads. Clamping to a single global `n`
  // instead is what let an attention-sinks prologue's `key < num_sinks` bound
  // truncate the sliding-window phase.
  auto one = [&](mlir::Value v, const char *name) {
    unsigned idx = _varCount++;
    os << ind << "int _fa_r" << idx << " = ";
    translateValueOrVarName(v);
    os << ";\n";
    os << ind << "uint " << name << " = (uint)max(_fa_r" << idx << ", 0);\n";
    return std::string(name);
  };
  // Stable names rather than `v<n>`: these two bound the key sweep, and a
  // reader of the generated MSL should be able to find them. Each phase emits
  // inside its own block, so the names do not collide across phases.
  std::string beg = one(yield.getStart(), "_fa_kbeg");
  std::string end = one(yield.getEnd(), "_fa_kend");
  return {beg, end};
}

// Simdgroup body: both matmuls on the matrix unit, with the score transform
// coming from the op's region instead of the hard-coded scale+mask+softmax that
// a per-variant emitter would bake in. The staged S tile is exactly where a
// per-(row, key) transform belongs — the row and key indices are both in hand
// there — which is what makes one emitter able to serve every variant.
//
// Applies only when the working set fits threadgroup memory; `translate` falls
// back to the scalar body otherwise, so this is a fast path, never a gate.
void ModuleTranslation::emitFusedAttentionMma_(
    mlir::triton::metal::FusedAttentionOp op) {
  const int64_t BM = op.getBm(), BN = op.getBn(), BD = op.getBd();
  const bool softmax =
      op.getNorm() == mlir::triton::metal::AttnNorm::OnlineSoftmax;
  const bool featureTiled = op.getFeatureTiled();
  const int64_t SZ_Q = BM * BD, SZ_KTV = BD * BN, SZ_S = BM * BN;
  const int64_t mT = BM / 8, nT = BN / 8, dT = BD / 8;

  auto bufName = [&](mlir::Value m) -> std::string {
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
      op.emitError() << "metal.fused_attention: operand does not resolve to a "
                        "kernel buffer; refusing to emit";
      _emitFailed = true;
      return "<unresolved>";
    }
    return "v" + std::to_string(it->second);
  };
  const std::string Q = bufName(op.getQ()), K = bufName(op.getK()),
                    V = bufName(op.getV()), O = bufName(op.getOut());
  const std::string M = bufName(op.getM()) + "[0]";
  const std::string N = bufName(op.getN()) + "[0]";
  const std::string DH = bufName(op.getDHead()) + "[0]";
  const std::string SQ = bufName(op.getStrideQ()) + "[0]";
  const std::string SK = bufName(op.getStrideK()) + "[0]";
  const std::string SV = bufName(op.getStrideV()) + "[0]";
  const std::string SO = bufName(op.getStrideO()) + "[0]";
  auto headParams = op.getHeadParams();
  const bool independentHeads = !headParams.empty();
  std::string SQH, SKH, SVH, SOH, GROUPS;
  if (independentHeads) {
    SQH = bufName(headParams[0]) + "[0]";
    SKH = bufName(headParams[1]) + "[0]";
    SVH = bufName(headParams[2]) + "[0]";
    SOH = bufName(headParams[3]) + "[0]";
    GROUPS = bufName(headParams[4]) + "[0]";
  }
  auto S = [](int64_t x) { return std::to_string(x); };
  const char *E = op.getSoftmaxNaturalExp() ? "exp" : "exp2";

  auto &os = _output;
  os << "\n  // ---- metal.fused_attention (simdgroup dots, region score "
        "transform) ----\n";
  os << "  threadgroup float _fa_qbuf[" << S(SZ_Q) << "];\n";
  os << "  threadgroup float _fa_ktbuf[" << S(SZ_KTV) << "];\n";
  os << "  threadgroup float _fa_vbuf[" << S(SZ_KTV) << "];\n";
  os << "  threadgroup float _fa_sbuf[" << S(SZ_S) << "];\n";
  os << "  threadgroup float _fa_pbuf[" << S(SZ_S) << "];\n";
  os << "  threadgroup float _fa_obuf[" << S(SZ_Q) << "];\n";
  os << "  threadgroup float _fa_otbuf[" << S(SZ_Q) << "];\n";
  if (softmax) {
    os << "  threadgroup float _fa_rmax[" << S(BM) << "];\n";
    os << "  threadgroup float _fa_rsum[" << S(BM) << "];\n";
  }
  os << "  {\n";
  os << "  uint _fa_lane = ltid.x & 31u;\n";
  os << "  bool _fa_active = ltid.x < 32u;\n";
  os << "  uint _fa_M = " << M << ";\n";
  os << "  uint _fa_N = " << N << ";\n";
  // `d_head` holds the FULL feature width when a head split is present — a
  // head-split kernel computes `d_model / h` rather than being passed it, so
  // the quotient has no buffer to point at. Same contract as
  // the head-split kernel's own `d_head = d_model // h`.
  if (op.getH())
    os << "  uint _fa_dh = " << DH << " / " << bufName(op.getH()) << "[0];\n";
  else
    os << "  uint _fa_dh = " << DH << ";\n";
  os << "  uint _fa_sq = " << SQ << ";\n";
  os << "  uint _fa_sk = " << SK << ";\n";
  os << "  uint _fa_sv = " << SV << ";\n";
  os << "  uint _fa_so = " << SO << ";\n";
  if (independentHeads) {
    os << "  uint _fa_qhoff = tgid.y * " << SQH << ";\n";
    os << "  uint _fa_khoff = (tgid.y / " << GROUPS << ") * " << SKH
       << ";\n";
    os << "  uint _fa_vhoff = (tgid.y / " << GROUPS << ") * " << SVH
       << ";\n";
    os << "  uint _fa_ohoff = tgid.y * " << SOH << ";\n";
  }
  if (op.getH())
    os << "  uint _fa_col = tgid.y * _fa_dh;\n";
  else if (featureTiled)
    os << "  uint _fa_col = tgid.y * " << S(BD) << "u;\n";
  else
    os << "  uint _fa_col = 0u;\n";
  if (featureTiled)
    os << "  uint _fa_out_d = (_fa_col < _fa_dh) ? min(" << S(BD)
       << "u, _fa_dh - _fa_col) : 0u;\n";
  os << "  uint _fa_rowoff = tgid.x * " << S(BM) << "u;\n";
  // Stage Q and zero the accumulator / running state.
  os << "  if (_fa_active) {\n";
  os << "    for (uint c = _fa_lane; c < " << S(SZ_Q) << "u; c += 32u) {\n";
  os << "      uint q = c / " << S(BD) << "u; uint d = c % " << S(BD)
     << "u; uint row = _fa_rowoff + q;\n";
  if (!featureTiled)
    os << "      _fa_qbuf[c] = (row < _fa_M && d < _fa_dh) ? " << Q
       << "[" << (independentHeads ? "_fa_qhoff + " : "")
       << "row * _fa_sq + _fa_col + d] : 0.0f;\n";
  os << "      _fa_obuf[c] = 0.0f;\n";
  os << "    }\n";
  if (softmax)
    os << "    if (_fa_lane < " << S(BM)
       << "u) { _fa_rmax[_fa_lane] = -INFINITY; _fa_rsum[_fa_lane] = 0.0f; }\n";
  os << "  }\n";
  os << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  // The key range this program sweeps, computed by the op's key-bounds region.
  // A causal kernel's `min((pid+1)*bm, n)` is that region's yielded `end`.
  auto [_fa_kbeg, _fa_kend] =
      emitKeyBoundsRegion_(op, "(int)tgid.x", "0 /*phase*/", "(int)_fa_M",
                           "(int)_fa_N", "  ");
  os << "  for (uint kb = " << _fa_kbeg << "; kb < " << _fa_kend
     << "; kb += " << S(BN) << "u) {\n";
  if (featureTiled) {
    // V belongs to the y-selected output feature tile, while Q/K belong to a
    // separate full-d_head reduction domain. Clear S once per key block, then
    // accumulate one BD-wide Q/K chunk at a time so runtime d_head may exceed
    // the statically-sized threadgroup tiles.
    os << "    if (_fa_active) {\n";
    os << "      for (uint c = _fa_lane; c < " << S(SZ_KTV)
       << "u; c += 32u) {\n";
    os << "        uint key = c / " << S(BD) << "u; uint d = c % " << S(BD)
       << "u; uint kk = kb + key;\n";
    os << "        _fa_vbuf[c] = (kk < _fa_N && d < _fa_out_d) ? " << V
       << "[kk * _fa_sv + _fa_col + d] : 0.0f;\n";
    os << "      }\n";
    os << "    }\n";
    os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os << "    // ---- feature-tiled QK full-dhead sweep ----\n";
    os << "    for (uint _fa_dc = 0u; _fa_dc < _fa_dh; _fa_dc += " << S(BD)
       << "u) {\n";
    os << "      if (_fa_active) {\n";
    os << "        for (uint c = _fa_lane; c < " << S(SZ_Q)
       << "u; c += 32u) {\n";
    os << "          uint q = c / " << S(BD) << "u; uint d = c % " << S(BD)
       << "u; uint row = _fa_rowoff + q;\n";
    os << "          _fa_qbuf[c] = (row < _fa_M && _fa_dc + d < _fa_dh) ? "
       << Q << "[row * _fa_sq + _fa_dc + d] : 0.0f;\n";
    os << "        }\n";
    os << "        for (uint c = _fa_lane; c < " << S(SZ_KTV)
       << "u; c += 32u) {\n";
    os << "          uint d = c / " << S(BN) << "u; uint key = c % " << S(BN)
       << "u; uint kk = kb + key;\n";
    os << "          _fa_ktbuf[c] = (kk < _fa_N && _fa_dc + d < _fa_dh) ? "
       << K << "[(_fa_dc + d) * _fa_sk + kk] : 0.0f;\n";
    os << "        }\n";
    os << "      }\n";
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os << "      if (_fa_active) {\n";
    os << "        for (uint mi = 0; mi < " << S(mT) << "u; ++mi)\n";
    os << "        for (uint ni = 0; ni < " << S(nT) << "u; ++ni) {\n";
    // The zero constructor and in-place MAC are both required on Apple family
    // 9. Only later chunks reload the partial score tile.
    os << "          simdgroup_float8x8 acc(0.0f);\n";
    os << "          if (_fa_dc != 0u) simdgroup_load(acc, "
          "&_fa_sbuf[(mi*8u)*"
       << S(BN) << "u + ni*8u], " << S(BN) << ");\n";
    os << "          for (uint ki = 0; ki < " << S(dT) << "u; ++ki) {\n";
    os << "            simdgroup_float8x8 a, b;\n";
    os << "            simdgroup_load(a, &_fa_qbuf[(mi*8u)*" << S(BD)
       << "u + ki*8u], " << S(BD) << ");\n";
    os << "            simdgroup_load(b, &_fa_ktbuf[(ki*8u)*" << S(BN)
       << "u + ni*8u], " << S(BN) << ");\n";
    os << "            simdgroup_multiply_accumulate(acc, a, b, acc);\n";
    os << "          }\n";
    os << "          simdgroup_store(acc, &_fa_sbuf[(mi*8u)*" << S(BN)
       << "u + ni*8u], " << S(BN) << ");\n";
    os << "        }\n";
    os << "      }\n";
    os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os << "    }\n";
  } else {
    // Stage K^T and V for this key block.
    os << "    if (_fa_active) {\n";
    os << "      for (uint c = _fa_lane; c < " << S(SZ_KTV)
       << "u; c += 32u) {\n";
    os << "        uint d = c / " << S(BN) << "u; uint key = c % " << S(BN)
       << "u; uint kk = kb + key;\n";
    os << "        _fa_ktbuf[c] = (kk < _fa_N && d < _fa_dh) ? " << K
       << "[" << (independentHeads ? "_fa_khoff + " : "")
       << "kk * _fa_sk + _fa_col + d] : 0.0f;\n";
    os << "      }\n";
    os << "      for (uint c = _fa_lane; c < " << S(SZ_KTV)
       << "u; c += 32u) {\n";
    os << "        uint key = c / " << S(BD) << "u; uint d = c % " << S(BD)
       << "u; uint kk = kb + key;\n";
    os << "        _fa_vbuf[c] = (kk < _fa_N && d < _fa_dh) ? " << V
       << "[" << (independentHeads ? "_fa_vhoff + " : "")
       << "kk * _fa_sv + _fa_col + d] : 0.0f;\n";
    os << "      }\n";
    os << "    }\n";
    os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    // Dot 1: S = Q @ K^T, straight into the staged S tile.
    os << "    if (_fa_active) {\n";
    os << "      for (uint mi = 0; mi < " << S(mT) << "u; ++mi)\n";
    os << "      for (uint ni = 0; ni < " << S(nT) << "u; ++ni) {\n";
    os << "        simdgroup_float8x8 acc(0.0f);\n";
    os << "        for (uint ki = 0; ki < " << S(dT) << "u; ++ki) {\n";
    os << "          simdgroup_float8x8 a, b;\n";
    os << "          simdgroup_load(a, &_fa_qbuf[(mi*8u)*" << S(BD)
       << "u + ki*8u], " << S(BD) << ");\n";
    os << "          simdgroup_load(b, &_fa_ktbuf[(ki*8u)*" << S(BN)
       << "u + ni*8u], " << S(BN) << ");\n";
    os << "          simdgroup_multiply_accumulate(acc, a, b, acc);\n";
    os << "        }\n";
    os << "        simdgroup_store(acc, &_fa_sbuf[(mi*8u)*" << S(BN)
       << "u + ni*8u], " << S(BN) << ");\n";
    os << "      }\n";
    os << "    }\n";
    os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  }
  // The score transform, one query row per lane. Emitted ONCE, writing its
  // result back over the S tile, so the softmax passes below re-read it instead
  // of re-evaluating the region.
  os << "    if (_fa_active) {\n";
  os << "      uint q = _fa_lane; uint row = _fa_rowoff + q;\n";
  // `q < BM` FIRST: every per-row buffer is sized by BM and `row < M` does not
  // imply it when BM < 32.
  os << "      if (q < " << S(BM) << "u && row < _fa_M) {\n";
  os << "        for (uint kk = 0; kk < " << S(BN) << "u; ++kk) {\n";
  os << "          uint _fa_key = kb + kk;\n";
  os << "          if (_fa_key < " << _fa_kend << ") {\n";
  {
    std::string sc = "_fa_sbuf[q*" + S(BN) + "u + kk]";
    std::string w = emitScoreRegion_(op, sc, "(int)row", "(int)_fa_key",
                                     "0 /*phase*/", "            ");
    os << "            _fa_pbuf[q*" << S(BN) << "u + kk] = " << w << ";\n";
  }
  os << "          } else {\n";
  // Out-of-range keys must not contribute: a zero weight under norm=none, and a
  // logit of -inf (which exponentiates to zero) under the online softmax.
  os << "            _fa_pbuf[q*" << S(BN)
     << "u + kk] = " << (softmax ? "-INFINITY" : "0.0f") << ";\n";
  os << "          }\n";
  os << "        }\n";
  if (softmax) {
    os << "        float m_cur = -INFINITY;\n";
    os << "        for (uint kk = 0; kk < " << S(BN)
       << "u; ++kk) m_cur = max(m_cur, _fa_pbuf[q*" << S(BN) << "u + kk]);\n";
    os << "        float m_old = _fa_rmax[q];\n";
    os << "        float m_new = max(m_old, m_cur);\n";
    // Both guards keep -inf minus -inf (= NaN) out of the running state: the
    // first for a block where every key was masked, the second per key.
    os << "        float scaler = (m_old == m_new) ? 1.0f : " << E
       << "(m_old - m_new);\n";
    os << "        float denom = 0.0f;\n";
    os << "        for (uint kk = 0; kk < " << S(BN) << "u; ++kk) {\n";
    os << "          float l = _fa_pbuf[q*" << S(BN) << "u + kk];\n";
    os << "          float p = (l == -INFINITY || m_new == -INFINITY) ? 0.0f : "
       << E << "(l - m_new);\n";
    os << "          _fa_pbuf[q*" << S(BN) << "u + kk] = p; denom += p;\n";
    os << "        }\n";
    os << "        _fa_rsum[q] = _fa_rsum[q]*scaler + denom;\n";
    os << "        _fa_rmax[q] = m_new;\n";
    if (featureTiled)
      os << "        for (uint d = 0; d < _fa_out_d; ++d) _fa_obuf[q*"
         << S(BD) << "u + d] *= scaler;\n";
    else
      os << "        for (uint d = 0; d < " << S(BD) << "u; ++d) _fa_obuf[q*"
         << S(BD) << "u + d] *= scaler;\n";
  }
  os << "      } else if (q < " << S(BM) << "u) {\n";
  os << "        for (uint kk = 0; kk < " << S(BN) << "u; ++kk) _fa_pbuf[q*"
     << S(BN) << "u + kk] = 0.0f;\n";
  os << "      }\n";
  os << "    }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  // Dot 2: O_tile = P @ V.
  os << "    if (_fa_active) {\n";
  os << "      for (uint mi = 0; mi < " << S(mT) << "u; ++mi)\n";
  os << "      for (uint di = 0; di < " << S(dT) << "u; ++di) {\n";
  os << "        simdgroup_float8x8 acc(0.0f);\n";
  os << "        for (uint ki = 0; ki < " << S(nT) << "u; ++ki) {\n";
  os << "          simdgroup_float8x8 a, b;\n";
  os << "          simdgroup_load(a, &_fa_pbuf[(mi*8u)*" << S(BN)
     << "u + ki*8u], " << S(BN) << ");\n";
  os << "          simdgroup_load(b, &_fa_vbuf[(ki*8u)*" << S(BD)
     << "u + di*8u], " << S(BD) << ");\n";
  os << "          simdgroup_multiply_accumulate(acc, a, b, acc);\n";
  os << "        }\n";
  os << "        simdgroup_store(acc, &_fa_otbuf[(mi*8u)*" << S(BD)
     << "u + di*8u], " << S(BD) << ");\n";
  os << "      }\n";
  os << "    }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "    if (_fa_active) { for (uint c = _fa_lane; c < " << S(SZ_Q)
     << "u; c += 32u) _fa_obuf[c] += _fa_otbuf[c]; }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "  }\n";
  os << "  if (_fa_active) {\n";
  os << "    uint q = _fa_lane; uint row = _fa_rowoff + q;\n";
  os << "    if (q < " << S(BM) << "u && row < _fa_M) {\n";
  if (softmax) {
    os << "      float denom = _fa_rsum[q];\n";
    if (op.getSafeDenominatorOne())
      os << "      denom = (denom == 0.0f) ? 1.0f : denom;\n";
    if (featureTiled)
      os << "      for (uint d = 0; d < _fa_out_d; ++d)\n";
    else
      os << "      for (uint d = 0; d < _fa_dh; ++d)\n";
    os << "        " << O << "["
       << (independentHeads ? "_fa_ohoff + " : "")
       << "row * _fa_so + _fa_col + d] = _fa_obuf[q*"
       << S(BD) << "u + d] / denom;\n";
  } else {
    if (featureTiled)
      os << "      for (uint d = 0; d < _fa_out_d; ++d)\n";
    else
      os << "      for (uint d = 0; d < _fa_dh; ++d)\n";
    os << "        " << O << "["
       << (independentHeads ? "_fa_ohoff + " : "")
       << "row * _fa_so + _fa_col + d] = _fa_obuf[q*"
       << S(BD) << "u + d];\n";
  }
  os << "    }\n";
  os << "  }\n";
  os << "  }";
}

void ModuleTranslation::translate(mlir::triton::metal::FusedAttentionOp op) {
  // Two bodies, same op and same region — the dialect's existing
  // `MatmulKind::Scalar|Mma` split. The simdgroup body needs
  // `3*bm*bd + 2*bd*bn + 2*bm*bn (+ 2*bm)` floats of threadgroup memory against
  // Apple's 32 KiB; when that does not fit, the scalar body runs instead rather
  // than the op declining. That is the whole point of keeping the scalar tier:
  // a shape the fast path cannot hold still compiles and is still correct.
  //
  // Lifting the ceiling means blocking bm/bn/bd inside this body, which is
  // separate follow-up work — today a 64x64x64 tile (28800 floats) takes the
  // scalar path.
  const int64_t BM = op.getBm(), BN = op.getBn(), BD = op.getBd();
  const int64_t need =
      3 * BM * BD + 2 * BD * BN + 2 * BM * BN +
      (op.getNorm() == mlir::triton::metal::AttnNorm::OnlineSoftmax ? 2 * BM
                                                                    : 0);
  // 8192 floats == Apple's 32 KiB.
  // Safe to spend in full here because the op replaces the ENTIRE kernel body,
  // so nothing else in the kernel holds threadgroup memory.
  //
  // Several key phases take the scalar body too. The simdgroup body stages a
  // whole BN-wide key BLOCK per iteration, so a phase whose range is neither
  // block-aligned nor block-sized would need its own staging pass; the scalar
  // body walks keys one at a time and simply runs the phases back to back. That
  // is not a regression against the hand-written body this replaces, which was
  // per-key for the same reason.
  const bool fits = need <= 8192 && BM <= 32 && BM % 8 == 0 && BN % 8 == 0 &&
                    BD % 8 == 0 && op.getNumPhases() == 1;
  if (fits && !::getenv("TRITON_METAL_FUSED_ATTN_SCALAR"))
    return emitFusedAttentionMma_(op);
  return emitFusedAttentionScalar_(op);
}

void ModuleTranslation::emitFusedAttentionScalar_(
    mlir::triton::metal::FusedAttentionOp op) {
  // Correctness-first body: one query row per lane, keys walked one at a time,
  // the score kept in a REGISTER — which is what lets an arbitrary score region
  // be evaluated inline with no S/P staging and no cross-lane traffic.
  //
  // The query-row block is chunked so the threadgroup working set fits Apple's
  // 32 KiB rather than being capped by `bm` (the ceiling that forced
  // the predecessor op to decline `bm > 32` and `d_head > 64` outright).
  const int64_t BM = op.getBm(), BD = op.getBd();
  const bool softmax =
      op.getNorm() == mlir::triton::metal::AttnNorm::OnlineSoftmax;
  const bool featureTiled = op.getFeatureTiled();

  // 2 tiles of CH*BD floats (Q staging + O accumulator) plus 2*CH of running
  // state. Budget 7168 floats (28 KiB) leaves headroom for allocas the rest of
  // the kernel may hold. One query row per lane caps a chunk at 32.
  int64_t CH = 32;
  while (CH > 1 && 2 * CH * BD + 2 * CH > 7168)
    CH /= 2;
  if (CH > BM)
    CH = BM;
  const int64_t SZ = CH * BD;

  auto bufName = [&](mlir::Value m) -> std::string {
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
      // NEVER fall back to buffer 0: a wrong buffer is a silent wrong answer.
      op.emitError() << "metal.fused_attention: operand does not resolve to a "
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
  const std::string DH = bufName(op.getDHead()) + "[0]";
  const std::string SQ = bufName(op.getStrideQ()) + "[0]";
  const std::string SK = bufName(op.getStrideK()) + "[0]";
  const std::string SV = bufName(op.getStrideV()) + "[0]";
  const std::string SO = bufName(op.getStrideO()) + "[0]";
  auto headParams = op.getHeadParams();
  const bool independentHeads = !headParams.empty();
  std::string SQH, SKH, SVH, SOH, GROUPS;
  if (independentHeads) {
    SQH = bufName(headParams[0]) + "[0]";
    SKH = bufName(headParams[1]) + "[0]";
    SVH = bufName(headParams[2]) + "[0]";
    SOH = bufName(headParams[3]) + "[0]";
    GROUPS = bufName(headParams[4]) + "[0]";
  }
  auto S = [](int64_t x) { return std::to_string(x); };

  auto &os = _output;
  os << "\n  // ---- metal.fused_attention (per-key, region score transform) "
        "----\n";
  os << "  threadgroup float _fa_qbuf[" << S(SZ) << "];\n";
  os << "  threadgroup float _fa_obuf[" << S(SZ) << "];\n";
  if (softmax) {
    os << "  threadgroup float _fa_rmax[" << S(CH) << "];\n";
    os << "  threadgroup float _fa_rsum[" << S(CH) << "];\n";
  }
  os << "  {\n";
  os << "  uint _fa_lane = ltid.x & 31u;\n";
  os << "  bool _fa_active = ltid.x < 32u;\n";
  os << "  uint _fa_M = " << M << ";\n";
  os << "  uint _fa_N = " << N << ";\n";
  // `d_head` holds the FULL feature width when a head split is present — a
  // head-split kernel computes `d_model / h` rather than being passed it, so
  // the quotient has no buffer to point at. Same contract as
  // the head-split kernel's own `d_head = d_model // h`.
  if (op.getH())
    os << "  uint _fa_dh = " << DH << " / " << bufName(op.getH()) << "[0];\n";
  else
    os << "  uint _fa_dh = " << DH << ";\n";
  os << "  uint _fa_sq = " << SQ << ";\n";
  os << "  uint _fa_sk = " << SK << ";\n";
  os << "  uint _fa_sv = " << SV << ";\n";
  os << "  uint _fa_so = " << SO << ";\n";
  if (independentHeads) {
    os << "  uint _fa_qhoff = tgid.y * " << SQH << ";\n";
    os << "  uint _fa_khoff = (tgid.y / " << GROUPS << ") * " << SKH
       << ";\n";
    os << "  uint _fa_vhoff = (tgid.y / " << GROUPS << ") * " << SVH
       << ";\n";
    os << "  uint _fa_ohoff = tgid.y * " << SOH << ";\n";
  }
  // The y grid is either a head selector or an output-feature tile selector.
  if (op.getH())
    os << "  uint _fa_col = tgid.y * _fa_dh;\n";
  else if (featureTiled)
    os << "  uint _fa_col = tgid.y * " << S(BD) << "u;\n";
  else
    os << "  uint _fa_col = 0u;\n";
  if (featureTiled)
    os << "  uint _fa_out_d = (_fa_col < _fa_dh) ? min(" << S(BD)
       << "u, _fa_dh - _fa_col) : 0u;\n";
  else
    os << "  uint _fa_out_d = _fa_dh;\n";
  os << "  uint _fa_rowoff = tgid.x * " << S(BM) << "u;\n";
  // The chunk loop is OUTSIDE the single-warp guard so every warp reaches every
  // barrier; only the body is warp-0's.
  os << "  for (uint _fa_c0 = 0u; _fa_c0 < " << S(BM) << "u; _fa_c0 += "
     << S(CH) << "u) {\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "    if (_fa_active) {\n";
  os << "      for (uint c = _fa_lane; c < " << S(SZ) << "u; c += 32u) {\n";
  os << "        uint qq = c / " << S(BD) << "u; uint d = c % " << S(BD)
     << "u;\n";
  os << "        uint row = _fa_rowoff + _fa_c0 + qq;\n";
  os << "        _fa_qbuf[c] = (qq + _fa_c0 < " << S(BM)
     << "u && row < _fa_M && d < _fa_out_d) ? " << Q
     << "[" << (independentHeads ? "_fa_qhoff + " : "")
     << "row * _fa_sq + _fa_col + d] : 0.0f;\n";
  os << "        _fa_obuf[c] = 0.0f;\n";
  os << "      }\n";
  if (softmax) {
    os << "      if (_fa_lane < " << S(CH)
       << "u) { _fa_rmax[_fa_lane] = -INFINITY; _fa_rsum[_fa_lane] = 0.0f; }\n";
  }
  os << "    }\n";
  os << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  os << "    if (_fa_active) {\n";
  os << "      uint _fa_q = _fa_lane;\n";
  os << "      uint _fa_row = _fa_rowoff + _fa_c0 + _fa_q;\n";
  // `_fa_q < CH` FIRST: every per-row buffer is sized by CH, and `row < M` does
  // not imply it when CH < 32 (the FA emitter learned this the hard way).
  os << "      if (_fa_q < " << S(CH) << "u && _fa_c0 + _fa_q < " << S(BM)
     << "u && _fa_row < _fa_M) {\n";
  // One sweep per key phase, unrolled at emit time because `num_phases` is a
  // compile-time attribute -- which also lets `phase` reach both regions as a
  // literal, so a `select(phase == k, ...)` mask folds away in the Metal
  // compiler rather than costing a branch per key.
  //
  // The phases share the running (accumulator, sum, max) state and run in
  // source order. That is what makes them composable at all: an online softmax
  // merges any partition of the key set, so visiting a sink block and then a
  // sliding window gives the same answer as one sweep over their union -- while
  // REPRODUCING the source's key set instead of arguing about it.
  for (int64_t ph = 0; ph < op.getNumPhases(); ++ph) {
    const std::string P = std::to_string(ph);
    os << "        // --- key phase " << P << " ---\n";
    os << "        {\n";
    auto [kbeg, kend] = emitKeyBoundsRegion_(op, "(int)tgid.x", P + " /*phase*/",
                                             "(int)_fa_M", "(int)_fa_N",
                                             "        ");
  os << "        for (uint _fa_key = " << kbeg << "; _fa_key < " << kend
     << "; ++_fa_key) {\n";
  os << "          float _fa_a = 0.0f;\n";
  os << "          for (uint d = 0; d < _fa_dh; ++d)\n";
  if (featureTiled)
    os << "            _fa_a += " << Q << "[_fa_row * _fa_sq + d] * " << K
       << "[d * _fa_sk + _fa_key];\n";
  else
    os << "            _fa_a += _fa_qbuf[_fa_q * " << S(BD) << "u + d] * " << K
       << "[" << (independentHeads ? "_fa_khoff + " : "")
       << "_fa_key * _fa_sk + _fa_col + d];\n";
  // ---- the score transform, straight out of the op's region ----
  std::string w = emitScoreRegion_(op, "_fa_a", "(int)_fa_row", "(int)_fa_key",
                                   P + " /*phase*/", "          ");
  if (!softmax) {
    // norm = none: the transformed score IS the weight. The region is
    // responsible for zeroing keys that must not contribute, exactly as the
    // source kernel's mask does.
    os << "          for (uint d = 0; d < _fa_out_d; ++d)\n";
    os << "            _fa_obuf[_fa_q * " << S(BD) << "u + d] += " << w << " * "
       << V << "[" << (independentHeads ? "_fa_vhoff + " : "")
       << "_fa_key * _fa_sv + _fa_col + d];\n";
  } else {
    // norm = online_softmax: the transformed score is a logit. `(m_old ==
    // m_new) ? 1` keeps exp2(-inf - -inf) = NaN out of the running state on the
    // first visited key, where m_old is -INFINITY.
    // The exponential must be the one the source used: a kernel that folds
    // log2(e) into its scale uses exp2, one that scales by plain 1/sqrt(d) uses
    // exp, and the region has already produced the logit so the emitter cannot
    // rescale after the fact.
    const char *E = op.getSoftmaxNaturalExp() ? "exp" : "exp2";
    os << "          float _fa_mold = _fa_rmax[_fa_q];\n";
    os << "          float _fa_mnew = max(_fa_mold, " << w << ");\n";
    os << "          float _fa_sc = (_fa_mold == _fa_mnew) ? 1.0f : " << E
       << "(_fa_mold - _fa_mnew);\n";
    // A region that masks by yielding -inf reaches here with `w == -INFINITY`,
    // and on the first visited key `_fa_mnew` is -INFINITY too. `exp(-inf -
    // -inf)` is NaN, and one NaN poisons the whole row's accumulator -- so the
    // masked-out weight is forced to zero rather than computed.
    os << "          float _fa_p = (" << w << " == -INFINITY || _fa_mnew == "
          "-INFINITY) ? 0.0f : "
       << E << "(" << w << " - _fa_mnew);\n";
    os << "          _fa_rsum[_fa_q] = _fa_rsum[_fa_q] * _fa_sc + _fa_p;\n";
    os << "          _fa_rmax[_fa_q] = _fa_mnew;\n";
    os << "          for (uint d = 0; d < _fa_out_d; ++d)\n";
    os << "            _fa_obuf[_fa_q * " << S(BD) << "u + d] = _fa_obuf[_fa_q "
          "* " << S(BD) << "u + d] * _fa_sc + _fa_p * " << V
       << "[" << (independentHeads ? "_fa_vhoff + " : "")
       << "_fa_key * _fa_sv + _fa_col + d];\n";
  }
  os << "        }\n";
  os << "        }\n";
  }
  if (softmax) {
    os << "        float _fa_den = _fa_rsum[_fa_q];\n";
    if (op.getSafeDenominatorOne())
      os << "        _fa_den = (_fa_den == 0.0f) ? 1.0f : _fa_den;\n";
    os << "        for (uint d = 0; d < _fa_out_d; ++d)\n";
    os << "          " << O << "["
       << (independentHeads ? "_fa_ohoff + " : "")
       << "_fa_row * _fa_so + _fa_col + d] = "
       << "_fa_obuf[_fa_q * " << S(BD) << "u + d] / _fa_den;\n";
  } else {
    os << "        for (uint d = 0; d < _fa_out_d; ++d)\n";
    os << "          " << O << "["
       << (independentHeads ? "_fa_ohoff + " : "")
       << "_fa_row * _fa_so + _fa_col + d] = "
       << "_fa_obuf[_fa_q * " << S(BD) << "u + d];\n";
  }
  os << "      }\n";
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

void ModuleTranslation::translate(
    mlir::triton::metal::Int4WeightOnlyMatmulOp op) {
  auto bufName = [&](mlir::Value m) -> std::string {
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
      op.emitError() << "metal.int4_weight_only_matmul: operand does not "
                        "resolve to a kernel buffer";
      _emitFailed = true;
      return std::string("<unresolved>");
    }
    return "v" + std::to_string(it->second);
  };

  const std::string X = bufName(op.getX());
  const std::string WQ = bufName(op.getWq());
  const std::string SCALES = bufName(op.getScales());
  const std::string OUT = bufName(op.getOut());
  const std::string M = bufName(op.getM()) + "[0]";
  const std::string N = bufName(op.getN()) + "[0]";
  const std::string K = bufName(op.getK()) + "[0]";
  const std::string SXM = bufName(op.getStrideXm()) + "[0]";
  const std::string SWQN = bufName(op.getStrideWqn()) + "[0]";
  const std::string SSN = bufName(op.getStrideSn()) + "[0]";
  const std::string SYM = bufName(op.getStrideYm()) + "[0]";
  auto S = [](int64_t x) { return std::to_string(x); };

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "// ---- metal.int4_weight_only_matmul scalar fallback ----\n";
    indent();
    _output << "uint _i4_lane = ltid.x;\n";
    indent();
    _output << "uint _i4_M = " << M << ";\n";
    indent();
    _output << "uint _i4_N = " << N << ";\n";
    indent();
    _output << "uint _i4_K = " << K << ";\n";
    indent();
    _output << "uint _i4_sxm = " << SXM << ";\n";
    indent();
    _output << "uint _i4_swqn = " << SWQN << ";\n";
    indent();
    _output << "uint _i4_ssn = " << SSN << ";\n";
    indent();
    _output << "uint _i4_sym = " << SYM << ";\n";
    indent();
    _output << "uint _i4_row0 = tgid.x * " << S(op.getBm()) << "u;\n";
    indent();
    _output << "uint _i4_col0 = tgid.y * " << S(op.getBn()) << "u;\n";
    indent();
    _output << "for (uint _i4_e = _i4_lane; _i4_e < "
            << S(op.getBm() * op.getBn()) << "u; _i4_e += 128u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "uint _i4_rm = _i4_e / " << S(op.getBn()) << "u;\n";
      indent();
      _output << "uint _i4_cn = _i4_e % " << S(op.getBn()) << "u;\n";
      indent();
      _output << "uint _i4_m = _i4_row0 + _i4_rm;\n";
      indent();
      _output << "uint _i4_n = _i4_col0 + _i4_cn;\n";
      indent();
      _output << "if (_i4_m < _i4_M && _i4_n < _i4_N) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "float _i4_acc = 0.0f;\n";
        indent();
        _output << "for (uint _i4_kk = 0u; _i4_kk < _i4_K; ++_i4_kk) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "uchar _i4_pack = " << WQ
                  << "[_i4_n * _i4_swqn + (_i4_kk >> 1u)];\n";
          indent();
          _output << "uint _i4_q = ((_i4_kk & 1u) == 0u) ? "
                     "((uint(_i4_pack) >> 4u) & 15u) : "
                     "(uint(_i4_pack) & 15u);\n";
          indent();
          _output << "float _i4_scale = " << SCALES
                  << "[_i4_n * _i4_ssn + (_i4_kk / "
                  << S(op.getGroupSize()) << "u)];\n";
          indent();
          _output << "_i4_acc += float(" << X
                  << "[_i4_m * _i4_sxm + _i4_kk]) * "
                     "(float(int(_i4_q) - 8) * _i4_scale);\n";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << OUT << "[_i4_m * _i4_sym + _i4_n] = half(_i4_acc);";
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

void ModuleTranslation::translate(
    mlir::triton::metal::Int8QuantizedMatmulOp op) {
  auto bufName = [&](mlir::Value m) -> std::string {
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
      op.emitError() << "metal.int8_quantized_matmul: operand does not "
                        "resolve to a kernel buffer";
      _emitFailed = true;
      return std::string("<unresolved>");
    }
    return "v" + std::to_string(it->second);
  };

  const std::string A = bufName(op.getA());
  const std::string B = bufName(op.getB());
  const std::string OUT = bufName(op.getOut());
  const std::string M = bufName(op.getM()) + "[0]";
  const std::string N = bufName(op.getN()) + "[0]";
  const std::string K = bufName(op.getK()) + "[0]";
  const std::string SCALE_A = bufName(op.getScaleA()) + "[0]";
  const std::string SCALE_B = bufName(op.getScaleB()) + "[0]";
  const std::string SCALE_C = bufName(op.getScaleC()) + "[0]";
  const std::string ZERO_A = bufName(op.getZeroPointA()) + "[0]";
  const std::string ZERO_B = bufName(op.getZeroPointB()) + "[0]";
  const std::string ZERO_C = bufName(op.getZeroPointC()) + "[0]";
  auto S = [](int64_t x) { return std::to_string(x); };

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "// ---- metal.int8_quantized_matmul scalar fallback ----\n";
    indent();
    _output << "uint _i8_lane = ltid.x;\n";
    indent();
    _output << "uint _i8_M = " << M << ";\n";
    indent();
    _output << "uint _i8_N = " << N << ";\n";
    indent();
    _output << "uint _i8_K = " << K << ";\n";
    indent();
    _output << "float _i8_scale = " << SCALE_A << " * " << SCALE_B
            << " / " << SCALE_C << ";\n";
    indent();
    _output << "int _i8_zpa = as_type<int>(" << ZERO_A << ");\n";
    indent();
    _output << "int _i8_zpb = as_type<int>(" << ZERO_B << ");\n";
    indent();
    _output << "int _i8_zpc = as_type<int>(" << ZERO_C << ");\n";
    indent();
    _output << "uint _i8_row0 = tgid.x * " << S(op.getBm()) << "u;\n";
    indent();
    _output << "uint _i8_col0 = tgid.y * " << S(op.getBn()) << "u;\n";
    indent();
    _output << "for (uint _i8_e = _i8_lane; _i8_e < "
            << S(op.getBm() * op.getBn()) << "u; _i8_e += 128u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "uint _i8_m = _i8_row0 + _i8_e / " << S(op.getBn())
              << "u;\n";
      indent();
      _output << "uint _i8_n = _i8_col0 + _i8_e % " << S(op.getBn())
              << "u;\n";
      indent();
      _output << "if (_i8_m < _i8_M && _i8_n < _i8_N) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "int _i8_acc = 0;\n";
        indent();
        _output << "int _i8_sum_a = 0;\n";
        indent();
        _output << "int _i8_sum_b = 0;\n";
        indent();
        _output << "for (uint _i8_kk = 0u; _i8_kk < _i8_K; ++_i8_kk) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "int _i8_av = int(" << A
                  << "[_i8_m * _i8_K + _i8_kk]);\n";
          indent();
          _output << "int _i8_bv = int(" << B
                  << "[_i8_kk * _i8_N + _i8_n]);\n";
          indent();
          _output << "_i8_acc += _i8_av * _i8_bv;\n";
          indent();
          _output << "_i8_sum_a += _i8_av;\n";
          indent();
          _output << "_i8_sum_b += _i8_bv;";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "int _i8_corrected = _i8_acc - _i8_sum_a * _i8_zpb - "
                   "_i8_sum_b * _i8_zpa + int(_i8_K) * _i8_zpa * _i8_zpb;\n";
        indent();
        _output << "float _i8_q = floor(float(_i8_corrected) * _i8_scale + "
                   "0.5f) + float(_i8_zpc);\n";
        indent();
        _output << "_i8_q = clamp(_i8_q, -128.0f, 127.0f);\n";
        indent();
        _output << OUT << "[_i8_m * _i8_N + _i8_n] = int8_t(_i8_q);";
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

void ModuleTranslation::translate(
    mlir::triton::metal::LinearAttentionPreprocessOp op) {
  auto bufName = [&](mlir::Value value) -> std::string {
    for (;;) {
      while (auto cast =
                 value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (cast.getInputs().size() != 1)
          break;
        value = cast.getInputs()[0];
      }
      if (auto get =
              value.getDefiningOp<mlir::triton::metal::GetElementOp>()) {
        value = get.getMemref();
        continue;
      }
      break;
    }
    auto it = _buffers.find(value.getAsOpaquePointer());
    if (it == _buffers.end()) {
      op.emitError() << "metal.linear_attention_preprocess: operand does not "
                        "resolve to a kernel buffer";
      _emitFailed = true;
      return std::string("<unresolved>");
    }
    return "v" + std::to_string(it->second);
  };

  const std::string KT = bufName(op.getKt());
  const std::string V = bufName(op.getV());
  const std::string KV = bufName(op.getKv());
  const std::string KSUM = bufName(op.getKsum());
  const std::string M = bufName(op.getM()) + "[0]";
  const std::string D = bufName(op.getD()) + "[0]";

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "// ---- metal.linear_attention_preprocess scalar fallback ----\n";
    indent();
    _output << "uint _lap_lane = ltid.x;\n";
    indent();
    _output << "uint _lap_M = " << M << ";\n";
    indent();
    _output << "uint _lap_D = " << D << ";\n";
    indent();
    _output << "uint _lap_row_begin = tgid.x * 256u;\n";
    indent();
    _output << "uint _lap_row_end = min(_lap_row_begin + 256u, _lap_M);\n";
    indent();
    _output << "uint _lap_matrix_elems = _lap_D * _lap_D;\n";
    indent();
    _output << "for (uint _lap_e = _lap_lane; _lap_e < "
               "_lap_matrix_elems + _lap_D; _lap_e += 512u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "if (_lap_e < _lap_matrix_elems) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint _lap_k = _lap_e / _lap_D;\n";
        indent();
        _output << "uint _lap_col = _lap_e % _lap_D;\n";
        indent();
        _output << "float _lap_acc = 0.0f;\n";
        indent();
        _output << "for (uint _lap_row = _lap_row_begin; "
                   "_lap_row < _lap_row_end; ++_lap_row) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "float _lap_x = " << KT
                  << "[_lap_k * _lap_M + _lap_row];\n";
          indent();
          _output << "float _lap_phi = (_lap_x > 0.0f) ? "
                     "(_lap_x + 1.0f) : exp(_lap_x);\n";
          indent();
          _output << "_lap_acc += _lap_phi * " << V
                  << "[_lap_row * _lap_D + _lap_col];";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "atomic_fetch_add_explicit((device atomic_float*)&" << KV
                << "[_lap_e], _lap_acc, memory_order_relaxed);";
      }
      _output << "\n";
      indent();
      _output << "} else {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "uint _lap_k = _lap_e - _lap_matrix_elems;\n";
        indent();
        _output << "float _lap_acc = 0.0f;\n";
        indent();
        _output << "for (uint _lap_row = _lap_row_begin; "
                   "_lap_row < _lap_row_end; ++_lap_row) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "float _lap_x = " << KT
                  << "[_lap_k * _lap_M + _lap_row];\n";
          indent();
          _output << "_lap_acc += (_lap_x > 0.0f) ? "
                     "(_lap_x + 1.0f) : exp(_lap_x);";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << "atomic_fetch_add_explicit((device atomic_float*)&"
                << KSUM
                << "[_lap_k], _lap_acc, memory_order_relaxed);";
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

void ModuleTranslation::translate(
    mlir::triton::metal::LinearAttentionApplyOp op) {
  auto bufName = [&](mlir::Value value) -> std::string {
    for (;;) {
      while (auto cast =
                 value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (cast.getInputs().size() != 1)
          break;
        value = cast.getInputs()[0];
      }
      if (auto get =
              value.getDefiningOp<mlir::triton::metal::GetElementOp>()) {
        value = get.getMemref();
        continue;
      }
      break;
    }
    auto it = _buffers.find(value.getAsOpaquePointer());
    if (it == _buffers.end()) {
      op.emitError() << "metal.linear_attention_apply: operand does not "
                        "resolve to a kernel buffer";
      _emitFailed = true;
      return std::string("<unresolved>");
    }
    return "v" + std::to_string(it->second);
  };

  const std::string Q = bufName(op.getQ());
  const std::string KV = bufName(op.getKv());
  const std::string KSUM = bufName(op.getKsum());
  const std::string OUT = bufName(op.getOut());
  const std::string M = bufName(op.getM()) + "[0]";
  const std::string D = bufName(op.getD()) + "[0]";

  _output << "{";
  {
    INDENT();
    _output << "\n";
    indent();
    _output << "// ---- metal.linear_attention_apply scalar fallback ----\n";
    indent();
    _output << "uint _laa_lane = ltid.x;\n";
    indent();
    _output << "uint _laa_M = " << M << ";\n";
    indent();
    _output << "uint _laa_D = " << D << ";\n";
    indent();
    _output << "uint _laa_row0 = tgid.y * 64u;\n";
    indent();
    _output << "uint _laa_col0 = tgid.x * 64u;\n";
    indent();
    _output << "for (uint _laa_e = _laa_lane; _laa_e < 4096u; "
               "_laa_e += 256u) {";
    {
      INDENT();
      _output << "\n";
      indent();
      _output << "uint _laa_row = _laa_row0 + _laa_e / 64u;\n";
      indent();
      _output << "uint _laa_col = _laa_col0 + _laa_e % 64u;\n";
      indent();
      _output << "if (_laa_row < _laa_M && _laa_col < _laa_D) {";
      {
        INDENT();
        _output << "\n";
        indent();
        _output << "float _laa_numer = 0.0f;\n";
        indent();
        _output << "float _laa_denom = 1.0e-5f;\n";
        indent();
        _output << "for (uint _laa_k = 0u; _laa_k < _laa_D; ++_laa_k) {";
        {
          INDENT();
          _output << "\n";
          indent();
          _output << "float _laa_x = " << Q
                  << "[_laa_row * _laa_D + _laa_k];\n";
          indent();
          _output << "float _laa_phi = (_laa_x > 0.0f) ? "
                     "(_laa_x + 1.0f) : exp(_laa_x);\n";
          indent();
          _output << "_laa_numer += _laa_phi * " << KV
                  << "[_laa_k * _laa_D + _laa_col];\n";
          indent();
          _output << "_laa_denom += _laa_phi * " << KSUM << "[_laa_k];";
        }
        _output << "\n";
        indent();
        _output << "}\n";
        indent();
        _output << OUT << "[_laa_row * _laa_D + _laa_col] = "
                << "_laa_numer / _laa_denom;";
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
    // Read-after-overwrite guard. A single-use `metal.get_element` is inlined
    // at its USE, which is a different program point from its IR position: if
    // anything writes that buffer in between, the inlined read observes the NEW
    // contents. `_gae_carry_kernel` in
    // `leet-triton/medium-parallel_reverse_scan_gae.py` is the minimal case —
    // `local = work[b]; work[b] = carry; carry = local + coeff*carry` emitted
    // the load AFTER the store, so `local` read back the carry just written and
    // every block carry came out 0, with no diagnostic anywhere.
    //
    // A later non-Pure op holding the same memref is taken as a possible writer
    // (metal.store, atomic_rmw/cas, tg_store_indexed, the threadgroup scan
    // primitives). Reads are Pure, so this never trips on another read.
    // Deliberately index-blind: the store's index is usually a DIFFERENT SSA
    // value computing the same address (the conversion re-derives address
    // arithmetic per use), so comparing indices would silently miss the very
    // case this guards. Aliasing between two distinct memrefs is out of scope
    // here, as everywhere else in this backend.
    llvm::DenseMap<void *, int> lastMemRefWriter;
    {
      int writerIdx = 0;
      for (auto &op : region.getOps()) {
        op.walk([&](mlir::Operation *nested) {
          if (mlir::isMemoryEffectFree(nested))
            return;
          for (mlir::Value operand : nested->getOperands())
            if (llvm::isa<mlir::triton::metal::MetalMemRefType>(
                    operand.getType()))
              lastMemRefWriter[operand.getAsOpaquePointer()] = writerIdx;
        });
        ++writerIdx;
      }
    }
    int opIdx = -1;
    for (auto &op : region.getOps()) {
      ++opIdx;
      bool readOverwrittenLater = false;
      if (auto getEl =
              llvm::dyn_cast<mlir::triton::metal::GetElementOp>(&op)) {
        // `metal.alloca` is exempt. The TritonGPU pipeline never emits one; it
        // belongs to the linalg→metal path, which uses a 1-element private
        // alloca as a MUTABLE VARIABLE cell and reads it with `get_element`
        // placed once, ahead of the stores that update it. There, re-evaluating
        // the read at each use is the intended semantics, not a hazard.
        bool isPrivateVar =
            getEl.getMemref()
                .getDefiningOp<mlir::triton::metal::AllocaOp>() != nullptr;
        auto it =
            lastMemRefWriter.find(getEl.getMemref().getAsOpaquePointer());
        readOverwrittenLater = !isPrivateVar &&
                               it != lastMemRefWriter.end() &&
                               it->second > opIdx;
      }
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
      // `readOverwrittenLater` pins a device/threadgroup read the same way when
      // a later op in this region overwrites the buffer it reads.
      if (op.getNumResults() == 1 && !op.getResult(0).use_empty() &&
          (!op.getResult(0).hasOneUse() || op.hasAttr("metal.materialize") ||
           readOverwrittenLater) &&
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
                    mlir::math::FloorOp,
                    mlir::math::SqrtOp, mlir::math::LogOp, mlir::math::SinOp,
                    mlir::math::CosOp, mlir::math::ErfOp, mlir::math::RsqrtOp,
                    mlir::triton::metal::BinaryExpOp,
                    mlir::triton::metal::UnaryExpOp,
                    mlir::triton::metal::FmaOp,
                    mlir::triton::metal::ClampFOp,
                    mlir::triton::metal::MathIntrinsicOp,
                    mlir::triton::metal::Fp8ConvertOp,
                    mlir::triton::metal::MulHiUIOp,
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
  // Emit an operand with the signedness the CONSUMING OP asks for (see
  // `signednessCast`). Routes through translateValueOrVarName so a block
  // argument or one result of a multi-result op still resolves.
  auto emitAs = [&](mlir::Value v, bool wantSigned) {
    llvm::StringRef cast = signednessCast(v.getType(), wantSigned);
    _output << cast << "(";
    translateValueOrVarName(v);
    _output << ")";
  };
  llvm::TypeSwitch<Operation *>(opInst)
      .Case<mlir::triton::metal::ConstantOp, mlir::triton::metal::GetElementOp,
            mlir::triton::metal::TgLoadIndexedOp,
            mlir::triton::metal::ThreadIdOp,
            mlir::triton::metal::ThreadgroupIdOp,
            mlir::triton::metal::ThreadgroupsPerGridOp,
            mlir::triton::metal::CastOp, mlir::triton::metal::BitcastOp,
            mlir::triton::metal::UnaryExpOp, mlir::triton::metal::BinaryExpOp,
            mlir::triton::metal::FmaOp, mlir::triton::metal::ClampFOp,
            mlir::triton::metal::MathIntrinsicOp,
            mlir::triton::metal::Fp8ConvertOp,
            mlir::triton::metal::MulHiUIOp,
            mlir::triton::metal::YieldWhileOp>([&](auto &op) { translate(op); })
      .Case<mlir::arith::CmpIOp>([&](mlir::arith::CmpIOp op) {
        // Emit `(lhs <pred> rhs)` matching arith.cmpi semantics. Only the
        // predicates needed by the masked vector-add path are wired; extend
        // as new fixtures land.
        using P = mlir::arith::CmpIPredicate;
        const char *opStr = nullptr;
        // MSL spells the signed and unsigned relations identically, so the
        // predicate's signedness has to reach the operands as a cast — see
        // `signednessCast`. eq/ne compare bit patterns and need neither.
        bool wantSigned = true;
        switch (op.getPredicate()) {
        case P::eq:  opStr = " == "; break;
        case P::ne:  opStr = " != "; break;
        case P::slt: opStr = " < ";  break;
        case P::sle: opStr = " <= "; break;
        case P::sgt: opStr = " > ";  break;
        case P::sge: opStr = " >= "; break;
        case P::ult: opStr = " < ";  wantSigned = false; break;
        case P::ule: opStr = " <= "; wantSigned = false; break;
        case P::ugt: opStr = " > ";  wantSigned = false; break;
        case P::uge: opStr = " >= "; wantSigned = false; break;
        }
        // Operands may be block arguments (e.g. an scf.for induction var fed
        // straight into a comparison by the computed-cone reduce evaluator),
        // so route through translateValueOrVarName, not translateValue.
        _output << "(";
        emitAs(op.getLhs(), wantSigned);
        _output << opStr;
        emitAs(op.getRhs(), wantSigned);
        _output << ")";
      })
      .Case<mlir::arith::CmpFOp>([&](mlir::arith::CmpFOp op) {
        // Emit an exact scalar implementation of every arith.cmpf predicate.
        // Plain MSL relations already implement ordered comparisons (except
        // ONE) and unordered not-equal. The remaining predicates need
        // explicit isnan terms to preserve MLIR's NaN semantics.
        using P = mlir::arith::CmpFPredicate;
        auto emitRelation = [&](llvm::StringRef relation) {
          _output << "(";
          translateValueOrVarName(op.getLhs());
          _output << relation;
          translateValueOrVarName(op.getRhs());
          _output << ")";
        };
        auto emitIsNan = [&](mlir::Value value, bool negate) {
          if (negate)
            _output << "!";
          _output << "metal::isnan(";
          translateValueOrVarName(value);
          _output << ")";
        };
        auto emitUnorderedRelation = [&](llvm::StringRef relation) {
          _output << "(";
          emitIsNan(op.getLhs(), false);
          _output << " || ";
          emitIsNan(op.getRhs(), false);
          _output << " || ";
          translateValueOrVarName(op.getLhs());
          _output << relation;
          translateValueOrVarName(op.getRhs());
          _output << ")";
        };

        switch (op.getPredicate()) {
        case P::OEQ: emitRelation(" == "); break;
        case P::OGT: emitRelation(" > "); break;
        case P::OGE: emitRelation(" >= "); break;
        case P::OLT: emitRelation(" < "); break;
        case P::OLE: emitRelation(" <= "); break;
        case P::ONE:
          _output << "(";
          emitIsNan(op.getLhs(), true);
          _output << " && ";
          emitIsNan(op.getRhs(), true);
          _output << " && ";
          translateValueOrVarName(op.getLhs());
          _output << " != ";
          translateValueOrVarName(op.getRhs());
          _output << ")";
          break;
        case P::UEQ: emitUnorderedRelation(" == "); break;
        case P::UGT: emitUnorderedRelation(" > "); break;
        case P::UGE: emitUnorderedRelation(" >= "); break;
        case P::ULT: emitUnorderedRelation(" < "); break;
        case P::ULE: emitUnorderedRelation(" <= "); break;
        // MSL `!=` is true if either operand is NaN, exactly matching UNE.
        case P::UNE: emitRelation(" != "); break;
        case P::ORD:
          _output << "(";
          emitIsNan(op.getLhs(), true);
          _output << " && ";
          emitIsNan(op.getRhs(), true);
          _output << ")";
          break;
        case P::UNO:
          _output << "(";
          emitIsNan(op.getLhs(), false);
          _output << " || ";
          emitIsNan(op.getRhs(), false);
          _output << ")";
          break;
        case P::AlwaysTrue: _output << "true"; break;
        case P::AlwaysFalse: _output << "false"; break;
        }
      })
      .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp op) {
        // arith.constant (signless integer or float) survives in the masked
        // path as the upper-bound N of the comparison. Emit as a literal.
        if (auto v = llvm::dyn_cast<IntegerAttr>(op.getValue())) {
          // A one-bit APInt prints as -1 for `true`, and -1 is not `true` in
          // arithmetic: `cond ^ -1` is nonzero for BOTH values of cond, so a
          // negated predicate came out always-true. `tl.device_assert` fired on
          // every passing kernel because of it.
          if (v.getType().isInteger(1))
            _output << (v.getValue().isZero() ? "false" : "true");
          else
            _output << v.getValue();
        } else if (auto v = llvm::dyn_cast<FloatAttr>(op.getValue()))
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
            // A cast may consume one specific result of a multi-result scf.for
            // (the i32 top-k index carried by the MoE selection loop). Resolve
            // the Value first so the per-result `_buffers` mapping selects the
            // right accumulator instead of trying to render the whole scf.for
            // as an expression.
            translateValueOrVarName(op.getInputs()[0]);
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
      .Case<mlir::arith::MaxNumFOp>([&](mlir::arith::MaxNumFOp op) {
        // Scalar tl.maximum can survive conversion when it combines an
        // scf.for iter_arg with a reduced tile (online softmax). Tensor forms
        // lower to metal.binary_exp; spell the scalar form directly in MSL.
        _output << "metal::max(";
        translateValueOrVarName(op.getLhs());
        _output << ", ";
        translateValueOrVarName(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::MaximumFOp>([&](mlir::arith::MaximumFOp op) {
        _output << "metal::max(";
        translateValueOrVarName(op.getLhs());
        _output << ", ";
        translateValueOrVarName(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::MinNumFOp>([&](mlir::arith::MinNumFOp op) {
        _output << "metal::min(";
        translateValueOrVarName(op.getLhs());
        _output << ", ";
        translateValueOrVarName(op.getRhs());
        _output << ")";
      })
      .Case<mlir::arith::MinimumFOp>([&](mlir::arith::MinimumFOp op) {
        _output << "metal::min(";
        translateValueOrVarName(op.getLhs());
        _output << ", ";
        translateValueOrVarName(op.getRhs());
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
      .Case<mlir::math::FloorOp>([&](mlir::math::FloorOp op) {
        // Scalar tl.floor survives TritonGPU-to-Metal conversion unchanged.
        // This occurs when a runtime scalar controls index arithmetic, such as
        // gaussian blur's `floor(kernel_size / 2)`. MSL provides an exact
        // floor overload in the metal namespace.
        _output << "metal::floor(";
        translateValueOrVarName(op.getOperand());
        _output << ")";
      })
      .Case<mlir::math::CeilOp>([&](mlir::math::CeilOp op) {
        // Scalar tl.ceil survives conversion for runtime loop bounds, e.g.
        // ceil(N / BLOCK_SIZE) in chunked kernels.
        _output << "metal::ceil(";
        translateValueOrVarName(op.getOperand());
        _output << ")";
      })
      .Case<mlir::arith::DivSIOp>([&](mlir::arith::DivSIOp op) {
        // Used by 2D MakeRangeLowering: `row = idx / BLOCK_N`. MSL integer
        // division follows C semantics, which matches arith.divsi ONLY when
        // both operands are read as signed — a signless device buffer is
        // declared `uint32_t`, so the cast is what makes `x // 2` truncate
        // toward zero instead of dividing 4294967295 by 2.
        _output << "(";
        emitAs(op.getLhs(), /*wantSigned=*/true);
        _output << " / ";
        emitAs(op.getRhs(), /*wantSigned=*/true);
        _output << ")";
      })
      .Case<mlir::arith::RemSIOp>([&](mlir::arith::RemSIOp op) {
        // Used by 2D MakeRangeLowering: `col = idx % BLOCK_N`.
        _output << "(";
        emitAs(op.getLhs(), /*wantSigned=*/true);
        _output << " % ";
        emitAs(op.getRhs(), /*wantSigned=*/true);
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
      // Signed/unsigned semantics come from the OP, spelled onto the operands
      // by `emitAs` — NOT from the operand's own C type, which is `uint32_t`
      // for any signless device buffer.
      // See `.omc/specs/deep-interview-leet-triton-l2-int-arith-broad.md`.
      .Case<mlir::arith::ShRSIOp>([&](mlir::arith::ShRSIOp op) {
        // Arithmetic (sign-propagating) shift: MSL gives one only if the
        // shifted operand is signed.
        _output << "(";
        emitAs(op.getLhs(), /*wantSigned=*/true);
        _output << " >> ";
        emitAs(op.getRhs(), /*wantSigned=*/true);
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
        // Unsigned by construction — spelled explicitly so the result does not
        // depend on how the operand's buffer happens to be declared.
        _output << "(";
        emitAs(op.getLhs(), /*wantSigned=*/false);
        _output << " / ";
        emitAs(op.getRhs(), /*wantSigned=*/false);
        _output << ")";
      })
      .Case<mlir::arith::RemUIOp>([&](mlir::arith::RemUIOp op) {
        _output << "(";
        emitAs(op.getLhs(), /*wantSigned=*/false);
        _output << " % ";
        emitAs(op.getRhs(), /*wantSigned=*/false);
        _output << ")";
      })
      .Case<mlir::arith::MinSIOp, mlir::arith::MinUIOp, mlir::arith::MaxSIOp,
            mlir::arith::MaxUIOp>([&](Operation *op) {
        // MSL `min`/`max` (used by e.g. tl.swizzle2d's group-size clamp). Cast
        // both operands to the op's signedness so the overload isn't ambiguous
        // when one operand is `tgid.x` (uint) and the other signed arithmetic.
        // Cast at the operand's own WIDTH — the old hardcoded `(int)`/`(uint)`
        // truncated an i64 min/max to 32 bits.
        bool isMin =
            mlir::isa<mlir::arith::MinSIOp, mlir::arith::MinUIOp>(op);
        bool isSigned =
            mlir::isa<mlir::arith::MinSIOp, mlir::arith::MaxSIOp>(op);
        _output << (isMin ? "min(" : "max(");
        emitAs(op->getOperand(0), isSigned);
        _output << ", ";
        emitAs(op->getOperand(1), isSigned);
        _output << ")";
      })
      .Case<mlir::arith::ShRUIOp>([&](mlir::arith::ShRUIOp op) {
        // Logical (zero-filling) shift — requires an unsigned left operand.
        _output << "(";
        emitAs(op.getLhs(), /*wantSigned=*/false);
        _output << " >> ";
        emitAs(op.getRhs(), /*wantSigned=*/false);
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
        //
        // The result cast alone is not enough: for sitofp/uitofp the INPUT's
        // signedness decides which number the integer bits denote, and for
        // extsi/extui it decides sign- vs zero-extension. A signless device
        // buffer is declared `uint32_t`, so an uncast input made
        // `x.to(tl.float32)` on a negative i32 produce 2^32 instead of -1.
        // The float-input conversions (fptosi/fptoui/extf/truncf) and trunci
        // (which only keeps low bits) are unaffected.
        mlir::Operation *conv = op.getOperation();
        bool wantSigned =
            mlir::isa<mlir::arith::SIToFPOp, mlir::arith::ExtSIOp>(conv);
        bool inputSignednessMatters =
            wantSigned ||
            mlir::isa<mlir::arith::UIToFPOp, mlir::arith::ExtUIOp>(conv);
        _output << "(" << typeToString(op.getType()) << ")(";
        if (inputSignednessMatters)
          emitAs(op.getIn(), wantSigned);
        else
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
  // Wall 15 dispatch: the argument may be a BlockArgument with no defining op —
  // an scf.for iter_arg, or a `metal.fused_attention` score-region arg, where
  // `float(%row)` and `float(%param)` are both ordinary. `translateValue` would
  // null-deref on those.
  translateValueOrVarName(op.getArgument());
  _output << ")";
}

void ModuleTranslation::translate(mlir::triton::metal::BitcastOp op) {
  _output << "as_type<" << typeToString(op.getType()) << ">(";
  // Integer arithmetic narrower than `int` is promoted by C/MSL expression
  // rules.  Reassert the MLIR source type at the as_type boundary so a ui16
  // shift cannot become a 32-bit argument to as_type<half>.
  _output << typeToString(op.getArgument().getType()) << "(";
  translateValueOrVarName(op.getArgument());
  _output << "))";
}

void ModuleTranslation::translate(mlir::triton::metal::UnaryExpOp op) {
  // Wall 15 dispatch: the argument may be a BlockArgument (an scf.for iter_arg,
  // or a `metal.fused_attention` score-region arg such as `exp2(%score)`), in
  // which case there is no defining op and `translateValue` would null-deref.
  // `metal.binary_exp` already routes through this helper.
  auto translateArgument = [&] { translateValueOrVarName(op.getArgument()); };

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
  case OP::exp2Op:
    // `tl.exp2` is the base-2 exponential every attention kernel in this tree
    // uses for its softmax (the scale carries log2(e)), and the decay mask in
    // medium-decaying_causal_attention.py relies on exp2(-inf) == 0. Rewriting
    // it as exp(x * ln2) would perturb both, so it maps to MSL `exp2` directly.
    _output << "metal::precise::exp2(";
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
  case OP::log2Op:
    _output << "metal::precise::log2(";
    translateArgument();
    _output << ")";
    break;
  case OP::absOp:
    _output << "metal::fabs(";
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
  case OP::bitAndOp:
    _output << "&";
    break;
  case OP::bitOrOp:
    _output << "|";
    break;
  case OP::bitXorOp:
    _output << "^";
    break;
  case OP::maxOp:
  case OP::minOp:
    llvm_unreachable("min/maxOp handled above via function-call form");
  }

  _output << " (";
  translateValueOrVarName(op.getRhs());
  _output << ")";
}

void ModuleTranslation::translate(mlir::triton::metal::FmaOp op) {
  _output << "metal::precise::fma(";
  translateValueOrVarName(op.getA());
  _output << ", ";
  translateValueOrVarName(op.getB());
  _output << ", ";
  translateValueOrVarName(op.getC());
  _output << ")";
}

void ModuleTranslation::translate(mlir::triton::metal::Fp8ConvertOp op) {
  // The format's parameters travel as literals so one preamble helper serves
  // both: e4m3fn is (3 mantissa bits, 4 exponent bits, bias 7, no infinity)
  // and e5m2 is (2, 5, 15, has infinity).
  const bool e5m2 = op.getE5m2();
  if (op.getToFp8())
    _output << "__triton_f32_to_fp8(";
  else
    _output << "__triton_fp8_to_f32(";
  translateValueOrVarName(op.getValue());
  if (e5m2)
    _output << ", 2u, 5u, 15u";
  else
    _output << ", 3u, 4u, 7u";
  if (op.getToFp8())
    _output << (e5m2 ? ", true" : ", false");
  _output << ")";
}

void ModuleTranslation::translate(mlir::triton::metal::MathIntrinsicOp op) {
  // `callee` is emitted verbatim. It is only ever set from the fixed symbol
  // table in ExternElementwiseLowering (see TritonGPUToMetal.cpp), never from
  // anything a kernel author controls.
  _output << op.getCallee().str() << "(";
  llvm::interleaveComma(op.getArgs(), _output,
                        [&](mlir::Value arg) { translateValueOrVarName(arg); });
  _output << ")";
}

void ModuleTranslation::translate(mlir::triton::metal::ClampFOp op) {
  auto emitClamp = [&]() {
    _output << "metal::fmin(metal::fmax(";
    translateValueOrVarName(op.getX());
    _output << ", ";
    translateValueOrVarName(op.getMin());
    _output << "), ";
    translateValueOrVarName(op.getMax());
    _output << ")";
  };

  if (!op.getPropagateNan()) {
    emitClamp();
    return;
  }

  // Triton PropagateNan::ALL applies only to x; NaN bounds are undefined.
  // Preserve x itself so its NaN payload/sign are not unnecessarily replaced.
  _output << "(metal::isnan(";
  translateValueOrVarName(op.getX());
  _output << ") ? ";
  translateValueOrVarName(op.getX());
  _output << " : ";
  emitClamp();
  _output << ")";
}

void ModuleTranslation::translate(mlir::triton::metal::MulHiUIOp op) {
  auto intTy = mlir::cast<mlir::IntegerType>(op.getResult().getType());
  // Widen through an unsigned cast at the operand's OWN width. A signless i32
  // constant is printed as a SIGNED literal (0xD2511F53 comes out as
  // -766436013), and `(uint64_t)(-766436013)` SIGN-extends to
  // 0xFFFFFFFFD2511F53, so the high half of the product was garbage. Only
  // constant operands were affected — a value loaded from a device buffer is
  // already declared `uint32_t` — which is why a standalone `tl.umulhi` on
  // loaded data was exact while philox (`tl.rand`/`tl.randint`/`tl.randn`,
  // whose round multipliers are constants) silently produced the wrong stream.
  auto emitWide = [&](mlir::Value v) {
    _output << "((uint64_t)" << signednessCast(v.getType(), /*wantSigned=*/false)
            << "(";
    translateValueOrVarName(v);
    _output << "))";
  };
  if (intTy.getWidth() == 32) {
    _output << "(uint32_t)((";
    emitWide(op.getX());
    _output << " * ";
    emitWide(op.getY());
    _output << ") >> 32u)";
    return;
  }

  // Hacker's Delight unsigned 64x64 high-product decomposition, expanded as
  // one pure expression because value translation occurs inside its consumer.
  // Let x0/y0 be low 32-bit limbs and x1/y1 be high limbs:
  //   t = x1*y0 + high(x0*y0)
  //   hi = x1*y1 + high(t) + high(low(t) + x0*y1)
  // Each intermediate fits in uint64_t, so this is exact without uint128.
  auto emitX0 = [&] {
    _output << "(";
    emitWide(op.getX());
    _output << " & 0xfffffffful)";
  };
  auto emitY0 = [&] {
    _output << "(";
    emitWide(op.getY());
    _output << " & 0xfffffffful)";
  };
  auto emitX1 = [&] {
    _output << "(";
    emitWide(op.getX());
    _output << " >> 32u)";
  };
  auto emitY1 = [&] {
    _output << "(";
    emitWide(op.getY());
    _output << " >> 32u)";
  };
  auto emitT = [&] {
    _output << "(";
    emitX1();
    _output << " * ";
    emitY0();
    _output << " + ((";
    emitX0();
    _output << " * ";
    emitY0();
    _output << ") >> 32u))";
  };

  _output << "(";
  emitX1();
  _output << " * ";
  emitY1();
  _output << " + (";
  emitT();
  _output << " >> 32u) + (((";
  emitT();
  _output << " & 0xfffffffful) + (";
  emitX0();
  _output << " * ";
  emitY1();
  _output << ")) >> 32u))";
}

void ModuleTranslation::translate(mlir::triton::metal::YieldWhileOp op) {
  translateValue(op.getCondition().getDefiningOp());
}
