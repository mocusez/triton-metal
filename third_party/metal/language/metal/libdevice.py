"""Metal's `triton.language.extra.libdevice`.

Selected by `MetalBackend.get_module_map()`, so a kernel written against
`triton.language.extra.libdevice` compiles on Metal with the same call sites it
uses on CUDA. Each function emits `tt.extern_elementwise` with a `__metal_*`
symbol that `TritonExternElementwiseLowering` (third_party/metal/lib/Conversion/
TritonGPUToMetal/TritonGPUToMetal.cpp) maps to an MSL intrinsic.

Only functions with a real MSL intrinsic are declared. Every name here was
verified to exist by compiling `metal::<name>(...)` through
`torch.mps.compile_shader` — MSL's math surface is close to but not identical to
CUDA's, and a wrong name would only fail at kernel-load time.

Deliberately ABSENT because MSL has no equivalent (using them is a clean
compile-time error rather than a broken shader): cbrt, rcbrt, erfc, erfcx,
erfinv, erfcinv, expm1, log1p, hypot, rhypot, norm3d/norm4d, lgamma, tgamma,
remainder, normcdf/normcdfinv, the Bessel family (j0/j1/y0/y1/jn/yn,
cyl_bessel_i0/i1), and every f64 entry point — MPS has no double precision at
all, so those cannot be expressed on this backend.

Generated shape mirrors third_party/nvidia/language/cuda/libdevice.py.
"""

from triton.language import core


@core.extern
def abs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_abs", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_floor", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_ceil", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_trunc", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_rint", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nearbyint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_nearbyint", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_round", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_sqrt", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_rsqrt", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_exp", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_exp2", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_exp10", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_log", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_log2", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_log10", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_sin", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_cos", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_tan", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_sinh", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_cosh", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_tanh", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_asin", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_acos", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_atan", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_asinh", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_acosh", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_atanh", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinpi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_sinpi", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cospi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_cospi", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanpi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_tanpi", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def saturatef(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_saturatef", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_erf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_sinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_fast_sinf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_cosf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_fast_cosf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_fast_tanf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_expf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_fast_expf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_exp10f(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_fast_exp10f", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_logf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_fast_logf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log2f(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_fast_log2f", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log10f(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_fast_log10f", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_isnan", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_isinf", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isfinited(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_isfinite", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def signbit(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__metal_signbit", core.dtype("int1")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_atan2", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def copysign(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_copysign", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fdim(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_fdim", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_fmod", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_pow", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def powr(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_powr", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nextafter(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_nextafter", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmax(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_fmax", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmin(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_fmin", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_powf(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_fast_powf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_fast_dividef", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")):
                ("__metal_ldexp", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
                ("__metal_fma", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def clz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__metal_clz", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def popc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__metal_popc", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def brev(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__metal_brev", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)
