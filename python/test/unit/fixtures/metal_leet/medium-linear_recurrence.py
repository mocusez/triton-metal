import torch
import triton
import triton.language as tl


# ------------------------------------------------------------
# Affine transform composition
#
# left  : h -> a_l * h + x_l
# right : h -> a_r * h + x_r
#
# right(left(h))
#   = a_r * a_l * h + a_r * x_l + x_r
# ------------------------------------------------------------
@triton.jit
def affine_combine(a_l, x_l, a_r, x_r):
    a = a_l * a_r
    x = a_r * x_l + x_r
    return a, x


# ------------------------------------------------------------
# Stage 1:
# Scan each BLOCK-sized chunk independently.
#
# Outputs:
#   h_local[t] = recurrence inside this chunk assuming
#                incoming state = 0
#
#   a_prefix[t] = product of recurrence coefficients from
#                 chunk start through t
#
#   chunk_a[c], chunk_x[c] = affine transform of whole chunk
#
# Grid:
#   (B, ceil(L / BLOCK))
# ------------------------------------------------------------
@triton.jit
def local_scan_kernel(
    a_ptr,
    x_ptr,
    h_ptr,
    a_prefix_ptr,
    chunk_a_ptr,
    chunk_x_ptr,
    L: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)

    start = c * BLOCK
    offs = tl.arange(0, BLOCK)
    t = start + offs

    mask = t < L
    idx = b * L + t

    a = tl.load(a_ptr + idx, mask=mask, other=1.0).to(tl.float32)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    # Important boundary condition:
    #
    #   h[b, 0] = x[b, 0]
    #
    # rather than
    #
    #   h[b, 0] = a[b,0] * 0 + x[b,0]
    #
    # Numerically those are the same, but setting a[0] = 0
    # makes the first affine transform explicitly independent
    # of any incoming state.
    is_global_first = t == 0
    a = tl.where(is_global_first, 0.0, a)

    # Prefix affine composition inside this chunk.
    a_pref, x_pref = tl.associative_scan(
        (a, x),
        axis=0,
        combine_fn=affine_combine,
    )

    # Temporary local recurrence result.
    tl.store(h_ptr + idx, x_pref, mask=mask)

    # Needed later to inject the state from the preceding chunk.
    tl.store(a_prefix_ptr + idx, a_pref, mask=mask)

    # One lane stores the transform representing this whole chunk.
    remaining = L - start
    valid_count = tl.minimum(remaining, BLOCK)
    last = offs == (valid_count - 1)
    last = last & mask

    chunk_idx = b * N_CHUNKS + c

    tl.store(
        chunk_a_ptr + chunk_idx + offs * 0,
        a_pref,
        mask=last,
    )
    tl.store(
        chunk_x_ptr + chunk_idx + offs * 0,
        x_pref,
        mask=last,
    )


# ------------------------------------------------------------
# Stage 2:
# Scan the per-chunk transforms.
#
# After this kernel:
#   chunk_x[b, c] = true h value at the end of chunk c
#
# Only one program is needed per batch row because the number
# of chunks is small:
#
# BLOCK=256:
#   L=16384 -> 64 chunks
#   L=65536 -> 256 chunks
# ------------------------------------------------------------
@triton.jit
def chunk_scan_kernel(
    chunk_a_ptr,
    chunk_x_ptr,
    N_CHUNKS: tl.constexpr,
    CHUNK_SCAN_BLOCK: tl.constexpr,
):
    b = tl.program_id(0)

    offs = tl.arange(0, CHUNK_SCAN_BLOCK)
    mask = offs < N_CHUNKS

    idx = b * N_CHUNKS + offs

    a = tl.load(
        chunk_a_ptr + idx,
        mask=mask,
        other=1.0,
    ).to(tl.float32)

    x = tl.load(
        chunk_x_ptr + idx,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    _, x_scan = tl.associative_scan(
        (a, x),
        axis=0,
        combine_fn=affine_combine,
    )

    # Save actual state at end of every chunk.
    tl.store(
        chunk_x_ptr + idx,
        x_scan,
        mask=mask,
    )


# ------------------------------------------------------------
# Stage 3:
# Inject the state from the previous chunk:
#
# local result:
#     q[t]
#
# local coefficient product:
#     p[t]
#
# incoming state from previous chunk:
#     s
#
# final:
#     h[t] = p[t] * s + q[t]
#
# Chunk 0 already starts with the true initial condition, so
# nothing needs to be added there.
# ------------------------------------------------------------
@triton.jit
def fixup_kernel(
    h_ptr,
    a_prefix_ptr,
    chunk_x_ptr,
    L: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)

    # First chunk is already globally correct.
    if c == 0:
        return

    offs = tl.arange(0, BLOCK)
    t = c * BLOCK + offs

    mask = t < L
    idx = b * L + t

    local_h = tl.load(
        h_ptr + idx,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    a_pref = tl.load(
        a_prefix_ptr + idx,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # chunk_x[c - 1] after stage 2 is exactly the state entering
    # the current chunk.
    incoming = tl.load(
        chunk_x_ptr + b * N_CHUNKS + (c - 1)
    ).to(tl.float32)

    out = a_pref * incoming + local_h

    tl.store(
        h_ptr + idx,
        out,
        mask=mask,
    )


# ------------------------------------------------------------
# solve
#
# If the benchmark template already supplies h, use this form.
# The result is written directly into h as required.
# ------------------------------------------------------------
def solve(a: torch.Tensor, x: torch.Tensor, h: torch.Tensor, B: int, L: int):
    BLOCK = 256

    n_chunks = triton.cdiv(L, BLOCK)

    a_prefix = torch.empty_like(a)

    chunk_a = torch.empty(
        (B, n_chunks),
        device=a.device,
        dtype=torch.float32,
    )

    chunk_x = torch.empty(
        (B, n_chunks),
        device=a.device,
        dtype=torch.float32,
    )

    local_scan_kernel[(B, n_chunks)](
        a,
        x,
        h,
        a_prefix,
        chunk_a,
        chunk_x,
        L=L,
        N_CHUNKS=n_chunks,
        BLOCK=BLOCK,
        num_warps=4,
    )

    CHUNK_SCAN_BLOCK = triton.next_power_of_2(n_chunks)

    chunk_scan_kernel[(B,)](
        chunk_a,
        chunk_x,
        N_CHUNKS=n_chunks,
        CHUNK_SCAN_BLOCK=CHUNK_SCAN_BLOCK,
        num_warps=4,
    )

    fixup_kernel[(B, n_chunks)](
        h,
        a_prefix,
        chunk_x,
        L=L,
        N_CHUNKS=n_chunks,
        BLOCK=BLOCK,
        num_warps=4,
    )