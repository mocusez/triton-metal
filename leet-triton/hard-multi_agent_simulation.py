import torch
import triton
import triton.language as tl

@triton.jit
def swarm_kernel(
    agents_ptr,
    agents_next_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    off = tl.arange(0, 2)
    pos = tl.load(agents_ptr + pid * 4 + off)
    vel = tl.load(agents_ptr + pid * 4 + 2 + off)

    n_loops = tl.ceil(N / BLOCK_SIZE).to(tl.int32)
    d_thresh = 25.0

    v_avg = tl.zeros((2,), tl.float32)
    n_neigh = 0.0

    for i in range(n_loops):
        agent_off = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        msk = (agent_off < N) & (agent_off != pid)
        block_off = (agent_off * 4)[:, None] + off

        pos_neigh = tl.load(agents_ptr + block_off, mask = msk[:, None])
        vel_neigh = tl.load(agents_ptr + block_off + 2, mask = msk[:, None])

        diff = pos[None, :] - pos_neigh
        dp = tl.sum(diff * diff, axis=1)
        neighs = ((dp < d_thresh) & msk).to(tl.float32)

        v_avg += tl.sum(vel_neigh * neighs[:, None], axis = 0)
        n_neigh += tl.sum(neighs, axis = 0)

    v_avg = v_avg / tl.maximum(n_neigh, 1e-6)
    v_avg = tl.where(n_neigh > 0, v_avg, vel)

    v_new = vel + 0.05 * (v_avg - vel)
    p_new = pos + v_new

    tl.store(agents_next_ptr + pid * 4 + off, p_new)
    tl.store(agents_next_ptr + pid * 4 + off + 2, v_new)



# agents, agents_next are tensors on the GPU
def solve(agents: torch.Tensor, agents_next: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (N, )

    swarm_kernel[grid](
        agents,
        agents_next,
        N,
        BLOCK_SIZE
    )
