import triton
import triton.language as tl

@triton.jit
def _speculative_decoding_verification_kernel(
    draft_tokens_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    uniform_samples_ptr,
    output_tokens_ptr,
    B, T, V,
    BLOCK_SIZE_V: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    b = pid

    # 1. 默认将该 Sequence 的整个输出行初始化为 0
    for idx in range(T + 1):
        tl.store(output_tokens_ptr + b * (T + 1) + idx, 0)

    accepted_all = True

    # 2. 从左到右依次处理位置
    for i in range(T):
        if accepted_all:
            t_i = tl.load(draft_tokens_ptr + b * T + i)

            p_i_ti = tl.load(draft_probs_ptr + b * T * V + i * V + t_i)
            q_i_ti = tl.load(target_probs_ptr + b * T * V + i * V + t_i)

            alpha_i = q_i_ti / p_i_ti
            if alpha_i > 1.0:
                alpha_i = 1.0

            u_i = tl.load(uniform_samples_ptr + b * (T + 1) + i)

            if u_i < alpha_i:
                # 接受 Token
                tl.store(output_tokens_ptr + b * (T + 1) + i, t_i)
            else:
                # 拒绝 Token：标记状态，中断后续处理
                accepted_all = False

                # Pass 1: 计算 adj 分布的总和
                sum_adj = 0.0
                for v_offset in range(0, V, BLOCK_SIZE_V):
                    v_idx = v_offset + tl.arange(0, BLOCK_SIZE_V)
                    v_mask = v_idx < V

                    p_ptr = draft_probs_ptr + b * T * V + i * V + v_idx
                    q_ptr = target_probs_ptr + b * T * V + i * V + v_idx

                    p_val = tl.load(p_ptr, mask=v_mask, other=0.0)
                    q_val = tl.load(q_ptr, mask=v_mask, other=0.0)

                    adj_val = tl.where(q_val > p_val, q_val - p_val, 0.0)
                    adj_val = tl.where(v_mask, adj_val, 0.0)
                    sum_adj += tl.sum(adj_val, axis=0)

                r = tl.load(uniform_samples_ptr + b * (T + 1) + T)
                is_uniform = sum_adj <= 0.0
                target_r = tl.where(is_uniform, r * V, r * sum_adj)

                # Pass 2: Inverse CDF 寻找目标 Token
                running_sum = 0.0
                chosen_k = V  # 初始标记为 V (越界值表示未找到)

                for v_offset in range(0, V, BLOCK_SIZE_V):
                    v_idx = v_offset + tl.arange(0, BLOCK_SIZE_V)
                    v_mask = v_idx < V

                    if is_uniform:
                        adj_val = tl.where(v_mask, 1.0, 0.0)
                    else:
                        p_ptr = draft_probs_ptr + b * T * V + i * V + v_idx
                        q_ptr = target_probs_ptr + b * T * V + i * V + v_idx
                        p_val = tl.load(p_ptr, mask=v_mask, other=0.0)
                        q_val = tl.load(q_ptr, mask=v_mask, other=0.0)
                        adj_val = tl.where(q_val > p_val, q_val - p_val, 0.0)
                        adj_val = tl.where(v_mask, adj_val, 0.0)

                    chunk_cumsum = tl.cumsum(adj_val, axis=0)
                    total_cumsum = running_sum + chunk_cumsum

                    cond = (total_cumsum >= target_r) & v_mask
                    v_idx_selected = tl.where(cond, v_idx, V)
                    min_idx = tl.min(v_idx_selected, axis=0)

                    chosen_k = tl.where((chosen_k == V) & (min_idx < V), min_idx, chosen_k)

                    running_sum += tl.sum(adj_val, axis=0)

                # 容错与防越界处理：使用整数安全的 minimum/maximum 替代 clamp
                chosen_k = tl.where(chosen_k == V, V - 1, chosen_k)
                chosen_k = tl.maximum(0, tl.minimum(chosen_k, V - 1))

                tl.store(output_tokens_ptr + b * (T + 1) + i, chosen_k)

    # 3. 如果所有 T 个都被接受，提取 Bonus Token
    if accepted_all:
        r = tl.load(uniform_samples_ptr + b * (T + 1) + T)
        running_sum = 0.0
        bonus_k = V

        for v_offset in range(0, V, BLOCK_SIZE_V):
            v_idx = v_offset + tl.arange(0, BLOCK_SIZE_V)
            v_mask = v_idx < V

            q_ptr = target_probs_ptr + b * T * V + (T - 1) * V + v_idx
            q_val = tl.load(q_ptr, mask=v_mask, other=0.0)
            q_val = tl.where(v_mask, q_val, 0.0)

            chunk_cumsum = tl.cumsum(q_val, axis=0)
            total_cumsum = running_sum + chunk_cumsum

            cond = (total_cumsum >= r) & v_mask
            v_idx_selected = tl.where(cond, v_idx, V)
            min_idx = tl.min(v_idx_selected, axis=0)

            bonus_k = tl.where((bonus_k == V) & (min_idx < V), min_idx, bonus_k)

            running_sum += tl.sum(q_val, axis=0)

        bonus_k = tl.where(bonus_k == V, V - 1, bonus_k)
        # 同样替换掉这里的 clamp
        bonus_k = tl.maximum(0, tl.minimum(bonus_k, V - 1))

        tl.store(output_tokens_ptr + b * (T + 1) + T, bonus_k)


def solve(draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V):
    grid = (B,)
    _speculative_decoding_verification_kernel[grid](
        draft_tokens,
        draft_probs,
        target_probs,
        uniform_samples,
        output_tokens,
        B, T, V,
        BLOCK_SIZE_V=1024
    )
