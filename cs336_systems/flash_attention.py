import math

import torch

import triton
import triton.language as tl

@torch.compile
def flash_attention_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_dim = Q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # D = rowsum(O ∘ dO)
    # 使用 FP32 计算 reduction，增强数值稳定性。
    D = torch.sum(
        O.float() * dO.float(),
        dim=-1,
    )

    # 式 (13)：S = QK^T / sqrt(d)
    S = torch.matmul(
        Q,
        K.transpose(-1, -2),
    )
    S = S * scale

    if is_causal:
        num_queries = Q.shape[-2]
        num_keys = K.shape[-2]

        query_indices = torch.arange(
            num_queries,
            device=Q.device,
        )
        key_indices = torch.arange(
            num_keys,
            device=Q.device,
        )

        causal_mask = (
            query_indices[:, None]
            >= key_indices[None, :]
        )

        S = S.masked_fill(
            ~causal_mask.unsqueeze(0),
            -1.0e6,
        )

    # 式 (14)：使用 forward 保存的 L 重算 P。
    # 这里没有再次调用 softmax。
    P = torch.exp(
        S.float() - L.float().unsqueeze(-1)
    )

    # 式 (15)：dV = P^T dO
    # matmul 前转回输入类型，使 BF16 输入可以使用 Tensor Cores。
    dV = torch.matmul(
        P.to(dO.dtype).transpose(-1, -2),
        dO,
    )

    # 式 (16)：dP = dO V^T
    dP = torch.matmul(
        dO,
        V.transpose(-1, -2),
    )

    # 式 (17)：dS = P ∘ (dP - D)
    dS = P * (
        dP.float() - D.unsqueeze(-1)
    )

    # 让后续矩阵乘法使用输入的数据类型。
    dS_for_matmul = dS.to(Q.dtype)

    # 式 (18)：dQ = dS K / sqrt(d)
    dQ = torch.matmul(
        dS_for_matmul,
        K,
    )
    dQ = dQ * scale

    # 式 (19)：dK = dS^T Q / sqrt(d)
    dK = torch.matmul(
        dS_for_matmul.transpose(-1, -2),
        Q,
    )
    dK = dK * scale

    return dQ, dK, dV


class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        block_q = 16
        block_k = 16

        batch_size, num_queries, head_dim = Q.shape
        num_keys = K.shape[-2]
        value_dim = V.shape[-1]

        if num_queries % block_q != 0:
            raise ValueError("num_queries must be divisible by block_q")
        if num_keys % block_k != 0:
            raise ValueError("num_keys must be divisible by block_k")

        scale = 1.0 / math.sqrt(head_dim)

        # 最终输出保持与输入相同的数据类型。
        O = torch.empty(
            (batch_size, num_queries, value_dim),
            device=Q.device,
            dtype=Q.dtype,
        )

        # online softmax 的统计量使用 FP32，提高数值稳定性。
        L = torch.empty(
            (batch_size, num_queries),
            device=Q.device,
            dtype=torch.float32,
        )

        for query_start in range(0, num_queries, block_q):
            query_end = query_start + block_q
            query_block = Q[:, query_start:query_end].float()

            output_accumulator = torch.zeros(
                (batch_size, block_q, value_dim),
                device=Q.device,
                dtype=torch.float32,
            )
            row_sum = torch.zeros(
                (batch_size, block_q),
                device=Q.device,
                dtype=torch.float32,
            )
            row_max = torch.full(
                (batch_size, block_q),
                -torch.inf,
                device=Q.device,
                dtype=torch.float32,
            )

            for key_start in range(0, num_keys, block_k):
                key_end = key_start + block_k

                key_block = K[:, key_start:key_end].float()
                value_block = V[:, key_start:key_end].float()

                scores = torch.matmul(
                    query_block,
                    key_block.transpose(-1, -2),
                )
                scores = scores * scale

                if is_causal:
                    query_positions = torch.arange(
                        query_start,
                        query_end,
                        device=Q.device,
                    )
                    key_positions = torch.arange(
                        key_start,
                        key_end,
                        device=Q.device,
                    )
                    causal_mask = (
                        query_positions[:, None]
                        < key_positions[None, :]
                    )
                    scores = scores.masked_fill(
                        causal_mask.unsqueeze(0),
                        -torch.inf,
                    )

                new_row_max = torch.maximum(
                    row_max,
                    scores.max(dim=-1).values,
                )

                old_scale = torch.exp(row_max - new_row_max)
                probabilities = torch.exp(
                    scores - new_row_max.unsqueeze(-1)
                )

                row_sum = (
                    old_scale * row_sum
                    + probabilities.sum(dim=-1)
                )

                output_accumulator = (
                    old_scale.unsqueeze(-1) * output_accumulator
                    + torch.matmul(probabilities, value_block)
                )

                row_max = new_row_max

            normalized_output = (
                output_accumulator / row_sum.unsqueeze(-1)
            )

            O[:, query_start:query_end] = normalized_output.to(Q.dtype)
            L[:, query_start:query_end] = row_max + torch.log(row_sum)

        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)

        # 测试期望 apply() 只返回 O，L 从 saved_tensors 中读取。
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors

        dQ, dK, dV = flash_attention_backward(
            Q,
            K,
            V,
            O,
            dO,
            L,
            ctx.is_causal,
        )

        # forward 有 Q、K、V、is_causal 四个输入。
        # bool 参数没有梯度，所以最后返回 None。
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # grid 的两个维度分别表示 query tile 和 batch。
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # 一个 program 只加载一个 query tile。
    q = tl.load(Q_block_ptr)

    # Online softmax 状态，全部使用 FP32。
    row_max = tl.full(
        (Q_TILE_SIZE,),
        -float("inf"),
        tl.float32,
    )
    row_sum = tl.zeros(
        (Q_TILE_SIZE,),
        tl.float32,
    )
    output_accumulator = tl.zeros(
        (Q_TILE_SIZE, D),
        tl.float32,
    )

    query_offsets = (
        query_tile_index * Q_TILE_SIZE
        + tl.arange(0, Q_TILE_SIZE)
    )

    # kernel 中唯一的循环：遍历 key/value tiles。
    for key_start in range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        scores = tl.dot(
            q,
            tl.trans(k),
            input_precision="ieee",
        )
        scores = scores * scale

        if IS_CAUSAL:
            key_offsets = (
                key_start
                + tl.arange(0, K_TILE_SIZE)
            )

            causal_mask = (
                query_offsets[:, None]
                >= key_offsets[None, :]
            )

            scores = tl.where(
                causal_mask,
                scores,
                -float("inf"),
            )

        new_row_max = tl.maximum(
            row_max,
            tl.max(scores, axis=1),
        )

        old_scale = tl.exp(row_max - new_row_max)

        probabilities = tl.exp(
            scores - new_row_max[:, None]
        )

        row_sum = (
            old_scale * row_sum
            + tl.sum(probabilities, axis=1)
        )

        output_accumulator = (
            output_accumulator
            * old_scale[:, None]
        )

        # 将 probabilities 转成和 V 相同的数据类型后再做 dot。
        probabilities = probabilities.to(v.dtype)

        output_accumulator = tl.dot(
            probabilities,
            v,
            acc=output_accumulator,
            input_precision="ieee",
        )

        row_max = new_row_max

        # 按题目要求，在循环末尾移动 block pointers。
        K_block_ptr = tl.advance(
            K_block_ptr,
            (K_TILE_SIZE, 0),
        )
        V_block_ptr = tl.advance(
            V_block_ptr,
            (K_TILE_SIZE, 0),
        )

    output_accumulator = (
        output_accumulator
        / row_sum[:, None]
    )
    logsumexp = row_max + tl.log(row_sum)

    # 写回 HBM 前转换为输出 tensor 的数据类型。
    tl.store(
        O_block_ptr,
        output_accumulator.to(
            O_block_ptr.type.element_ty
        ),
    )
    tl.store(L_block_ptr, logsumexp)


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not (Q.is_cuda and K.is_cuda and V.is_cuda):
            raise ValueError(
                "The Triton implementation requires CUDA tensors."
            )

        batch_size, num_queries, head_dim = Q.shape
        num_keys = K.shape[-2]

        if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0]:
            raise ValueError("Q, K, and V must have the same batch size.")

        if K.shape[-1] != head_dim or V.shape[-1] != head_dim:
            raise ValueError("Q, K, and V must have the same head dimension.")

        query_tile_size = 16
        key_tile_size = 16

        if num_queries % query_tile_size != 0:
            raise ValueError(
                "num_queries must be divisible by query_tile_size."
            )

        if num_keys % key_tile_size != 0:
            raise ValueError(
                "num_keys must be divisible by key_tile_size."
            )

        O = torch.empty_like(Q)

        # L 按题目要求使用 FP32。
        L = torch.empty(
            (batch_size, num_queries),
            device=Q.device,
            dtype=torch.float32,
        )

        scale = 1.0 / math.sqrt(head_dim)

        # (T_q, batch_size)
        grid = (
            triton.cdiv(num_queries, query_tile_size),
            batch_size,
        )

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),
            num_queries,
            num_keys,
            scale,
            IS_CAUSAL=is_causal,
            D=head_dim,
            Q_TILE_SIZE=query_tile_size,
            K_TILE_SIZE=key_tile_size,
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors

        dQ, dK, dV = flash_attention_backward(
            Q,
            K,
            V,
            O,
            dO,
            L,
            ctx.is_causal,
        )

        return dQ, dK, dV, None