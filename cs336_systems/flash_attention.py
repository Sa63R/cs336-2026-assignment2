import math

import torch




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
        raise NotImplementedError(
            "FlashAttention2PyTorch backward is implemented in a later task."
        )