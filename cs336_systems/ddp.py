from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    """
    最朴素的 Distributed Data Parallel wrapper。

    - 初始化时从 rank 0 广播模型状态；
    - forward 直接调用被包装的模型；
    - backward 完成后逐参数同步梯度；
    - 不进行 communication/computation overlap；
    - 不进行 gradient bucketing。
    """

    def __init__(self, module: nn.Module):
        super().__init__()

        if not dist.is_available():
            raise RuntimeError(
                "torch.distributed is not available"
            )

        if not dist.is_initialized():
            raise RuntimeError(
                "The process group must be initialized "
                "before constructing NaiveDDP."
            )

        # 将 module 赋给 self.module 会自动注册为子模块。
        self.module = module

        self._broadcast_module_state()

    def _broadcast_module_state(self) -> None:
        """
        让所有 rank 的参数和 buffer 与 rank 0 完全一致。
        """
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(
                    parameter,
                    src=0,
                )

            # 当前 ToyModel 没有重要 buffer，
            # 但完整的 module state 还应包含 buffers。
            for buffer in self.module.buffers():
                dist.broadcast(
                    buffer,
                    src=0,
                )

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.module(
            *args,
            **kwargs,
        )

    def finish_gradient_synchronization(
        self,
    ) -> None:
        """
        对每个 parameter gradient 单独执行同步 all-reduce。
        """
        world_size = dist.get_world_size()

        for parameter in self.module.parameters():
            gradient = parameter.grad

            # requires_grad=False 或未参与当前计算的参数没有梯度。
            if gradient is None:
                continue

            # all_reduce 是原地操作：
            # gradient <- 所有 rank 的 gradient 之和
            dist.all_reduce(
                gradient,
                op=dist.ReduceOp.SUM,
                async_op=False,
            )

            # 将梯度和转换为所有 rank 梯度的平均值。
            gradient.div_(world_size)