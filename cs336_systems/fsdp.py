from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from cs336_basics.model import Embedding, Linear


@dataclass
class _ShardedParameter:
    name: str
    parameter: nn.Parameter
    original_shape: torch.Size
    original_numel: int
    shard_numel: int
    master_shard: torch.Tensor
    communication_buffer: torch.Tensor | None = None
    full_weight: torch.Tensor | None = None
    saved_shard_grad: torch.Tensor | None = None


@dataclass
class _PendingShardedGradient:
    info: _ShardedParameter
    work: dist.Work
    communication_input: torch.Tensor
    reduced_shard: torch.Tensor
    previous_shard_grad: torch.Tensor | None


class FullyShardedDataParallel(nn.Module):
    """A small educational FSDP wrapper for cs336 Linear and Embedding.

    Linear/Embedding weights are stored as flat FP32 shards outside compute.
    Hooks materialize one complete weight immediately before its layer needs it,
    then release it after forward and after its backward gradient is sharded.
    Parameters belonging to all other module types remain replicated.
    """

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("FullyShardedDataParallel requires an initialized process group")
        if compute_dtype is not None and not compute_dtype.is_floating_point:
            raise ValueError("compute_dtype must be a floating-point dtype")

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self._sharded_parameters: list[_ShardedParameter] = []
        self._sharded_by_parameter_id: dict[int, _ShardedParameter] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._pending_sharded_gradients: list[_PendingShardedGradient] = []
        self._pending_replicated_gradients: list[tuple[dist.Work, nn.Parameter]] = []

        self._initialize_parameter_shards()
        self._register_module_hooks()
        self._register_gradient_hooks()

    def _initialize_parameter_shards(self) -> None:
        parameter_names = {id(parameter): name for name, parameter in self.module.named_parameters()}

        # First discover every unique Linear/Embedding weight. A shared weight
        # receives only one shard even if more than one module references it.
        for child_module in self.module.modules():
            if not isinstance(child_module, (Linear, Embedding)):
                continue
            parameter = child_module.weight
            if not isinstance(parameter, nn.Parameter):
                raise TypeError("Linear/Embedding weight must be an nn.Parameter")
            if id(parameter) in self._sharded_by_parameter_id:
                continue

            full_weight = parameter.detach().to(dtype=torch.float32).contiguous()
            dist.broadcast(full_weight, src=0)
            full_flat = full_weight.flatten()
            shard_numel = (full_flat.numel() + self.world_size - 1) // self.world_size
            padded_numel = shard_numel * self.world_size
            if padded_numel != full_flat.numel():
                padded = torch.zeros(padded_numel, dtype=full_flat.dtype, device=full_flat.device)
                padded[: full_flat.numel()].copy_(full_flat)
                full_flat = padded

            shard_start = self.rank * shard_numel
            master_shard = full_flat.narrow(0, shard_start, shard_numel).clone()
            parameter.data = master_shard

            info = _ShardedParameter(
                name=parameter_names[id(parameter)],
                parameter=parameter,
                original_shape=full_weight.shape,
                original_numel=full_weight.numel(),
                shard_numel=shard_numel,
                master_shard=master_shard,
            )
            self._sharded_parameters.append(info)
            # 这也是个字典，往上看就理解了
            self._sharded_by_parameter_id[id(parameter)] = info

        # Parameters and buffers not managed by FSDP remain replicated. Rank 0
        # is authoritative so callers do not need to seed every rank equally.
        with torch.no_grad():
            for parameter in self.module.parameters():
                if id(parameter) not in self._sharded_by_parameter_id:
                    dist.broadcast(parameter, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)

    def _register_module_hooks(self) -> None:
        for child_module in self.module.modules():
            if not isinstance(child_module, (Linear, Embedding)):
                continue
            info = self._sharded_by_parameter_id[id(child_module.weight)]
            self._hook_handles.append(
                child_module.register_forward_pre_hook(partial(self._before_forward, info)),
            )
            self._hook_handles.append(
                child_module.register_forward_hook(
                    partial(self._after_forward, info),
                    always_call=True,
                ),
            )

    def _register_gradient_hooks(self) -> None:
        for parameter in self.module.parameters():
            if not parameter.requires_grad:
                continue
            info = self._sharded_by_parameter_id.get(id(parameter))
            if info is None:
                hook = parameter.register_post_accumulate_grad_hook(self._reduce_replicated_gradient)
            else:
                hook = parameter.register_post_accumulate_grad_hook(partial(self._reduce_sharded_gradient, info))
            self._hook_handles.append(hook)

    def _all_gather_weight(self, info: _ShardedParameter) -> None:
        if info.full_weight is not None:
            return

        communication_dtype = self.compute_dtype or info.master_shard.dtype
        communication_shard = info.master_shard.to(dtype=communication_dtype).contiguous()
        if info.communication_buffer is None:
            info.communication_buffer = torch.empty(
                info.shard_numel * self.world_size,
                dtype=communication_dtype,
                device=communication_shard.device,
            )
        else:
            # Forward resharding shrinks this storage to zero. Re-grow the same
            # storage object so autograd's saved views become valid again for
            # backward without retaining a full parameter between the passes.
            storage_bytes = info.communication_buffer.numel() * info.communication_buffer.element_size()
            info.communication_buffer.untyped_storage().resize_(storage_bytes)

        dist.all_gather_into_tensor(info.communication_buffer, communication_shard)
        full_weight = info.communication_buffer[: info.original_numel].view(info.original_shape)
        info.full_weight = full_weight
        info.parameter.data = full_weight

    def _release_weight(self, info: _ShardedParameter) -> None:
        info.parameter.data = info.master_shard
        info.full_weight = None
        if info.communication_buffer is not None:
            # Views saved by autograd keep the Storage object alive. Resizing
            # that storage to zero is what actually releases the full weight;
            # merely dropping this class's Tensor reference would not.
            info.communication_buffer.untyped_storage().resize_(0)

    def _before_forward(
        self,
        info: _ShardedParameter,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
    ) -> None:
        self._all_gather_weight(info)

    def _after_forward(
        self,
        info: _ShardedParameter,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if isinstance(_output, torch.Tensor) and _output.requires_grad:
            # A Tensor hook runs when the output gradient becomes available,
            # immediately before autograd executes this layer's backward.
            _output.register_hook(partial(self._before_backward, info))
        self._release_weight(info)

    def _before_backward(
        self,
        info: _ShardedParameter,
        output_gradient: torch.Tensor,
    ) -> torch.Tensor:
        # Gradient accumulation from an earlier backward has shard shape and
        # cannot coexist with the temporarily materialized full parameter.
        if info.parameter.grad is not None and info.parameter.grad.numel() == info.shard_numel:
            info.saved_shard_grad = info.parameter.grad
            info.parameter.grad = None
        self._all_gather_weight(info)
        return output_gradient

    def _reduce_sharded_gradient(
        self,
        info: _ShardedParameter,
        parameter: nn.Parameter,
    ) -> None:
        full_gradient = parameter.grad
        if full_gradient is None:
            self._release_weight(info)
            return

        communication_dtype = self.compute_dtype or info.master_shard.dtype
        padded_gradient = torch.zeros(
            info.shard_numel * self.world_size,
            dtype=communication_dtype,
            device=full_gradient.device,
        )
        padded_gradient[: info.original_numel].copy_(full_gradient.flatten().to(communication_dtype))
        reduced_shard = torch.empty(
            info.shard_numel,
            dtype=communication_dtype,
            device=full_gradient.device,
        )
        work = dist.reduce_scatter_tensor(
            reduced_shard,
            padded_gradient,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )
        self._pending_sharded_gradients.append(
            _PendingShardedGradient(
                info=info,
                work=work,
                communication_input=padded_gradient,
                reduced_shard=reduced_shard,
                previous_shard_grad=info.saved_shard_grad,
            ),
        )
        info.saved_shard_grad = None
        parameter.grad = None
        self._release_weight(info)

    def _reduce_replicated_gradient(self, parameter: nn.Parameter) -> None:
        if parameter.grad is None:
            return
        work = dist.all_reduce(
            parameter.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )
        self._pending_replicated_gradients.append((work, parameter))

    def finish_gradient_synchronization(self) -> None:
        """Wait for gradient communication and expose optimizer-ready grads."""

        for pending in self._pending_sharded_gradients:
            pending.work.wait()
            shard_gradient = pending.reduced_shard.to(dtype=pending.info.master_shard.dtype)
            shard_gradient.div_(self.world_size)
            if pending.previous_shard_grad is not None:
                shard_gradient.add_(pending.previous_shard_grad)
            if pending.info.parameter.grad is None:
                pending.info.parameter.grad = shard_gradient
            else:
                pending.info.parameter.grad.add_(shard_gradient)

        for work, parameter in self._pending_replicated_gradients:
            work.wait()
            if parameter.grad is not None:
                parameter.grad.div_(self.world_size)

        self._pending_sharded_gradients.clear()
        self._pending_replicated_gradients.clear()

    def gather_full_parameters(self) -> dict[str, torch.Tensor]:
        """Collect complete FP32 parameters without changing shard storage."""

        if self._pending_sharded_gradients or self._pending_replicated_gradients:
            self.finish_gradient_synchronization()

        sharded_by_id = self._sharded_by_parameter_id
        result: dict[str, torch.Tensor] = {}
        for name, parameter in self.module.named_parameters():
            info = sharded_by_id.get(id(parameter))
            if info is None:
                result[name] = parameter.detach().clone()
                continue

            gathered = torch.empty(
                info.shard_numel * self.world_size,
                dtype=info.master_shard.dtype,
                device=info.master_shard.device,
            )
            dist.all_gather_into_tensor(gathered, info.master_shard)
            result[name] = gathered[: info.original_numel].view(info.original_shape).clone()
        return result

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)
