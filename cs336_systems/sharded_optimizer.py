from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """Shard an optimizer's parameters and state across distributed ranks.

    Every rank keeps the complete model, but the wrapped optimizer owns and
    updates only a deterministic subset of its parameters.  After each local
    optimizer step, the owner broadcasts every updated parameter so all model
    replicas become identical again.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[Optimizer],
        **kwargs: Any,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires an initialized process group")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self._optimizer_kwargs = dict(kwargs)

        # Optimizer.__init__ calls self.add_param_group(), so all attributes
        # used by our override must exist before invoking the superclass.
        self._initialized = False
        self._parameter_owner: dict[torch.Tensor, int] = {}
        self._rank_numel = [0 for _ in range(self.world_size)]
        self._all_parameters: list[torch.Tensor] = []
        self._local_group_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self._local_optimizer: Optimizer | None = None

        # The wrapper deliberately tracks every parameter.  This makes the
        # inherited zero_grad() clear gradients on the full model replica.
        super().__init__(params, kwargs)

        for group in self.param_groups:
            self._assign_parameters(group["params"])
        self._rebuild_all_parameters()
        self._create_local_optimizer()
        self._initialized = True

    @property
    def local_optimizer(self) -> Optimizer | None:
        """The optimizer containing this rank's parameter/state shard."""

        return self._local_optimizer

    def _assign_parameters(self, parameters: Iterable[torch.Tensor]) -> None:
        """Assign new parameters greedily by size for deterministic balance."""

        for parameter in parameters:
            owner = min(range(self.world_size), key=lambda rank: (self._rank_numel[rank], rank))
            self._parameter_owner[parameter] = owner
            self._rank_numel[owner] += parameter.numel()

    def _rebuild_all_parameters(self) -> None:
        self._all_parameters = [parameter for group in self.param_groups for parameter in group["params"]]

    def _local_copy_of_group(self, group: dict[str, Any]) -> dict[str, Any] | None:
        local_indices = [index for index, parameter in enumerate(group["params"]) if self._parameter_owner[parameter] == self.rank]
        if not local_indices:
            return None

        local_group = {
            key: value 
            for key, value in group.items() 
            if key not in ("params", "param_names")
            }
        local_group["params"] = [group["params"][index] for index in local_indices]
        if "param_names" in group:
            local_group["param_names"] = [group["param_names"][index] for index in local_indices]
        return local_group

    def _create_local_optimizer(self) -> None:
        local_groups: list[dict[str, Any]] = []
        global_groups: list[dict[str, Any]] = []
        for global_group in self.param_groups:
            local_group = self._local_copy_of_group(global_group)
            if local_group is not None:
                global_groups.append(global_group)
                local_groups.append(local_group)

        # A rank can legitimately receive no parameters when world_size is
        # larger than the number of tensors.  It still participates in every
        # broadcast, but has no local optimizer state.
        if not local_groups:
            return

        self._local_optimizer = self.optimizer_cls(local_groups, **self._optimizer_kwargs)
        self._local_group_pairs = list(zip(global_groups, self._local_optimizer.param_groups, strict=True))

        # Expose the local shard through the standard Optimizer.state API.
        self.state = self._local_optimizer.state
        self.defaults = self._local_optimizer.defaults
        self._fill_global_group_defaults()

    def _fill_global_group_defaults(self) -> None:
        for group in self.param_groups:
            for name, default in self.defaults.items():
                group.setdefault(name, default)

    def _sync_local_group_options(self) -> None:
        """Propagate scheduler/user changes from public groups to local ones."""

        for global_group, local_group in self._local_group_pairs:
            for name, value in global_group.items():
                if name not in ("params", "param_names"):
                    local_group[name] = value

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add and deterministically shard a parameter group on every rank."""

        super().add_param_group(param_group)
        if not self._initialized:
            return

        global_group = self.param_groups[-1]
        self._assign_parameters(global_group["params"])
        self._rebuild_all_parameters()
        local_group = self._local_copy_of_group(global_group)
        if local_group is None:
            return

        if self._local_optimizer is None:
            self._local_optimizer = self.optimizer_cls([local_group], **self._optimizer_kwargs)
            self.state = self._local_optimizer.state
            self.defaults = self._local_optimizer.defaults
            self._fill_global_group_defaults()
        else:
            self._local_optimizer.add_param_group(local_group)

        self._local_group_pairs.append((global_group, self._local_optimizer.param_groups[-1]))

    @overload
    def step(self, closure: None = None, **kwargs: Any) -> None: ...

    @overload
    def step(self, closure: Callable[[], float], **kwargs: Any) -> float: ...

    def step(
        self,
        closure: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> float | None:
        """Update this rank's shard, then synchronize all updated parameters."""

        if self._local_optimizer is None:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
        else:
            self._sync_local_group_options()
            loss = self._local_optimizer.step(closure=closure, **kwargs)  # type: ignore[no-matching-overload]

        # All ranks execute the same collectives in the same parameter order.
        # The parameter owner is the only rank that performed its update.
        with torch.no_grad():
            for parameter in self._all_parameters:
                dist.broadcast(parameter, src=self._parameter_owner[parameter])

        return loss
