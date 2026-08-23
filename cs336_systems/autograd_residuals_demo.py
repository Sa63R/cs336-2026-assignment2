import torch
from torch import nn


x = torch.randn((4, 512, 2560), requires_grad=True)


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device),
        )
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps,
        )
        x = x * rms
        return self.weight * x


def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(
        f"Saving residual: shape={shape}, "
        f"dtype={dtype}, grad_fn={grad_fn}"
    )
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(
        f"Loading residual: shape={shape}, "
        f"dtype={dtype}, grad_fn={grad_fn}"
    )
    return t


ln = RMSNorm(x.shape[-1])

with torch.autograd.graph.saved_tensors_hooks(
    pack_hook,
    unpack_hook,
):
    y = ln(x)
    y.sum().backward()


#  这代码有点烧脑。。。不过最后总算是看懂了