
import torch

from tgelu import tgelu3


def siluLike(x: torch.Tensor) -> torch.Tensor:
    const = 0.004409
    part2 = 1 + const * torch.log1p(x**2)
    return x * torch.sigmoid(x) * part2


def silu_tgelu3_hybrid_like_025(x: torch.Tensor) -> torch.Tensor:
    return 0.25 * (siluLike(x) + tgelu3(x))
