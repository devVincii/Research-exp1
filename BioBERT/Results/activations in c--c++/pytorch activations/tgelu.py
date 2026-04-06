import math
import torch

def student_t3_cdf(x: torch.Tensor) -> torch.Tensor:
    sqrt3 = torch.as_tensor(math.sqrt(3.0), dtype=x.dtype, device=x.device)
    pi = torch.as_tensor(math.pi, dtype=x.dtype, device=x.device)
    return 0.5 + (1.0 / pi) * (
        torch.atan(x / sqrt3) + (sqrt3 * x) / (x * x + 3.0)
    )

def tgelu3(x: torch.Tensor) -> torch.Tensor:
    return x * student_t3_cdf(x)

