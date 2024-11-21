"""Custom normalization layers."""

from typing import Optional

import torch
import torch.nn as nn

from sarathi._kernels_C import kernels
from sarathi._model_executor_C.model_executor import RMSNorm as RMSNormC
from sarathi.metrics.cuda_timer import CudaTimer


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
        use_native_execution_backend: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_native_execution_backend = use_native_execution_backend

        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)

        if self.use_native_execution_backend:
            self.native_handler = RMSNormC(self.weight, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_native_execution_backend:
            return self.native_handler.forward(x)

        with self._norm_timer:
            out = torch.empty_like(x)
            kernels.rms_norm(
                out,
                x,
                self.weight.data,
                self.variance_epsilon,
            )
            return out
