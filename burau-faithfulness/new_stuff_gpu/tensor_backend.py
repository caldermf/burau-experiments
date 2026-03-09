from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TensorBackend:
    kind: str
    lib: Any
    device: str | None = None

    @property
    def is_torch(self) -> bool:
        return self.kind == "torch"

    def array(self, data, dtype=np.int32):
        if self.is_torch:
            torch_dtype = self._torch_dtype(dtype)
            return self.lib.tensor(data, dtype=torch_dtype, device=self.device)
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=np.int32):
        if self.is_torch:
            torch_dtype = self._torch_dtype(dtype)
            return self.lib.zeros(shape, dtype=torch_dtype, device=self.device)
        return np.zeros(shape, dtype=dtype)

    def full(self, shape, value, dtype=np.int32):
        if self.is_torch:
            torch_dtype = self._torch_dtype(dtype)
            return self.lib.full(shape, value, dtype=torch_dtype, device=self.device)
        return np.full(shape, value, dtype=dtype)

    def arange(self, n, dtype=np.int32):
        if self.is_torch:
            torch_dtype = self._torch_dtype(dtype)
            return self.lib.arange(n, dtype=torch_dtype, device=self.device)
        return np.arange(n, dtype=dtype)

    def concat(self, seq, axis=0):
        if self.is_torch:
            return self.lib.cat(seq, dim=axis)
        return np.concatenate(seq, axis=axis)

    def flip(self, arr, axis):
        if self.is_torch:
            return self.lib.flip(arr, dims=(axis,))
        return np.flip(arr, axis=axis)

    def argmax(self, arr, axis):
        if self.is_torch:
            if arr.dtype == self.lib.bool:
                arr = arr.to(self.lib.int64)
            return self.lib.argmax(arr, dim=axis)
        return np.argmax(arr, axis=axis)

    def sum(self, arr, axis=None):
        if self.is_torch:
            return self.lib.sum(arr, dim=axis)
        return np.sum(arr, axis=axis)

    def any(self, arr, axis=None):
        if self.is_torch:
            return self.lib.any(arr, dim=axis)
        return np.any(arr, axis=axis)

    def gather_last_axis(self, arr, indices):
        if self.is_torch:
            if indices.dtype != self.lib.int64:
                indices = indices.to(self.lib.int64)
            expanded = indices.unsqueeze(1).expand(-1, arr.shape[1], -1)
            return self.lib.gather(arr, 2, expanded)
        expanded = indices[:, None, :]
        return np.take_along_axis(arr, expanded, axis=2)

    def mod(self, arr, p: int):
        if p == 0:
            return arr
        return arr % p

    def to_numpy(self, arr):
        if self.is_torch:
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def eval(self, *arrs):
        if self.is_torch and self.device == "cuda":
            self.lib.cuda.synchronize()

    def _torch_dtype(self, dtype):
        if dtype == np.int16:
            return self.lib.int16
        if dtype == np.int32:
            return self.lib.int32
        if dtype == np.int64:
            return self.lib.int64
        raise ValueError(f"Unsupported dtype {dtype}")


def get_backend(name: str = "auto", device: str = "auto") -> TensorBackend:
    if name in {"auto", "torch"}:
        try:
            import torch

            if device == "auto":
                if torch.cuda.is_available():
                    return TensorBackend("torch", torch, "cuda")
                return TensorBackend("torch", torch, "cpu")

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

            return TensorBackend("torch", torch, device)
        except Exception:
            if name == "torch":
                raise

    return TensorBackend("numpy", np, None)
