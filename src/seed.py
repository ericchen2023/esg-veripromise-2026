"""Global seeding utilities for deterministic experiments.

Determinism level (Phase 1 reproducibility hardening, 2026-05-21 audit follow-up):
  - Python ``random`` / NumPy / ``torch.manual_seed`` / ``torch.cuda.manual_seed_all``: always
  - cuDNN deterministic algorithms + benchmark off: enabled by default
  - ``torch.use_deterministic_algorithms(True)`` + ``CUBLAS_WORKSPACE_CONFIG=:4096:8``:
    enabled by default; can be relaxed by setting env ``ESG_STRICT_DETERMINISM=0``
    (e.g., when an op without a deterministic implementation must be tolerated).

Stored OOF/ckpt artifacts produced before this hardening remain bit-exact and
the ensemble score (AP-D4 = 0.71608) is recomputable from those artifacts.
Any *new* training run started after this hardening is meant to be reproducible
to within numerical noise across machines that share the same PyTorch / CUDA
build, hardware class, and pinned dependency versions.
"""
from __future__ import annotations

import os
import random

import numpy as np


_STRICT = os.environ.get("ESG_STRICT_DETERMINISM", "1").strip() not in ("0", "false", "False")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility.

    Also enables cuDNN / cuBLAS deterministic flags unless strict determinism is
    explicitly disabled via ``ESG_STRICT_DETERMINISM=0``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _STRICT:
        # Required for ``torch.use_deterministic_algorithms(True)`` with cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if _STRICT:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                # Older torch builds may not support ``warn_only``.
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
    except ImportError:
        pass
