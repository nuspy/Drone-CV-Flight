"""Single place where all randomness is seeded, for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def rng(seed: int) -> np.random.Generator:
    """Independent generator for a subsystem; never touches the global state."""
    return np.random.default_rng(seed)
