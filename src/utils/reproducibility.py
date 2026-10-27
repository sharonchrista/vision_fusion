"""
Reproducibility helpers — seed everything for T-ITS 3-run requirement.
"""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops where available (slight perf cost — acceptable for paper runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Pass to DataLoader for reproducible multi-process data loading."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
