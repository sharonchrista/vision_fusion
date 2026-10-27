"""
Unified logger — writes to console, TensorBoard, and optionally W&B.
"""
import logging
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

_logger: logging.Logger | None = None
_tb_writer: SummaryWriter | None = None
_wandb_run = None


def setup_logger(log_dir: str | Path, use_wandb: bool = False,
                 wandb_project: str = "", run_name: str = "") -> logging.Logger:
    global _logger, _tb_writer, _wandb_run

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger = logging.getLogger("liquidfusion")
    _logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    _logger.addHandler(sh)
    _logger.addHandler(fh)

    _tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    if use_wandb:
        try:
            import wandb
            _wandb_run = wandb.init(project=wandb_project, name=run_name,
                                    resume="allow")
        except ImportError:
            _logger.warning("wandb not installed — skipping W&B logging.")

    return _logger


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("liquidfusion")
        _logger.setLevel(logging.INFO)
        _logger.addHandler(logging.StreamHandler(sys.stdout))
    return _logger


def log_scalar(tag: str, value: float, step: int) -> None:
    if _tb_writer:
        _tb_writer.add_scalar(tag, value, step)
    if _wandb_run:
        import wandb
        wandb.log({tag: value}, step=step)


def log_scalars(metrics: dict, step: int) -> None:
    for k, v in metrics.items():
        log_scalar(k, v, step)


def close_logger() -> None:
    if _tb_writer:
        _tb_writer.close()
    if _wandb_run:
        import wandb
        wandb.finish()
