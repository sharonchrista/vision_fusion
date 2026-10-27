"""
Ablation runner — trains all 6 model variants required for T-ITS.

Variants (defined in config.yaml under ablations):
  1. camera_only
  2. camera_lidar
  3. camera_lidar_radar
  4. full_no_delta_t
  5. full_lstm
  6. full_model  (main result)

Run:
    python scripts/run_ablations.py --config configs/config.yaml
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.utils.config import load_config, config_to_dict
from src.utils.logger import setup_logger, get_logger
from src.utils.reproducibility import seed_everything

log = get_logger()


def patch_config_for_ablation(base_cfg, ablation: SimpleNamespace) -> SimpleNamespace:
    """
    Returns a modified copy of base_cfg with ablation overrides applied.
    Modifications:
      - modalities: disable unused encoders (zero their feat_dim to 0 or skip)
      - ltc: swap to GRU if ablation.ltc == False
      - inject_delta_t: disable Δt injection
    """
    cfg_dict = config_to_dict(base_cfg)
    cfg = copy.deepcopy(base_cfg)

    modalities = ablation.modalities
    # Zero out feat_dims for disabled modalities so fused_dim stays consistent
    # (we keep the architecture identical but the encoder outputs zeros)
    cfg.ablation_modalities   = modalities
    cfg.ablation_use_ltc      = getattr(ablation, "ltc", True)
    cfg.ablation_inject_dt    = getattr(ablation, "inject_delta_t", True)
    cfg.ablation_name         = ablation.name

    return cfg


def run_single_ablation(ablation_cfg, run_idx: int) -> dict:
    """
    Imports and runs a single training + evaluation run.
    Returns final val metrics.
    """
    from src.training.dataset import build_dataloader
    from src.models.liquidfusionnet import LiquidFusionNet
    from src.training.trainer import Trainer
    from src.evaluation.evaluator import Evaluator

    name = ablation_cfg.ablation_name
    log.info(f"\n{'='*60}")
    log.info(f"  Ablation: {name}  |  run {run_idx+1}/3")
    log.info(f"{'='*60}")

    # Override checkpoint dir per ablation + run
    ablation_cfg.paths.checkpoints = str(
        Path(ablation_cfg.paths.checkpoints) / name / f"run{run_idx}"
    )
    ablation_cfg.paths.logs = str(
        Path(ablation_cfg.paths.logs) / name / f"run{run_idx}"
    )
    setup_logger(ablation_cfg.paths.logs)

    seed_everything(ablation_cfg.project.seed + run_idx)

    train_loader = build_dataloader(ablation_cfg, split="train")
    val_loader   = build_dataloader(ablation_cfg, split="val")

    trainer = Trainer(ablation_cfg, rank=0, world_size=1)
    trainer.fit(train_loader, val_loader)

    return {"ablation": name, "run": run_idx, "best_val": trainer.best_val}


def main(cfg_path: str) -> None:
    base_cfg   = load_config(cfg_path)
    ablations  = base_cfg.ablations   # list of SimpleNamespace

    all_results = []
    metrics_dir = Path(base_cfg.paths.metrics) / "ablations"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for ablation in ablations:
        abl_results = []
        for run_idx in range(3):   # T-ITS: 3 runs for statistical significance
            abl_cfg = patch_config_for_ablation(base_cfg, ablation)
            result  = run_single_ablation(abl_cfg, run_idx)
            abl_results.append(result)

        # Compute mean ± std over 3 runs
        vals = [r["best_val"] for r in abl_results]
        summary = {
            "ablation":    ablation.name,
            "val_loss_mean": round(float(sum(vals)/len(vals)), 4),
            "val_loss_std":  round(float(
                (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals))**0.5
            ), 4),
            "runs": abl_results,
        }
        all_results.append(summary)
        log.info(
            f"[{ablation.name}] "
            f"val_loss = {summary['val_loss_mean']:.4f} ± {summary['val_loss_std']:.4f}"
        )

    # Save full ablation table
    out_path = metrics_dir / "ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nAblation summary saved → {out_path}")

    # Print T-ITS-ready table
    print("\n" + "="*70)
    print(f"{'Variant':<28} {'Val loss (mean±std)':>22}")
    print("-"*70)
    for r in all_results:
        print(f"{r['ablation']:<28} "
              f"{r['val_loss_mean']:>10.4f} ± {r['val_loss_std']:<8.4f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
