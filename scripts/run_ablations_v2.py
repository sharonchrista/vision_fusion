"""
run_ablations_v2.py  —  Reviewer-corrected ablation runner.

Fixes for R2.1 (Information Fusion reviewers):
  - ALL variants run for identical 80 epochs (not 20)
  - Val loss recorded at every checkpoint (not just train loss)
  - 3 random seeds per variant: 42, 123, 456
  - Mean ± std reported across seeds
  - Welch t-test vs full_model for significance
  - Early-stopping patience identical across variants (patience=15)
  - Output: results/metrics/ablations/ablation_summary_v2.json

Variants (from config.yaml ablations section):
  1. camera_only
  2. camera_lidar
  3. camera_lidar_radar
  4. full_no_delta_t        ← KEY: must show Δt helps at 80 epochs
  5. full_lstm              ← KEY: must show LTC helps at 80 epochs
  6. full_model             ← baseline everything compared to

Usage:
    # Dry run (2 epochs) to verify pipeline:
    python scripts/run_ablations_v2.py --config configs/config.yaml --dry-run

    # Full run (80 epochs × 3 seeds × 6 variants = 18 training runs):
    python scripts/run_ablations_v2.py --config configs/config.yaml

    # Resume interrupted run (skips completed variants):
    python scripts/run_ablations_v2.py --config configs/config.yaml --resume

    # Run a single variant (for debugging):
    python scripts/run_ablations_v2.py --config configs/config.yaml \
        --only full_no_delta_t
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from src.utils.config import load_config, config_to_dict
from src.utils.logger import setup_logger, get_logger
from src.utils.reproducibility import seed_everything

log = get_logger()

# ── Seeds used across all variants ──────────────────────────────────────────
SEEDS = [42, 123, 456]

# ── Reviewer-required epoch budget ──────────────────────────────────────────
EPOCHS = 80          # identical for ALL variants — R2.1 fix
PATIENCE = 15        # early-stopping patience (val checks, not epochs)
VAL_EVERY = 2        # validate every N epochs (same as base config)


# ── Statistical helpers ──────────────────────────────────────────────────────

def mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n
    return mu, math.sqrt(var)


def welch_t_pvalue(a: list[float], b: list[float]) -> float:
    """
    Welch's t-test p-value (two-sided).
    Returns float in [0, 1]; < 0.05 = significant.
    Reference: Welch (1947), Biometrika.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")

    mu_a, mu_b = sum(a) / na, sum(b) / nb
    var_a = sum((x - mu_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mu_b) ** 2 for x in b) / (nb - 1)

    se = math.sqrt(var_a / na + var_b / nb)
    if se == 0:
        return 0.0

    t = (mu_a - mu_b) / se

    # Welch-Satterthwaite degrees of freedom
    df_num = (var_a / na + var_b / nb) ** 2
    df_den = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    df = df_num / df_den if df_den > 0 else 1.0

    # Approximate p-value using incomplete beta function (scipy not always avail)
    try:
        from scipy import stats
        p = float(stats.t.sf(abs(t), df) * 2)
    except ImportError:
        # Rough Cornish-Fisher approximation when scipy absent
        # Good enough for reporting; note in paper if used
        z = abs(t)
        p = math.exp(-0.717 * z - 0.416 * z * z)  # approximation
        p = min(1.0, max(0.0, p))
        log.warning("scipy not found — using approximate p-value. "
                    "Install scipy for exact results.")
    return p


# ── Config patching ──────────────────────────────────────────────────────────

def patch_config(base_cfg, ablation, seed: int, run_idx: int,
                 dry_run: bool = False):
    """
    Returns a deep-copy of base_cfg with ablation overrides applied.
    All training hyperparameters (lr, wd, schedule) are kept identical
    to the full model — only architecture is changed.
    """
    cfg = copy.deepcopy(base_cfg)

    # ── Epoch budget: identical for all variants (R2.1 fix) ──
    cfg.training.epochs          = 2 if dry_run else EPOCHS
    cfg.training.patience        = PATIENCE
    cfg.training.val_every_n_epochs = VAL_EVERY
    cfg.training.save_every_n_epochs = 10

    # ── Architecture switches ──
    cfg.ablation_modalities  = ablation.modalities
    cfg.ablation_use_ltc          = getattr(ablation, "ltc", True)
    cfg.ablation_inject_dt        = getattr(ablation, "inject_delta_t", True)
    cfg.ablation_use_transformer  = getattr(ablation, "use_transformer", False)
    cfg.ablation_name        = ablation.name

    # ── Per-run paths ──
    tag = f"{ablation.name}/seed{seed}/run{run_idx}"
    cfg.paths.checkpoints = str(
        Path(base_cfg.paths.checkpoints) / "ablations_v2" / tag
    )
    cfg.paths.logs = str(
        Path(base_cfg.paths.logs) / "ablations_v2" / tag
    )
    Path(cfg.paths.checkpoints).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.logs).mkdir(parents=True, exist_ok=True)

    # ── Reproducibility ──
    cfg.project.seed = seed

    # ── Disable wandb for ablation runs to reduce overhead ──
    cfg.logging.use_wandb = False

    return cfg


# ── Single training run ──────────────────────────────────────────────────────

def run_single(abl_cfg, seed: int, run_idx: int) -> dict:
    """
    Train one variant with one seed.
    Returns dict with val_loss history and best_val_loss.
    """
    from src.training.dataset import build_dataloader
    from src.models.liquidfusionnet import LiquidFusionNet
    from src.training.trainer import Trainer

    name = abl_cfg.ablation_name
    log.info(f"\n{'─'*60}")
    log.info(f"  {name}  |  seed={seed}  |  run {run_idx+1}/{len(SEEDS)}")
    log.info(f"  epochs={abl_cfg.training.epochs}  "
             f"ltc={abl_cfg.ablation_use_ltc}  "
             f"Δt={abl_cfg.ablation_inject_dt}  "
             f"modalities={abl_cfg.ablation_modalities}")
    log.info(f"{'─'*60}")

    seed_everything(seed)

    train_loader = build_dataloader(abl_cfg, split="train")
    val_loader   = build_dataloader(abl_cfg, split="val")

    trainer = Trainer(abl_cfg, rank=0, world_size=1)
    t0 = time.time()
    trainer.fit(train_loader, val_loader)
    elapsed = time.time() - t0

    # trainer.best_val must be the VALIDATION loss at best checkpoint
    # trainer.val_history must be list of (epoch, val_loss) tuples
    best_val  = float(trainer.best_val)
    val_hist  = getattr(trainer, "val_history", [])
    best_ep   = getattr(trainer, "best_epoch", -1)

    log.info(f"  ✓ {name} seed={seed}: "
             f"best_val={best_val:.4f} at epoch {best_ep} "
             f"({elapsed/60:.1f} min)")

    return {
        "ablation":    name,
        "seed":        seed,
        "run_idx":     run_idx,
        "best_val":    best_val,
        "best_epoch":  best_ep,
        "val_history": val_hist,
        "elapsed_min": round(elapsed / 60, 1),
        "epochs_run":  abl_cfg.training.epochs,
        "modalities":  abl_cfg.ablation_modalities,
        "use_ltc":     abl_cfg.ablation_use_ltc,
        "inject_dt":   abl_cfg.ablation_inject_dt,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(cfg_path: str, dry_run: bool = False,
         resume: bool = False, only: str | None = None) -> None:

    base_cfg  = load_config(cfg_path)
    ablations = base_cfg.ablations

    metrics_dir = Path(base_cfg.paths.metrics) / "ablations_v2"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    partial_path = metrics_dir / "partial_results.json"
    summary_path = metrics_dir / "ablation_summary_v2.json"

    # Load partial results if resuming
    completed: dict[str, list] = {}
    if resume and partial_path.exists():
        with open(partial_path) as f:
            completed = json.load(f)
        log.info(f"Resuming — found {sum(len(v) for v in completed.values())} "
                 f"completed runs across {len(completed)} variants")

    all_summaries = []

    for ablation in ablations:
        name = ablation.name

        # --only filter
        if only and name != only:
            continue

        log.info(f"\n{'='*60}")
        log.info(f"  VARIANT: {name}")
        log.info(f"{'='*60}")

        run_results = completed.get(name, [])
        completed_seeds = {r["seed"] for r in run_results}

        for run_idx, seed in enumerate(SEEDS):
            if seed in completed_seeds:
                log.info(f"  Skipping {name} seed={seed} (already done)")
                continue

            abl_cfg = patch_config(base_cfg, ablation, seed, run_idx,
                                   dry_run=dry_run)
            setup_logger(abl_cfg.paths.logs)

            result = run_single(abl_cfg, seed, run_idx)
            run_results.append(result)

            # Save partial results after every run
            completed[name] = run_results
            with open(partial_path, "w") as f:
                json.dump(completed, f, indent=2)

        # ── Aggregate across 3 seeds ──────────────────────────────────────
        vals = [r["best_val"] for r in run_results]
        mu, std = mean_std(vals)

        summary = {
            "ablation":        name,
            "modalities":      ablation.modalities,
            "use_ltc":         getattr(ablation, "ltc", True),
            "inject_dt":       getattr(ablation, "inject_delta_t", True),
            "epochs":          2 if dry_run else EPOCHS,
            "seeds":           SEEDS,
            "val_loss_mean":   round(mu, 4),
            "val_loss_std":    round(std, 4),
            "val_loss_per_seed": [round(v, 4) for v in vals],
            "runs":            run_results,
        }
        all_summaries.append(summary)

        log.info(f"\n  [{name}] "
                 f"val_loss = {mu:.4f} ± {std:.4f}  "
                 f"(seeds: {[round(v,4) for v in vals]})")

    # ── Significance tests vs full_model ─────────────────────────────────
    full_vals = next(
        (s["val_loss_per_seed"] for s in all_summaries
         if s["ablation"] == "full_model"), None
    )
    if full_vals:
        for s in all_summaries:
            if s["ablation"] == "full_model":
                s["p_vs_full"] = None
                s["significant"] = None
                continue
            p = welch_t_pvalue(s["val_loss_per_seed"], full_vals)
            s["p_vs_full"] = round(p, 4) if not math.isnan(p) else None
            s["significant"] = bool(p < 0.05) if not math.isnan(p) else None

    # ── Save final summary ────────────────────────────────────────────────
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    log.info(f"\nFull ablation summary saved → {summary_path}")

    # ── Print paper-ready table ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Variant':<25} {'Modalities':<25} {'Val loss':<20} {'p vs full':<12} {'Sig?'}")
    print("-" * 80)
    for s in all_summaries:
        mods = "+".join(m[:3] for m in s["modalities"])
        loss = f"{s['val_loss_mean']:.4f} ± {s['val_loss_std']:.4f}"
        p    = f"{s['p_vs_full']:.4f}" if s.get("p_vs_full") is not None else "—"
        sig  = "✓" if s.get("significant") else ("—" if s.get("significant") is None else "✗")
        print(f"{s['ablation']:<25} {mods:<25} {loss:<20} {p:<12} {sig}")
    print("=" * 80)
    print(f"\nAll epochs: {2 if dry_run else EPOCHS}  |  "
          f"Seeds: {SEEDS}  |  "
          f"Early-stop patience: {PATIENCE} val-checks")
    print(f"Results saved to: {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reviewer-corrected ablation runner (R2.1 fix)"
    )
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 epochs per variant to verify pipeline")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip completed variants/seeds")
    parser.add_argument("--only",    type=str, default=None,
                        help="Run only one named variant (e.g. full_no_delta_t)")
    args = parser.parse_args()

    main(args.config,
         dry_run=args.dry_run,
         resume=args.resume,
         only=args.only)