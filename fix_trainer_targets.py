"""
fix_trainer_targets.py — Patch trainer.py to unpack (batch, targets) from
the dataloader and pass targets to the loss function.

Run with:
    python fix_trainer_targets.py
"""
import sys
from pathlib import Path

TRAINER = Path("/home/sharon/vision_fusion/src/training/trainer.py")

if not TRAINER.exists():
    print(f"ERROR: {TRAINER} not found", file=sys.stderr)
    sys.exit(1)

src = TRAINER.read_text()

# ── PATCH 1: _train_epoch — unpack targets, move to device, pass to criterion ─
OLD1 = '''\
        for batch in loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs, _ = self.model(batch)
                loss, breakdown = self.criterion(outputs, {})'''

NEW1 = '''\
        for batch_and_targets in loader:
            # dataset returns (batch, targets); targets may be {} for unlabelled data
            if isinstance(batch_and_targets, (list, tuple)) and len(batch_and_targets) == 2:
                batch, targets = batch_and_targets
            else:
                batch, targets = batch_and_targets, {}
            batch   = self._to_device(batch)
            targets = self._to_device(targets)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs, _ = self.model(batch)
                loss, breakdown = self.criterion(outputs, targets)'''

if OLD1 not in src:
    print(f"ERROR: Patch 1 anchor not found", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD1, NEW1, 1)
print("Patch 1 applied: _train_epoch unpacks (batch, targets)")


# ── PATCH 2: _val_epoch — same unpack ────────────────────────────────────────
OLD2 = '''\
        for batch in loader:
            batch = self._to_device(batch)
            with autocast(enabled=self.use_amp):
                outputs, _ = self.model(batch)
                _, breakdown = self.criterion(outputs, {})'''

NEW2 = '''\
        for batch_and_targets in loader:
            if isinstance(batch_and_targets, (list, tuple)) and len(batch_and_targets) == 2:
                batch, targets = batch_and_targets
            else:
                batch, targets = batch_and_targets, {}
            batch   = self._to_device(batch)
            targets = self._to_device(targets)
            with autocast(enabled=self.use_amp):
                outputs, _ = self.model(batch)
                _, breakdown = self.criterion(outputs, targets)'''

if OLD2 not in src:
    print(f"ERROR: Patch 2 anchor not found", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD2, NEW2, 1)
print("Patch 2 applied: _val_epoch unpacks (batch, targets)")


TRAINER.write_text(src)
print(f"\nBoth patches applied → {TRAINER}")
print("\nNow run the dry-run:")
print("  PYTHONPATH=/home/sharon/vision_fusion \\")
print("    python scripts/run_ablations_v2.py --dry-run 2>&1 | tail -30")