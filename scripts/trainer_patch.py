import sys
from pathlib import Path

TRAINER = Path("/home/sharon/vision_fusion/src/training/trainer.py")

if not TRAINER.exists():
    print(f"ERROR: {TRAINER} not found", file=sys.stderr)
    sys.exit(1)

src = TRAINER.read_text()

OLD1 = '        self.best_val     = float("inf")'
NEW1 = '''        self.best_val     = float("inf")
        self.best_epoch   = -1
        self.val_history  = []
        self._patience    = getattr(getattr(cfg, "training", cfg), "patience", 15)
        self._no_improve  = 0'''

if OLD1 not in src:
    print(f"ERROR: Patch 1 anchor not found"); sys.exit(1)
src = src.replace(OLD1, NEW1, 1)
print("Patch 1 OK")

OLD2 = '''                    if val_metrics["loss_total"] < self.best_val:
                        self.best_val = val_metrics["loss_total"]
                        self._save_checkpoint(epoch, val_metrics, tag="best")'''
NEW2 = '''                    self.val_history.append([epoch, float(val_metrics["loss_total"])])
                    if val_metrics["loss_total"] < self.best_val:
                        self.best_val    = val_metrics["loss_total"]
                        self.best_epoch  = epoch
                        self._no_improve = 0
                        self._save_checkpoint(epoch, val_metrics, tag="best")
                    else:
                        self._no_improve += 1
                        log.info(f"  No improvement: {self._no_improve}/{self._patience} patience checks")'''

if OLD2 not in src:
    print(f"ERROR: Patch 2 anchor not found"); sys.exit(1)
src = src.replace(OLD2, NEW2, 1)
print("Patch 2 OK")

OLD3 = '''            # ── Periodic checkpoint ──
            if self.is_master and (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, {}, tag=f"epoch{epoch+1:03d}")'''
NEW3 = '''            # ── Early stopping ──
            stop_flag = torch.tensor(
                1 if (self.is_master and self._no_improve >= self._patience) else 0,
                device=self.device
            )
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(stop_flag, src=0)
            if stop_flag.item():
                if self.is_master:
                    log.info(f"Early stopping at epoch {epoch+1}: best val={self.best_val:.4f} at epoch {self.best_epoch+1}.")
                break

            # ── Periodic checkpoint ──
            if self.is_master and (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, {}, tag=f"epoch{epoch+1:03d}")'''

if OLD3 not in src:
    print(f"ERROR: Patch 3 anchor not found"); sys.exit(1)
src = src.replace(OLD3, NEW3, 1)
print("Patch 3 OK")

TRAINER.write_text(src)
print(f"Done → {TRAINER}")
