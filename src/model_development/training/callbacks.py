import os
import torch
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING
from src.model_development.utils.providers.logger_provider import global_logger

if TYPE_CHECKING:
    from src.model_development.training.trainer import MorpheusTrainer


class Callback:
    def on_train_begin(self, trainer: "MorpheusTrainer") -> None: pass
    def on_train_end(self, trainer: "MorpheusTrainer") -> None: pass
    def on_epoch_begin(self, trainer: "MorpheusTrainer", epoch: int) -> None: pass
    def on_epoch_end(self, trainer: "MorpheusTrainer", epoch: int, metrics: Dict) -> None: pass
    def on_step_end(self, trainer: "MorpheusTrainer", step: int, metrics: Dict) -> None: pass


class CallbackHandler:
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
        self.should_stop = False

    def on_train_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer, epoch):
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer, epoch, metrics):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_step_end(self, trainer, step, metrics):
        for cb in self.callbacks:
            cb.on_step_end(trainer, step, metrics)


class LossWeightSchedulerCallback(Callback):
    def __init__(
            self,
            aux_start: float = 0.50,
            aux_end: float = 0.05,
            aux_decay: float = 0.85,
    ):
        self.aux_start = aux_start
        self.aux_end = aux_end
        self.aux_decay = aux_decay

    def on_epoch_end(self, trainer, epoch, metrics):
        new_aux = max(
            self.aux_end,
            self.aux_start * (self.aux_decay ** epoch),
        )
        old_aux = trainer.loss_weights.get("aux", new_aux)
        trainer.loss_weights["aux"] = new_aux
        global_logger.info(
            f"[LossWeightScheduler] Epoch {epoch+1}: "
            f"aux {old_aux:.4f} -> {new_aux:.4f} "
            f"| sgns={trainer.loss_weights['sgns']:.2f} "
            f"| ctr={trainer.loss_weights['contrastive']:.2f} "
            f"| mlm={trainer.loss_weights['mlm']:.2f}"
        )


class CheckpointCallback(Callback):
    def __init__(
            self,
            checkpoint_dir: str = "checkpoints/",
            run_name: str = "morpheus",
            save_every_n: int = 1,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = run_name
        self.save_every_n = save_every_n
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer, epoch, metrics):
        if (epoch + 1) % self.save_every_n == 0:
            path = self.checkpoint_dir / f"{self.run_name}_epoch{epoch+1}.pt"
            trainer.save_checkpoint(str(path))

        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            best_path = self.checkpoint_dir / f"{self.run_name}_best.pt"
            trainer.save_checkpoint(str(best_path))
            global_logger.info(
                f"[Checkpoint] Best model updated: val_loss={val_loss:.4f}"
            )

    def on_train_end(self, trainer):
        final_path = self.checkpoint_dir / f"{self.run_name}_final.pt"
        trainer.save_checkpoint(str(final_path))
        global_logger.info(f"[Checkpoint] Final model saved: {final_path}")


class LoggerCallback(Callback):
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, trainer, step, metrics):
        if step % self.log_every_n_steps == 0:
            global_logger.info(
                f"[Step {step:>7}] "
                f"total={metrics.get('total_loss', 0):.4f} | "
                f"aux={metrics.get('aux_loss', 0):.4f} | "
                f"sgns={metrics.get('sgns_sgns_loss', 0):.4f} | "
                f"ctr={metrics.get('ctr_contrastive_loss', 0):.4f} | "
                f"mlm={metrics.get('mlm_mlm_loss', 0):.4f} | "
                f"grad={metrics.get('grad_norm', 0):.3f} | "
                f"lr={metrics.get('lr', 0):.2e}"
            )

    def on_epoch_end(self, trainer, epoch, metrics):
        global_logger.info(
            f"\n{'='*70}\n"
            f"[Epoch {epoch+1:>3}] "
            f"train={metrics.get('epoch_loss', 0):.4f} | "
            f"val={metrics.get('val_loss', 0):.4f} | "
            f"val_aux={metrics.get('val_aux_loss', 0):.4f} | "
            f"val_sgns={metrics.get('val_sgns_loss', 0):.4f} | "
            f"val_ctr={metrics.get('val_contrastive_loss', 0):.4f} | "
            f"val_mlm={metrics.get('val_mlm_loss', 0):.4f} | "
            f"time={metrics.get('epoch_time_s', 0):.1f}s\n"
            f"{'='*70}"
        )


class EarlyStoppingCallback(Callback):
    def __init__(
            self,
            patience: int = 5,
            min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, trainer, epoch, metrics):
        val_loss = metrics.get("val_loss", float("inf"))

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            global_logger.info(
                f"[EarlyStopping] No improvement: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                global_logger.info(
                    f"[EarlyStopping] {self.patience} epochs no improvement — stopping."
                )
                trainer.callback_handler.should_stop = True


class WandBCallback(Callback):
    def __init__(
            self,
            project: str,
            name: str,
            config: Dict = None,
            tags: List[str] = None,
            log_every_n: int = 50,
            notes: str = None,
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.log_every_n = log_every_n
        self.notes = notes
        self._run = None

    def on_train_begin(self, trainer):
        try:
            import wandb
            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                resume="allow",
            )
            total_params = sum(p.numel() for p in trainer.model.parameters())
            if trainer.mlm_head is not None:
                total_params += sum(p.numel() for p in trainer.mlm_head.parameters())
            total_params += sum(p.numel() for p in trainer.sgns_loss.parameters())
            wandb.config.update({
                "total_parameters": total_params,
                "device": str(trainer.device),
            })
            global_logger.info(f"[WandB] Run started: {self._run.url}")
        except ImportError:
            global_logger.warning("[WandB] wandb not installed — pip install wandb")
        except Exception as e:
            global_logger.error(f"[WandB] Init failed: {e}")

    def on_step_end(self, trainer, step, metrics):
        if self._run is None:
            return
        if step % self.log_every_n == 0:
            try:
                import wandb
                wandb.log({
                    "train/total_loss": metrics.get("total_loss", 0),
                    "train/aux_loss": metrics.get("aux_loss", 0),
                    "train/sgns_loss": metrics.get("sgns_sgns_loss", 0),
                    "train/contrastive_loss": metrics.get("ctr_contrastive_loss", 0),
                    "train/mlm_loss": metrics.get("mlm_mlm_loss", 0),
                    "train/grad_norm": metrics.get("grad_norm", 0),
                    "train/lr": metrics.get("lr", 0),
                    "train/w_aux": metrics.get("w_aux", 0),
                }, step=step)
            except Exception as e:
                global_logger.error(f"[WandB] Step log error: {e}")

    def on_epoch_end(self, trainer, epoch, metrics):
        if self._run is None:
            return
        try:
            import wandb
            log_dict = {
                "epoch/train_loss": metrics.get("epoch_loss", 0),
                "epoch/val_loss": metrics.get("val_loss", 0),
                "epoch/val_aux_loss": metrics.get("val_aux_loss", 0),
                "epoch/val_sgns_loss": metrics.get("val_sgns_loss", 0),
                "epoch/val_contrastive_loss": metrics.get("val_contrastive_loss", 0),
                "epoch/val_mlm_loss": metrics.get("val_mlm_loss", 0),
                "epoch/lr": trainer.optimizer.param_groups[0]["lr"],
                "epoch/epoch_time_s": metrics.get("epoch_time_s", 0),
                "epoch/aux_weight": trainer.loss_weights.get("aux", 0),
                "epoch": epoch + 1,
            }

            bm = self._compute_boundary_metrics(trainer)
            if bm is not None:
                log_dict["epoch/boundary_accuracy"] = bm["accuracy"]
                log_dict["epoch/boundary_precision"] = bm["precision"]
                log_dict["epoch/boundary_recall"] = bm["recall"]
                log_dict["epoch/boundary_f1"] = bm["f1"]
                global_logger.info(
                    f"[WandB] Boundary metrics: "
                    f"acc={bm['accuracy']:.3f} P={bm['precision']:.3f} "
                    f"R={bm['recall']:.3f} F1={bm['f1']:.3f}"
                )

            wandb.log(log_dict, step=trainer.global_step)

        except Exception as e:
            global_logger.error(f"[WandB] Epoch log error: {e}")

    def _compute_boundary_metrics(self, trainer) -> Optional[Dict]:
        try:
            import torch
            from torch.utils.data import DataLoader

            loader = DataLoader(
                trainer.val_dataset,
                batch_size=trainer.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

            trainer.model.eval()
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            n_seen = 0
            max_batches = 16

            with torch.no_grad():
                for batch in loader:
                    if n_seen >= max_batches:
                        break
                    n_seen += 1
                    char_ids = batch["char_ids"].to(trainer.device)
                    case_flags = batch["case_flags"].to(trainer.device)
                    real_lengths = batch["real_lengths"].to(trainer.device)
                    labels = batch["morfessor_labels"].to(trainer.device)
                    B, T, L = char_ids.shape
                    L_minus_1 = labels.size(-1)
                    flat_ids = char_ids.view(B * T, L)
                    flat_case = case_flags.view(B * T, L)
                    flat_lens = real_lengths.view(B * T)
                    flat_labels = labels.view(B * T, L_minus_1)
                    flat_pad = (flat_ids == trainer.model.char_encoder.encoder_helper._PAD_ID)

                    out = trainer.model(
                        char_ids=flat_ids,
                        case_flags=flat_case,
                        real_lengths=flat_lens,
                        padding_mask=flat_pad,
                    )
                    hard = out["hard_boundaries"].long()
                    valid = (~flat_pad[:, :-1]) & (~flat_pad[:, 1:])
                    pred = hard[valid]
                    true = flat_labels[valid]
                    tp += ((pred == 1) & (true == 1)).sum().item()
                    fp += ((pred == 1) & (true == 0)).sum().item()
                    fn += ((pred == 0) & (true == 1)).sum().item()
                    tn += ((pred == 0) & (true == 0)).sum().item()

            trainer.model.train()
            total = tp + fp + fn + tn
            acc = (tp + tn) / max(total, 1)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            return {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        except Exception as e:
            global_logger.error(f"[WandB] Boundary metrics failed: {e}")
            return None

    def on_train_end(self, trainer):
        if self._run is None:
            return
        try:
            import wandb
            artifact = wandb.Artifact(
                name=f"{self.name}_model",
                type="model",
                description="Morpheus v2 trained model",
            )
            best_path = Path(trainer.config.checkpoint_dir) / f"{trainer.config.run_name}_best.pt"
            if best_path.exists():
                artifact.add_file(str(best_path))
                self._run.log_artifact(artifact)
            wandb.finish()
            global_logger.info("[WandB] Run finished.")
        except Exception as e:
            global_logger.error(f"[WandB] Finish error: {e}")
