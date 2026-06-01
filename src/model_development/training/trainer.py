import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from src.model_development.model.morpheus import Morpheus
from src.model_development.model.mlm_head import WordMLMHead
from src.model_development.training.loss import (
    SkipGramLoss,
    RootFamilyContrastiveLoss,
    MorpheusCombinedLoss,
)
from src.model_development.training.dataset import (
    MorpheusSentenceDataset,
    get_sentence_loader,
    build_sentence_cache,
)
from src.model_development.training.callbacks import (
    Callback,
    CallbackHandler,
    CheckpointCallback,
    LossWeightSchedulerCallback,
    LoggerCallback,
    EarlyStoppingCallback,
    WandBCallback,
)
from src.model_development.utils.providers.logger_provider import global_logger


@dataclass
class TrainingConfig:
    train_cache_path: str = "cache/train_sentences.pt"
    val_cache_path: str = "cache/val_sentences.pt"
    word_vocab_path: str = "cache/word_vocab.pt"
    checkpoint_dir: str = "checkpoints/"
    run_name: str = "morpheus_v2"

    n_epochs: int = 20
    batch_size: int = 64
    grad_accum_steps: int = 2
    learning_rate: float = 7e-5
    warmup_steps: int = 1000
    weight_decay: float = 1e-2
    grad_clip: float = 0.3
    num_workers: int = 4
    adam_beta2: float = 0.98
    adam_eps: float = 1e-6
    lr_min_ratio: float = 0.2

    char_dim: int = 256
    char_embed_dim: int = 56
    case_embed_dim: int = 8
    n_layers_encoder: int = 2
    n_layers_detector: int = 2
    num_heads: int = 4
    max_word_len: int = 32
    max_sent_len: int = 24
    max_segs: int = 8
    dropout: float = 0.1
    threshold: float = 0.5
    pos_weight: float = 2.5
    count_loss_w: float = 0.1

    use_mlm: bool = True
    mlm_ctx_layers: int = 2
    mlm_dec_layers: int = 1
    mlm_mask_rate: float = 0.15

    sgns_n_negatives: int = 5
    sgns_window: int = 5

    ctr_temperature: float = 0.10

    aux_weight_start: float = 0.5
    aux_weight_end: float = 0.10
    aux_weight_decay: float = 0.92

    sgns_weight: float = 0.7
    ctr_weight: float = 0.5
    mlm_weight: float = 0.7

    use_amp: bool = True

    patience: int = 5
    min_delta: float = 1e-4

    log_every_n_steps: int = 100
    save_every_n_epochs: int = 1

    wandb_project: str = "morpheus-turkish"
    wandb_tags: List[str] = field(default_factory=lambda: ["turkish", "morphology", "v2"])
    wandb_notes: str = "Morpheus v2: SGNS + root-contrastive + MLM + deep supervision"


class MorpheusTrainer:
    def __init__(
            self,
            config: TrainingConfig,
            use_wandb: bool = False,
            callbacks: List[Callback] = None,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_logger.info(f"[Trainer] Device: {self.device}")

        self.train_dataset = MorpheusSentenceDataset(config.train_cache_path)
        self.val_dataset = MorpheusSentenceDataset(config.val_cache_path)

        self.model = Morpheus(
            char_dim=config.char_dim,
            char_embed_dim=config.char_embed_dim,
            case_embed_dim=config.case_embed_dim,
            n_layers_encoder=config.n_layers_encoder,
            n_layers_detector=config.n_layers_detector,
            num_heads=config.num_heads,
            max_word_len=config.max_word_len,
            max_segs=config.max_segs,
            dropout=config.dropout,
            threshold=config.threshold,
            pos_weight=config.pos_weight,
            count_loss_w=config.count_loss_w,
        ).to(self.device)
        self.model.parameter_summary()

        if config.use_mlm:
            self.mlm_head = WordMLMHead(
                char_vocab_size=self.model.char_encoder.encoder_helper.char_vocab_size,
                dim=config.char_dim,
                n_ctx_layers=config.mlm_ctx_layers,
                n_dec_layers=config.mlm_dec_layers,
                n_heads=config.num_heads,
                max_sent_len=config.max_sent_len,
                max_word_len=config.max_word_len,
                dropout=config.dropout,
                mask_rate=config.mlm_mask_rate,
                pad_id=self.model.char_encoder.encoder_helper._PAD_ID,
                bos_id=self.model.char_encoder.encoder_helper._BOS_ID,
            ).to(self.device)
            mlm_params = sum(p.numel() for p in self.mlm_head.parameters())
            global_logger.info(f"[Trainer] MLM head params: {mlm_params:,}")
        else:
            self.mlm_head = None

        self.sgns_loss = SkipGramLoss(
            vocab_size=self.train_dataset.word_vocab_size,
            dim=config.char_dim,
            n_negatives=config.sgns_n_negatives,
            window=config.sgns_window,
        ).to(self.device)
        if Path(config.word_vocab_path).exists():
            wv = torch.load(config.word_vocab_path)
            self.sgns_loss.set_unigram_distribution(wv["freqs"])
            global_logger.info(
                f"[Trainer] SGNS unigram distribution set (vocab={len(wv['vocab'])})"
            )

        self.contrastive_loss = RootFamilyContrastiveLoss(
            temperature=config.ctr_temperature,
        ).to(self.device)

        self.combined_loss = MorpheusCombinedLoss(
            sgns_loss=self.sgns_loss,
            contrastive_loss=self.contrastive_loss,
            mlm_head=self.mlm_head,
        )

        self.loss_weights = {
            "aux": config.aux_weight_start,
            "sgns": config.sgns_weight,
            "contrastive": config.ctr_weight,
            "mlm": config.mlm_weight if config.use_mlm else 0.0,
        }

        module_list = [self.model, self.sgns_loss]
        if self.mlm_head is not None:
            module_list.append(self.mlm_head)

        decay_params = []
        no_decay_params = []
        for module in module_list:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    name.endswith(".bias")
                    or "norm" in name.lower()
                    or "embedding" in name.lower()
                    or name == "mask_token"
                ):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.all_params = decay_params + no_decay_params
        global_logger.info(
            f"[Trainer] Param groups: decay={len(decay_params):,} "
            f"no_decay={len(no_decay_params):,}"
        )

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, config.adam_beta2),
            eps=config.adam_eps,
        )

        steps_per_epoch = max(1, len(self.train_dataset) // (config.batch_size * config.grad_accum_steps))
        total_steps = config.n_epochs * steps_per_epoch
        warmup_steps = min(config.warmup_steps, total_steps // 4)

        lr_floor = config.lr_min_ratio
        lr_span = 1.0 - lr_floor

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return lr_floor + lr_span * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=_lr_lambda,
        )

        self.scaler = GradScaler(
            device="cuda",
            enabled=(config.use_amp and self.device.type == "cuda"),
            init_scale=2**10,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=4000,
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        cb_list = [
            LossWeightSchedulerCallback(
                aux_start=config.aux_weight_start,
                aux_end=config.aux_weight_end,
                aux_decay=config.aux_weight_decay,
            ),
            CheckpointCallback(
                checkpoint_dir=config.checkpoint_dir,
                run_name=config.run_name,
                save_every_n=config.save_every_n_epochs,
            ),
            LoggerCallback(log_every_n_steps=config.log_every_n_steps),
            EarlyStoppingCallback(
                patience=config.patience,
                min_delta=config.min_delta,
            ),
        ]

        if use_wandb:
            cb_list.append(WandBCallback(
                project=config.wandb_project,
                name=config.run_name,
                config=config.__dict__,
                tags=config.wandb_tags,
                notes=config.wandb_notes,
                log_every_n=config.log_every_n_steps,
            ))

        if callbacks:
            cb_list.extend(callbacks)

        self.callback_handler = CallbackHandler(cb_list)

    def _get_loader(self, dataset: MorpheusSentenceDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

    def _compute_loss(self, batch: Dict) -> Dict:
        char_ids = batch["char_ids"].to(self.device, non_blocking=True)
        case_flags = batch["case_flags"].to(self.device, non_blocking=True)
        real_lengths = batch["real_lengths"].to(self.device, non_blocking=True)
        morfessor_labels = batch["morfessor_labels"].to(self.device, non_blocking=True)
        confidence = batch["confidence"].to(self.device, non_blocking=True)
        word_ids = batch["word_ids"].to(self.device, non_blocking=True)
        root_ids = batch["root_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

        B, T, L = char_ids.shape
        L_minus_1 = morfessor_labels.size(-1)

        flat_char_ids = char_ids.view(B * T, L)
        flat_case = case_flags.view(B * T, L)
        flat_real_lens = real_lengths.view(B * T)
        flat_labels = morfessor_labels.view(B * T, L_minus_1)
        flat_conf = confidence.view(B * T)
        flat_padding = (flat_char_ids == self.model.char_encoder.encoder_helper._PAD_ID)

        out = self.model(
            char_ids=flat_char_ids,
            case_flags=flat_case,
            real_lengths=flat_real_lens,
            padding_mask=flat_padding,
            morfessor_labels=flat_labels,
            confidence_weights=flat_conf,
        )

        total_loss, loss_dict = self.combined_loss(
            morpheus_output=out,
            word_ids=word_ids,
            root_ids=root_ids,
            attention_mask=attention_mask,
            target_char_ids=char_ids,
            weights=self.loss_weights,
        )
        return {"loss": total_loss, "info": loss_dict}

    def _run_epoch(self, loader: DataLoader) -> Dict:
        epoch_loss = 0.0
        n_micro = 0
        t0 = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        accum = self.config.grad_accum_steps

        running_info: Dict[str, float] = {}

        nan_skips = 0
        for micro_step, batch in enumerate(loader):
            with autocast(device_type="cuda", enabled=self.config.use_amp and self.device.type == "cuda"):
                out = self._compute_loss(batch)
                loss = out["loss"] / accum

            if not torch.isfinite(loss):
                nan_skips += 1
                if nan_skips % 25 == 1:
                    info = out["info"]
                    culprits = []
                    for k in ("aux_loss", "sgns_sgns_loss", "ctr_contrastive_loss", "mlm_mlm_loss"):
                        v = info.get(k, 0.0)
                        if not (v == v) or v in (float("inf"), float("-inf")):
                            culprits.append(k)
                    cur_scale = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
                    global_logger.warning(
                        f"[Trainer] Non-finite loss at step {self.global_step} "
                        f"(skipped {nan_skips}; scale={cur_scale:.0f}; "
                        f"culprits={culprits or 'unknown'})"
                    )
                if self.scaler.is_enabled():
                    new_scale = max(1.0, self.scaler.get_scale() * 0.5)
                    self.scaler._scale.fill_(new_scale)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            n_micro += 1
            epoch_loss += out["loss"].item()

            for k, v in out["info"].items():
                if isinstance(v, (int, float)):
                    running_info[k] = running_info.get(k, 0.0) + float(v)

            if (micro_step + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.all_params, self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                self.global_step += 1

                step_metrics = {k: v / accum for k, v in running_info.items()}
                step_metrics["grad_norm"] = grad_norm.item()
                step_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                self.callback_handler.on_step_end(self, self.global_step, step_metrics)
                running_info = {}

        return {
            "epoch_loss": epoch_loss / max(n_micro, 1),
            "epoch_time_s": time.time() - t0,
            "steps": self.global_step,
        }

    @torch.no_grad()
    def _val_epoch(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        if self.mlm_head is not None:
            self.mlm_head.eval()

        total_loss = 0.0
        total_aux = 0.0
        total_ctr = 0.0
        total_sgns = 0.0
        total_mlm = 0.0
        n = 0

        for batch in val_loader:
            with autocast(device_type="cuda", enabled=self.config.use_amp and self.device.type == "cuda"):
                out = self._compute_loss(batch)

            total_loss += out["loss"].item()
            info = out["info"]
            total_aux += info.get("aux_loss", 0.0)
            total_ctr += info.get("ctr_contrastive_loss", 0.0)
            total_sgns += info.get("sgns_sgns_loss", 0.0)
            total_mlm += info.get("mlm_mlm_loss", 0.0)
            n += 1

        self.model.train()
        if self.mlm_head is not None:
            self.mlm_head.train()

        return {
            "val_loss": total_loss / max(n, 1),
            "val_aux_loss": total_aux / max(n, 1),
            "val_contrastive_loss": total_ctr / max(n, 1),
            "val_sgns_loss": total_sgns / max(n, 1),
            "val_mlm_loss": total_mlm / max(n, 1),
        }

    def train(self) -> None:
        train_loader = self._get_loader(self.train_dataset, shuffle=True)
        val_loader = self._get_loader(self.val_dataset, shuffle=False)
        global_logger.info(f"[Trainer] Train stats: {self.train_dataset.stats()}")
        global_logger.info(f"[Trainer] Val stats: {self.val_dataset.stats()}")

        self.callback_handler.on_train_begin(self)
        global_logger.info(
            f"[Trainer] Training start: {self.config.n_epochs} epochs, "
            f"bs={self.config.batch_size}, accum={self.config.grad_accum_steps}"
        )

        for epoch in range(self.current_epoch, self.config.n_epochs):
            self.current_epoch = epoch
            self.model.train()
            if self.mlm_head is not None:
                self.mlm_head.train()

            self.callback_handler.on_epoch_begin(self, epoch)
            epoch_metrics = self._run_epoch(train_loader)
            val_metrics = self._val_epoch(val_loader)
            epoch_metrics.update(val_metrics)
            self.callback_handler.on_epoch_end(self, epoch, epoch_metrics)

            if self.callback_handler.should_stop:
                global_logger.info("[Trainer] Early stop triggered.")
                break

        self.callback_handler.on_train_end(self)
        global_logger.info("[Trainer] Training done.")

    def save_checkpoint(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "sgns_state": self.sgns_loss.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "loss_weights": self.loss_weights,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.mlm_head is not None:
            ckpt["mlm_head_state"] = self.mlm_head.state_dict()
        torch.save(ckpt, path)
        global_logger.info(f"[Trainer] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.sgns_loss.load_state_dict(ckpt["sgns_state"])
        if self.mlm_head is not None and "mlm_head_state" in ckpt:
            self.mlm_head.load_state_dict(ckpt["mlm_head_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.current_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.loss_weights = ckpt.get("loss_weights", self.loss_weights)
        self.best_val_loss = ckpt["best_val_loss"]
        global_logger.info(
            f"[Trainer] Loaded checkpoint: {path} "
            f"(epoch {ckpt['epoch']}, step {ckpt['global_step']})"
        )


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    train_txt = str(base / "benchmarker/dataset/splits/train.txt")
    test_txt = str(base / "benchmarker/dataset/splits/test.txt")
    morfessor_path = str(base / "benchmarker/results/morfessor_model.bin")
    word_vocab_path = str(base / "benchmarker/dataset/splits/word_vocab.pt")
    root_vocab_path = str(base / "benchmarker/dataset/splits/root_vocab.pt")
    train_cache = str(base / "benchmarker/dataset/splits/train_sentences.pt")
    test_cache = str(base / "benchmarker/dataset/splits/test_sentences.pt")
    checkpoint_dir = str(Path(__file__).parent / "checkpoints")

    build_sentence_cache(
        txt_path=train_txt,
        cache_path=train_cache,
        word_vocab_path=word_vocab_path,
        root_vocab_path=root_vocab_path,
        morfessor_path=morfessor_path,
        max_sentences=200_000,
    )

    build_sentence_cache(
        txt_path=test_txt,
        cache_path=test_cache,
        word_vocab_path=word_vocab_path,
        root_vocab_path=root_vocab_path,
        morfessor_path=morfessor_path,
        max_sentences=20_000,
    )

    config = TrainingConfig(
        train_cache_path=train_cache,
        val_cache_path=test_cache,
        word_vocab_path=word_vocab_path,
        checkpoint_dir=checkpoint_dir,
        run_name="turkish_morpheus",
        batch_size=64,
        grad_accum_steps=2,
        n_epochs=18,
        learning_rate=1e-4,
        warmup_steps=1000,
        use_amp=True,
        num_workers=4,
        use_mlm=True,
    )

    trainer = MorpheusTrainer(config, use_wandb=True)
    trainer.train()
