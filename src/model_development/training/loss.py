import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class SkipGramLoss(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            dim: int,
            n_negatives: int = 5,
            window: int = 5,
            unk_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_negatives = n_negatives
        self.window = window
        self.unk_id = unk_id

        self.context_embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.context_embedding.weight, std=0.02)

        uniform = torch.ones(vocab_size, dtype=torch.float32)
        uniform[unk_id] = 0.0
        uniform = uniform / uniform.sum().clamp(min=1e-8)
        self.register_buffer("neg_dist", uniform)

    def set_unigram_distribution(self, freqs: torch.Tensor) -> None:
        dist = freqs.float() ** 0.75
        if dist.size(0) != self.vocab_size:
            raise ValueError(
                f"freqs size {dist.size(0)} != vocab_size {self.vocab_size}"
            )
        dist[self.unk_id] = 0.0
        dist = dist / dist.sum().clamp(min=1e-8)
        self.neg_dist.data.copy_(dist.to(self.neg_dist.device))

    def forward(
            self,
            word_embeddings: torch.Tensor,
            word_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        word_embeddings = word_embeddings.float()
        B, T, D = word_embeddings.shape
        device = word_embeddings.device

        total_pos_loss = word_embeddings.sum() * 0.0
        total_neg_loss = word_embeddings.sum() * 0.0
        n_pairs = 0

        for offset in range(1, self.window + 1):
            for direction in (-offset, offset):
                if direction < 0:
                    d = -direction
                    centers = word_embeddings[:, d:, :]
                    ctx_ids = word_ids[:, :-d]
                    valid = attention_mask[:, d:] & attention_mask[:, :-d]
                else:
                    d = direction
                    centers = word_embeddings[:, :-d, :]
                    ctx_ids = word_ids[:, d:]
                    valid = attention_mask[:, :-d] & attention_mask[:, d:]

                valid = valid & (ctx_ids != self.unk_id)
                n_valid = int(valid.sum().item())
                if n_valid == 0:
                    continue

                pos_ctx = self.context_embedding(ctx_ids).float()
                pos_score = (centers * pos_ctx).sum(-1).clamp(min=-30.0, max=30.0)
                pos_l = -F.logsigmoid(pos_score)
                total_pos_loss = total_pos_loss + (pos_l * valid.float()).sum()
                n_pairs += n_valid

                neg_ids = torch.multinomial(
                    self.neg_dist,
                    n_valid * self.n_negatives,
                    replacement=True,
                ).view(n_valid, self.n_negatives)

                valid_centers = centers[valid]
                neg_ctx = self.context_embedding(neg_ids).float()
                neg_score = (valid_centers.unsqueeze(1) * neg_ctx).sum(-1).clamp(min=-30.0, max=30.0)
                total_neg_loss = total_neg_loss + (-F.logsigmoid(-neg_score)).sum()

        if n_pairs == 0:
            zero = word_embeddings.sum() * 0.0
            return zero, {"sgns_pos": 0.0, "sgns_neg": 0.0, "sgns_loss": 0.0, "n_pairs": 0}

        pos_mean = total_pos_loss / n_pairs
        neg_mean = total_neg_loss / (n_pairs * self.n_negatives)
        total = pos_mean + neg_mean

        return total, {
            "sgns_pos": pos_mean.item(),
            "sgns_neg": neg_mean.item(),
            "sgns_loss": total.item(),
            "n_pairs": n_pairs,
        }


class RootFamilyContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
            self,
            word_embeddings: torch.Tensor,
            root_ids: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        word_embeddings = word_embeddings.float()
        device = word_embeddings.device
        N = word_embeddings.size(0)

        if N < 2 or valid_mask.sum() < 2:
            zero = word_embeddings.sum() * 0.0
            return zero, {"contrastive_loss": 0.0, "n_anchors": 0, "avg_n_pos": 0.0}

        emb = F.normalize(word_embeddings, dim=-1)
        sim = (emb @ emb.t()) / self.temperature

        eye = torch.eye(N, device=device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e4)

        invalid_row = ~valid_mask.unsqueeze(0)
        invalid_col = ~valid_mask.unsqueeze(1)
        sim = sim.masked_fill(invalid_row | invalid_col, -1e4)

        same_root = (root_ids.unsqueeze(0) == root_ids.unsqueeze(1))
        nonzero_root = (root_ids > 0).unsqueeze(0) & (root_ids > 0).unsqueeze(1)
        pos_mask = same_root & nonzero_root & ~eye & valid_mask.unsqueeze(0) & valid_mask.unsqueeze(1)

        has_pos = pos_mask.any(dim=-1) & valid_mask
        n_anchors = int(has_pos.sum().item())
        if n_anchors == 0:
            zero = word_embeddings.sum() * 0.0
            return zero, {"contrastive_loss": 0.0, "n_anchors": 0, "avg_n_pos": 0.0}

        log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
        n_pos_per = pos_mask.float().sum(dim=-1).clamp(min=1.0)
        loss_per = -(log_prob * pos_mask.float()).sum(dim=-1) / n_pos_per

        loss = (loss_per * has_pos.float()).sum() / max(n_anchors, 1)

        avg_n_pos = (n_pos_per * has_pos.float()).sum().item() / max(n_anchors, 1)

        return loss, {
            "contrastive_loss": loss.item(),
            "n_anchors": n_anchors,
            "avg_n_pos": avg_n_pos,
        }


class MorpheusCombinedLoss(nn.Module):
    def __init__(
            self,
            sgns_loss: SkipGramLoss,
            contrastive_loss: RootFamilyContrastiveLoss,
            mlm_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.sgns_loss = sgns_loss
        self.contrastive_loss = contrastive_loss
        self.mlm_head = mlm_head

    def forward(
            self,
            morpheus_output: Dict,
            word_ids: torch.Tensor,
            root_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            target_char_ids: torch.Tensor,
            weights: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict]:
        word_embs_flat = morpheus_output["word_embeddings"]
        B, T = attention_mask.shape
        D = word_embs_flat.size(-1)
        word_embs = word_embs_flat.view(B, T, D)

        aux_loss = morpheus_output["aux_loss"]
        aux_loss_value = aux_loss if aux_loss is not None else word_embs.sum() * 0.0

        sgns_l, sgns_info = self.sgns_loss(word_embs, word_ids, attention_mask)

        valid_flat = attention_mask.view(-1)
        root_flat = root_ids.view(-1)
        ctr_l, ctr_info = self.contrastive_loss(word_embs_flat, root_flat, valid_flat)

        if self.mlm_head is not None:
            mlm_l, mlm_info = self.mlm_head(word_embs, attention_mask, target_char_ids)
        else:
            mlm_l = word_embs.sum() * 0.0
            mlm_info = {"mlm_loss": 0.0, "n_masked": 0}

        w_aux = weights.get("aux", 0.5)
        w_sgns = weights.get("sgns", 1.0)
        w_ctr = weights.get("contrastive", 0.5)
        w_mlm = weights.get("mlm", 1.0)

        total = (
            w_aux * aux_loss_value
            + w_sgns * sgns_l
            + w_ctr * ctr_l
            + w_mlm * mlm_l
        )

        loss_dict = {
            "total_loss": total.item(),
            "aux_loss": aux_loss_value.item() if isinstance(aux_loss_value, torch.Tensor) else float(aux_loss_value),
            "w_aux": w_aux,
            "w_sgns": w_sgns,
            "w_contrastive": w_ctr,
            "w_mlm": w_mlm,
            **{f"sgns_{k}": v for k, v in sgns_info.items()},
            **{f"ctr_{k}": v for k, v in ctr_info.items()},
            **{f"mlm_{k}": v for k, v in mlm_info.items()},
        }

        if morpheus_output.get("loss_dict") is not None:
            for k, v in morpheus_output["loss_dict"].items():
                if not isinstance(v, list):
                    loss_dict[f"boundary_{k}"] = v

        return total, loss_dict
