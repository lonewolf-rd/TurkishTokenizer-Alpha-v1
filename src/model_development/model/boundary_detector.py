from typing import Tuple, Optional, List
import torch.nn.functional as F
import torch.nn as nn
import torch


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 64, base: int = 10000):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)

        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len * 2)

        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([
            x1 * cos[..., :half] - x2 * sin[..., :half],
            x1 * sin[..., half:] + x2 * cos[..., half:],
        ], dim=-1)


class BoundaryAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 4,
            dropout: float = 0.1,
    ):
        super().__init__()

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=64)

        self.boundary_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        residual = x

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]

        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if padding_mask is not None:
            pad = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        if padding_mask is not None:
            query_mask = padding_mask.unsqueeze(1).unsqueeze(-1)
            attn = attn.masked_fill(query_mask, 0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj(out)
        out = self.norm(out + residual)

        pairs = torch.cat([out[:, :-1, :], out[:, 1:, :]], dim=-1)
        boundary_logits = self.boundary_scorer(pairs).squeeze(-1)

        return out, boundary_logits


class MorfemAuxLoss(nn.Module):
    def __init__(
            self,
            pos_weight: float = 4.0,
            count_loss_w: float = 0.3,
    ):
        super().__init__()
        self.count_loss_w = count_loss_w
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(
            self,
            boundary_logits: torch.Tensor,
            morfessor_labels: torch.Tensor,
            confidence_weights: torch.Tensor = None,
            valid_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:

        boundary_logits = boundary_logits.float().clamp(min=-30.0, max=30.0)

        bce = F.binary_cross_entropy_with_logits(
            boundary_logits,
            morfessor_labels.float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if valid_mask is not None:
            vm = valid_mask.float()
            valid_per_word = vm.sum(dim=-1)
            bce_per_word = (bce * vm).sum(dim=-1) / valid_per_word.clamp(min=1.0)
            word_has_content = (valid_per_word > 0).float()
        else:
            bce_per_word = bce.mean(dim=-1)
            word_has_content = torch.ones_like(bce_per_word)

        pred_count = torch.sigmoid(boundary_logits)
        if valid_mask is not None:
            pred_count = (pred_count * valid_mask.float()).sum(dim=-1)
            true_count = (morfessor_labels.float() * valid_mask.float()).sum(dim=-1)
        else:
            pred_count = pred_count.sum(dim=-1)
            true_count = morfessor_labels.float().sum(dim=-1)

        count_per_word = (pred_count - true_count).pow(2)

        if confidence_weights is not None:
            per_word_loss = confidence_weights * bce_per_word + self.count_loss_w * count_per_word
            real_word_mask = (confidence_weights > 0).float() * word_has_content
        else:
            per_word_loss = bce_per_word + self.count_loss_w * count_per_word
            real_word_mask = word_has_content

        n_real = real_word_mask.sum().clamp(min=1.0)
        total_loss = (per_word_loss * real_word_mask).sum() / n_real

        bce_mean = (bce_per_word * real_word_mask).sum() / n_real
        count_mean = (count_per_word * real_word_mask).sum() / n_real

        loss_dict = {
            "bce_loss": bce_mean.item(),
            "count_loss": count_mean.item(),
            "total_aux": total_loss.item(),
        }

        return total_loss, loss_dict


class BoundaryDetector(nn.Module):
    def __init__(
            self,
            char_dim: int = 256,
            num_heads: int = 4,
            n_layers: int = 2,
            threshold: float = 0.5,
            dropout: float = 0.1,
            pos_weight: float = 4.0,
            count_loss_w: float = 0.3,
            depth_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.threshold = threshold
        self.n_layers = n_layers

        if depth_weights is None:
            depth_weights = [0.85 + 0.15 * (i + 1) / n_layers for i in range(n_layers)]
        assert len(depth_weights) == n_layers
        self.register_buffer(
            "depth_weights",
            torch.tensor(depth_weights, dtype=torch.float32),
        )

        self.layers = nn.ModuleList([
            BoundaryAttention(
                dim=char_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(char_dim),
                nn.Linear(char_dim, char_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(char_dim * 2, char_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        self.aux_loss_fn = MorfemAuxLoss(
            pos_weight=pos_weight,
            count_loss_w=count_loss_w,
        )

    def forward(
            self,
            char_context: torch.Tensor,
            padding_mask: torch.Tensor = None,
            morfessor_labels: torch.Tensor = None,
            confidence_weights: torch.Tensor = None,
    ) -> dict:
        x = char_context
        all_logits: List[torch.Tensor] = []

        for attn, ffn in zip(self.layers, self.ffns):
            x, layer_logits = attn(x, padding_mask)
            x = x + ffn(x)
            all_logits.append(layer_logits)

        final_logits = all_logits[-1]
        boundary_probs = torch.sigmoid(final_logits)
        hard_boundaries = boundary_probs > self.threshold

        result = {
            "boundary_logits": final_logits,
            "boundary_probs": boundary_probs,
            "hard_boundaries": hard_boundaries,
            "all_logits": all_logits,
            "aux_loss": None,
            "loss_dict": None,
        }

        if morfessor_labels is not None:
            if padding_mask is not None:
                valid = ~padding_mask[:, :-1] & ~padding_mask[:, 1:]
            else:
                valid = None

            total_aux = char_context.new_zeros(())
            agg_dict = {"bce_loss": 0.0, "count_loss": 0.0, "per_layer_aux": []}
            for layer_idx, logits in enumerate(all_logits):
                layer_loss, layer_dict = self.aux_loss_fn(
                    boundary_logits=logits,
                    morfessor_labels=morfessor_labels,
                    confidence_weights=confidence_weights,
                    valid_mask=valid,
                )
                w = self.depth_weights[layer_idx]
                total_aux = total_aux + w * layer_loss
                agg_dict["bce_loss"] += w.item() * layer_dict["bce_loss"]
                agg_dict["count_loss"] += w.item() * layer_dict["count_loss"]
                agg_dict["per_layer_aux"].append(layer_dict["total_aux"])

            w_sum = self.depth_weights.sum().item()
            agg_dict["bce_loss"] /= max(w_sum, 1e-8)
            agg_dict["count_loss"] /= max(w_sum, 1e-8)
            agg_dict["total_aux"] = total_aux.item()

            result["aux_loss"] = total_aux
            result["loss_dict"] = agg_dict

        return result

    def parameter_count(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "boundary_attention": count(self.layers),
            "ffns": count(self.ffns),
            "aux_loss_fn": count(self.aux_loss_fn),
            "total": count(self),
        }
