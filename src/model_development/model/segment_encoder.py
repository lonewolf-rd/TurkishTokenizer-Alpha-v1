import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SegmentEncoder(nn.Module):
    def __init__(
            self,
            dim: int = 256,
            max_segs: int = 8,
            min_seg_mass: float = 1e-2,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.max_segs = max_segs
        self.min_seg_mass = min_seg_mass

        self.char_scorer = nn.Linear(dim, 1, bias=False)

        self.fusion_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _soft_membership(
            boundary_probs: torch.Tensor,
            padding_mask: torch.Tensor,
            max_segs: int,
    ) -> torch.Tensor:
        B, L_minus_1 = boundary_probs.shape
        L = L_minus_1 + 1
        device = boundary_probs.device
        dtype = boundary_probs.dtype

        p = F.pad(boundary_probs, (1, 0), value=0.0)

        zero_idx = torch.zeros(B, dtype=torch.long, device=device)
        f_init = F.one_hot(zero_idx, num_classes=max_segs).to(dtype)

        f_list = [f_init]
        for i in range(1, L):
            p_i = p[:, i].unsqueeze(-1)
            prev = f_list[-1]
            no_b = prev * (1.0 - p_i)
            b_shifted = F.pad(prev[:, :-1], (1, 0), value=0.0)
            b = b_shifted * p_i
            overflow_mass = prev[:, -1:] * p_i
            sink_pad = F.pad(overflow_mass, (max_segs - 1, 0), value=0.0)
            f_list.append(no_b + b + sink_pad)

        f = torch.stack(f_list, dim=1)

        if padding_mask is not None:
            f = f.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return f

    def _pool(
            self,
            char_context: torch.Tensor,
            membership: torch.Tensor,
            padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        char_scores = self.char_scorer(char_context).squeeze(-1).float()
        char_scores = char_scores.masked_fill(padding_mask, -1e4)
        char_scores = char_scores.clamp(min=-30.0, max=30.0)

        max_scores = char_scores.max(dim=-1, keepdim=True).values
        exp_scores = (char_scores - max_scores).exp().unsqueeze(-1)

        weight = membership.float() * exp_scores
        weight_sum = weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weight = weight / weight_sum

        weight = weight.to(char_context.dtype)
        seg_vecs = torch.einsum("bls,bld->bsd", weight, char_context)

        seg_mass = membership.sum(dim=1)
        seg_valid = (seg_mass > self.min_seg_mass).float()

        return seg_vecs, seg_mass, seg_valid

    def forward(
            self,
            char_context: torch.Tensor,
            boundary_probs: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            real_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = char_context.shape
        device = char_context.device

        if padding_mask is None:
            padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        max_segs = min(self.max_segs, L)
        membership = self._soft_membership(boundary_probs, padding_mask, max_segs)

        seg_vecs, seg_mass, seg_valid = self._pool(char_context, membership, padding_mask)

        n_segs = seg_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        word_emb = (seg_vecs * seg_valid.unsqueeze(-1)).sum(dim=1) / n_segs

        word_emb = self.dropout(word_emb)
        word_emb = self.fusion_ffn(word_emb)
        word_emb = self.norm(word_emb)

        return word_emb

    def forward_with_segments(
            self,
            char_context: torch.Tensor,
            boundary_probs: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        B, L, D = char_context.shape
        device = char_context.device

        if padding_mask is None:
            padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        max_segs = min(self.max_segs, L)
        membership = self._soft_membership(boundary_probs, padding_mask, max_segs)
        seg_vecs, seg_mass, seg_valid = self._pool(char_context, membership, padding_mask)

        n_segs = seg_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        word_emb = (seg_vecs * seg_valid.unsqueeze(-1)).sum(dim=1) / n_segs

        word_emb = self.dropout(word_emb)
        word_emb = self.fusion_ffn(word_emb)
        word_emb = self.norm(word_emb)

        return {
            "word_embedding": word_emb,
            "segment_vectors": seg_vecs,
            "segment_mass": seg_mass,
            "segment_valid": seg_valid,
            "membership": membership,
        }

    def parameter_count(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "char_scorer": count(self.char_scorer),
            "fusion_ffn": count(self.fusion_ffn),
            "total": count(self),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L, D = 4, 32, 256
    char_context = torch.randn(B, L, D, device=device)

    boundary_probs_hard = torch.zeros(B, L - 1, device=device)
    boundary_probs_hard[:, [4, 9, 15]] = 1.0

    boundary_probs_soft = torch.sigmoid(torch.randn(B, L - 1, device=device))

    padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    padding_mask[:, 20:] = True

    encoder = SegmentEncoder(dim=D, max_segs=8).to(device)
    print(f"Param count: {encoder.parameter_count()}")

    out_hard = encoder(char_context, boundary_probs_hard, padding_mask)
    out_soft = encoder(char_context, boundary_probs_soft, padding_mask)
    print(f"Hard boundaries out: {out_hard.shape}")
    print(f"Soft boundaries out: {out_soft.shape}")

    info = encoder.forward_with_segments(char_context, boundary_probs_hard, padding_mask)
    print(f"Segment mass (hard, batch 0): {info['segment_mass'][0].tolist()}")
    print(f"Segment valid (hard, batch 0): {info['segment_valid'][0].tolist()}")

    out_hard.sum().backward()
    print(f"Backward OK (grad on char_scorer: {encoder.char_scorer.weight.grad is not None})")
