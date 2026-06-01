import torch
import torch.nn as nn
from typing import Dict, Optional

from src.model_development.model.char_encoder import CharEncoder
from src.model_development.model.boundary_detector import BoundaryDetector
from src.model_development.model.segment_encoder import SegmentEncoder


class Morpheus(nn.Module):
    def __init__(
            self,
            char_dim: int = 256,
            char_embed_dim: int = 56,
            case_embed_dim: int = 8,
            n_layers_encoder: int = 2,
            n_layers_detector: int = 2,
            num_heads: int = 4,
            max_word_len: int = 32,
            max_segs: int = 8,
            dropout: float = 0.1,
            threshold: float = 0.5,
            pos_weight: float = 4.0,
            count_loss_w: float = 0.3,
    ):
        super().__init__()

        self.char_dim = char_dim
        self.threshold = threshold

        self.char_encoder = CharEncoder(
            char_embed_dim=char_embed_dim,
            case_embed_dim=case_embed_dim,
            char_dim=char_dim,
            n_attn_layers=n_layers_encoder,
            num_heads=num_heads,
            max_word_len=max_word_len,
            dropout=dropout,
        )

        self.boundary_detector = BoundaryDetector(
            char_dim=char_dim,
            num_heads=num_heads,
            n_layers=n_layers_detector,
            threshold=threshold,
            dropout=dropout,
            pos_weight=pos_weight,
            count_loss_w=count_loss_w,
        )

        self.segment_encoder = SegmentEncoder(
            dim=char_dim,
            max_segs=max_segs,
            dropout=dropout,
        )

    def forward(
            self,
            char_ids: torch.Tensor,
            case_flags: torch.Tensor,
            real_lengths: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            morfessor_labels: Optional[torch.Tensor] = None,
            confidence_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        if padding_mask is None:
            padding_mask = (char_ids == self.char_encoder.encoder_helper._PAD_ID)

        char_context = self.char_encoder(
            char_ids=char_ids,
            case_flags=case_flags,
            padding_mask=padding_mask,
        )

        detector_out = self.boundary_detector(
            char_context=char_context,
            padding_mask=padding_mask,
            morfessor_labels=morfessor_labels,
            confidence_weights=confidence_weights,
        )

        word_embeddings = self.segment_encoder(
            char_context=char_context,
            boundary_probs=detector_out["boundary_probs"],
            padding_mask=padding_mask,
            real_lengths=real_lengths,
        )

        return {
            "word_embeddings": word_embeddings,
            "char_context": char_context,
            "boundary_probs": detector_out["boundary_probs"],
            "hard_boundaries": detector_out["hard_boundaries"],
            "aux_loss": detector_out["aux_loss"],
            "loss_dict": detector_out["loss_dict"],
        }

    def parameter_summary(self) -> None:
        print(f"\n{'Module':<22} | {'Parameters':>12}")
        print("-" * 38)
        total = 0
        for name, child in self.named_children():
            p = sum(p.numel() for p in child.parameters())
            print(f"{name:<22} | {p:>12,}")
            total += p
        print("-" * 38)
        print(f"{'TOTAL':<22} | {total:>12,}\n")


if __name__ == "__main__":
    from src.model_development.model.char_encoder import CharEncoderHelper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    helper = CharEncoderHelper()

    model = Morpheus(
        char_dim=256,
        n_layers_encoder=2,
        n_layers_detector=2,
        dropout=0.1,
        threshold=0.5,
    ).to(device)

    model.parameter_summary()
    test_words = [
        "muhasebeleştirme",
        "gidebileceklerindenmişsiniz",
        "İstanbul",
        "ev",
        "teknoloji",
    ]

    B = len(test_words)
    MAX_LEN = 32

    all_ids, all_flags, all_real_lens, all_labels = [], [], [], []

    for word in test_words:
        ids, flags, rl = helper.word_to_char_ids(word, max_len=MAX_LEN)
        all_ids.append(ids)
        all_flags.append(flags)
        all_real_lens.append(rl)

        label = [0] * (MAX_LEN - 1)
        for i in range(3, MAX_LEN - 1, 4):
            label[i] = 1
        all_labels.append(label)

    char_ids = torch.tensor(all_ids, device=device)
    case_flags = torch.tensor(all_flags, device=device)
    real_lengths = torch.tensor(all_real_lens, device=device)
    morfessor_labels = torch.tensor(all_labels, device=device)
    confidence_w = torch.ones(B, device=device) * 0.9
    padding_mask = (char_ids == helper._PAD_ID)

    model.train()
    out = model(
        char_ids=char_ids,
        case_flags=case_flags,
        real_lengths=real_lengths,
        padding_mask=padding_mask,
        morfessor_labels=morfessor_labels,
        confidence_weights=confidence_w,
    )

    print("Training forward:")
    print(f"   word_embeddings : {out['word_embeddings'].shape}")
    print(f"   boundary_probs  : {out['boundary_probs'].shape}")
    print(f"   aux_loss        : {out['aux_loss'].item():.4f}")

    dummy = out["word_embeddings"].mean()
    total = dummy + 0.5 * out["aux_loss"]
    total.backward()
    print(f"   backward OK")

    model.eval()
    with torch.no_grad():
        out_inf = model(
            char_ids=char_ids,
            case_flags=case_flags,
            real_lengths=real_lengths,
            padding_mask=padding_mask,
        )

    print("\nInference forward:")
    print(f"   word_embeddings : {out_inf['word_embeddings'].shape}")
    for i, word in enumerate(test_words):
        n_b = out_inf["hard_boundaries"][i].sum().item()
        print(f"   '{word}' -> {int(n_b) + 1} segment")
