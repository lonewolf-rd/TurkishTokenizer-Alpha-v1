import sys
import torch
from pathlib import Path

from src.model_development.model.morpheus import Morpheus
from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.training.dataset import MorfessorWrapper
from src.model_development.training.trainer import TrainingConfig

sys.modules["__main__"].TrainingConfig = TrainingConfig


def _format_with_pipe(word: str, boundary_positions, total_chars: int) -> str:
    parts = []
    current = ""
    for i in range(total_chars):
        current += word[i]
        if (i + 1) in boundary_positions and i + 1 < total_chars:
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return " | ".join(parts)


def diagnose(words, morpheus_model, morfessor, helper, device, threshold=0.5, max_word_len=32):
    print(f"\n{'='*100}")
    print(f"{'WORD':<22} {'MORFESSOR SEG':<35} {'TRUE BNDS':<14} {'MODEL BNDS':<14} {'MODEL OUTPUT':<25}")
    print("=" * 100)

    for word in words:
        ids, flags, rl = helper.word_to_char_ids(word, max_len=max_word_len)
        char_ids = torch.tensor([ids], device=device)
        case_flags = torch.tensor([flags], device=device)
        real_lengths = torch.tensor([rl], device=device)

        with torch.no_grad():
            out = morpheus_model(
                char_ids=char_ids,
                case_flags=case_flags,
                real_lengths=real_lengths,
            )

        boundary_probs = out["boundary_probs"][0].cpu().tolist()
        hard_b = [p > threshold for p in boundary_probs]

        n_chars = len(word)
        morfessor_segs, _ = morfessor.segment(word)

        true_bnd_positions = set()
        pos = 0
        for seg in morfessor_segs[:-1]:
            pos += len(seg)
            true_bnd_positions.add(pos)

        model_bnd_positions = set()
        for i in range(1, min(n_chars + 1, len(hard_b))):
            if hard_b[i]:
                model_bnd_positions.add(i)

        true_str = ",".join(str(p) for p in sorted(true_bnd_positions))
        model_str = ",".join(str(p) for p in sorted(model_bnd_positions))

        model_output = _format_with_pipe(word, model_bnd_positions, n_chars)

        print(
            f"{word:<22} "
            f"{' | '.join(morfessor_segs):<35} "
            f"{true_str:<14} "
            f"{model_str:<14} "
            f"{model_output:<25}"
        )

    print("=" * 100)
    print("\nINTERPRETATION:")
    print("  TRUE BNDS  = morphologically correct boundary positions (in word indexing, 0=start)")
    print("  MODEL BNDS = positions where boundary_probs[i] > threshold, i = char_ids index")
    print("  If MODEL BNDS = TRUE BNDS + 1 systematically, off-by-one bug confirmed")
    print("  (e.g. true=[3,5] model=[4,6] → +1 shift consistent with the suspected bug)")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.parent
    checkpoint_path = base / "src/model_development/training/checkpoints/morpheus_v2_epoch13.pt"
    morfessor_path = base / "src/benchmarker/results/morfessor_model.bin"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = Morpheus(
        char_dim=cfg.char_dim,
        char_embed_dim=cfg.char_embed_dim,
        case_embed_dim=cfg.case_embed_dim,
        n_layers_encoder=cfg.n_layers_encoder,
        n_layers_detector=cfg.n_layers_detector,
        num_heads=cfg.num_heads,
        max_word_len=cfg.max_word_len,
        max_segs=cfg.max_segs,
        dropout=cfg.dropout,
        threshold=cfg.threshold,
        pos_weight=cfg.pos_weight,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    morfessor = MorfessorWrapper(str(morfessor_path))
    helper = CharEncoderHelper()

    test_words = [
        "yanımda",
        "kitaplar",
        "kitaplarımı",
        "bizim",
        "ülkemizde",
        "çokça",
        "ekonomisi",
        "çeyrekte",
        "evlerimizdekiler",
        "gidiyorum",
        "muvaffakiyetsizleştiriciler",
        "geliyorum",
        "evler",
        "evde",
        "kitap",
        "ev",
    ]

    diagnose(test_words, model, morfessor, helper, device)
