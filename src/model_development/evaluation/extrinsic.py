import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import defaultdict
from src.model_development.utils.providers.logger_provider import global_logger


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _train_probe(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        out_dim: int,
        epochs: int = 30,
        lr: float = 1e-2,
        batch_size: int = 256,
        device: torch.device = None,
) -> Tuple[float, float]:
    if device is None:
        device = train_x.device
    probe = LinearProbe(train_x.size(-1), out_dim).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    n = train_x.size(0)
    best_val = 0.0
    best_train = 0.0
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        total = 0
        correct = 0
        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            preds = logits.argmax(-1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / max(total, 1)

        probe.eval()
        with torch.no_grad():
            val_preds = probe(val_x).argmax(-1)
            val_acc = (val_preds == val_y).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_train = train_acc

    return best_train, best_val


def morpheme_count_probe(
        embeddings: torch.Tensor,
        morpheme_counts: torch.Tensor,
        device: torch.device,
        n_classes: int = 6,
        train_frac: float = 0.8,
        seed: int = 0,
) -> Dict[str, float]:
    counts = morpheme_counts.clamp(1, n_classes) - 1
    y = counts.long().to(device)
    x = embeddings.to(device)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = x.size(0)
    perm = torch.randperm(n, generator=gen)
    n_train = int(n * train_frac)

    train_x = x[perm[:n_train]]
    train_y = y[perm[:n_train]]
    val_x = x[perm[n_train:]]
    val_y = y[perm[n_train:]]

    train_acc, val_acc = _train_probe(
        train_x, train_y, val_x, val_y,
        out_dim=n_classes, device=device,
    )

    return {
        "morpheme_count_train_acc": round(train_acc, 4),
        "morpheme_count_val_acc": round(val_acc, 4),
        "morpheme_count_n_classes": n_classes,
        "morpheme_count_n_samples": int(n),
    }


def same_root_probe(
        embeddings: torch.Tensor,
        words: List[str],
        word_to_root: Dict[str, str],
        device: torch.device,
        n_pairs: int = 10000,
        train_frac: float = 0.8,
        seed: int = 0,
) -> Dict[str, float]:
    gen = torch.Generator().manual_seed(seed)

    root_groups: Dict[str, List[int]] = defaultdict(list)
    for i, w in enumerate(words):
        r = word_to_root.get(w)
        if r and r != "<UNK>":
            root_groups[r].append(i)

    multi_root_keys = [r for r, idxs in root_groups.items() if len(idxs) >= 2]
    if len(multi_root_keys) < 2:
        return {
            "same_root_train_acc": 0.0,
            "same_root_val_acc": 0.0,
            "same_root_n_pairs": 0,
        }

    pos_pairs: List[Tuple[int, int]] = []
    neg_pairs: List[Tuple[int, int]] = []

    while len(pos_pairs) < n_pairs // 2:
        ki = torch.randint(0, len(multi_root_keys), (1,), generator=gen).item()
        idxs = root_groups[multi_root_keys[ki]]
        ij = torch.randint(0, len(idxs), (2,), generator=gen).tolist()
        if ij[0] != ij[1]:
            pos_pairs.append((idxs[ij[0]], idxs[ij[1]]))

    V = embeddings.size(0)
    while len(neg_pairs) < n_pairs // 2:
        ij = torch.randint(0, V, (2,), generator=gen).tolist()
        if ij[0] == ij[1]:
            continue
        r_i = word_to_root.get(words[ij[0]])
        r_j = word_to_root.get(words[ij[1]])
        if r_i and r_j and r_i != r_j and r_i != "<UNK>" and r_j != "<UNK>":
            neg_pairs.append((ij[0], ij[1]))

    pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    perm = torch.randperm(len(pairs), generator=gen).tolist()
    pairs = [pairs[i] for i in perm]
    labels = [labels[i] for i in perm]

    emb_norm = F.normalize(embeddings, dim=-1)
    feats = []
    for i, j in pairs:
        e1 = emb_norm[i]
        e2 = emb_norm[j]
        feat = torch.cat([e1, e2, (e1 - e2).abs(), e1 * e2])
        feats.append(feat)

    x = torch.stack(feats).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    n = x.size(0)
    n_train = int(n * train_frac)
    train_x, train_y = x[:n_train], y[:n_train]
    val_x, val_y = x[n_train:], y[n_train:]

    train_acc, val_acc = _train_probe(
        train_x, train_y, val_x, val_y,
        out_dim=2, device=device,
    )

    return {
        "same_root_train_acc": round(train_acc, 4),
        "same_root_val_acc": round(val_acc, 4),
        "same_root_n_pairs": len(pairs),
    }
