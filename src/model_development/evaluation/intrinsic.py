import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.utils.providers.logger_provider import global_logger


@torch.no_grad()
def embed_word_list(
        model,
        helper: CharEncoderHelper,
        words: List[str],
        device: torch.device,
        max_word_len: int = 32,
        batch_size: int = 256,
) -> torch.Tensor:
    model.eval()
    all_embs = []

    for start in range(0, len(words), batch_size):
        chunk = words[start: start + batch_size]
        ids_list, flags_list, lens_list = [], [], []
        for w in chunk:
            ids, flags, rl = helper.word_to_char_ids(w, max_len=max_word_len)
            ids_list.append(ids)
            flags_list.append(flags)
            lens_list.append(rl)

        char_ids = torch.tensor(ids_list, device=device)
        case_flags = torch.tensor(flags_list, device=device)
        real_lens = torch.tensor(lens_list, device=device)

        out = model(
            char_ids=char_ids,
            case_flags=case_flags,
            real_lengths=real_lens,
        )
        all_embs.append(out["word_embeddings"].detach().float().cpu())

    return torch.cat(all_embs, dim=0)


def root_cluster_coherence(
        word_embeddings: torch.Tensor,
        words: List[str],
        word_to_root: Dict[str, str],
        n_neg_samples: int = 2000,
        min_group_size: int = 3,
        seed: int = 0,
) -> Dict[str, float]:
    word_to_idx = {w: i for i, w in enumerate(words)}

    root_groups: Dict[str, List[int]] = defaultdict(list)
    for w in words:
        r = word_to_root.get(w)
        if r and r != "<UNK>":
            root_groups[r].append(word_to_idx[w])

    emb = F.normalize(word_embeddings, dim=-1)

    intra_sims: List[float] = []
    group_sizes: List[int] = []
    for root, idxs in root_groups.items():
        if len(idxs) < min_group_size:
            continue
        ix = torch.tensor(idxs)
        g = emb[ix]
        sim = g @ g.t()
        mask = ~torch.eye(len(idxs), dtype=torch.bool)
        intra_sims.append(sim[mask].mean().item())
        group_sizes.append(len(idxs))

    gen = torch.Generator().manual_seed(seed)
    V = emb.size(0)
    inter_sims: List[float] = []
    for _ in range(n_neg_samples):
        ij = torch.randint(0, V, (2,), generator=gen).tolist()
        if ij[0] != ij[1]:
            r_i = word_to_root.get(words[ij[0]])
            r_j = word_to_root.get(words[ij[1]])
            if r_i and r_j and r_i != r_j:
                inter_sims.append((emb[ij[0]] @ emb[ij[1]]).item())

    intra = sum(intra_sims) / max(len(intra_sims), 1)
    inter = sum(inter_sims) / max(len(inter_sims), 1)

    return {
        "intra_root_cosine": round(intra, 4),
        "inter_root_cosine": round(inter, 4),
        "delta": round(intra - inter, 4),
        "n_groups_evaluated": len(intra_sims),
        "n_inter_pairs": len(inter_sims),
        "mean_group_size": round(sum(group_sizes) / max(len(group_sizes), 1), 2),
    }


def build_analogy_pairs(
        word_to_segments: Dict[str, List[str]],
        min_examples_per_suffix: int = 8,
        max_pairs_per_suffix: int = 100,
        seed: int = 0,
) -> List[Tuple[str, str, str, str]]:
    suffix_to_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for word, segs in word_to_segments.items():
        if len(segs) != 2:
            continue
        root, suffix = segs[0], segs[1]
        if root not in word_to_segments:
            continue
        if word_to_segments[root] != [root]:
            continue
        suffix_to_pairs[suffix].append((root, word))

    rng = torch.Generator().manual_seed(seed)
    analogies: List[Tuple[str, str, str, str]] = []
    for suffix, pairs in suffix_to_pairs.items():
        if len(pairs) < min_examples_per_suffix:
            continue
        n = len(pairs)
        n_quartets = min(max_pairs_per_suffix, n * (n - 1) // 2)
        for _ in range(n_quartets):
            i, j = torch.randint(0, n, (2,), generator=rng).tolist()
            if i == j:
                continue
            a, a_p = pairs[i]
            b, b_p = pairs[j]
            analogies.append((a, a_p, b, b_p))
    return analogies


def morphological_analogy_accuracy(
        word_embeddings: torch.Tensor,
        words: List[str],
        analogies: List[Tuple[str, str, str, str]],
        top_k: int = 10,
) -> Dict[str, float]:
    word_to_idx = {w: i for i, w in enumerate(words)}
    emb = F.normalize(word_embeddings, dim=-1)

    correct = 0
    total = 0
    rank_sum = 0
    rank_count = 0

    for a, a_p, b, b_p in analogies:
        if not all(w in word_to_idx for w in (a, a_p, b, b_p)):
            continue
        ia, iap, ib, ibp = (word_to_idx[a], word_to_idx[a_p],
                            word_to_idx[b], word_to_idx[b_p])
        target = emb[iap] - emb[ia] + emb[ib]
        target = F.normalize(target, dim=-1)
        sims = emb @ target
        sims[ia] = -1e4
        sims[iap] = -1e4
        sims[ib] = -1e4

        top = sims.topk(top_k).indices.tolist()
        if ibp in top:
            correct += 1
        rank = (sims > sims[ibp]).sum().item() + 1
        rank_sum += rank
        rank_count += 1
        total += 1

    return {
        f"analogy_top{top_k}_acc": round(correct / max(total, 1), 4),
        "analogy_mean_rank": round(rank_sum / max(rank_count, 1), 2),
        "analogy_n_evaluated": total,
    }


def nearest_neighbors(
        word_embeddings: torch.Tensor,
        words: List[str],
        query_words: List[str],
        k: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    word_to_idx = {w: i for i, w in enumerate(words)}
    emb = F.normalize(word_embeddings, dim=-1)

    results: Dict[str, List[Tuple[str, float]]] = {}
    for q in query_words:
        if q not in word_to_idx:
            results[q] = []
            continue
        qi = word_to_idx[q]
        sims = emb @ emb[qi]
        sims[qi] = -1e4
        top_vals, top_idx = sims.topk(k)
        results[q] = [(words[i], round(v.item(), 4)) for i, v in zip(top_idx.tolist(), top_vals)]
    return results
