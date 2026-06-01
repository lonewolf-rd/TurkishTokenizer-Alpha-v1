import os
import re
import time
import torch
import morfessor
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter
from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.utils.providers.logger_provider import global_logger
from src.model_development.utils.text_utils import turkish_lower


class MorfessorWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load()

    def _load(self):
        try:
            io = morfessor.MorfessorIO()
            self.model = io.read_binary_model_file(self.model_path)
            global_logger.info(f"[MorfessorWrapper] Loaded model: {self.model_path}")
        except Exception as e:
            global_logger.error(f"[MorfessorWrapper] Load failed: {e}")
            raise

    def segment(self, word: str) -> Tuple[List[str], float]:
        try:
            segs, score = self.model.viterbi_segment(turkish_lower(word))
            if len(segs) == 1:
                confidence = 1.0
            else:
                avg_seg_len = sum(len(s) for s in segs) / len(segs)
                confidence = min(avg_seg_len / 4.0, 1.0)
                confidence = max(confidence, 0.3)
            return segs, confidence
        except Exception:
            return [turkish_lower(word)], 0.3

    def get_boundary_labels(
            self,
            word: str,
            max_len: int = 32,
            add_bos: bool = True,
            add_eos: bool = True,
    ) -> Tuple[List[int], float, str]:
        segs, confidence = self.segment(word)

        boundaries = set()
        pos = (1 if add_bos else 0) - 1
        for seg in segs[:-1]:
            pos += len(seg)
            boundaries.add(pos)

        labels = [0] * (max_len - 1)
        for b in boundaries:
            if 0 <= b < max_len - 1:
                labels[b] = 1

        root = segs[0] if segs else turkish_lower(word)
        return labels, confidence, root


def clean_word_preserve_case(word: str) -> Optional[str]:
    word = re.sub(r"[^\w]", "", word, flags=re.UNICODE)
    word = re.sub(r"[0-9]", "", word)
    if len(word) < 2 or len(word) > 30:
        return None
    if not all(c.isalpha() for c in word):
        return None
    return word


def build_word_vocab(
        txt_path: str,
        top_k: int = 50000,
        min_freq: int = 5,
) -> Tuple[Dict[str, int], torch.Tensor]:
    global_logger.info(f"[build_word_vocab] Counting words in {txt_path}")
    counter = Counter()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            for w in line.strip().split():
                cleaned = clean_word_preserve_case(w)
                if cleaned:
                    counter[turkish_lower(cleaned)] += 1

    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: -x[1])
    items = items[: top_k - 1]

    vocab = {"<UNK>": 0}
    freqs = [0]
    for i, (w, c) in enumerate(items):
        vocab[w] = i + 1
        freqs.append(c)

    global_logger.info(
        f"[build_word_vocab] Built vocab (size={len(vocab)}, "
        f"min_freq={min_freq}, top_word='{items[0][0] if items else None}')"
    )
    return vocab, torch.tensor(freqs, dtype=torch.float32)


def build_root_vocab(
        word_freqs: Counter,
        morfessor_path: str,
        top_k: int = 20000,
        min_freq: int = 3,
) -> Dict[str, int]:
    global_logger.info(f"[build_root_vocab] Loading Morfessor: {morfessor_path}")
    wrapper = MorfessorWrapper(morfessor_path)

    root_counter = Counter()
    for word, cnt in word_freqs.items():
        segs, _ = wrapper.segment(word)
        if segs:
            root_counter[segs[0]] += cnt

    items = [(r, c) for r, c in root_counter.items() if c >= min_freq]
    items.sort(key=lambda x: -x[1])
    items = items[: top_k - 1]

    vocab = {"<UNK>": 0}
    for i, (r, _) in enumerate(items):
        vocab[r] = i + 1

    global_logger.info(
        f"[build_root_vocab] Built root vocab (size={len(vocab)}, "
        f"top_root='{items[0][0] if items else None}')"
    )
    return vocab


def build_sentence_cache(
        txt_path: str,
        cache_path: str,
        word_vocab_path: str,
        root_vocab_path: str,
        morfessor_path: str = None,
        max_word_len: int = 32,
        max_sent_len: int = 24,
        min_sent_words: int = 3,
        max_sentences: int = 200_000,
        batch_log: int = 5000,
) -> None:
    cache_path_obj = Path(cache_path)
    if cache_path_obj.exists():
        global_logger.info(f"[build_sentence_cache] Cache exists, skipping: {cache_path}")
        return

    if morfessor_path is None:
        morfessor_path = str(
            Path(__file__).parent.parent.parent / "benchmarker/results/morfessor_model.bin"
        )

    global_logger.info(f"[build_sentence_cache] Start: {txt_path}")
    t0 = time.time()

    global_logger.info("[build_sentence_cache] Counting word frequencies...")
    word_counter = Counter()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            for w in line.strip().split():
                cleaned = clean_word_preserve_case(w)
                if cleaned:
                    word_counter[turkish_lower(cleaned)] += 1

    if Path(word_vocab_path).exists():
        global_logger.info(f"[build_sentence_cache] Loading word vocab: {word_vocab_path}")
        word_vocab = torch.load(word_vocab_path)["vocab"]
        word_freqs = torch.load(word_vocab_path)["freqs"]
    else:
        word_vocab, word_freqs = build_word_vocab(txt_path)
        Path(word_vocab_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"vocab": word_vocab, "freqs": word_freqs}, word_vocab_path)
        global_logger.info(f"[build_sentence_cache] Saved word vocab: {word_vocab_path}")

    if Path(root_vocab_path).exists():
        global_logger.info(f"[build_sentence_cache] Loading root vocab: {root_vocab_path}")
        root_vocab = torch.load(root_vocab_path)
    else:
        root_vocab = build_root_vocab(word_counter, morfessor_path)
        Path(root_vocab_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(root_vocab, root_vocab_path)
        global_logger.info(f"[build_sentence_cache] Saved root vocab: {root_vocab_path}")

    wrapper = MorfessorWrapper(morfessor_path)
    helper = CharEncoderHelper()

    char_ids_all = []
    case_flags_all = []
    real_lens_all = []
    labels_all = []
    confs_all = []
    word_ids_all = []
    root_ids_all = []
    masks_all = []

    n_sents = 0
    total_lines = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            raw_words = line.strip().split()
            cleaned_words = []
            for w in raw_words:
                cw = clean_word_preserve_case(w)
                if cw:
                    cleaned_words.append(cw)

            if len(cleaned_words) < min_sent_words:
                continue

            for chunk_start in range(0, len(cleaned_words), max_sent_len):
                chunk = cleaned_words[chunk_start: chunk_start + max_sent_len]
                if len(chunk) < min_sent_words:
                    continue

                s_char_ids = []
                s_case_flags = []
                s_real_lens = []
                s_labels = []
                s_confs = []
                s_word_ids = []
                s_root_ids = []

                for w in chunk:
                    ids, flags, rl = helper.word_to_char_ids(w, max_len=max_word_len)
                    labels, conf, root = wrapper.get_boundary_labels(w, max_len=max_word_len)
                    wid = word_vocab.get(turkish_lower(w), 0)
                    rid = root_vocab.get(root, 0)
                    s_char_ids.append(ids)
                    s_case_flags.append(flags)
                    s_real_lens.append(rl)
                    s_labels.append(labels)
                    s_confs.append(conf)
                    s_word_ids.append(wid)
                    s_root_ids.append(rid)

                actual = len(chunk)
                pad_word = [helper._PAD_ID] * max_word_len
                pad_case = [0] * max_word_len
                pad_label = [0] * (max_word_len - 1)
                while len(s_char_ids) < max_sent_len:
                    s_char_ids.append(pad_word)
                    s_case_flags.append(pad_case)
                    s_real_lens.append(0)
                    s_labels.append(pad_label)
                    s_confs.append(0.0)
                    s_word_ids.append(0)
                    s_root_ids.append(0)

                mask = [True] * actual + [False] * (max_sent_len - actual)

                char_ids_all.append(s_char_ids)
                case_flags_all.append(s_case_flags)
                real_lens_all.append(s_real_lens)
                labels_all.append(s_labels)
                confs_all.append(s_confs)
                word_ids_all.append(s_word_ids)
                root_ids_all.append(s_root_ids)
                masks_all.append(mask)

                n_sents += 1
                if n_sents >= max_sentences:
                    break

                if n_sents % batch_log == 0:
                    elapsed = time.time() - t0
                    global_logger.info(
                        f"[build_sentence_cache] {n_sents:>7,} sentences "
                        f"| {elapsed:.1f}s elapsed"
                    )

            if n_sents >= max_sentences:
                break

    global_logger.info(f"[build_sentence_cache] Packing tensors (N={n_sents})...")

    cache = {
        "char_ids": torch.tensor(char_ids_all, dtype=torch.long),
        "case_flags": torch.tensor(case_flags_all, dtype=torch.long),
        "real_lengths": torch.tensor(real_lens_all, dtype=torch.long),
        "morfessor_labels": torch.tensor(labels_all, dtype=torch.long),
        "confidence": torch.tensor(confs_all, dtype=torch.float32),
        "word_ids": torch.tensor(word_ids_all, dtype=torch.long),
        "root_ids": torch.tensor(root_ids_all, dtype=torch.long),
        "attention_mask": torch.tensor(masks_all, dtype=torch.bool),
        "word_vocab_size": len(word_vocab),
        "root_vocab_size": len(root_vocab),
        "max_word_len": max_word_len,
        "max_sent_len": max_sent_len,
    }

    cache_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)

    elapsed = time.time() - t0
    global_logger.info(
        f"[build_sentence_cache] Done. {n_sents:,} sentences "
        f"| {elapsed:.1f}s | size: {cache_path_obj.stat().st_size / 1e6:.1f} MB"
    )


class MorpheusSentenceDataset(Dataset):
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        global_logger.info(f"[MorpheusSentenceDataset] Loading: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")

        self.char_ids = cache["char_ids"]
        self.case_flags = cache["case_flags"]
        self.real_lengths = cache["real_lengths"]
        self.morfessor_labels = cache["morfessor_labels"]
        self.confidence = cache["confidence"]
        self.word_ids = cache["word_ids"]
        self.root_ids = cache["root_ids"]
        self.attention_mask = cache["attention_mask"]
        self.word_vocab_size = cache["word_vocab_size"]
        self.root_vocab_size = cache["root_vocab_size"]
        self.max_word_len = cache["max_word_len"]
        self.max_sent_len = cache["max_sent_len"]

        global_logger.info(
            f"[MorpheusSentenceDataset] Loaded {len(self.char_ids):,} sentences "
            f"| shape={self.char_ids.shape}"
        )

    def __len__(self) -> int:
        return self.char_ids.size(0)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "char_ids": self.char_ids[idx],
            "case_flags": self.case_flags[idx],
            "real_lengths": self.real_lengths[idx],
            "morfessor_labels": self.morfessor_labels[idx],
            "confidence": self.confidence[idx],
            "word_ids": self.word_ids[idx],
            "root_ids": self.root_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

    def stats(self) -> Dict:
        n = self.char_ids.size(0)
        n_words = self.attention_mask.float().sum().item()
        avg_words_per_sent = n_words / max(n, 1)
        avg_word_len = (self.real_lengths.float() * self.attention_mask.float()).sum().item() / max(n_words, 1)
        avg_boundaries = (
            self.morfessor_labels.float().sum(dim=-1) * self.attention_mask.float()
        ).sum().item() / max(n_words, 1)
        return {
            "n_sentences": n,
            "n_words_total": int(n_words),
            "avg_words_per_sent": round(avg_words_per_sent, 2),
            "avg_word_len": round(avg_word_len, 2),
            "avg_n_morphemes": round(avg_boundaries + 1, 2),
            "word_vocab_size": self.word_vocab_size,
            "root_vocab_size": self.root_vocab_size,
        }


def get_sentence_loader(
        cache_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 2,
        pin_memory: bool = True,
        drop_last: bool = True,
) -> DataLoader:
    dataset = MorpheusSentenceDataset(cache_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    global_logger.info(
        f"[get_sentence_loader] Loader ready: {len(dataset):,} sentences "
        f"| batch_size={batch_size} | {len(loader):,} batches"
    )
    return loader


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    train_txt = str(base / "benchmarker/dataset/splits/train.txt")
    test_txt = str(base / "benchmarker/dataset/splits/test.txt")
    morfessor_path = str(base / "benchmarker/results/morfessor_model.bin")

    word_vocab_path = str(base / "benchmarker/dataset/splits/word_vocab.pt")
    root_vocab_path = str(base / "benchmarker/dataset/splits/root_vocab.pt")

    train_cache = str(base / "benchmarker/dataset/splits/train_sentences.pt")
    test_cache = str(base / "benchmarker/dataset/splits/test_sentences.pt")

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

    ds = MorpheusSentenceDataset(train_cache)
    print(ds.stats())
