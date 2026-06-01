import json
import re
import sys
import time
import torch
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from src.model_development.model.morpheus import Morpheus
from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.utils.text_utils import turkish_lower
from src.model_development.utils.providers.logger_provider import global_logger


WORD_BOUNDARY = "▁"


SPECIAL_TOKENS = {
    "<UNK>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<PAD>": 3,
    "<MASK>": 4,
}


PRIORITY_SUFFIXES: Tuple[str, ...] = (
    "lar", "ler",
    "da", "de", "ta", "te",
    "dan", "den", "tan", "ten",
    "ya", "ye",
    "yı", "yi", "yu", "yü",
    "ın", "in", "un", "ün",
    "nın", "nin", "nun", "nün",
    "ım", "im", "um", "üm",
    "mız", "miz", "muz", "müz",
    "nız", "niz", "nuz", "nüz",
    "ları", "leri",
    "ki",
    "ça", "çe", "ca", "ce",
    "dı", "di", "du", "dü",
    "tı", "ti", "tu", "tü",
    "yor", "yordu", "yormuş",
    "acak", "ecek",
    "ar", "er", "ır", "ir", "ur", "ür",
    "sın", "sin", "sun", "sün",
    "ız", "iz", "uz", "üz",
    "sınız", "siniz", "sunuz", "sünüz",
    "ma", "me",
    "mı", "mi", "mu", "mü",
    "tır", "tir", "tur", "tür",
    "dır", "dir", "dur", "dür",
    "ıl", "il", "ul", "ül",
    "lık", "lik", "luk", "lük",
    "cı", "ci", "cu", "cü",
    "çı", "çi", "çu", "çü",
    "sız", "siz", "suz", "süz",
    "lı", "li", "lu", "lü",
    "mış", "miş", "muş", "müş",
    "sa", "se",
    "yse", "ysa",
    "ken",
    "iken",
    "deki", "daki", "teki", "taki",
    "miydi", "mıydı", "muydu", "müydü",
    "den", "dan", "ten", "tan",
    "yi", "yı", "yu", "yü",
    "ce", "ca",
)


def _clean_word(word: str) -> Optional[str]:
    w = re.sub(r"[^\w]", "", word, flags=re.UNICODE)
    if len(w) < 1 or len(w) > 30:
        return None
    return w


class MorpheusTokenizer:
    def __init__(
            self,
            morpheus_model: Optional[Morpheus] = None,
            vocab: Optional[Dict[str, int]] = None,
            max_word_len: int = 32,
            preserve_case: bool = True,
            device: Optional[torch.device] = None,
            batch_size: int = 256,
            legacy_boundary_shift: bool = False,
    ):
        self.helper = CharEncoderHelper()
        self.model = morpheus_model
        self.vocab: Dict[str, int] = vocab or dict(SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.max_word_len = max_word_len
        self.preserve_case = preserve_case
        self.batch_size = batch_size
        self.legacy_boundary_shift = legacy_boundary_shift

        self.unk_id = self.vocab.get("<UNK>", 0)
        self.bos_id = self.vocab.get("<BOS>", 1)
        self.eos_id = self.vocab.get("<EOS>", 2)
        self.pad_id = self.vocab.get("<PAD>", 3)
        self.mask_id = self.vocab.get("<MASK>", 4)

        if device is None and morpheus_model is not None:
            self.device = next(morpheus_model.parameters()).device
        else:
            self.device = device or torch.device("cpu")

        self._segment_cache: Dict[str, List[str]] = {}

        if morpheus_model is not None:
            morpheus_model.eval()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @torch.no_grad()
    def _morpheus_segment_batch(
            self,
            words: List[str],
    ) -> List[List[str]]:
        if not words:
            return []
        if self.model is None:
            raise RuntimeError("MorpheusTokenizer initialized without a model; cannot segment new words")

        ids_list, flags_list, real_lens, lengths = [], [], [], []
        for w in words:
            ids, flags, rl = self.helper.word_to_char_ids(w, max_len=self.max_word_len)
            ids_list.append(ids)
            flags_list.append(flags)
            real_lens.append(rl)
            lengths.append(len(w))

        char_ids = torch.tensor(ids_list, device=self.device)
        case_flags = torch.tensor(flags_list, device=self.device)
        real_length_t = torch.tensor(real_lens, device=self.device)

        out = self.model(
            char_ids=char_ids,
            case_flags=case_flags,
            real_lengths=real_length_t,
        )
        hard_boundaries = out["hard_boundaries"].cpu().tolist()

        results: List[List[str]] = []
        for w, bnds, rl, wlen in zip(words, hard_boundaries, real_lens, lengths):
            segs = self._boundaries_to_segments(w, bnds, rl, wlen)
            results.append(segs)
        return results

    def _boundaries_to_segments(
            self,
            word: str,
            hard_boundaries: List[bool],
            real_length: int,
            word_length: int,
    ) -> List[str]:
        n_chars = min(word_length, real_length - 2)
        if n_chars <= 0:
            return [word] if word else []

        shift = 2 if self.legacy_boundary_shift else 1

        boundary_at_char: List[bool] = []
        for i in range(n_chars - 1):
            bnd_idx = i + shift
            if 0 <= bnd_idx < len(hard_boundaries):
                boundary_at_char.append(bool(hard_boundaries[bnd_idx]))
            else:
                boundary_at_char.append(False)

        segments: List[str] = []
        current = word[0]
        for i in range(1, n_chars):
            if boundary_at_char[i - 1]:
                segments.append(current)
                current = word[i]
            else:
                current += word[i]

        if n_chars < word_length:
            current += word[n_chars:]

        if current:
            segments.append(current)

        return segments

    def segment_word(self, word: str) -> List[str]:
        if not word:
            return []
        key = word if self.preserve_case else turkish_lower(word)
        if key in self._segment_cache:
            return self._segment_cache[key]
        segs = self._morpheus_segment_batch([word])[0]
        self._segment_cache[key] = segs
        return segs

    def segment_words_batched(self, words: List[str]) -> List[List[str]]:
        results: List[Optional[List[str]]] = [None] * len(words)
        need_segment: List[int] = []
        need_words: List[str] = []

        for i, w in enumerate(words):
            if not w:
                results[i] = []
                continue
            key = w if self.preserve_case else turkish_lower(w)
            cached = self._segment_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                need_segment.append(i)
                need_words.append(w)

        for start in range(0, len(need_words), self.batch_size):
            chunk = need_words[start: start + self.batch_size]
            chunk_segs = self._morpheus_segment_batch(chunk)
            for w, segs in zip(chunk, chunk_segs):
                key = w if self.preserve_case else turkish_lower(w)
                self._segment_cache[key] = segs
            for local_i, segs in enumerate(chunk_segs):
                global_i = need_segment[start + local_i]
                results[global_i] = segs

        return [r if r is not None else [] for r in results]

    def _segment_to_ids(self, segment: str, is_word_start: bool) -> List[int]:
        token = (WORD_BOUNDARY + segment) if is_word_start else segment
        if token in self.vocab:
            return [self.vocab[token]]

        char_tokens = list(token)
        out: List[int] = []
        for ch in char_tokens:
            out.append(self.vocab.get(ch, self.unk_id))
        return out

    def tokenize(self, text: str) -> List[str]:
        words = text.split()
        cleaned_words: List[str] = []
        for w in words:
            cw = _clean_word(w)
            if cw:
                cleaned_words.append(cw)

        all_segments = self.segment_words_batched(cleaned_words)

        tokens: List[str] = []
        for segs in all_segments:
            if not segs:
                continue
            tokens.append(WORD_BOUNDARY + segs[0])
            for seg in segs[1:]:
                tokens.append(seg)
        return tokens

    def encode_as_pieces(self, text: str) -> List[str]:
        return self.tokenize(text)

    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
    ) -> List[int]:
        tokens = self.tokenize(text)
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for tok in tokens:
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            else:
                for ch in tok:
                    ids.append(self.vocab.get(ch, self.unk_id))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def encode_batch(
            self,
            texts: List[str],
            add_special_tokens: bool = True,
    ) -> List[List[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode(
            self,
            ids: List[int],
            skip_special_tokens: bool = True,
            skip_unk: bool = True,
    ) -> str:
        specials = {self.bos_id, self.eos_id, self.pad_id, self.mask_id}
        if skip_unk:
            specials.add(self.unk_id)
        pieces: List[str] = []
        for i in ids:
            if skip_special_tokens and i in specials:
                continue
            tok = self.id_to_token.get(i, "")
            pieces.append(tok)
        text = "".join(pieces).replace(WORD_BOUNDARY, " ").strip()
        return text

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        config = {
            "vocab_size": len(self.vocab),
            "max_word_len": self.max_word_len,
            "preserve_case": self.preserve_case,
            "legacy_boundary_shift": self.legacy_boundary_shift,
            "word_boundary": WORD_BOUNDARY,
            "special_tokens": {
                "unk_id": self.unk_id,
                "bos_id": self.bos_id,
                "eos_id": self.eos_id,
                "pad_id": self.pad_id,
                "mask_id": self.mask_id,
            },
        }
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        global_logger.info(f"[MorpheusTokenizer] Saved tokenizer to {path}")

    @classmethod
    def load(
            cls,
            path: Union[str, Path],
            morpheus_model: Optional[Morpheus] = None,
            device: Optional[torch.device] = None,
    ) -> "MorpheusTokenizer":
        path = Path(path)
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(path / "tokenizer_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        tok = cls(
            morpheus_model=morpheus_model,
            vocab=vocab,
            max_word_len=config.get("max_word_len", 32),
            preserve_case=config.get("preserve_case", True),
            legacy_boundary_shift=config.get("legacy_boundary_shift", False),
            device=device,
        )
        global_logger.info(f"[MorpheusTokenizer] Loaded tokenizer from {path} (vocab={len(vocab)})")
        return tok

    def vocab_stats(self) -> Dict:
        char_tokens = [t for t in self.vocab if len(t) == 1 and t != WORD_BOUNDARY]
        word_start_tokens = [t for t in self.vocab if t.startswith(WORD_BOUNDARY)]
        morpheme_tokens = [
            t for t in self.vocab
            if not t.startswith(WORD_BOUNDARY)
            and len(t) > 1
            and t not in SPECIAL_TOKENS
        ]
        return {
            "vocab_size": len(self.vocab),
            "specials": len(SPECIAL_TOKENS),
            "single_chars": len(char_tokens),
            "word_starts": len(word_start_tokens),
            "morphemes": len(morpheme_tokens),
        }


def build_morpheus_vocab(
        morpheus_model: Morpheus,
        corpus_path: str,
        vocab_size: int = 50_000,
        min_freq: int = 2,
        max_words_to_segment: Optional[int] = None,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
        char_fallback: bool = True,
        preserve_case: bool = True,
        legacy_boundary_shift: bool = False,
        inject_priority_suffixes: bool = True,
        log_every: int = 5000,
) -> Dict[str, int]:
    if device is None:
        device = next(morpheus_model.parameters()).device

    morpheus_model.eval()

    global_logger.info(f"[build_morpheus_vocab] Counting words in {corpus_path}")
    word_counter: Counter = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            for w in line.strip().split():
                cw = _clean_word(w)
                if cw is None:
                    continue
                if not preserve_case:
                    cw = turkish_lower(cw)
                word_counter[cw] += 1

    unique_words = [w for w, c in word_counter.items() if c >= min_freq]
    unique_words.sort(key=lambda w: -word_counter[w])
    if max_words_to_segment is not None:
        unique_words = unique_words[: max_words_to_segment]

    global_logger.info(
        f"[build_morpheus_vocab] {len(unique_words):,} unique words (min_freq={min_freq}) "
        f"out of {len(word_counter):,} total"
    )

    bootstrap_tok = MorpheusTokenizer(
        morpheus_model=morpheus_model,
        vocab=dict(SPECIAL_TOKENS),
        preserve_case=preserve_case,
        device=device,
        batch_size=batch_size,
        legacy_boundary_shift=legacy_boundary_shift,
    )

    seg_counter: Counter = Counter()
    t0 = time.time()

    for i in range(0, len(unique_words), batch_size):
        batch = unique_words[i: i + batch_size]
        batch_segs = bootstrap_tok._morpheus_segment_batch(batch)
        for w, segs in zip(batch, batch_segs):
            freq = word_counter[w]
            for j, seg in enumerate(segs):
                token = (WORD_BOUNDARY + seg) if j == 0 else seg
                seg_counter[token] += freq

        if (i // batch_size + 1) % (log_every // batch_size + 1) == 0:
            elapsed = time.time() - t0
            done = min(i + batch_size, len(unique_words))
            rate = done / max(elapsed, 1e-6)
            remaining = (len(unique_words) - done) / max(rate, 1e-6)
            global_logger.info(
                f"[build_morpheus_vocab] {done:>7,} / {len(unique_words):,} "
                f"| {elapsed:.1f}s | ETA {remaining:.0f}s"
            )

    vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
    next_id = len(vocab)

    if char_fallback:
        for ch in bootstrap_tok.helper._TURKISH_CHARS:
            if ch in (" ", "\n", "\t"):
                continue
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1

    if inject_priority_suffixes:
        n_injected = 0
        for suf in PRIORITY_SUFFIXES:
            if suf not in vocab:
                vocab[suf] = next_id
                next_id += 1
                n_injected += 1
        global_logger.info(
            f"[build_morpheus_vocab] Injected {n_injected} priority Turkish suffixes"
        )

    slots_left = vocab_size - len(vocab)
    if slots_left <= 0:
        global_logger.warning(
            f"[build_morpheus_vocab] vocab_size={vocab_size} too small "
            f"for specials + char fallback ({len(vocab)})"
        )
        return vocab

    candidates = seg_counter.most_common(slots_left * 2)
    added = 0
    for token, freq in candidates:
        if freq < min_freq:
            break
        if token in vocab:
            continue
        vocab[token] = next_id
        next_id += 1
        added += 1
        if added >= slots_left:
            break

    elapsed = time.time() - t0
    global_logger.info(
        f"[build_morpheus_vocab] Done. vocab_size={len(vocab)} "
        f"| specials={len(SPECIAL_TOKENS)} | total time {elapsed:.1f}s"
    )
    return vocab


def export_corpus_tokenized(
        tokenizer: MorpheusTokenizer,
        corpus_path: str,
        output_path: str,
        log_every: int = 50_000,
) -> Dict:
    n_lines = 0
    n_words = 0
    n_tokens = 0
    n_unk = 0
    n_chars = 0
    t0 = time.time()

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            tokens = tokenizer.tokenize(line)
            n_lines += 1
            n_words += len(line.split())
            n_chars += len(line)

            id_list: List[int] = []
            for tok in tokens:
                if tok in tokenizer.vocab:
                    id_list.append(tokenizer.vocab[tok])
                else:
                    for ch in tok:
                        cid = tokenizer.vocab.get(ch, tokenizer.unk_id)
                        if cid == tokenizer.unk_id:
                            n_unk += 1
                        id_list.append(cid)
            n_tokens += len(id_list)
            fout.write(" ".join(str(i) for i in id_list) + "\n")

            if n_lines % log_every == 0:
                elapsed = time.time() - t0
                global_logger.info(
                    f"[export_corpus_tokenized] {n_lines:>9,} lines "
                    f"| {elapsed:.0f}s | {n_tokens / max(n_words, 1):.3f} tokens/word"
                )

    stats = {
        "n_lines": n_lines,
        "n_words": n_words,
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "unk_count": n_unk,
        "fertility": round(n_tokens / max(n_words, 1), 4),
        "compression": round(n_chars / max(n_tokens, 1), 4),
        "unk_rate": round(n_unk / max(n_tokens, 1) * 100, 4),
    }
    global_logger.info(f"[export_corpus_tokenized] Done. Stats: {stats}")
    return stats


if __name__ == "__main__":
    sys.modules["__main__"].__dict__.setdefault(
        "TrainingConfig",
        __import__(
            "src.model_development.training.trainer",
            fromlist=["TrainingConfig"],
        ).TrainingConfig,
    )

    base = Path(__file__).parent.parent.parent.parent
    checkpoint_path = str(
        base / "src/model_development/training/checkpoints/morpheus_v2_epoch13.pt"
    )
    corpus_path = str(base / "src/benchmarker/dataset/splits/train.txt")
    output_dir = base / "src/model_development/tokenizer/morpheus_50k"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_logger.info(f"[main] Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")

    if cfg is not None:
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
            count_loss_w=getattr(cfg, "count_loss_w", 0.3),
        )
    else:
        model = Morpheus()

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    legacy_shift = False

    global_logger.info(
        f"[main] Building MorpheusTokenizer vocab (64k, legacy_boundary_shift={legacy_shift})..."
    )
    vocab = build_morpheus_vocab(
        morpheus_model=model,
        corpus_path=corpus_path,
        vocab_size=64_000,
        min_freq=2,
        device=device,
        legacy_boundary_shift=legacy_shift,
    )

    tokenizer = MorpheusTokenizer(
        morpheus_model=model,
        vocab=vocab,
        device=device,
        legacy_boundary_shift=legacy_shift,
    )
    tokenizer.save(output_dir)

    global_logger.info(f"[main] Vocab stats: {tokenizer.vocab_stats()}")

    sample_sentences = [
        "Bugün İstanbul'a gidiyorum ve kitaplarımı yanımda götüreceğim.",
        "Muvaffakiyetsizleştiriciler bizim ülkemizde çokça bulunur.",
        "Ev | ler | imiz | de | ki | kitap | lar",
        "ABD ekonomisi son çeyrekte büyüme gösterdi.",
    ]

    print("\n=== Sample Tokenizations ===")
    for s in sample_sentences:
        tokens = tokenizer.tokenize(s)
        ids = tokenizer.encode(s, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"\nOriginal : {s}")
        print(f"Tokens   : {tokens}")
        print(f"N tokens : {len(tokens)}")
        print(f"Decoded  : {decoded}")
