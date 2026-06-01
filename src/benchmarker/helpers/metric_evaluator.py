import time
import math
from collections import Counter
from typing import List, Callable, Set, Dict, Any
from src.benchmarker.utils.providers.logger_provider import global_logger
from src.benchmarker.utils.text_utils import turkish_lower


class MetricEvaluator:
    def __init__(self, morfessor_model=None):
        self.morfessor_model = morfessor_model

    @staticmethod
    def get_clean_surface(tok: str) -> str:
        # SentencePiece ' ', ByteBPE '▁', WordPiece '##'
        return tok.replace(" ", "").replace("▁", "").replace("##", "")

    def _get_morfessor_boundaries(self, word: str) -> Set[int]:
        if not self.morfessor_model:
            return set()
        try:
            segments, _ = self.morfessor_model.viterbi_segment(turkish_lower(word))
            boundaries = set()
            pos = 0
            for s in segments[:-1]:
                pos += len(s)
                boundaries.add(pos)
            return boundaries
        except Exception:
            return set()

    def get_boundaries(self, tokenizer_obj: Any, word: str) -> Set[int]:
        boundaries = set()
        pos = 0
        word_lower = turkish_lower(word)

        if hasattr(tokenizer_obj, "encode_as_pieces"):
            tokens = tokenizer_obj.encode_as_pieces(word_lower)
            for t in tokens[:-1]:
                clean_t = t.replace(" ", "").replace("▁", "")
                if clean_t:
                    pos += len(clean_t)
                    boundaries.add(pos)
            return boundaries

        elif hasattr(tokenizer_obj, "encode"):
            enc = tokenizer_obj.encode(word_lower)
            tokens = enc.tokens
            for t in tokens[:-1]:
                clean_t = t.replace("##", "")
                pos += len(clean_t)
                boundaries.add(pos)
            return boundaries

        return boundaries

    def compute_metrics(
            self,
            name: str,
            tokenizer_obj: Any,
            encode_fn: Callable[[str], List[str]],
            vocab: Set[str],
            test_sents: List[str]
    ) -> Dict[str, Any]:

        global_logger.info(f"[MetricEvaluator](compute_metrics) Evaluating: {name}")

        results = {"name": name}
        all_tokens = []
        all_words_e = []
        unk_count = 0
        total_chars = 0
        total_boundary_hits = 0
        total_boundary_total = 0

        t0 = time.time()
        for sent in test_sents:
            tokens = encode_fn(sent)
            if not tokens:
                continue

            all_tokens.extend(tokens)
            words = sent.split()
            all_words_e.extend(words)
            total_chars += len(sent)

            for tok in tokens:
                if tok not in vocab:
                    unk_count += 1

            for word in words:
                if len(word) < 4:
                    continue

                word_lower = turkish_lower(word)
                ref_boundaries = self._get_morfessor_boundaries(word_lower)
                if not ref_boundaries:
                    continue

                pred_boundaries = self.get_boundaries(tokenizer_obj, word_lower)

                hits = len(ref_boundaries & pred_boundaries)
                total_boundary_hits += hits
                total_boundary_total += len(ref_boundaries)

        encode_time_ms = (time.time() - t0) * 1000
        n_tokens = len(all_tokens)
        n_words = len(all_words_e)
        n_chars = total_chars

        results["fertility"] = round(n_tokens / n_words, 4) if n_words else 0
        results["oov_rate"] = round(unk_count / n_tokens * 100, 4) if n_tokens else 0
        results["compression"] = round(n_chars / n_tokens, 4) if n_tokens else 0
        results["vocab_coverage"] = round((1 - unk_count / n_tokens) * 100, 4) if n_tokens else 0

        tok_freq = Counter(all_tokens)
        total = sum(tok_freq.values())
        if total > 0:
            entropy = -sum((c / total) * math.log2(c / total) for c in tok_freq.values() if c > 0)
            results["subword_entropy"] = round(entropy, 4)
        else:
            results["subword_entropy"] = 0

        if total_boundary_total > 0:
            results["morfem_alignment"] = round(total_boundary_hits / total_boundary_total * 100, 2)
        else:
            results["morfem_alignment"] = 0.0

        results["encode_ms"] = round(encode_time_ms, 1)
        results["vocab_size"] = len(vocab)
        results["n_tokens"] = n_tokens

        global_logger.info(f"[MetricEvaluator](compute_metrics) {name} Done. "
                           f"MAS: {results['morfem_alignment']}%, Fertility: {results['fertility']}")

        return results

