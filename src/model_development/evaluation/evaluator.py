import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Callable, Tuple
from src.model_development.model.morpheus import Morpheus
from src.model_development.model.char_encoder import CharEncoderHelper
from src.model_development.training.dataset import (
    MorfessorWrapper,
    clean_word_preserve_case,
)
from src.model_development.training.trainer import TrainingConfig
from src.model_development.evaluation.intrinsic import (
    embed_word_list,
    root_cluster_coherence,
    morphological_analogy_accuracy,
    nearest_neighbors,
    build_analogy_pairs,
)
from src.model_development.evaluation.extrinsic import (
    same_root_probe,
    morpheme_count_probe,
)
from src.model_development.utils.providers.logger_provider import global_logger

sys.modules["__main__"].TrainingConfig = TrainingConfig


class MorpheusEvaluator:
    def __init__(
            self,
            checkpoint_path: Optional[str] = None,
            morfessor_path: str = None,
            test_corpus_path: str = None,
            output_dir: str = "evaluation_results",
            max_eval_vocab: int = 20_000,
            min_word_freq: int = 3,
            device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = checkpoint_path
        self.morfessor_path = morfessor_path
        self.test_corpus_path = test_corpus_path
        self.max_eval_vocab = max_eval_vocab
        self.min_word_freq = min_word_freq

        self.helper = CharEncoderHelper()
        self.model: Optional[Morpheus] = None
        self.config = None
        self.morfessor = MorfessorWrapper(morfessor_path)

        self.eval_words: List[str] = []
        self.word_to_root: Dict[str, str] = {}
        self.word_to_segments: Dict[str, List[str]] = {}
        self.word_morpheme_counts: torch.Tensor = None
        self.confidence: torch.Tensor = None

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        ckpt_path = checkpoint_path or self.checkpoint_path
        if ckpt_path is None:
            global_logger.info("[Evaluator] No checkpoint — using random init Morpheus")
            self.model = Morpheus().to(self.device)
            self.model.eval()
            return

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config")
        self.config = cfg

        if cfg is not None:
            self.model = Morpheus(
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
            self.model = Morpheus()

        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        global_logger.info(
            f"[Evaluator] Loaded checkpoint: {ckpt_path} "
            f"(epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')})"
        )

    def build_eval_vocab(self) -> None:
        if self.test_corpus_path is None:
            raise ValueError("test_corpus_path required to build eval vocab")

        global_logger.info(f"[Evaluator] Counting words in {self.test_corpus_path}")
        counter: Counter = Counter()
        with open(self.test_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.strip().split():
                    cleaned = clean_word_preserve_case(w)
                    if cleaned:
                        counter[cleaned.lower()] += 1

        items = [(w, c) for w, c in counter.items() if c >= self.min_word_freq]
        items.sort(key=lambda x: -x[1])
        items = items[: self.max_eval_vocab]

        self.eval_words = [w for w, _ in items]
        global_logger.info(f"[Evaluator] Eval vocab: {len(self.eval_words)} words")

        global_logger.info("[Evaluator] Computing Morfessor segmentations...")
        morpheme_counts = []
        confidences = []
        for w in self.eval_words:
            segs, conf = self.morfessor.segment(w)
            self.word_to_segments[w] = segs
            self.word_to_root[w] = segs[0] if segs else w
            morpheme_counts.append(len(segs))
            confidences.append(conf)

        self.word_morpheme_counts = torch.tensor(morpheme_counts, dtype=torch.long)
        self.confidence = torch.tensor(confidences, dtype=torch.float)

    def compute_embeddings(self) -> torch.Tensor:
        global_logger.info(f"[Evaluator] Embedding {len(self.eval_words)} words...")
        return embed_word_list(
            model=self.model,
            helper=self.helper,
            words=self.eval_words,
            device=self.device,
        )

    def run(
            self,
            query_words: Optional[List[str]] = None,
            tag: str = "morpheus",
    ) -> Dict[str, float]:
        if self.model is None:
            self.load_model()
        if not self.eval_words:
            self.build_eval_vocab()

        embeddings = self.compute_embeddings()
        return self.run_on_embeddings(embeddings, query_words=query_words, tag=tag)

    def run_on_embeddings(
            self,
            embeddings: torch.Tensor,
            query_words: Optional[List[str]] = None,
            tag: str = "embeddings",
    ) -> Dict[str, float]:
        if not self.eval_words:
            self.build_eval_vocab()
        assert embeddings.size(0) == len(self.eval_words), (
            f"Embedding count {embeddings.size(0)} != vocab size {len(self.eval_words)}"
        )

        results: Dict[str, float] = {"tag": tag}

        coh = root_cluster_coherence(
            word_embeddings=embeddings,
            words=self.eval_words,
            word_to_root=self.word_to_root,
        )
        results.update(coh)
        global_logger.info(f"[Evaluator/{tag}] root_cluster_coherence: {coh}")

        analogies = build_analogy_pairs(self.word_to_segments)
        global_logger.info(f"[Evaluator/{tag}] Built {len(analogies)} analogy quartets")
        if analogies:
            ana = morphological_analogy_accuracy(
                word_embeddings=embeddings,
                words=self.eval_words,
                analogies=analogies,
                top_k=10,
            )
            results.update(ana)
            global_logger.info(f"[Evaluator/{tag}] morphological_analogy: {ana}")

        same_root = same_root_probe(
            embeddings=embeddings,
            words=self.eval_words,
            word_to_root=self.word_to_root,
            device=self.device,
        )
        results.update(same_root)
        global_logger.info(f"[Evaluator/{tag}] same_root_probe: {same_root}")

        morph_count = morpheme_count_probe(
            embeddings=embeddings,
            morpheme_counts=self.word_morpheme_counts,
            device=self.device,
        )
        results.update(morph_count)
        global_logger.info(f"[Evaluator/{tag}] morpheme_count_probe: {morph_count}")

        if query_words:
            nn_results = nearest_neighbors(
                word_embeddings=embeddings,
                words=self.eval_words,
                query_words=query_words,
                k=10,
            )
            self._write_nn_report(nn_results, tag)

        self._write_csv(results, tag)
        return results

    def compare_against_random(
            self,
            query_words: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        global_logger.info("[Evaluator] === Trained Morpheus ===")
        trained = self.run(query_words=query_words, tag="morpheus_trained")
        rows.append(trained)

        global_logger.info("[Evaluator] === Random init baseline ===")
        random_model = Morpheus(
            char_dim=self.config.char_dim if self.config else 256,
            char_embed_dim=self.config.char_embed_dim if self.config else 56,
            case_embed_dim=self.config.case_embed_dim if self.config else 8,
        ).to(self.device)
        random_model.eval()
        saved_model = self.model
        self.model = random_model
        random_results = self.run(query_words=query_words, tag="morpheus_random")
        rows.append(random_results)
        self.model = saved_model

        df = pd.DataFrame(rows)
        df_path = self.output_dir / "comparison_random_vs_trained.csv"
        df.to_csv(df_path, index=False)
        global_logger.info(f"[Evaluator] Comparison written: {df_path}")
        self._plot_comparison(df)
        return df

    def evaluate_external_embedder(
            self,
            embed_fn: Callable[[List[str]], torch.Tensor],
            tag: str = "external",
            query_words: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        if not self.eval_words:
            self.build_eval_vocab()
        global_logger.info(f"[Evaluator/{tag}] Running external embedder")
        emb = embed_fn(self.eval_words)
        return self.run_on_embeddings(emb, query_words=query_words, tag=tag)

    def _write_csv(self, results: Dict, tag: str) -> None:
        path = self.output_dir / f"results_{tag}.csv"
        pd.DataFrame([results]).to_csv(path, index=False)
        global_logger.info(f"[Evaluator/{tag}] Results: {path}")

    def _write_nn_report(self, nn_results: Dict, tag: str) -> None:
        path = self.output_dir / f"nearest_neighbors_{tag}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for q, neighbors in nn_results.items():
                f.write(f"\n=== {q} ===\n")
                for word, sim in neighbors:
                    f.write(f"  {word:25s} {sim:.4f}\n")
        global_logger.info(f"[Evaluator/{tag}] NN report: {path}")

    def _plot_comparison(self, df: pd.DataFrame) -> None:
        try:
            metric_cols = [
                "delta",
                "analogy_top10_acc",
                "same_root_val_acc",
                "morpheme_count_val_acc",
            ]
            metric_cols = [c for c in metric_cols if c in df.columns]
            if not metric_cols:
                return

            fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4))
            if len(metric_cols) == 1:
                axes = [axes]
            sns.set_theme(style="whitegrid")
            for ax, col in zip(axes, metric_cols):
                sns.barplot(data=df, x="tag", y=col, ax=ax)
                ax.set_title(col)
                ax.set_xlabel("")
                for i, v in enumerate(df[col]):
                    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            png_path = self.output_dir / "comparison.png"
            plt.savefig(png_path, dpi=120, bbox_inches="tight")
            plt.close()
            global_logger.info(f"[Evaluator] Comparison plot: {png_path}")
        except Exception as e:
            global_logger.error(f"[Evaluator] Plot failed: {e}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    checkpoint = str(
        Path(__file__).parent.parent / "training" / "checkpoints" / "morpheus_v2_epoch13.pt"
    )
    morfessor_path = str(base / "benchmarker/results/morfessor_model.bin")
    test_corpus = str(base / "benchmarker/dataset/splits/test.txt")

    query_words = [
        "kitaplar", "kitapta", "kitabı",
        "geliyorum", "gidiyorum", "yapıyorum",
        "muhasebeleştirme",
    ]

    evaluator = MorpheusEvaluator(
        checkpoint_path=checkpoint if Path(checkpoint).exists() else None,
        morfessor_path=morfessor_path,
        test_corpus_path=test_corpus,
        output_dir=str(Path(__file__).parent.parent / "evaluation_results"),
        max_eval_vocab=15_000,
    )

    df = evaluator.compare_against_random(query_words=query_words)
    print("\n=== Final Comparison ===")
    print(df.to_string(index=False))
