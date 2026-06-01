import pandas as pd
from typing import List
from pathlib import Path
from src.benchmarker.utils.providers.logger_provider import global_logger
from src.benchmarker.utils.text_utils import turkish_lower
from src.benchmarker.helpers.tokenizer_trainer import TokenizerTrainer
from src.benchmarker.helpers.metric_evaluator import MetricEvaluator
from src.benchmarker.helpers.visualizer import ResultVisualizer
from src.benchmarker.utils.providers.config_provider import config_provider

class TokenizerBenchmarker:
    _DEFAULT_RESULT_PATH: Path = Path(__file__).parent.parent / "results"

    def __init__(self, test_sentences: List[str], output_dir: Path = _DEFAULT_RESULT_PATH):
        self.test_sentences = test_sentences
        self.output_dir = output_dir

        self.trainer = TokenizerTrainer()
        self.visualizer = ResultVisualizer(output_dir=str(self.output_dir))
        self.evaluator = None

        self.all_results = []



    def run_benchmark(self, vocab_sizes: List[int] = config_provider.cfg.training.vocab_size):

        global_logger.info("[Benchmarker] Training Morfessor Baseline...")
        morfessor_model = self.trainer.train_morfessor()
        self.evaluator = MetricEvaluator(morfessor_model=morfessor_model)

        sp_tasks = [("BPE", self.trainer.train_bpe),
            ("Unigram", self.trainer.train_unigram),
            ("ByteBPE", self.trainer.train_byte_bpe)]

        for prefix, train_fn in sp_tasks:
            global_logger.info(f"[Benchmarker] Processing {prefix} models...")
            models = train_fn(vocab_sizes=vocab_sizes)

            for i, model in enumerate(models):
                vs = vocab_sizes[i]
                res = self.evaluator.compute_metrics(
                    name=f"{prefix}-{vs // 1000}K",
                    tokenizer_obj=model,
                    encode_fn=lambda x, m=model: m.encode_as_pieces(turkish_lower(x)),
                    vocab=set(model.id_to_piece(i) for i in range(model.get_piece_size())),
                    test_sents=self.test_sentences
                )
                self.all_results.append(res)

        global_logger.info("[Benchmarker] Processing WordPiece models...")
        wp_models = self.trainer.train_wordpiece(vocab_sizes=vocab_sizes)
        for i, model in enumerate(wp_models):
            vs = vocab_sizes[i]
            res = self.evaluator.compute_metrics(
                name=f"WordPiece-{vs // 1000}K",
                tokenizer_obj=model,
                encode_fn=lambda x, m=model: model.encode(turkish_lower(x)).tokens,
                vocab=set(model.get_vocab().keys()),
                test_sents=self.test_sentences
            )
            self.all_results.append(res)

        df = pd.DataFrame(self.all_results)
        df_scored = self.visualizer.calculate_weighted_winner(df)
        self.visualizer.run_all(df_scored)

        winner = df_scored.iloc[0]["name"]
        global_logger.info(f"BENCHMARK COMPLETE. Winner: {winner}")

        return df_scored


if __name__ == "__main__":
    HARD_TURKISH_WORDS = [
        "gidebileceklerindenmişsiniz",
        "karşılaştırılamayacak",
        "üniversitelerindeki",
        "bilgisayarlarımızın",
        "ekonomik",
        "yapamayacağız",
        "görevlendirilemeyeceklerinden",
        "anlaşılamamaktadır",
        "hükümet",
        "teknoloji",
        "evlerdekiler",
        "gidiyorum",
        "çalışmalarımızdan",
        "muhasebeleştirme",
        "bilgisayarlarımızın",
        "kitap",
        "ev",
    ]
    benchmarker = TokenizerBenchmarker(test_sentences=HARD_TURKISH_WORDS)
    results_df = benchmarker.run_benchmark()