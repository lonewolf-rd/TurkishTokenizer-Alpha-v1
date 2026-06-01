from tokenizers import (Tokenizer,models,trainers,pre_tokenizers,normalizers,decoders)
from src.benchmarker.utils.providers.config_provider import config_provider
from src.benchmarker.utils.providers.logger_provider import global_logger
from src.benchmarker.utils.text_utils import turkish_lower
from collections import Counter
from pathlib import Path
from typing import Union, Tuple, List
import os, morfessor
import sentencepiece as spm


class TokenizerTrainer:
    _DEFAULT_CORPUS_PATH: Path = Path(__file__).parent.parent / "dataset" / "splits" / "train.txt"
    _DEFAULT_OUTPUT_PATH: Path = Path(__file__).parent.parent / "results"


    def __init__(self):
        self.vocab_size = config_provider.cfg.training.vocab_size
        self.turkish_sample_list: List[str] = [
            "ev",
            "kitap",
            "su",
            "ay",
            "göz",
            "baş",
            "evler",
            "kitaplar",
            "gözler",
            "evde",
            "evlerde",
            "evden",
            "evlerden",
            "eve",
            "evler",
            "evim",
            "evimiz",
            "evimde",
            "evlerimiz",
            "evlerimizde",
            "evlerimizdeki",
            "evlerimizdekiler",
            "kitabım",
            "kitabımdaki",
            "geldi",
            "geldim",
            "geliyorum",
            "geliyorduk",
            "gelmedim",
            "gelirim",
            "gelirsen",
            "gelecek",
            "gelmişti",
            "yaptırmak",
            "yapılmıştı",
            "görünmek",
            "görünmüyor",
            "İstanbul",
            "İstanbul'da",
            "Irak",
            "TBMM",
            "muhasebeleştirme",
            "muvaffakiyetsizleştiriciler",
            "karşılaştırılamayacak",
            "gidebileceklerindenmişsiniz",
            "üniversitelerindeki",
            "bilgisayarlarımızın",
            "çalışmalarımızdan",
            "anlaşılamamaktadır",
            "görevlendirilemeyeceklerinden",
        ]
        os.environ["HF_TOKEN"] = config_provider.cfg.huggingface.access_token

    def prepare_word_count(self) -> List[Tuple[int, str]]:
        try:
            word_counter = Counter()
            global_logger.info(f"[TokenizerTrainer](prepare_word_count) Reading corpus from {self._DEFAULT_CORPUS_PATH}")

            with open(self._DEFAULT_CORPUS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    word_counter.update(turkish_lower(line).split())

            training_data = [(count, word) for word, count in word_counter.items()]
            global_logger.info(f"[TokenizerTrainer](prepare_word_count) Unique words prepared: {len(training_data):,}")
            return training_data

        except Exception as err:
            global_logger.error(f"[TokenizerTrainer](prepare_word_count) Error: {str(err)}")
            raise err

    def load_morfessor(self) -> Union[morfessor.BaselineModel, None]:
        model_file = self._DEFAULT_OUTPUT_PATH / "morfessor_model.bin"
        io_handler = morfessor.MorfessorIO()
        try:
            if os.path.exists(model_file):
                global_logger.info(f"[TokenizerTrainer](train_morfessor) Found existing model at {model_file}. Loading...")
                morph_model = io_handler.read_binary_model_file(str(model_file))
                global_logger.info("[TokenizerTrainer](train_morfessor) Morfessor model loaded successfully from disk.")

                self._run_sanity_check(morph_model)
                return morph_model
            else:
                return None
        except Exception as e:
            global_logger.error(
                f"[TokenizerTrainer](train_morfessor) Could not load existing model: {e}. Re-training...")

    def train_morfessor(self, use_online: bool = True) -> morfessor.BaselineModel:
        model = self.load_morfessor()
        if model:
            return model

        try:
            morph_train_data = self.prepare_word_count()
            morph_model = morfessor.BaselineModel(
                corpusweight=1.0,
                use_skips=False,
            )

            global_logger.info("[TokenizerTrainer] Batch training başlıyor (corpusweight=1.0, max_epochs=20)...")
            morph_model.train_batch(
                morph_train_data,
                finish_threshold=0.005,
                max_epochs=20,
            )

            if use_online:
                global_logger.info("[TokenizerTrainer] Online refinement başlıyor (max_epochs=5)...")
                morph_model.train_online(
                    iter(morph_train_data),
                    max_epochs=5,
                )

            model_file = self._DEFAULT_OUTPUT_PATH / "morfessor_model.bin"
            io_handler = morfessor.MorfessorIO()
            io_handler.write_binary_model_file(str(model_file), morph_model)

            global_logger.info(f"[TokenizerTrainer] Model kaydedildi: {model_file}")
            self._run_sanity_check(morph_model)

            return morph_model

        except Exception as err:
            global_logger.error(f"[TokenizerTrainer] Training failed: {err}")
            raise


    def _run_sanity_check(self, model: morfessor.BaselineModel):
        global_logger.info("[TokenizerTrainer](sanity_check) Testing model on Turkish morphology:")
        for word in self.turkish_sample_list:
            segments, _ = model.viterbi_segment(turkish_lower(word))
            global_logger.info(f"  {word:35s} -> {' | '.join(segments)}")

    def train_bpe(self, vocab_sizes: List[int] = config_provider.cfg.training.vocab_size) -> List[spm.SentencePieceProcessor]:
        trained_processors = []
        try:
            global_logger.info(f"[TokenizerTrainer](train_bpe) Starting BPE training for sizes: {vocab_sizes}")

            for vs in vocab_sizes:
                model_prefix = self._DEFAULT_OUTPUT_PATH / f"bpe_{vs}"
                global_logger.info(f"[TokenizerTrainer](train_bpe) Training BPE-{vs // 1000}K...")

                spm.SentencePieceTrainer.train(
                    input=str(self._DEFAULT_CORPUS_PATH),
                    model_prefix=str(model_prefix),
                    vocab_size=vs,
                    model_type=config_provider.cfg.bpe_configs.model_type,
                    character_coverage=config_provider.cfg.bpe_configs.coverage,
                    byte_fallback=True,
                    split_digits=True,
                    normalization_rule_name="nmt_nfkc_cf",
                    input_sentence_size=config_provider.cfg.bpe_configs.input_sentence_size,
                    shuffle_input_sentence=True,
                    num_threads=config_provider.cfg.bpe_configs.num_threads,
                    pad_id=config_provider.cfg.bpe_configs.pad_id,
                )
                sp = spm.SentencePieceProcessor()
                sp.load(f"{str(model_prefix)}.model")
                trained_processors.append(sp)

                global_logger.info(f"[TokenizerTrainer](train_bpe) BPE-{vs // 1000}K training complete and loaded.")

                sample_text = self.turkish_sample_list[0]
                tokens = sp.encode_as_pieces(sample_text.lower())
                global_logger.info(f" [Sample] {sample_text} -> {' '.join(tokens)}")

            return trained_processors

        except Exception as err:
            global_logger.error(f"[TokenizerTrainer](train_bpe) BPE training failed: {str(err)}")
            raise err

    def train_unigram(
            self,
            vocab_sizes: List[int] = config_provider.cfg.training.vocab_size
    ) -> List[spm.SentencePieceProcessor]:
        trained_processors = []
        try:
            global_logger.info(f"[TokenizerTrainer](train_unigram) Starting Unigram training for sizes: {vocab_sizes}")

            for vs in vocab_sizes:
                model_prefix = self._DEFAULT_OUTPUT_PATH / f"unigram_{vs}"
                global_logger.info(f"[TokenizerTrainer](train_bpe) Training BPE-{vs // 1000}K...")

                spm.SentencePieceTrainer.train(
                    input=str(self._DEFAULT_CORPUS_PATH),
                    model_prefix=str(model_prefix),
                    vocab_size=vs,
                    model_type="unigram",
                    character_coverage=0.9999,
                    byte_fallback=True,
                    split_digits=True,
                    normalization_rule_name="nmt_nfkc_cf",
                    input_sentence_size=5_000_000,
                    shuffle_input_sentence=True,
                    num_threads=6,
                    pad_id=3,
                    shrinking_factor=0.75,
                    num_sub_iterations=2,
                )
                sp = spm.SentencePieceProcessor()
                sp.load(f"{str(model_prefix)}.model")
                trained_processors.append(sp)

                global_logger.info(f"[TokenizerTrainer](train_unigram) Unigram-{vs // 1000}K training complete and loaded.")

                sample_text = self.turkish_sample_list[0]
                tokens = sp.encode_as_pieces(sample_text.lower())
                global_logger.info(f" [Sample] {sample_text} -> {' '.join(tokens)}")

            return trained_processors

        except Exception as err:
            global_logger.error(f"[TokenizerTrainer](train_bpe) BPE training failed: {str(err)}")
            raise err

    def train_wordpiece(self, vocab_sizes: List[int] = config_provider.cfg.training.vocab_size) -> List[Tokenizer]:
        trained_tokenizers = []
        try:
            global_logger.info(
                f"[TokenizerTrainer](train_wordpiece) Starting WordPiece training for sizes: {vocab_sizes}")

            for vs in vocab_sizes:
                global_logger.info(f"[TokenizerTrainer](train_wordpiece) Training WordPiece-{vs // 1000}K...")

                tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
                tokenizer.normalizer = normalizers.Sequence([
                    normalizers.NFD(),
                    normalizers.Lowercase(),
                    normalizers.StripAccents(),
                ])
                tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
                tokenizer.decoder = decoders.WordPiece(prefix="##")

                trainer = trainers.WordPieceTrainer(
                    vocab_size=vs,
                    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                    min_frequency=2,
                    continuing_subword_prefix="##",
                )

                tokenizer.train([str(self._DEFAULT_CORPUS_PATH)], trainer=trainer)

                save_path = self._DEFAULT_OUTPUT_PATH / f"wordpiece_{vs}.json"
                tokenizer.save(str(save_path))
                trained_tokenizers.append(tokenizer)

                global_logger.info(
                    f"[TokenizerTrainer](train_wordpiece) WordPiece-{vs // 1000}K training complete and saved to {save_path.name}")

                sample_text = self.turkish_sample_list[0]
                tokens = tokenizer.encode(sample_text.lower()).tokens
                global_logger.info(f"  [Sample] {sample_text} -> {' '.join(tokens)}")

            return trained_tokenizers

        except Exception as err:
            global_logger.error(f"[TokenizerTrainer](train_wordpiece) WordPiece training failed: {str(err)}")
            raise err

    def train_byte_bpe(
            self,
            vocab_sizes: List[int] = config_provider.cfg.training.vocab_size
    ) -> List[spm.SentencePieceProcessor]:
        trained_processors = []
        try:
            global_logger.info(
                f"[TokenizerTrainer](train_byte_bpe) Starting Byte-level BPE training for sizes: {vocab_sizes}")

            for vs in vocab_sizes:
                model_prefix = self._DEFAULT_OUTPUT_PATH / f"byte_bpe_{vs}"
                global_logger.info(f"[TokenizerTrainer](train_byte_bpe) Training ByteBPE-{vs // 1000}K...")

                spm.SentencePieceTrainer.train(
                    input=str(self._DEFAULT_CORPUS_PATH),
                    model_prefix=str(model_prefix),
                    vocab_size=vs,
                    model_type="bpe",
                    character_coverage=0.9995,
                    byte_fallback=True,
                    split_digits=True,
                    normalization_rule_name="nmt_nfkc",
                    input_sentence_size=5_000_000,
                    shuffle_input_sentence=True,
                    num_threads=6,
                    pad_id=3,
                )

                sp = spm.SentencePieceProcessor()
                sp.load(f"{str(model_prefix)}.model")
                trained_processors.append(sp)

                global_logger.info(f"[TokenizerTrainer](train_byte_bpe) ByteBPE-{vs // 1000}K complete and loaded.")

                sample_text = self.turkish_sample_list[0]
                tokens = sp.encode_as_pieces(sample_text.lower())
                global_logger.info(f" [Sample] {sample_text} -> {' '.join(tokens)}")

            return trained_processors

        except Exception as err:
            global_logger.error(f"[TokenizerTrainer](train_byte_bpe) Byte-level BPE training failed: {str(err)}")
            raise err
