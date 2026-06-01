# Morpheus: Morpheme-Aware Neural Tokenization for Turkish

A research codebase studying the gap between statistical subword tokenization (BPE, ByteBPE, Unigram, WordPiece) and morphologically grounded word representations for **Turkish**, an agglutinative language whose semantic content is densely packed into productive suffix chains.

This repository contains:
1. **Morpheus** — a neural encoder that learns soft morpheme boundaries under Morfessor distillation and produces morphologically coherent fixed-dimensional word embeddings.
2. **A reproducible Turkish tokenization benchmark suite** — five vocab sizes × four classical tokenizers + Morfessor + Morpheus, evaluated under a unified metric set.
3. **An intrinsic + extrinsic evaluation framework** designed to test morphological consistency of word representations rather than only compression efficiency.

---

## Motivation

Modern large language models rely on subword tokenizers trained for compression and frequency, not linguistic structure. For high-resource fusional languages (English, French) this trade-off is acceptable. For **agglutinative languages**, however, semantically and grammatically loaded content lives in suffix chains:

```
evlerimizdekiler  →  ev + ler + imiz + de + ki + ler
                     root  PL    POSS-1PL LOC REL PL
                     ('the ones in our houses')
```

A frequency-based BPE will segment such words in ways that fragment morphemes (`evler | imizde | kiler`), forcing the model to relearn morphological generalization from distributional evidence. This is a known bottleneck for small-to-mid-scale models in agglutinative settings, where scale cannot fully compensate for inductive-bias mismatch.

### Research questions

1. Can a neural model learn *morphologically consistent* word representations from a statistical morphology teacher (Morfessor) plus distributional signal?
2. How does such a model compare to standard subword tokenizers under metrics that measure morphological alignment (not just compression)?
3. Does a Morpheus-derived tokenizer provide downstream sample-efficiency advantages over BPE for small Turkish language models?

This repository supplies the code, training recipe, and evaluation harness to answer (1) and (2) directly, and provides the tokenizer artifact to enable (3) as follow-up work.

---

## Key Contributions

- **Morpheus architecture**: character-level encoder → RoPE-attended boundary detector with deep supervision → vectorized Poisson-binomial soft segmentation → segment-aware attention pooling. The pipeline is end-to-end trainable and produces a fixed-dimensional word embedding from raw characters.
- **Multi-objective training recipe**: combines a decaying Morfessor distillation signal (boundary BCE + count regularization), Skip-Gram with Negative Sampling (SGNS) on a 50K context vocabulary, root-family contrastive (InfoNCE on Morfessor-derived root identities), and a word-level Masked LM with a character autoregressive decoder.
- **Differentiable soft segmentation**: exact Poisson-binomial dynamic-programming gives a closed-form probabilistic membership of each character to a segment, removing the gradient discontinuity of hard segmentation while preserving its inference-time semantics.
- **Turkish-specific design choices**: lowercase character vocabulary with a per-character case-flag side channel, Turkish-aware casefolding for the dotted/dotless `i` (`İ`/`I` → `i`/`ı`), and a curated priority list of common Turkish suffixes guaranteed in the tokenizer vocabulary.
- **Benchmark and evaluation framework**: a unified metric suite (Fertility, Compression, Morphological Alignment Score, Subword Entropy, root-cluster coherence, morphological analogy, linear probes) and a comparison harness that plugs any external embedder via a callable.

---


## Installation

```bash
python -m venv .venv
source .venv/bin/activate              # or .venv\Scripts\activate on Windows
pip install -e .
```

Python `>=3.13` is required. Dependencies are pinned in `pyproject.toml`. The codebase uses AMP fp16 throughout — any CUDA GPU with Tensor Cores will train end-to-end; we developed and tested on a 4 GB consumer card.

---

## Quick Start

The pipeline is staged: benchmarker outputs (Morfessor model + train/test splits) feed the model_development training.

```bash
# Stage 1: Wikipedia-tr → train/test split, train classical tokenizers + Morfessor
python -m src.benchmarker.helpers.benchmarker

# Stage 2: build sentence cache (uses Morfessor labels) and train Morpheus
python -m src.model_development.training.trainer

# Stage 3: build a 50K MorpheusTokenizer from the trained checkpoint
python -m src.model_development.tokenizer.morpheus_tokenizer

# Stage 4: run intrinsic + extrinsic evaluation against random-init baseline
python -m src.model_development.evaluation.evaluator
```

Each model module also exposes an `__main__` smoke test (e.g. `python -m src.model_development.model.morpheus`) running a forward+backward pass on synthetic data.

---

## Architecture

### Morpheus (Stage 2)

```
char_ids, case_flags
        │
        ▼
   CharEncoder              MultiScaleCNN (kernels 2..6) + 2 × LocalSelfAttention with RoPE
        │                   Output: (B, L, char_dim) context-aware character vectors
        ▼
  BoundaryDetector          2 × RoPE attention with deep supervision
        │                   Aux loss: weighted BCE + count regularization vs Morfessor labels
        │                   Output: boundary_probs (B, L−1)
        ▼
  SegmentEncoder            Poisson-binomial DP → soft segment membership (B, L, S)
        │                   Learned attention pooling per segment → segment vectors
        │                   Mean over valid segments + FFN + LayerNorm
        ▼
   word_embedding (B, char_dim)
```

**Soft segmentation via Poisson-binomial DP.** Given boundary probabilities `p_i ∈ [0,1]` between adjacent characters, the probability that character `i` belongs to segment `k` is the probability of observing exactly `k` boundaries before position `i` — a Poisson-binomial distribution. We compute this via the recurrence

```
f[i, k] = f[i-1, k] · (1 − p_i)   +   f[i-1, k-1] · p_i
```

This yields a differentiable membership matrix that converges to one-hot segment assignments as `p_i → {0, 1}`, recovering hard segmentation at inference time without an architectural switch.

**Position-aware boundary prediction.** Both CharEncoder's `LocalSelfAttention` and BoundaryDetector's `BoundaryAttention` apply Rotary Position Embedding (RoPE) on a shared `head_dim`. This is motivated by the fact that morpheme identity in Turkish depends on *position relative to the root* (e.g. the third suffix slot is structurally constrained to host certain morpheme types). Encoding this relative offset directly is more sample-efficient than recovering it from distributional evidence alone.

**Case as a side channel.** Rather than doubling the character vocabulary across uppercase/lowercase pairs, we lowercase the character input (Turkish-aware: `İ`→`i`, `I`→`ı`) and add a learned 2×8 case-flag embedding concatenated to the character embedding before projection. This halves the embedding rows and keeps morphologically equivalent forms (e.g. `İstanbul` vs `istanbul`) in the same orbit of the embedding space.

### Word MLM Head (auxiliary semantic objective)

A 2-layer transformer encoder operates over word embeddings within a sentence; 15 % of words are replaced with a learnable `[MASK]` token. For each masked position, a 1-layer transformer decoder generates the original word character-by-character, conditioned on the masked position's context vector. The cross-entropy on character predictions provides a vocabulary-free reconstruction signal that complements the discrete SGNS objective.

### Training Objectives

The total loss is a weighted sum of four signals:

```
L = w_aux · L_aux + w_sgns · L_sgns + w_ctr · L_contrastive + w_mlm · L_mlm
```

| Loss | Role | Weight schedule |
|---|---|---|
| `L_aux` | Boundary BCE + count MSE under Morfessor supervision; deep-supervised across detector layers | Decays from `0.50` to a `0.10` floor (slow geometric) |
| `L_sgns` | Skip-gram with negative sampling, ±5 window, 5 frequency-weighted negatives, 50K context vocab | Constant `0.7` |
| `L_contrastive` | InfoNCE on root identity (positives share the first Morfessor segment), temperature `0.10` | Constant `0.5` |
| `L_mlm` | Cross-entropy on character autoregressive reconstruction of masked words | Constant `0.7` |

The auxiliary aux schedule realizes a **curriculum**: in early epochs the Morfessor teacher dominates and the model learns where morpheme boundaries lie; as it decays, distributional signals (SGNS, MLM) take over to shape semantic geometry, with contrastive enforcing morphological consistency throughout.

Training uses AMP fp16 with a conservative GradScaler (init `2^10`, growth interval `4000`), AdamW with `β₂=0.98` for slower second-moment adaptation in the multi-objective setting, separate parameter groups so weight decay does not apply to biases/norms/embeddings, and gradient clipping at `0.3`. Loss components are computed in fp32 internally to avoid fp16 underflow in logsumexp/logsigmoid operations.

---

## Tokenizer

`MorpheusTokenizer` converts a trained Morpheus checkpoint into a discrete tokenizer compatible with standard LM pipelines:

- **Vocabulary construction**: segments every word in the training corpus once using Morpheus's hard boundary predictions, accumulates segment frequencies weighted by word frequency, and selects the top-K segments. Special tokens, the full Turkish character set as fallback, and a curated list of ~90 frequent Turkish suffixes (`-lar`, `-ler`, `-de`, `-da`, `-ki`, `-im`, `-mış`, etc.) are seeded into the vocabulary unconditionally to guarantee morphological coverage.
- **Word boundary**: SentencePiece convention (`▁` U+2581) marks the start of every word, making the format compatible with HuggingFace tokenizers and downstream LM training pipelines.
- **Out-of-vocabulary fallback**: unseen segments decompose character-by-character into vocab IDs, preserving information without loss.
- **Round-trip lossless**: `decode(encode(text))` recovers the original (lowercased) text exactly.

The tokenizer serializes to `vocab.json` + `tokenizer_config.json` for portability and can be loaded for inference without the Morpheus model when cached segmentations cover the input.

---

## Evaluation Framework

### Intrinsic metrics (no labeled data)

| Metric | What it measures |
|---|---|
| **Root cluster coherence** | Mean cosine similarity within a root family vs. across roots — the primary test of whether the embedding space geometrically organizes by morphological root identity. |
| **Morphological analogy** | Mikolov-style offset arithmetic on Morfessor-derived `(root, root+suffix)` quartets; reports top-K accuracy and mean rank. |
| **Nearest neighbors** | Qualitative inspection — given a query word, returns the K nearest neighbors with their cosine scores for visual sanity-checking. |

### Extrinsic metrics (frozen embeddings + linear probe)

| Metric | What it measures |
|---|---|
| **Same-root probe** | Binary linear classifier on `[e₁, e₂, |e₁−e₂|, e₁⊙e₂]` features predicts whether two word embeddings share a root. |
| **Morpheme count probe** | 6-class linear classifier predicts the number of morphemes from the word embedding. |

We document — and treat as a methodological caveat — that probes against Morfessor-derived labels are *partially circular* because Morfessor also supervises Morpheus during training. The headline metric (root cluster coherence) sidesteps this by measuring geometry directly rather than predictability.

### Comparison harness

`MorpheusEvaluator.compare_against_random()` runs the full metric set against a randomly initialized Morpheus to quantify the learning lift. `MorpheusEvaluator.evaluate_external_embedder(embed_fn)` accepts any callable `List[str] → Tensor` and runs the identical evaluation pipeline — this is the integration point for fastText, BERTurk, or any future baseline.

---

## Reproducibility Notes

- Boundary labels derived from Morfessor are produced by `MorfessorWrapper.get_boundary_labels` and use Turkish-aware lowercasing throughout the pipeline. A documented off-by-one bug in an earlier version of the label generator is fixed; older checkpoints can be used at inference via a `legacy_boundary_shift` flag in the tokenizer.
- All training-time stochasticity (SGNS negative sampling, MLM mask positions, dropout) reads from the default PyTorch RNG; seed control for full reproducibility is not yet exposed but is a planned addition.
- The Morfessor model is trained with `corpusweight=1.0` and 20 batch + 5 online epochs, using Turkish-aware lowercased word frequencies from the same corpus that supervises Morpheus.

---

## Status

Active research codebase. Empirical results (head-to-head comparison against BPE/Unigram/WordPiece baselines, sample-efficiency curves under a Morpheus-derived tokenizer for downstream Turkish LM pretraining) are forthcoming and will be reported in a separate technical write-up.

The architectural and training-recipe components — Morpheus, the Poisson-binomial soft segmentation, the multi-objective curriculum, the evaluation harness — are stable and reusable.

---

## Citation

If you use this codebase or build on the architectural ideas, please cite:

```bibtex
@misc{sakar2026morpheus,
  title  = {Morpheus: Morpheme-Aware Neural Tokenization for Turkish},
  author = {Şakar, Tolga},
  year   = {2026},
  note   = {lonewolfrd research notes}
}
```

---

## License

See `LICENSE`.

---

## Acknowledgments

- **Morfessor** (Creutz & Lagus 2002, 2007) as the unsupervised morphology teacher and reference for the Morphological Alignment Score.
- **SentencePiece**, **HuggingFace tokenizers** for the BPE/Unigram/WordPiece baselines.
- The Turkish NLP community for prior work on morphologically aware tokenization (BERTurk, Zemberek, TRMorph, and others) that motivated this study.
