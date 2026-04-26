# tr-morph-ngram

**Türkçe Morfolojik Tagger — N-gram tabanlı olasılıksal model (BOUN UD Treebank)**

Turkish morphological feature tagger using trigram language models and character n-gram emission. Trained and evaluated on BOUN UD Treebank v2.x.

---

## Motivation

Turkish suffixes don't just encode information about the word they attach to — they interact with morphological features of neighboring words in a sentence. This project tests the hypothesis that **cross-word suffix interactions** can be modeled probabilistically with n-grams.

---

## Architecture

```
HybridLM
├── WordTagEmission    — MLE from word-feats co-occurrence (BOUN train)
├── CharNgramEmission  — Suffix pattern → FEATS distribution (BOUN only, weight=3)
│   └── AMV Boost Rules  — Aspect/Mood/Voice deterministic boosts
├── NgramLM            — Trigram over FEATS sequences (cross-word context)
└── Viterbi Decoder    — Beam/full Viterbi with n-gram transition weights
```

**Key design decision:** CharNgramEmission is trained **only on BOUN** — mixing other UD treebanks (imst, kenet, penn) causes annotation schema conflicts that degrade Evident and VerbForm accuracy.

---

## Results (BOUN dev set)

| Feature | Accuracy |
|---------|----------|
| FEATS (overall) | **80.39%** |
| Tense | 85.85% |
| Evident | 84.81% |
| VerbForm | 76.88% |
| Aspect | 70.39% |
| Voice | 56.91% |
| Mood | 56.04% |

**Baseline** (unigram, no cross-word context): FEATS ~56% / Aspect ~56%

---

## Files

| File | Description |
|------|-------------|
| `trigram_morph.py` | Core model: HybridLM, CharNgramEmission, NgramLM, Viterbi, training & tuning |
| `unigram_morph.py` | Baseline unigram tagger |
| `morph/rules.py` | Suffix pattern rules & AMV boost tables |
| `morph/feats.py` | FEATS parsing utilities |
| `eval.py` | Full evaluation pipeline |
| `eval_unigram.py` | Unigram baseline eval |
| `dep_parser.py` | Rule-based dependency parser (experimental) |
| `mine_suffixes.py` | Suffix mining utilities |
| `TURKCE_MORFOLOJI_OGRETICI.md` | Turkish morphology tutorial (Turkish) |

---

## Usage

### Train & save model

```python
from trigram_morph import run_hybrid
model = run_hybrid(save_path="models/model_hybrid.pkl")
```

### Load & tag

```python
from trigram_morph import load_model, tag_sentence
model = load_model("models/model_hybrid.pkl")
tags = tag_sentence(model, "Kapı kırılmış .")
for word, feats in tags:
    print(f"{word:15} {feats}")
```

### Evaluate

```bash
python eval.py --model hybrid --decode viterbi
```

### Tune w_trans (multiprocessing)

```python
from trigram_morph import tune_w_trans
tune_w_trans(n_workers=4)
```

---

## Data

Model is trained on [BOUN UD Treebank](https://github.com/UniversalDependencies/UD_Turkish-BOUN) (not included — download separately and place in `data/`).

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n0` | 0.5 | WTE backoff exponent |
| `w_trans` | 0.07 | N-gram transition weight |
| `state_size` | 2 | Viterbi beam width |

---

## License

MIT
