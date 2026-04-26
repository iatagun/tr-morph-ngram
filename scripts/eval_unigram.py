"""
eval_unigram.py — Unigram baseline karşılaştırması
===================================================
Üç decoder karşılaştırır:
  1. Heuristik-sadece : suffix kuralı → ilk aday (bağlam yok)
  2. Sözcük-unigram  : eğitim verisinden word → en sık etiket (OOV → heuristik)
  3. Trigram Viterbi : mevcut modelimiz (referans)
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter

from taggers.ngram import (
    load_model, parse_feats,
    heuristic_candidates_weighted,
    viterbi_decode,
    upos_from_feats_word, lemmatize,
    DATA_DIR, BOS1, BOS2,
)


# ─── Gold CoNLL-U okuyucu ─────────────────────────────────────────────────────

def read_conllu(path: Path):
    sentences, current, text = [], [], ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# text ="):
                text = line[8:].strip(); continue
            if line.startswith("#") or line == "":
                if not line and current:
                    sentences.append({"text": text, "tokens": current})
                    current, text = [], ""
                continue
            parts = line.split("\t")
            if len(parts) < 8 or "-" in parts[0] or "." in parts[0]:
                continue
            feats_raw = parts[5]
            feats = parse_feats(feats_raw, reduced=False) if feats_raw != "_" else "NONE"
            current.append({"form": parts[1], "lemma": parts[2],
                             "upos": parts[3], "feats": feats})
    if current:
        sentences.append({"text": text, "tokens": current})
    return sentences


# ─── Sözcük → etiket frekans tablosu ─────────────────────────────────────────

def build_word_unigram(train_path: Path) -> dict:
    """Eğitim verisinden word.lower() → en sık FEATS etiketi"""
    freq: dict = defaultdict(Counter)
    for sent in read_conllu(train_path):
        for tok in sent["tokens"]:
            if tok["feats"] != "NONE":
                freq[tok["form"].lower()][tok["feats"]] += 1
    return {w: cnt.most_common(1)[0][0] for w, cnt in freq.items()}


# ─── Decode fonksiyonları ─────────────────────────────────────────────────────

def decode_heuristic_only(tokens):
    """Bağlam yok: her sözcük için ilk heuristik aday"""
    preds = []
    for tok in tokens:
        cands = heuristic_candidates_weighted(tok)
        # Ağırlığa göre sırala, en yüksek ağırlıklıyı al
        best = max(cands, key=lambda x: x[1])[0]
        preds.append(best)
    return preds


def decode_word_unigram(tokens, word_table):
    """Eğitim kelimesi → en sık etiket; OOV → heuristik ilk aday"""
    preds = []
    for tok in tokens:
        w = tok.lower()
        if w in word_table:
            preds.append(word_table[w])
        else:
            cands = heuristic_candidates_weighted(tok)
            preds.append(max(cands, key=lambda x: x[1])[0])
    return preds


def decode_trigram(model, tokens):
    pairs = viterbi_decode(model, tokens)
    return [tag for _, tag in pairs]


# ─── Değerlendirme ────────────────────────────────────────────────────────────

def evaluate_preds(sentences, preds_fn, label, max_sents=300):
    exact_c = exact_t = 0
    upos_c  = upos_t  = 0
    lemma_c = lemma_t = 0
    per_feat = defaultdict(lambda: {"c": 0, "t": 0})

    data = sentences[:max_sents]
    for sent in data:
        toks    = [t["form"] for t in sent["tokens"]]
        pred_fs = preds_fn(toks)
        for i, tok_data in enumerate(sent["tokens"]):
            gold_f = tok_data["feats"]
            pred_f = pred_fs[i] if i < len(pred_fs) else "NONE"

            exact_t += 1
            if pred_f == gold_f:
                exact_c += 1

            gold_kv = set(gold_f.split("|")) if gold_f != "NONE" else set()
            pred_kv = set(pred_f.split("|")) if pred_f != "NONE" else set()
            for kv in gold_kv:
                k = kv.split("=")[0]
                per_feat[k]["t"] += 1
                if kv in pred_kv:
                    per_feat[k]["c"] += 1

            pred_upos = upos_from_feats_word(tok_data["form"], pred_f)
            upos_t += 1
            if pred_upos == tok_data["upos"]:
                upos_c += 1

            pred_lem = lemmatize(tok_data["form"], pred_f)
            lemma_t += 1
            if pred_lem.lower() == tok_data["lemma"].lower():
                lemma_c += 1

    key_feats = ["Case", "Tense", "Aspect", "VerbForm", "Evident", "Mood",
                 "Polarity", "Voice", "Person[psor]", "Number[psor]"]

    print(f"\n  ═══ {label} ({'viterbi' if 'Trigram' in label else 'unigram'}) ═══")
    print(f"  FEATS exact : {exact_c:4d}/{exact_t} = {exact_c/exact_t*100:.2f}%")
    print(f"  UPOS        : {upos_c:4d}/{upos_t} = {upos_c/upos_t*100:.2f}%")
    print(f"  LEMMA       : {lemma_c:4d}/{lemma_t} = {lemma_c/lemma_t*100:.2f}%")
    print(f"  ── Per-feature ──")
    for k in key_feats:
        if k in per_feat:
            d = per_feat[k]
            print(f"    {k:<18}: {d['c']:3d}/{d['t']:3d} = {d['c']/d['t']*100:5.1f}%")

    return exact_c / exact_t


# ─── Ana ─────────────────────────────────────────────────────────────────────

def main():
    max_sents = 300
    train_path = DATA_DIR / "tr_boun-ud-train.conllu"
    dev_path   = DATA_DIR / "tr_boun-ud-dev.conllu"

    print("Gold veri yükleniyor...")
    dev_sents = read_conllu(dev_path)
    print(f"  {len(dev_sents)} cümle, {sum(len(s['tokens']) for s in dev_sents[:max_sents])} token")

    print("Sözcük-unigram tablosu oluşturuluyor (eğitim verisi)...")
    word_table = build_word_unigram(train_path)
    print(f"  {len(word_table)} eşsiz sözcük kaydedildi")

    print("Trigram modeli yükleniyor...")
    model = load_model("model_full")
    print("  ✓")

    # 1. Heuristik-sadece
    evaluate_preds(dev_sents,
                   lambda toks: decode_heuristic_only(toks),
                   "Heuristik-Sadece (bağlam yok)",
                   max_sents)

    # 2. Sözcük-unigram
    evaluate_preds(dev_sents,
                   lambda toks: decode_word_unigram(toks, word_table),
                   "Sözcük-Unigram (eğitim sözlüğü + OOV→heuristik)",
                   max_sents)

    # 3. Trigram Viterbi (referans)
    evaluate_preds(dev_sents,
                   lambda toks: decode_trigram(model, toks),
                   "Trigram Viterbi (mevcut model)",
                   max_sents)

    print()


if __name__ == "__main__":
    main()
