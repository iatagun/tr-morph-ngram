"""
eval.py — Türkçe Trigram Morfoloji Değerlendirici
==================================================
Gold CoNLL-U dosyasıyla karşılaştırma.

Kullanım:
  python eval.py                          # BOUN dev, greedy, model_full
  python eval.py --split test             # test seti
  python eval.py --decode viterbi         # Viterbi karşılaştırması
  python eval.py --model reduced          # reduced model
  python eval.py --max-sents 200          # ilk 200 cümle
  python eval.py --compare                # greedy vs viterbi yan yana
  python eval.py --dep-model model_dep    # istatistiksel dep parser kullan
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

from taggers import dep as _dep_parser

from data.conllu import read_conllu as _read_conllu
from taggers.ngram import (
    load_model,
    parse_feats,
    heuristic_candidates,
    rerank_candidates,
    viterbi_decode,
    upos_from_feats_word,
    dependency_parse,
    lemmatize,
    NgramLM,
    OrchestratorLM,
    HybridLM,
    DATA_DIR,
    BOS1, BOS2,
)
from taggers.unigram import UnigramLM


def load_orchestrator() -> OrchestratorLM:
    models = []
    for name in ["ngram5h"]:
        try:
            models.append(load_model(f"model_{name}"))
        except FileNotFoundError:
            pass
    try:
        models.append(UnigramLM.load("model_unigram"))
    except FileNotFoundError:
        pass
    return OrchestratorLM(models)


# ─── Gold CoNLL-U okuyucu ─────────────────────────────────────────────────────

def read_gold_conllu(path: Path):
    """
    CoNLL-U dosyasından cümleleri okur.
    Her cümle: {"text": ..., "tokens": [{"form","lemma","upos","feats","head","deprel"}, ...]}
    data.conllu.read_conllu'ya delegate eder.
    """
    result = []
    for sent in _read_conllu(path):
        tokens = []
        for tok in sent.tokens:
            feats = parse_feats(tok.feats_raw, reduced=False)
            tokens.append({
                "form":   tok.form,
                "lemma":  tok.lemma,
                "upos":   tok.upos,
                "feats":  feats,
                "head":   tok.head,
                "deprel": tok.deprel,
            })
        result.append({"text": sent.text, "tokens": tokens})
    return result


# ─── Model tahmin fonksiyonu ──────────────────────────────────────────────────

def predict(model, tokens, decode="greedy"):
    """
    Modelin tahmin etti?i FEATS etiketlerini d?nd?r?r.
    UnigramLM i?in do?rudan model.predict() ?a?r?l?r.
    NgramLM i?in greedy/viterbi decode.
    """
    if isinstance(model, UnigramLM):
        return model.predict(tokens)

    if isinstance(model, (NgramLM, OrchestratorLM, HybridLM)):
        if decode == "viterbi":
            pairs = model.decode_viterbi(tokens)
        else:
            pairs = model.decode_greedy(tokens)
        return [tag for _, tag in pairs]

    if decode == "viterbi":
        pairs = viterbi_decode(model, tokens)
        return [tag for _, tag in pairs]

    context = [BOS1, BOS2]
    preds   = []
    for tok in tokens:
        cands          = heuristic_candidates(tok)
        prev2, prev1   = context[-2], context[-1]
        ranked         = rerank_candidates(model, (prev2, prev1), cands)
        best_tag       = ranked[0][0]
        context.append(best_tag)
        preds.append(best_tag)
    return preds


# ??? De?erlendirme ????????????????????????????????????????????????????????????

def evaluate(model, sentences, decode="greedy", max_sents=None,
             dep_model=None):
    """
    Token d?zeyinde metrikler d?ner:
      feats_exact  : FEATS tam e?le?me
      feats_partial: En az 1 ?zellik do?ru
      per_feature  : {?zellik_ad?: {correct, total}}
      upos         : UPOS do?rulu?u
      dep_uas      : UAS (HEAD do?rulu?u, punct hari?)
      dep_las      : LAS (HEAD + DEPREL do?rulu?u, punct hari?)
      lemma        : LEMMA exact match
    """
    stats = {
        "feats_exact":   {"correct": 0, "total": 0},
        "feats_partial": {"correct": 0, "total": 0},
        "upos":          {"correct": 0, "total": 0},
        "dep_uas":       {"correct": 0, "total": 0},
        "dep_las":       {"correct": 0, "total": 0},
        "lemma":         {"correct": 0, "total": 0},
    }
    per_feat: dict = defaultdict(lambda: {"correct": 0, "total": 0})

    data = sentences[:max_sents] if max_sents else sentences

    for sent in data:
        toks    = [t["form"]   for t in sent["tokens"]]
        pred_fs = predict(model, toks, decode=decode)

        # Bağımlılık tahmini
        pred_upos_list = [upos_from_feats_word(t["form"], f)
                          for t, f in zip(sent["tokens"], pred_fs)]
        if dep_model is not None:
            dep_pred = _dep_parser.parse_sentence(
                dep_model, toks, pred_fs, pred_upos_list)
        else:
            dep_pred = dependency_parse(toks, pred_fs)

        for i, tok_data in enumerate(sent["tokens"]):
            gold_f = tok_data["feats"]
            pred_f = pred_fs[i]

            # ?? FEATS exact match ??
            stats["feats_exact"]["total"] += 1
            if pred_f == gold_f:
                stats["feats_exact"]["correct"] += 1

            # ?? FEATS partial match ??
            gold_pairs = set(gold_f.split("|")) if gold_f != "NONE" else set()
            pred_pairs = set(pred_f.split("|")) if pred_f != "NONE" else set()
            stats["feats_partial"]["total"] += 1
            if gold_pairs & pred_pairs:
                stats["feats_partial"]["correct"] += 1

            # ?? Per-feature accuracy ??
            for kv in gold_pairs:
                k, _, _ = kv.partition("=")
                per_feat[k]["total"] += 1
                if kv in pred_pairs:
                    per_feat[k]["correct"] += 1

            # ?? UPOS ??
            gold_upos = tok_data["upos"]
            pred_upos = upos_from_feats_word(tok_data["form"], pred_f)
            stats["upos"]["total"] += 1
            if pred_upos == gold_upos:
                stats["upos"]["correct"] += 1

            # ?? LEMMA ??
            pred_lemma = lemmatize(tok_data["form"], pred_f)
            stats["lemma"]["total"] += 1
            if pred_lemma.lower() == tok_data["lemma"].lower():
                stats["lemma"]["correct"] += 1

            # ?? Dependency (punct hari?) ??
            if gold_upos != "PUNCT":
                gold_head   = tok_data["head"]
                gold_deprel = tok_data["deprel"]
                pred_head, pred_deprel = dep_pred[i]
                stats["dep_uas"]["total"] += 1
                stats["dep_las"]["total"] += 1
                if pred_head == gold_head:
                    stats["dep_uas"]["correct"] += 1
                    if pred_deprel == gold_deprel:
                        stats["dep_las"]["correct"] += 1

    return stats, per_feat


# ??? Rapor yazd?r ?????????????????????????????????????????????????????????????

def _acc(d):
    if d["total"] == 0:
        return 0.0
    return 100 * d["correct"] / d["total"]


def print_report(stats, per_feat, decode, model_name, n_sents, n_toks):
    print(f"\n  {'?'*62}")
    print(f"  Model: {model_name}  |  Decoding: {decode.upper()}")
    print(f"  Sekt?r: {n_sents} c?mle, {n_toks} token")
    print(f"  {'?'*62}")
    print(f"  {'Metrik':<28} {'Do?ru':>8} {'Toplam':>8} {'Oran':>8}")
    print(f"  {'?'*62}")

    rows = [
        ("FEATS exact match",   "feats_exact"),
        ("FEATS partial match", "feats_partial"),
        ("UPOS",                "upos"),
        ("LEMMA",               "lemma"),
        ("Dep UAS (punct ex.)", "dep_uas"),
        ("Dep LAS (punct ex.)", "dep_las"),
    ]
    for label, key in rows:
        d = stats[key]
        print(f"  {label:<28} {d['correct']:>8,} {d['total']:>8,} {_acc(d):>7.2f}%")

    print(f"\n  {'?'*62}")
    print(f"  Per-feature accuracy (gold ?zellik baz?nda):")
    print(f"  {'?zellik':<24} {'Do?ru':>8} {'Toplam':>8} {'Oran':>8}")
    print(f"  {'?'*62}")

    for feat_key, d in sorted(per_feat.items(),
                               key=lambda x: -x[1]["total"]):
        if d["total"] < 5:
            continue
        print(f"  {feat_key:<24} {d['correct']:>8,} {d['total']:>8,} {_acc(d):>7.2f}%")

    print(f"  {'?'*62}\n")


# ??? Ana giri? ????????????????????????????????????????????????????????????????

def main():
    parser = argparse.ArgumentParser(
        description="T?rk?e trigram morfoloji de?erlendirici"
    )
    parser.add_argument("--model",     default="full",
                        help="Kullanılacak model adı — 'hybrid', 'hybrid_distil', 'ngram5' vb. "
                             "(models/model_<isim>.pkl dosyası yüklenir; varsayılan: full)")
    parser.add_argument("--split",     default="dev",
                        choices=["dev", "test", "train"],
                        help="BOUN treebank b?l?m? (varsay?lan: dev)")
    parser.add_argument("--decode",    default="greedy",
                        choices=["greedy", "viterbi"],
                        help="Decoding y?ntemi (varsay?lan: greedy)")
    parser.add_argument("--compare",   action="store_true",
                        help="Greedy ve Viterbi'yi yan yana kar??la?t?r")
    parser.add_argument("--max-sents", type=int, default=None,
                        metavar="N",
                        help="De?erlendirilecek maksimum c?mle say?s?")
    parser.add_argument("--dep-model", default=None,
                        metavar="NAME",
                        help="İstatistiksel dep parser modeli (ör. model_dep). "
                             "Belirtilmezse kural tabanlı kullanılır.")
    args = parser.parse_args()

    gold_path = DATA_DIR / f"tr_boun-ud-{args.split}.conllu"
    if not gold_path.exists():
        print(f"\n  HATA: {gold_path} bulunamad?."
              f"\n  ?nce 'python trigram_morph.py' ?al??t?r?n.\n")
        sys.exit(1)

    print(f"\n  Gold veri y?kleniyor: {gold_path.name} ...", end="")
    sentences = read_gold_conllu(gold_path)
    n_sents   = len(sentences[:args.max_sents]) if args.max_sents else len(sentences)
    n_toks    = sum(len(s["tokens"]) for s in sentences[:args.max_sents or len(sentences)])
    print(f" {n_sents} c?mle, {n_toks} token")

    model_name = f"model_{args.model}"
    print(f"  Model y?kleniyor: {model_name} ...", end="")
    if args.model == "unigram":
        model = UnigramLM.load(model_name)
    elif args.model == "orch":
        model = load_orchestrator()
    else:
        from taggers.ngram import load_model as _load
        model = _load(model_name)
    print(" OK")

    dep_model = None
    if args.dep_model:
        print(f"  Dep parser yükleniyor: {args.dep_model} ...", end="")
        dep_model = _dep_parser.load_parser(args.dep_model)
        print(" OK")

    if args.compare:
        for dec in ("greedy", "viterbi"):
            stats, per_feat = evaluate(
                model, sentences, decode=dec, max_sents=args.max_sents,
                dep_model=dep_model
            )
            print_report(stats, per_feat, dec, model_name, n_sents, n_toks)
    else:
        stats, per_feat = evaluate(
            model, sentences, decode=args.decode, max_sents=args.max_sents,
            dep_model=dep_model
        )
        print_report(stats, per_feat, args.decode, model_name, n_sents, n_toks)


if __name__ == "__main__":
    main()
