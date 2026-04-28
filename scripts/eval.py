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
  python eval.py --gold dosya.conllu      # özel CoNLL-U dosyası
  python eval.py --gold dosya.conllu --quality   # yalnızca kalite raporu
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter

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
    Modelin tahmin ettiği FEATS etiketlerini döndürür.
    UnigramLM için doğrudan model.predict() çağrılır.
    NgramLM için greedy/viterbi decode.
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


# ─── Değerlendirme ───────────────────────────────────────────────────────────

def evaluate(model, sentences, decode="greedy", max_sents=None,
             dep_model=None):
    """
    Token düzeyinde metrikler döner:
      feats_exact  : FEATS tam eşleşme
      feats_partial: En az 1 özellik doğru
      per_feature  : {özellik_adı: {correct, total}}
      upos         : UPOS doğruluğu
      dep_uas      : UAS (HEAD doğruluğu, punct hariç)
      dep_las      : LAS (HEAD + DEPREL doğruluğu, punct hariç)
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

            # ── FEATS exact match ──
            stats["feats_exact"]["total"] += 1
            if pred_f == gold_f:
                stats["feats_exact"]["correct"] += 1

            # ── FEATS partial match ──
            gold_pairs = set(gold_f.split("|")) if gold_f != "NONE" else set()
            pred_pairs = set(pred_f.split("|")) if pred_f != "NONE" else set()
            stats["feats_partial"]["total"] += 1
            if gold_pairs & pred_pairs:
                stats["feats_partial"]["correct"] += 1

            # ── Per-feature accuracy ──
            for kv in gold_pairs:
                k, _, _ = kv.partition("=")
                per_feat[k]["total"] += 1
                if kv in pred_pairs:
                    per_feat[k]["correct"] += 1

            # ── UPOS ──
            gold_upos = tok_data["upos"]
            pred_upos = upos_from_feats_word(tok_data["form"], pred_f)
            stats["upos"]["total"] += 1
            if pred_upos == gold_upos:
                stats["upos"]["correct"] += 1

            # ── LEMMA ──
            pred_lemma = lemmatize(tok_data["form"], pred_f)
            stats["lemma"]["total"] += 1
            if pred_lemma.lower() == tok_data["lemma"].lower():
                stats["lemma"]["correct"] += 1

            # ── Dependency (punct hariç) ──
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


# ─── Rapor yazdır ────────────────────────────────────────────────────────────

def _acc(d):
    if d["total"] == 0:
        return 0.0
    return 100 * d["correct"] / d["total"]


def print_report(stats, per_feat, decode, model_name, n_sents, n_toks):
    print(f"\n  {'─'*62}")
    print(f"  Model: {model_name}  |  Decoding: {decode.upper()}")
    print(f"  Sektör: {n_sents} cümle, {n_toks} token")
    print(f"  {'─'*62}")
    print(f"  {'Metrik':<28} {'Doğru':>8} {'Toplam':>8} {'Oran':>8}")
    print(f"  {'─'*62}")

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

    print(f"\n  {'─'*62}")
    print(f"  Per-feature accuracy (gold özellik bazında):")
    print(f"  {'Özellik':<24} {'Doğru':>8} {'Toplam':>8} {'Oran':>8}")
    print(f"  {'─'*62}")

    for feat_key, d in sorted(per_feat.items(),
                               key=lambda x: -x[1]["total"]):
        if d["total"] < 5:
            continue
        print(f"  {feat_key:<24} {d['correct']:>8,} {d['total']:>8,} {_acc(d):>7.2f}%")


# ─── Kalite denetimi ──────────────────────────────────────────────────────────

# Her UPOS için Türkçe'de beklenen temel özellik kümeleri
_UPOS_EXPECTED_FEATS: dict[str, set[str]] = {
    "VERB":  {"Tense", "Aspect", "Mood"},
    "AUX":   {"Tense", "Aspect", "Mood"},
    "NOUN":  {"Case", "Number"},
    "PROPN": {"Case", "Number"},
    "ADJ":   {"Case", "Number"},
    "PRON":  {"Case", "Number", "Person"},
    "DET":   {"Case", "Number"},
    "NUM":   {"Case", "Number"},
}

# VerbForm değerleri Tense/Mood/Aspect beklentisini kaldırır (fiil değil)
_VERBFORM_EXEMPTS_FINITE: frozenset[str] = frozenset({"Part", "Conv", "Vnoun"})

# Bar grafiği için maksimum sütun genişliği
_BAR_MAX_WIDTH = 30


def check_quality(sentences: list) -> dict:
    """
    CoNLL-U cümle listesinin kalite metriklerini hesaplar.

    Döner:
      n_sents          : cümle sayısı
      n_tokens         : token sayısı
      upos_coverage    : UPOS alanı dolu token sayısı
      feats_coverage   : FEATS alanı dolu token sayısı
      lemma_coverage   : LEMMA alanı dolu token sayısı
      dep_coverage     : DEPREL alanı dolu token sayısı
      upos_dist        : Counter(UPOS → count)
      feats_dist       : Counter(FEATS_key → count)  (özellik adı bazında)
      missing_required : Counter("UPOS:Feature" → kaç token'da eksik)
      dep_errors       : sınır dışı HEAD içeren token açıklamaları (ilk 10)
      sent_lengths     : cümle uzunluğu dağılımı (min/max/ortalama)
    """
    n_sents = len(sentences)
    n_tokens = 0
    upos_coverage = 0
    feats_coverage = 0
    lemma_coverage = 0
    dep_coverage = 0
    upos_dist: Counter = Counter()
    feats_dist: Counter = Counter()
    missing_required: Counter = Counter()
    dep_errors: list = []
    lengths: list = []

    for sent in sentences:
        toks = sent["tokens"]
        n = len(toks)
        lengths.append(n)
        n_tokens += n

        for i, tok in enumerate(toks):
            upos   = tok["upos"]
            feats  = tok["feats"]
            lemma  = tok["lemma"]
            head   = tok["head"]
            deprel = tok["deprel"]

            # Kapsam
            if upos and upos != "_":
                upos_coverage += 1
            if feats != "NONE":
                feats_coverage += 1
            if lemma and lemma != "_":
                lemma_coverage += 1
            if deprel and deprel != "_":
                dep_coverage += 1

            upos_dist[upos] += 1

            # Özellik adı dağılımı
            if feats != "NONE":
                for kv in feats.split("|"):
                    k = kv.split("=")[0]
                    feats_dist[k] += 1

            # UPOS-FEATS tutarlılık denetimi
            expected = _UPOS_EXPECTED_FEATS.get(upos, set())
            if expected:
                present_keys: set = set()
                feat_vals: dict = {}
                if feats != "NONE":
                    for kv in feats.split("|"):
                        k, _, v = kv.partition("=")
                        present_keys.add(k)
                        feat_vals[k] = v
                # VerbForm=Part/Conv/Vnoun → fiil değil; Tense/Aspect/Mood beklenmez
                if (upos in ("VERB", "AUX")
                        and feat_vals.get("VerbForm") in _VERBFORM_EXEMPTS_FINITE):
                    expected = set()
                for feat_key in expected - present_keys:
                    missing_required[f"{upos}:{feat_key}"] += 1

            # HEAD sınır denetimi
            if head < 0 or head > n:
                snippet = sent["text"][:40] if sent.get("text") else f"cümle-{len(lengths)}"
                dep_errors.append(
                    f"{snippet!r}  — token {i+1} ({tok['form']!r})  HEAD={head}  (n={n})"
                )

    avg_len = sum(lengths) / n_sents if n_sents else 0.0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    return {
        "n_sents":          n_sents,
        "n_tokens":         n_tokens,
        "upos_coverage":    upos_coverage,
        "feats_coverage":   feats_coverage,
        "lemma_coverage":   lemma_coverage,
        "dep_coverage":     dep_coverage,
        "upos_dist":        upos_dist,
        "feats_dist":       feats_dist,
        "missing_required": missing_required,
        "dep_errors":       dep_errors[:10],
        "sent_lengths":     {"min": min_len, "max": max_len, "avg": avg_len},
    }


def print_quality_report(qstats: dict, path: "Path | str") -> None:
    """Kalite raporunu ekrana yazdırır."""
    n = qstats["n_tokens"]

    def pct(k: str) -> float:
        return 100 * qstats[k] / n if n > 0 else 0.0

    print(f"\n  {'═'*62}")
    print(f"  KALİTE RAPORU: {Path(path).name}")
    print(f"  {'═'*62}")
    print(f"  Cümle sayısı   : {qstats['n_sents']:,}")
    print(f"  Token sayısı   : {n:,}")
    sl = qstats["sent_lengths"]
    print(f"  Cümle uzunluğu : ort. {sl['avg']:.1f}  |  min {sl['min']}  |  max {sl['max']}")

    print(f"\n  {'─'*62}")
    print(f"  Kapsam (Coverage)")
    print(f"  {'─'*62}")
    print(f"  {'Alan':<24} {'Sayı':>8} {'Oran':>8}")
    for label, key in [
        ("UPOS",   "upos_coverage"),
        ("FEATS",  "feats_coverage"),
        ("LEMMA",  "lemma_coverage"),
        ("DEPREL", "dep_coverage"),
    ]:
        print(f"  {label:<24} {qstats[key]:>8,} {pct(key):>7.2f}%")

    print(f"\n  {'─'*62}")
    print(f"  UPOS Dağılımı")
    print(f"  {'─'*62}")
    print(f"  {'UPOS':<12} {'Sayı':>8}  Oran")
    for upos, cnt in sorted(qstats["upos_dist"].items(), key=lambda x: -x[1]):
        bar = "█" * max(1, int(cnt / n * _BAR_MAX_WIDTH)) if n else ""
        print(f"  {upos:<12} {cnt:>8,}  {bar}")

    print(f"\n  {'─'*62}")
    print(f"  Morfolojik Özellik Dağılımı (Feature Coverage)")
    print(f"  {'─'*62}")
    print(f"  {'Özellik':<24} {'Sayı':>8}")
    for feat_key, cnt in sorted(qstats["feats_dist"].items(), key=lambda x: -x[1]):
        print(f"  {feat_key:<24} {cnt:>8,}")

    if qstats["missing_required"]:
        print(f"\n  {'─'*62}")
        print(f"  Eksik Beklenen Özellikler (UPOS:Feature — uyarı)")
        print(f"  {'─'*62}")
        for combo, cnt in sorted(
            qstats["missing_required"].items(), key=lambda x: -x[1]
        )[:20]:
            print(f"  {combo:<36} {cnt:>6,} token")

    if qstats["dep_errors"]:
        print(f"\n  {'─'*62}")
        print(f"  HEAD Sınır Dışı Hatalar (ilk {len(qstats['dep_errors'])})")
        print(f"  {'─'*62}")
        for err in qstats["dep_errors"]:
            print(f"  {err}")

    print(f"\n  {'═'*62}\n")


# ─── Ana giriş ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Türkçe trigram morfoloji değerlendirici"
    )
    parser.add_argument("--model",     default="full",
                        help="Kullanılacak model adı — 'hybrid', 'hybrid_distil', 'ngram5' vb. "
                             "(models/model_<isim>.pkl dosyası yüklenir; varsayılan: full)")
    parser.add_argument("--split",     default="dev",
                        choices=["dev", "test", "train"],
                        help="BOUN treebank bölümü (varsayılan: dev); --gold verilirse görmezden gelinir")
    parser.add_argument("--gold",      default=None,
                        metavar="DOSYA",
                        help="Değerlendirilecek CoNLL-U dosyası (herhangi bir yol). "
                             "Verilmezse BOUN treebank kullanılır.")
    parser.add_argument("--quality",   action="store_true",
                        help="Yalnızca kalite raporu üret; model yükleme ve doğruluk "
                             "değerlendirmesi yapılmaz.")
    parser.add_argument("--decode",    default="greedy",
                        choices=["greedy", "viterbi"],
                        help="Decoding yöntemi (varsayılan: greedy)")
    parser.add_argument("--compare",   action="store_true",
                        help="Greedy ve Viterbi'yi yan yana karşılaştır")
    parser.add_argument("--max-sents", type=int, default=None,
                        metavar="N",
                        help="Değerlendirilecek maksimum cümle sayısı")
    parser.add_argument("--dep-model", default=None,
                        metavar="NAME",
                        help="İstatistiksel dep parser modeli (ör. model_dep). "
                             "Belirtilmezse kural tabanlı kullanılır.")
    args = parser.parse_args()

    # ── Gold dosya yolu ──────────────────────────────────────────────────────
    if args.gold:
        gold_path = Path(args.gold)
    else:
        gold_path = DATA_DIR / f"tr_boun-ud-{args.split}.conllu"

    if not gold_path.exists():
        print(f"\n  HATA: {gold_path} bulunamadı.")
        if not args.gold:
            print(f"  Önce 'python trigram_morph.py' çalıştırın veya --gold ile dosya belirtin.")
        sys.exit(1)

    print(f"\n  Gold veri yükleniyor: {gold_path} ...", end="")
    sentences = read_gold_conllu(gold_path)
    n_sents   = len(sentences[:args.max_sents]) if args.max_sents else len(sentences)
    n_toks    = sum(len(s["tokens"]) for s in sentences[:args.max_sents or len(sentences)])
    print(f" {n_sents} cümle, {n_toks} token")

    # ── Kalite raporu ────────────────────────────────────────────────────────
    qstats = check_quality(sentences[:args.max_sents or len(sentences)])
    print_quality_report(qstats, gold_path)

    # --quality bayrağı varsa model yükleme ve doğruluk değerlendirmesi yapma
    if args.quality:
        return

    # ── Model yükleme ────────────────────────────────────────────────────────
    model_name = f"model_{args.model}"
    print(f"  Model yükleniyor: {model_name} ...", end="")
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
