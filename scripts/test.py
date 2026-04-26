"""
test.py — Türkçe Morfolojik Trigram Reranker: İnteraktif Test Arayüzü
======================================================================
Kullanım:
  python test.py                         # varsayılan: model_reduced.pkl
  python test.py --model full            # model_full.pkl
  python test.py --sentence "..."        # tek cümle, çıkış yok
  python test.py --format conllu -s "..." # CoNLL-U formatında çıktı

Gereksinim: önce 'python trigram_morph.py' çalıştırılmış olmalı
            (models/ klasöründe .pkl dosyaları oluşur).
"""

import argparse
import sys
import io
from pathlib import Path

from taggers import dep as _dep_parser
from taggers.ngram import (
    load_model,
    heuristic_candidates,
    heuristic_candidates_weighted,
    rerank_candidates,
    viterbi_decode,
    parse_feats,
    lemmatize,
    upos_from_feats_word,
    dependency_parse,
    NgramLM,
    OrchestratorLM,
    HybridLM,
    BOS1, BOS2,
    MODEL_DIR,
)
from taggers.unigram import UnigramLM


def load_orchestrator() -> OrchestratorLM:
    """
    Hibrit (kelime tablosu + N-gram bağlamı) modellerini orkestre eder.
    Saf heuristik modeller (ngram2-5 kelime tablosuz) dahil edilmez —
    onların yanlış heuristik adayları kelime tablosunun doğru sinyalini bastırır.
    """
    models = []
    # Kelime tablosu OLAN modeller: ngram5h (bağlam+tablo) + unigram (saf tablo)
    for name in ["ngram5h"]:
        try:
            models.append(load_model(f"model_{name}"))
        except FileNotFoundError:
            pass
    try:
        models.append(UnigramLM.load("model_unigram"))
    except FileNotFoundError:
        pass
    if not models:
        raise FileNotFoundError(
            "Orkestratör için hiçbir model bulunamadı.\n"
            "Önce 'python trigram_morph.py --corpus boun --order 5' çalıştırın."
        )
    return OrchestratorLM(models)


# ─── Tek cümle analizi ────────────────────────────────────────────────────────

def analyze(sentence: str, model, verbose: bool = True,
            decode: str = "greedy") -> list:
    """
    Cümleyi analiz eder.
    decode='greedy' → soldan sağa greedy
    decode='viterbi' → tüm cümle global optimizasyon
    Döner: [(token, seçilen_etiket, [(etiket, skor), ...]), ...]
    """
    tokens = sentence.split()

    # UnigramLM: word-lookup tabanlı tahmin
    if isinstance(model, UnigramLM):
        preds = model.predict(tokens)
        results = []
        for tok, tag in zip(tokens, preds):
            results.append((tok, tag, [(tag, 1.0)]))
        if verbose:
            for tok, tag, _ in results:
                print(f"\n  {'─'*60}")
                print(f"  Sözcük  : {tok}")
                print(f"  Seçilen : {tag}  [Unigram]")
        return results

    # NgramLM, OrchestratorLM veya HybridLM: bağlam tabanlı decode
    if isinstance(model, (NgramLM, OrchestratorLM, HybridLM)):
        if decode == "viterbi":
            pairs = model.decode_viterbi(tokens)
        else:
            pairs = model.decode_greedy(tokens)
        results = []
        for tok, tag in pairs:
            results.append((tok, tag, [(tag, 1.0)]))
        if verbose:
            if isinstance(model, OrchestratorLM):
                n_models = len(model.models)
                label = f"[Orkestratör ({n_models} model) / {decode.upper()}]"
            elif isinstance(model, HybridLM):
                label = f"[HybridLM {model.max_order}-gram w={model.w_trans} / {decode.upper()}]"
            else:
                order_label = f"{model.max_order}-gram"
                hybrid = "+Hibrit" if model.word_table else ""
                label = f"[{order_label}{hybrid} / {decode.upper()}]"
            for tok, tag, _ in results:
                print(f"\n  {'─'*60}")
                print(f"  Sözcük  : {tok}")
                print(f"  Seçilen : {tag}  {label}")
        return results

    if decode == "viterbi":
        pairs = viterbi_decode(model, tokens)
        results = []
        for tok, tag in pairs:
            cands  = heuristic_candidates(tok)
            dummy_context = (BOS1, BOS2)
            ranked = rerank_candidates(model, dummy_context, cands)
            results.append((tok, tag, ranked))
        if verbose:
            for tok, tag, ranked in results:
                best_score = max(s for _, s in ranked) if ranked else 1.0
                _print_token_result_viterbi(tok, tag, ranked, best_score)
        return results

    # greedy (TrigramLM)
    context = [BOS1, BOS2]
    results = []
    for tok in tokens:
        cands  = heuristic_candidates(tok)
        prev2, prev1 = context[-2], context[-1]
        ranked = rerank_candidates(model, (prev2, prev1), cands)
        best_tag, best_score = ranked[0]
        context.append(best_tag)
        results.append((tok, best_tag, ranked))
        if verbose:
            _print_token_result(tok, ranked, best_score)
    return results


def _print_token_result(tok: str, ranked: list, best_score: float):
    best_tag = ranked[0][0]
    print(f"\n  {'─'*60}")
    print(f"  Sözcük  : {tok}")
    print(f"  Seçilen : {best_tag}")
    if len(ranked) > 1:
        print(f"  Adaylar :")
        for tag, score in ranked:
            bar   = "▓" * int(score / best_score * 20)
            mark  = " ◄" if tag == best_tag else "  "
            delta = f"(×{score/best_score:.2f})" if tag != best_tag else "(×1.00)"
            print(f"    {mark} {bar:<20} {delta}  {tag}")


def _print_token_result_viterbi(tok: str, viterbi_tag: str,
                                ranked: list, best_score: float):
    print(f"\n  {'─'*60}")
    print(f"  Sözcük  : {tok}")
    print(f"  Seçilen : {viterbi_tag}  [Viterbi ✦]")
    if len(ranked) > 1:
        print(f"  Adaylar (yerel skor):")
        for tag, score in ranked:
            bar  = "▓" * int(score / (best_score + 1e-15) * 20)
            mark = " ◄" if tag == viterbi_tag else "  "
            print(f"    {mark} {bar:<20} {tag}")


def _feats_to_upos(feats: str, word: str = "") -> str:
    """FEATS + sözcük biçiminden UPOS tahmini (trigram_morph.upos_from_feats_word kullanır)."""
    return upos_from_feats_word(word, feats)


def format_conllu(sentence: str, results: list, sent_id: int = 1,
                  dep_model=None) -> str:
    """
    Analiz sonuçlarını CoNLL-U formatına çevirir.
    dep_model verilirse istatistiksel parser kullanılır; yoksa kural tabanlı.
    """
    tokens    = [token  for token, _, _  in results]
    feats_lst = [tag    for _, tag, _    in results]
    upos_lst  = [upos_from_feats_word(tok, tag) for tok, tag, _ in results]

    if dep_model is not None:
        dep_info = _dep_parser.parse_sentence(dep_model, tokens, feats_lst, upos_lst)
    else:
        dep_info = dependency_parse(tokens, feats_lst)

    lines = [
        f"# sent_id = {sent_id}",
        f"# text = {sentence}",
    ]
    for i, (token, tag, _) in enumerate(results, start=1):
        feats  = tag if tag != "NONE" else "_"
        upos   = upos_from_feats_word(token, tag)
        lemma  = lemmatize(token, tag)
        head, deprel = dep_info[i - 1]
        # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
        lines.append(
            f"{i}\t{token}\t{lemma}\t{upos}\t_\t{feats}\t{head}\t{deprel}\t_\t_"
        )
    lines.append("")
    return "\n".join(lines)




_FEATURE_TR = {
    "Case":     {"Nom": "Yalın",  "Gen": "İyelik/Tamlayan", "Dat": "Yönelme",
                 "Acc": "Belirtme","Loc": "Bulunma", "Abl": "Ayrılma",
                 "Ins": "Araç",   "Voc": "Seslenme"},
    "Number":   {"Sing": "Tekil", "Plur": "Çoğul"},
    "Person":   {"1": "1. Kişi",  "2": "2. Kişi", "3": "3. Kişi"},
    "Tense":    {"Past": "Geçmiş","Pres": "Şimdiki","Fut": "Gelecek"},
    "Mood":     {"Ind": "Haber",  "Cnd": "Koşul", "Opt": "Dilek",
                 "Imp": "Emir",   "Nec": "Gereklilik"},
    "VerbForm": {"Part": "Sıfat-fiil","Conv": "Bağ-fiil","Vnoun": "İsim-fiil"},
    "Polarity": {"Neg": "Olumsuz","Pos": "Olumlu"},
    "Evident":  {"Fh": "Dolaylı anlatım (miş)"},
}

def explain_tag(tag: str) -> str:
    """FEATS etiketi → Türkçe açıklama."""
    if tag in ("NONE", "<s1>", "<s2>", "</s>"):
        return tag
    parts = []
    for feat in tag.split("|"):
        k, _, v = feat.partition("=")
        tr_map = _FEATURE_TR.get(k, {})
        tr_v   = tr_map.get(v, v)
        parts.append(f"{k}={tr_v}")
    return "  |  ".join(parts)


# ─── İnteraktif döngü ─────────────────────────────────────────────────────────

def interactive_loop(model, show_explanation: bool = True, decode: str = "greedy"):
    # Windows terminal encoding
    if hasattr(sys.stdin, 'buffer'):
        stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    else:
        stdin = sys.stdin

    print("\n" + "═" * 62)
    print("  Türkçe Morfolojik Trigram Test Arayüzü")
    print("  Model:", getattr(model, '_name', 'trigram'),
          f"  Decoding: {decode.upper()}")
    print("  Komutlar: 'açıkla <etiket>'  •  'çıkış'  •  boş Enter")
    print("═" * 62)

    while True:
        sys.stdout.write("\n  > ")
        sys.stdout.flush()
        try:
            line = stdin.readline()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "":
            break
        line = line.strip()
        if not line or line.lower() in ("çıkış", "exit", "q", "quit"):
            break

        # Özel komut: açıkla
        if line.lower().startswith("açıkla ") or line.lower().startswith("acikla "):
            tag = line.split(" ", 1)[1].strip()
            print(f"\n  {tag}")
            print(f"  → {explain_tag(tag)}")
            continue

        # Cümle analizi
        print()
        results = analyze(line, model, verbose=True, decode=decode)

        # Özet satırı
        print(f"\n  {'─'*60}")
        print(f"  ÖZET: {line}")
        for tok, tag, _ in results:
            case = next((v for f in tag.split("|")
                         for k, _, v in [f.partition("=")]
                         if k == "Case"), "")
            mood = next((v for f in tag.split("|")
                         for k, _, v in [f.partition("=")]
                         if k == "Mood"), "")
            marker = _FEATURE_TR.get("Case", {}).get(case, case) or \
                     _FEATURE_TR.get("Mood", {}).get(mood, mood) or "—"
            print(f"  {tok:<20} {marker}")

    print("\n  Çıkılıyor...\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Türkçe morfolojik trigram reranker testi"
    )
    parser.add_argument(
        "--model", default="reduced",
        choices=["reduced", "full", "boun", "unigram",
                 "ngram2", "ngram3", "ngram4", "ngram5", "ngram5h",
                 "orch", "hybrid"],
        help="Model seçimi"
    )
    parser.add_argument(
        "--sentence", "-s", default=None,
        help="Tek cümle analiz et ve çık"
    )
    parser.add_argument(
        "--decode", "-d", default="greedy",
        choices=["greedy", "viterbi"],
        help="Decoding yöntemi: 'greedy' (varsayılan) veya 'viterbi'"
    )
    parser.add_argument(
        "--format", "-f", default="pretty",
        choices=["pretty", "conllu"],
        help="Çıktı formatı: 'pretty' (varsayılan) veya 'conllu'"
    )
    parser.add_argument(
        "--no-explain", action="store_true",
        help="Türkçe açıklamaları gizle"
    )
    args = parser.parse_args()

    model_name = f"model_{args.model}"
    if args.format == "pretty":
        print(f"\n  Model yükleniyor: {model_name} ...", end="", flush=True)
    try:
        if args.model == "unigram":
            model = UnigramLM.load(model_name)
        elif args.model == "orch":
            model = load_orchestrator()
        else:
            model = load_model(model_name)  # handles HybridLM via _Unpickler
        model._name = model_name
    except FileNotFoundError as e:
        print(f"\n\n  HATA: {e}\n")
        sys.exit(1)
    if args.format == "pretty":
        print(" OK")

    # İstatistiksel dep parser — en iyi model önce denenir
    dep_model = None
    for _dep_name in ("model_dep_pred20", "model_dep_pred", "model_dep"):
        dep_pkl = Path(__file__).parent.parent / "models" / f"{_dep_name}.pkl"
        if dep_pkl.exists():
            try:
                dep_model = _dep_parser.load_parser(_dep_name)
            except Exception:
                dep_model = None
            break

    if args.sentence:
        if args.format == "conllu":
            results = analyze(args.sentence, model, verbose=False, decode=args.decode)
            print(format_conllu(args.sentence, results, dep_model=dep_model))
        else:
            print()
            analyze(args.sentence, model, verbose=True, decode=args.decode)
            print()
    else:
        interactive_loop(model, show_explanation=not args.no_explain,
                         decode=args.decode)


if __name__ == "__main__":
    main()
