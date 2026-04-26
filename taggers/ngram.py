"""
Turkish Cross-Word Morphological Trigram Model
==============================================
Hipotez: Türkçe'de bir sözcüğün morfolojik etiketi (k),
yalnızca kendi iç yapısıyla değil; k-1 ve k-2. sözcüklerin
etiketleriyle de bağımlıdır.

Veri: UD Turkish-BOUN treebank (CoNLL-U)
Model: Unigram / Bigram / Trigram (linear interpolation backoff)
Değerlendirme: Perplexity + Next-tag accuracy (held-out sentences)
"""

import re
import math
import json
import pickle
import random
import urllib.request
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from data.conllu import (
    parse_feats as _parse_feats_impl,
    read_conllu as _read_conllu,
    REDUCED_KEYS,
    ConlluSentence,
)
from morph.rules import (
    _AMV_BOOST_RULES,
    _FUNCTION_WORDS,
    _PRONOUN_FEATS,
    _CCONJ_WORDS,
    _ADP_WORDS,
    _ADV_WORDS,
    _DET_WORDS,
    _PRON_WORDS,
    _PART_WORDS,
    _PUNCT_CHARS,
    _SUFFIX_RULES,
    _LEMMA_SUFFIX_STRIP,
    _CONSONANT_DEVOICE,
    _PRONOUN_LEMMA,
)

# ─── Sabitler ────────────────────────────────────────────────────────────────

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")


# ─── Model kaydet / yükle ─────────────────────────────────────────────────────

def save_model(model: "TrigramLM", name: str) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = path.stat().st_size / 1024
    print(f"  [kayıt] {path}  ({size_kb:.0f} KB)")
    return path


def load_model(name: str) -> "TrigramLM":
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {path}\n"
            f"Önce 'python -m taggers.ngram' çalıştırın."
        )

    import taggers.ngram as _tm

    class _Unpickler(pickle.Unpickler):
        """__main__ / trigram_morph / dep_parser で保存されたモデルを新モジュールへリダイレクト"""
        def find_class(self, module, classname):
            if module in ("__main__", "trigram_morph"):
                module = "taggers.ngram"
            elif module == "dep_parser":
                module = "taggers.dep"
            elif module == "unigram_morph":
                module = "taggers.unigram"
            return super().find_class(module, classname)

    with open(path, "rb") as f:
        return _Unpickler(f).load()


BOUN_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/master/tr_boun-ud-train.conllu",
    "dev":   "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/master/tr_boun-ud-dev.conllu",
    "test":  "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/master/tr_boun-ud-test.conllu",
}
IMST_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-train.conllu",
    "dev":   "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-dev.conllu",
    "test":  "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/tr_imst-ud-test.conllu",
}
KENET_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/master/tr_kenet-ud-train.conllu",
    "dev":   "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/master/tr_kenet-ud-dev.conllu",
    "test":  "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/master/tr_kenet-ud-test.conllu",
}
PENN_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Penn/master/tr_penn-ud-train.conllu",
    "dev":   "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Penn/master/tr_penn-ud-dev.conllu",
    "test":  "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Penn/master/tr_penn-ud-test.conllu",
}
TOURISM_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Tourism/master/tr_tourism-ud-train.conllu",
    "dev":   "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Tourism/master/tr_tourism-ud-dev.conllu",
    "test":  "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Tourism/master/tr_tourism-ud-test.conllu",
}
FRAMENET_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-FrameNet/master/tr_framenet-ud-train.conllu",
    "dev":   "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-FrameNet/master/tr_framenet-ud-dev.conllu",
    "test":  "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-FrameNet/master/tr_framenet-ud-test.conllu",
}

# Tüm treebank listesi: (kısa ad, URL sözlüğü)
ALL_TREEBANKS = [
    ("boun",     BOUN_URLS),
    ("imst",     IMST_URLS),
    ("kenet",    KENET_URLS),
    ("penn",     PENN_URLS),
    ("tourism",  TOURISM_URLS),
    ("framenet", FRAMENET_URLS),
]

# Cümle sınırı belirteçleri
BOS1 = "<s1>"   # i-2 pozisyonu (trigram başlangıcı)
BOS2 = "<s2>"   # i-1 pozisyonu
EOS  = "</s>"

# Bilinmeyen etiket
UNK = "<UNK>"

# Linear interpolation ağırlıkları (EM ile öğrenilebilir; şimdilik sabit)
LAMBDA3, LAMBDA2, LAMBDA1 = 0.6, 0.3, 0.1

# REDUCED_KEYS data.conllu'dan import edildi; backward-compat için burada da var.


# ─── Veri indirme ─────────────────────────────────────────────────────────────

def download_data():
    DATA_DIR.mkdir(exist_ok=True)
    for corpus, urls in ALL_TREEBANKS:
        for split, url in urls.items():
            path = DATA_DIR / f"tr_{corpus}-ud-{split}.conllu"
            if path.exists():
                print(f"  [cache] {path.name}")
                continue
            print(f"  [download] {path.name} ...", end="", flush=True)
            urllib.request.urlretrieve(url, path)
            print(" ✓")


# ─── CoNLL-U ayrıştırıcı ──────────────────────────────────────────────────────

# BOUN Treebank annotation normalisation: train split uses Aspect=Imp for -yor
# forms, but dev/test splits use Aspect=Prog for the same forms.
# Normalise to Aspect=Prog at training time so the model learns the correct label.
_YOR_RE = re.compile(r"[ıiuü]yor", re.IGNORECASE)


def _fix_yor_aspect(word: str, feats: str) -> str:
    """Normalise -yor verb forms: Aspect=Imp → Aspect=Prog (BOUN schema fix)."""
    if "Aspect=Imp" in feats and _YOR_RE.search(word):
        return feats.replace("Aspect=Imp", "Aspect=Prog")
    return feats


def parse_feats(feats_str: str, reduced: bool = False) -> str:
    """CoNLL-U FEATS sütununu kanonik string etikete dönüştürür. data.conllu'ya delegate."""
    return _parse_feats_impl(feats_str, reduced=reduced)


def parse_conllu(path: Path, reduced: bool = False) -> List[List[str]]:
    """
    CoNLL-U dosyasını cümle listesine dönüştürür.
    Her cümle: [BOS1, BOS2, tag1, tag2, ..., EOS]
    """
    result = []
    for sent in _read_conllu(path):
        tags = [BOS1, BOS2]
        for tok in sent.tokens:
            feats = parse_feats(tok.feats_raw, reduced=reduced)
            tags.append(_fix_yor_aspect(tok.form, feats))
        tags.append(EOS)
        result.append(tags)
    return result


def parse_conllu_upos(path: Path) -> List[List[str]]:
    """UPOS sekanslarını döner. Her cümle [BOS1, BOS2, upos1, ..., EOS] formatında."""
    result = []
    for sent in _read_conllu(path):
        tags = [BOS1, BOS2] + [tok.upos for tok in sent.tokens] + [EOS]
        result.append(tags)
    return result


def build_feats_to_upos(conllu_paths: List[Path]) -> Dict[str, str]:
    """
    Her FEATS dizgesi → eğitim verisinde en sık eşleştiği UPOS etiketi.
    BOS/EOS belirteçleri kendileriyle eşleştirilir.
    """
    counts: Dict[str, Counter] = defaultdict(Counter)
    for path in conllu_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#") or line == "":
                    continue
                parts = line.split("\t")
                if len(parts) < 6 or "-" in parts[0] or "." in parts[0]:
                    continue
                upos  = parts[3]
                feats = parse_feats(parts[5], reduced=False)
                counts[feats][upos] += 1
    result: Dict[str, str] = {
        feats: ctr.most_common(1)[0][0]
        for feats, ctr in counts.items()
    }
    # Özel belirteçler
    for tok in (BOS1, BOS2, EOS):
        result[tok] = tok
    return result




class NgramCounts:
    def __init__(self):
        self.unigram: Counter = Counter()
        self.bigram:  Counter = Counter()
        self.trigram: Counter = Counter()
        self.total_tokens = 0

    def fit(self, sentences: List[List[str]]):
        for sent in sentences:
            for i, tag in enumerate(sent):
                if tag in (BOS1, BOS2):
                    continue
                self.unigram[tag] += 1
                self.total_tokens += 1
                if i >= 1:
                    self.bigram[(sent[i-1], tag)] += 1
                if i >= 2:
                    self.trigram[(sent[i-2], sent[i-1], tag)] += 1

    def vocabulary(self) -> set:
        return set(self.unigram.keys())


# ─── Trigram dil modeli (linear interpolation) ────────────────────────────────

class TrigramLM:
    def __init__(self, counts: NgramCounts,
                 l3=LAMBDA3, l2=LAMBDA2, l1=LAMBDA1):
        self.counts = counts
        self.l3, self.l2, self.l1 = l3, l2, l1
        self.vocab = counts.vocabulary()
        self.V = len(self.vocab)

    def p_unigram(self, tag: str) -> float:
        """Add-1 (Laplace) unigram."""
        return (self.counts.unigram.get(tag, 0) + 1) / (self.counts.total_tokens + self.V)

    def p_bigram(self, prev: str, tag: str) -> float:
        """Add-1 bigram."""
        num = self.counts.bigram.get((prev, tag), 0) + 1
        den = self.counts.unigram.get(prev, 0) + self.V
        return num / den

    def p_trigram_ml(self, prev2: str, prev1: str, tag: str) -> float:
        """Maximum-likelihood trigram (sıfır olabilir)."""
        num = self.counts.trigram.get((prev2, prev1, tag), 0)
        den = self.counts.bigram.get((prev2, prev1), 0)
        return num / den if den > 0 else 0.0

    def p_interpolated(self, prev2: str, prev1: str, tag: str) -> float:
        """Linear interpolation: λ3·p3 + λ2·p2 + λ1·p1"""
        p3 = self.p_trigram_ml(prev2, prev1, tag)
        p2 = self.p_bigram(prev1, tag)
        p1 = self.p_unigram(tag)
        return self.l3 * p3 + self.l2 * p2 + self.l1 * p1

    def log_prob_sentence(self, sentence: List[str]) -> float:
        """Bir cümlenin log-olasılığı (BOS/EOS dahil)."""
        log_p = 0.0
        for i in range(2, len(sentence)):
            p = self.p_interpolated(sentence[i-2], sentence[i-1], sentence[i])
            log_p += math.log(p + 1e-300)
        return log_p

    def perplexity(self, sentences: List[List[str]]) -> float:
        """Held-out perplexity (cümle sınır tokenları hariç gerçek tokenlar)."""
        total_log = 0.0
        total_tokens = 0
        for sent in sentences:
            total_log += self.log_prob_sentence(sent)
            # BOS1, BOS2 dahil değil; EOS dahil
            total_tokens += len(sent) - 2
        return math.exp(-total_log / total_tokens)

    def next_tag_accuracy(self, sentences: List[List[str]],
                          order: int = 3) -> float:
        """
        order=1 → unigram (bağımsız), order=2 → bigram, order=3 → trigram
        EOS hariç gerçek tokenlarda doğruluk oranı döner.
        """
        correct = total = 0
        for sent in sentences:
            for i in range(2, len(sent) - 1):  # EOS'u atla
                prev2, prev1, gold = sent[i-2], sent[i-1], sent[i]
                if order == 1:
                    best_tag = max(self.vocab, key=lambda t: self.p_unigram(t))
                elif order == 2:
                    best_tag = max(self.vocab, key=lambda t: self.p_bigram(prev1, t))
                else:
                    best_tag = max(self.vocab, key=lambda t: self.p_interpolated(prev2, prev1, t))
                if best_tag == gold:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0.0


# ─── Aday reranker ────────────────────────────────────────────────────────────

def rerank_candidates(
    model: TrigramLM,
    context: Tuple[str, str],          # (tag_{k-2}, tag_{k-1})
    candidates: List[str],             # olası morfolojik etiketler
    following_tag: Optional[str] = None  # tag_{k+1} (varsa)
) -> List[Tuple[str, float]]:
    """
    Verilen bağlamda aday etiketleri skorla ve sırala.
    following_tag verilirse: P(candidate | context) * P(following | prev1=candidate)
    """
    prev2, prev1 = context
    scored = []
    for cand in candidates:
        score = model.p_interpolated(prev2, prev1, cand)
        if following_tag:
            # Bir sonraki sözcüğün etiketi de bilgiye dahil et
            score *= model.p_bigram(cand, following_tag)
        scored.append((cand, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ─── Viterbi decoder ─────────────────────────────────────────────────────────

def viterbi_decode(model: "TrigramLM", tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Viterbi algoritması ile tüm cümleyi global olarak optimize eder.
    Emisyon ağırlıkları (suffix match güveni) geçiş olasılıklarıyla birleştirilir.

    Durum : (t_{i-1}, t_i) çifti.
    Skor  : log P_interpolated(t_{i-2}, t_{i-1}, t_i) + log emission(t_i | word_i)
    Returns: [(token, best_tag), ...]
    """
    n = len(tokens)
    if n == 0:
        return []

    # Her token için (tag, emission_weight) listesi
    cands_w = [heuristic_candidates_weighted(tok) for tok in tokens]

    # viterbi[(t_prev, t_curr)] = en iyi log-olasılık
    # backptr[i][(t_prev, t_curr)] = bir önceki adımdaki t_pp
    viterbi: Dict[Tuple[str, str], float] = {}
    backptr: List[Dict[Tuple[str, str], str]] = [{}]

    # Başlangıç (i = 0)
    for t0, e0 in cands_w[0]:
        lp = (math.log(model.p_interpolated(BOS1, BOS2, t0) + 1e-300)
              + math.log(e0 + 1e-300))
        key = (BOS2, t0)
        if key not in viterbi or lp > viterbi[key]:
            viterbi[key] = lp
            backptr[0][key] = BOS1

    # Geçişler (i = 1 … n-1)
    for i in range(1, n):
        new_viterbi: Dict[Tuple[str, str], float] = {}
        new_bp:      Dict[Tuple[str, str], str]   = {}

        for t_curr, emit in cands_w[i]:
            e_lp = math.log(emit + 1e-300)
            for (t_pp, t_prev), prev_score in viterbi.items():
                lp = (prev_score
                      + math.log(model.p_interpolated(t_pp, t_prev, t_curr) + 1e-300)
                      + e_lp)
                key = (t_prev, t_curr)
                if key not in new_viterbi or lp > new_viterbi[key]:
                    new_viterbi[key] = lp
                    new_bp[key]      = t_pp

        viterbi = new_viterbi
        backptr.append(new_bp)

    # En iyi son durum
    if not viterbi:
        return [(tok, cands_w[i][0][0]) for i, tok in enumerate(tokens)]

    best_key = max(viterbi, key=lambda k: viterbi[k])

    # Geri iz sürme
    tags_rev: List[str] = []
    key = best_key
    for i in range(n - 1, -1, -1):
        _, t_curr = key
        tags_rev.append(t_curr)
        t_pp   = backptr[i][key]
        t_prev, _ = key
        key    = (t_pp, t_prev)

    return list(zip(tokens, reversed(tags_rev)))




def print_stats(name: str, sentences: List[List[str]], counts: NgramCounts):
    tokens = sum(len(s) - 2 for s in sentences)  # BOS1, BOS2 hariç
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Cümle sayısı   : {len(sentences):,}")
    print(f"  Token sayısı   : {tokens:,}")
    print(f"  Uniq etiket    : {len(counts.vocabulary()):,}")
    print(f"  Bigram tipi    : {len(counts.bigram):,}")
    print(f"  Trigram tipi   : {len(counts.trigram):,}")


# ─── Ana akış ─────────────────────────────────────────────────────────────────

def run(reduced: bool = False, corpora: list = None):
    """
    corpora: eğitilecek corpus kısa adlarının listesi. None → tüm treebank'lar.
    Örnek: run(corpora=["boun"])  → sadece BOUN verisiyle
    """
    if corpora is None:
        corpora = [name for name, _ in ALL_TREEBANKS]
    label = "REDUCED" if reduced else "FULL"
    src   = "+".join(corpora).upper()
    print(f"\n{'#'*55}")
    print(f"  MOD: {label} FEATS  ({src})")
    print(f"{'#'*55}")

    # 1. Belirtilen treebank'lardan train verisi
    train_sents = []
    for corpus in corpora:
        path = DATA_DIR / f"tr_{corpus}-ud-train.conllu"
        if path.exists():
            sents = parse_conllu(path, reduced)
            train_sents += sents
            print(f"  [{corpus:<8}] train: {len(sents):,} cümle")
        else:
            print(f"  [{corpus:<8}] ATLANADI (dosya yok: {path})")

    dev_sents  = parse_conllu(DATA_DIR / "tr_boun-ud-dev.conllu",  reduced)
    test_sents = parse_conllu(DATA_DIR / "tr_boun-ud-test.conllu", reduced)

    # 2. Sayımlar (sadece train)
    counts = NgramCounts()
    counts.fit(train_sents)
    print_stats(f"{label} — Train ({src})", train_sents, counts)
    print(f"  Dev cümle      : {len(dev_sents):,}")
    print(f"  Test cümle     : {len(test_sents):,}")

    # 3. Model
    model = TrigramLM(counts)

    # 4. Perplexity
    print(f"\n  --- Perplexity ({label} / {src}) ---")
    for split_name, split_data in [("Train", train_sents), ("Dev", dev_sents), ("Test", test_sents)]:
        ppl = model.perplexity(split_data)
        print(f"  {split_name:<6} perplexity : {ppl:.2f}")

    # 5. Next-tag accuracy karşılaştırması (Unigram vs Bigram vs Trigram)
    print(f"\n  --- Next-tag Accuracy: Unigram / Bigram / Trigram ({label}) ---")
    sample = dev_sents[:300]
    for order, name in [(1, "Unigram"), (2, "Bigram "), (3, "Trigram")]:
        acc = model.next_tag_accuracy(sample, order=order)
        print(f"  {name} : {acc*100:.2f}%")

    return model, counts


def demo_rerank(model_full: TrigramLM,
                train_sents_full: List[List[str]]):
    """
    Corpus'tan gerçek bağlamlar alınarak cross-word etiket
    bağımlılığı demonstrasyonu yapılır.
    """
    print(f"\n{'='*55}")
    print("  DEMO: Bağlamsal Morfolojik Reranking")
    print(f"{'='*55}")

    counts = model_full.counts

    def find_context(tag_contains: str, prev_contains: str, n: int = 1):
        """Gerçek trigram bağlamı corpus'tan çıkar."""
        results = []
        for sent in train_sents_full:
            for i in range(2, len(sent) - 1):
                if (prev_contains in sent[i-1]
                        and tag_contains in sent[i]
                        and sent[i-2] not in (BOS1, BOS2)):
                    results.append((sent[i-2], sent[i-1], sent[i]))
                    if len(results) >= n:
                        return results
        return results

    # 1. Belirtili tamlama: GEN → POSS
    print("\n  ► Belirtili tamlama: GEN → sonraki sözcük")
    poss_ctx = find_context("Person[psor]", "Case=Gen")
    if poss_ctx:
        prev2, prev1, gold = poss_ctx[0]
        candidates = [
            gold,
            re.sub(r"Number\[psor\]=\w+\|", "", gold).replace(
                "|Person[psor]=3", "").replace("Person[psor]=3|", ""),
            "Case=Acc|Number=Sing|Person=3",
        ]
        candidates = list(dict.fromkeys(c for c in candidates if c))
        ranked = rerank_candidates(model_full, (prev2, prev1), candidates)
        print(f"    Bağlam k-2: {prev2[:45]!r}")
        print(f"    Bağlam k-1: {prev1[:45]!r}  (GEN)")
        for tag, score in ranked:
            marker = "  ◄ GOLD" if tag == gold else ""
            print(f"      {score:.6f}  {tag[:65]}{marker}")

    # 2. Edat hükümeti: DAT → postposition / zarf
    print("\n  ► Edat hükümeti: DAT → NONE (postposition/adverb)")
    dat_ctx = find_context("NONE", "Case=Dat")
    if dat_ctx:
        prev2, prev1, gold = dat_ctx[0]
        candidates = [
            "NONE",
            "Case=Nom|Number=Sing|Person=3",
            "Case=Acc|Number=Sing|Person=3",
            "Case=Gen|Number=Sing|Person=3",
        ]
        ranked = rerank_candidates(model_full, (prev2, prev1), candidates)
        print(f"    Bağlam k-2: {prev2[:45]!r}")
        print(f"    Bağlam k-1: {prev1[:45]!r}  (DAT)")
        for tag, score in ranked:
            marker = "  ◄ GOLD" if tag == gold else ""
            print(f"      {score:.6f}  {tag[:65]}{marker}")

    # 3. Özne (NOM+Pers=1) → yüklem: BOUN'da Mood= finite; VerbForm=Fin yok
    print("\n  ► Özne-yüklem uyumu: NOM+1sg zamir → Fin fiil")
    verb_ctx = find_context("Mood=Ind|Number=Sing|Person=1",
                            "Case=Nom|Number=Sing|Person=1")
    if verb_ctx:
        prev2, prev1, gold = verb_ctx[0]
        alt3sg = re.sub(r"Person=\d", "Person=3", gold)
        alt3pl = re.sub(r"Number=Sing", "Number=Plur", alt3sg)
        candidates = list(dict.fromkeys([gold, alt3sg, alt3pl]))
        ranked = rerank_candidates(model_full, (prev2, prev1), candidates)
        print(f"    Bağlam k-2: {prev2[:45]!r}")
        print(f"    Bağlam k-1: {prev1[:45]!r}  (NOM 1sg)")
        for tag, score in ranked:
            marker = "  ◄ GOLD" if tag == gold else ""
            print(f"      {score:.6f}  {tag[:65]}{marker}")
    else:
        print("    (Pro-drop dil: 'ben' doğrudan bitişik örnek az.)")

    # 4. En güçlü cross-word bigram çiftleri (PMI)
    print(f"\n  ► En güçlü bigram çiftleri (PMI, min 20 örnek)")
    top = []
    for (prev, cur), cnt in counts.bigram.items():
        if prev in (BOS1, BOS2, EOS) or cur in (BOS1, BOS2, EOS):
            continue
        if cnt < 20:
            continue
        p_cur_given_prev = cnt / counts.unigram.get(prev, 1)
        p_cur = counts.unigram.get(cur, 0) / counts.total_tokens
        pmi = math.log(p_cur_given_prev / (p_cur + 1e-10) + 1e-10)
        top.append((pmi, prev, cur, p_cur_given_prev, cnt))
    top.sort(reverse=True)
    print(f"    {'Önceki etiket (k-1)':<40}  {'Sonraki etiket (k)':<40}  PMI    P(k|k-1)  N")
    for pmi, prev, cur, prob, cnt in top[:10]:
        print(f"    {prev[:40]:<40}  {cur[:40]:<40}  {pmi:5.2f}  {prob:.3f}  {cnt}")


# ─── Cross-domain değerlendirme ───────────────────────────────────────────────

def cross_domain_eval(reduced: bool = False):
    """
    Dört senaryo:
      BOUN-only  → BOUN test
      IMST-only  → IMST test
      BOUN train → IMST test   (cross-domain)
      IMST train → BOUN test   (cross-domain)
      Combined   → BOUN+IMST test
    """
    label = "REDUCED" if reduced else "FULL"
    print(f"\n{'#'*55}")
    print(f"  CROSS-DOMAIN DEĞERLENDİRME ({label})")
    print(f"{'#'*55}")

    def load(corpus, split):
        return parse_conllu(DATA_DIR / f"tr_{corpus}-ud-{split}.conllu", reduced)

    boun_tr = load("boun", "train")
    boun_te = load("boun", "test")
    imst_tr = load("imst", "train")
    imst_te = load("imst", "test")
    combined_tr = boun_tr + imst_tr
    combined_te = boun_te + imst_te

    header = f"  {'Eğitim':<18} {'Test':<18} {'PPL':>8}  {'Tok':>7}"
    print(header)
    print("  " + "-" * 55)

    configs = [
        ("BOUN",     boun_tr,     "BOUN",     boun_te),
        ("IMST",     imst_tr,     "IMST",     imst_te),
        ("BOUN",     boun_tr,     "IMST",     imst_te),
        ("IMST",     imst_tr,     "BOUN",     boun_te),
        ("COMBINED", combined_tr, "BOUN+IMST", combined_te),
    ]
    for train_name, train_d, test_name, test_d in configs:
        c = NgramCounts()
        c.fit(train_d)
        m = TrigramLM(c)
        ppl = m.perplexity(test_d)
        tok = sum(len(s) - 2 for s in test_d)
        print(f"  {train_name:<18} {test_name:<18} {ppl:>8.2f}  {tok:>7,}")

    return combined_tr, TrigramLM(NgramCounts())  # combined model opsiyonel


# ─── Uzun mesafe MI analizi ───────────────────────────────────────────────────

def long_range_mi(sentences: List[List[str]],
                  max_dist: int = 6,
                  min_count: int = 10) -> None:
    """
    Mesafe d = 1..max_dist için ortalama PMI hesaplar.
    Ayrıca NOM+Person → VerbForm=Fin çiftlerini özel analiz eder.
    """
    print(f"\n{'#'*55}")
    print("  UZUN MESAFE BAĞIMLILIK ANALİZİ")
    print(f"{'#'*55}")

    unigram: Counter = Counter()
    total = 0

    # Tüm token sayımı
    for sent in sentences:
        for tag in sent:
            if tag in (BOS1, BOS2, EOS):
                continue
            unigram[tag] += 1
            total += 1

    # Mesafeye göre skip-bigram sayımı ve PMI
    print(f"\n  {'Mesafe':>8}  {'Ortalama PMI':>14}  {'Çift sayısı':>12}")
    print("  " + "-" * 40)

    for dist in range(1, max_dist + 1):
        skip_counts: Counter = Counter()
        for sent in sentences:
            real = [t for t in sent if t not in (BOS1, BOS2, EOS)]
            for i in range(len(real) - dist):
                a, b = real[i], real[i + dist]
                skip_counts[(a, b)] += 1

        pmi_vals = []
        for (a, b), cnt in skip_counts.items():
            if cnt < min_count:
                continue
            p_ab = cnt / total
            p_a  = unigram[a] / total
            p_b  = unigram[b] / total
            if p_a > 0 and p_b > 0:
                pmi_vals.append(math.log(p_ab / (p_a * p_b)))

        avg_pmi = sum(pmi_vals) / len(pmi_vals) if pmi_vals else 0.0
        print(f"  {dist:>8}  {avg_pmi:>14.4f}  {len(pmi_vals):>12,}")

    # BOUN/IMST treebank'ta finite verbs: VerbForm=Fin yok;
    # Mood= özelliği finiteness göstergesidir (Ind, Cnd, Pot, Imp, Opt...).
    FIN_MARKER = "Mood="

    # Özne-yüklem uyumu: NOM isim/zamir → Fin fiil (Mood= içeren)
    print(f"\n  ► (A) NOM token → Fin fiil (Mood=) mesafe dağılımı (tüm NOM)")
    print(f"  {'Mesafe':>8}  {'Toplam NOM':>12}  {'NOM→Fin':>9}  {'Oran':>7}")
    print("  " + "-" * 44)

    for dist in range(1, max_dist + 1):
        nom_fin = nom_total = 0
        for sent in sentences:
            real = [t for t in sent if t not in (BOS1, BOS2, EOS)]
            for i in range(len(real) - dist):
                a, b = real[i], real[i + dist]
                if "Case=Nom" not in a:
                    continue
                nom_total += 1
                if FIN_MARKER in b:
                    nom_fin += 1
        rate = f"{nom_fin/nom_total*100:.2f}%" if nom_total else "-"
        print(f"  {dist:>8}  {nom_total:>12,}  {nom_fin:>9,}  {rate:>7}")

    print(f"\n  ► (B) Zamir NOM+Person=X → Fin+Person=X kişi uyumu (pro-drop'suz)")
    print(f"  {'Mesafe':>8}  {'1sg acc%':>10}  {'3sg acc%':>10}  {'toplam':>8}")
    print("  " + "-" * 44)

    for dist in range(1, max_dist + 1):
        hit_1 = miss_1 = hit_3 = miss_3 = 0
        for sent in sentences:
            real = [t for t in sent if t not in (BOS1, BOS2, EOS)]
            for i in range(len(real) - dist):
                a, b = real[i], real[i + dist]
                if "Case=Nom" not in a or FIN_MARKER not in b:
                    continue
                m_pa = re.search(r"Person=(\d)", a)
                m_pb = re.search(r"Person=(\d)", b)
                if not (m_pa and m_pb):
                    continue
                pa, pb = m_pa.group(1), m_pb.group(1)
                match = (pa == pb)
                if pa == "1":
                    hit_1 += match; miss_1 += not match
                elif pa == "3":
                    hit_3 += match; miss_3 += not match
        tot = hit_1 + miss_1 + hit_3 + miss_3
        a1 = f"{hit_1/(hit_1+miss_1)*100:.0f}%" if (hit_1+miss_1) else "-"
        a3 = f"{hit_3/(hit_3+miss_3)*100:.0f}%" if (hit_3+miss_3) else "-"
        print(f"  {dist:>8}  {a1:>10}  {a3:>10}  {tot:>8,}")



# ─── Genişletilmiş N-gram sayacı (4-gram / 5-gram) ────────────────────────────

class NgramCountsEx(NgramCounts):
    """NgramCounts'u 4-gram ve 5-gram desteğiyle genişletir.
    Mevcut TrigramLM ile PKL uyumluluğu korumak için ayrı sınıf.
    """
    # Cümle başındaki dolgu tokenları  (sent[0]=BOS1, sent[1]=BOS2)
    # 4-gram için i>=3; 5-gram için i>=4 — boundary coverage kabul edilir.

    def __init__(self, max_order: int = 5):
        super().__init__()
        self.max_order = max_order
        self.fourgram: Counter = Counter()
        self.fivegram: Counter = Counter()

    def fit(self, sentences: List[List[str]]):
        super().fit(sentences)   # unigram / bigram / trigram
        for sent in sentences:
            for i, tag in enumerate(sent):
                if tag in (BOS1, BOS2):
                    continue
                if self.max_order >= 4 and i >= 3:
                    self.fourgram[
                        (sent[i-3], sent[i-2], sent[i-1], tag)
                    ] += 1
                if self.max_order >= 5 and i >= 4:
                    self.fivegram[
                        (sent[i-4], sent[i-3], sent[i-2], sent[i-1], tag)
                    ] += 1


# ─── N-gram dil modeli (2..5) + hibrit aday üretimi ──────────────────────────

class NgramLM:
    """
    Lineer interpolasyon: P = Σ λ_n · P_n  (n=1..max_order)
    word_table verilirse hibrit aday üretimi aktif olur:
      bilinen sözcük → tablo etiketi 0.95 ağırlıkla;
      OOV → heuristik suffix kuralları.
    """

    # Varsayılan lambda ağırlıkları [λ1, λ2, λ3, λ4, λ5]
    _LAMBDA_DEFAULTS: Dict[int, List[float]] = {
        2: [0.30, 0.70, 0.00, 0.00, 0.00],
        3: [0.10, 0.30, 0.60, 0.00, 0.00],
        4: [0.05, 0.10, 0.25, 0.60, 0.00],
        5: [0.05, 0.05, 0.10, 0.25, 0.55],
    }

    def __init__(self,
                 counts: "NgramCountsEx",
                 max_order: int = 5,
                 lambdas: Optional[List[float]] = None,
                 word_table: Optional[dict] = None):
        self.counts    = counts
        self.max_order = max_order
        self.word_table: dict = word_table or {}
        self.vocab  = counts.vocabulary()
        self.V      = len(self.vocab)
        self.lambdas = (lambdas or
                        self._LAMBDA_DEFAULTS.get(max_order,
                                                   self._LAMBDA_DEFAULTS[3]))

    # ── Olasılık hesapları ───────────────────────────────────────────────────

    def p_unigram(self, tag: str) -> float:
        return (self.counts.unigram.get(tag, 0) + 1) / (
            self.counts.total_tokens + self.V)

    def p_bigram(self, prev1: str, tag: str) -> float:
        num = self.counts.bigram.get((prev1, tag), 0) + 1
        den = self.counts.unigram.get(prev1, 0) + self.V
        return num / den

    def p_trigram_ml(self, prev2: str, prev1: str, tag: str) -> float:
        num = self.counts.trigram.get((prev2, prev1, tag), 0)
        den = self.counts.bigram.get((prev2, prev1), 0)
        return num / den if den > 0 else 0.0

    def p_fourgram_ml(self, prev3: str, prev2: str,
                       prev1: str, tag: str) -> float:
        fg = getattr(self.counts, 'fourgram', None)
        if fg is None:
            return 0.0
        num = fg.get((prev3, prev2, prev1, tag), 0)
        den = self.counts.trigram.get((prev3, prev2, prev1), 0)
        return num / den if den > 0 else 0.0

    def p_fivegram_ml(self, prev4: str, prev3: str, prev2: str,
                       prev1: str, tag: str) -> float:
        fg5 = getattr(self.counts, 'fivegram', None)
        fg4 = getattr(self.counts, 'fourgram', None)
        if fg5 is None or fg4 is None:
            return 0.0
        num = fg5.get((prev4, prev3, prev2, prev1, tag), 0)
        den = fg4.get((prev4, prev3, prev2, prev1), 0)
        return num / den if den > 0 else 0.0

    def score(self, context: List[str], tag: str) -> float:
        """Tam interpolasyonlu skor. context = [oldest, ..., prev1]."""
        l1, l2, l3, l4, l5 = (self.lambdas + [0.0] * 5)[:5]
        prev1 = context[-1] if len(context) >= 1 else BOS2
        prev2 = context[-2] if len(context) >= 2 else BOS1
        prev3 = context[-3] if len(context) >= 3 else BOS1
        prev4 = context[-4] if len(context) >= 4 else BOS1

        p = (l1 * self.p_unigram(tag)
             + l2 * self.p_bigram(prev1, tag)
             + l3 * self.p_trigram_ml(prev2, prev1, tag)
             + l4 * self.p_fourgram_ml(prev3, prev2, prev1, tag)
             + l5 * self.p_fivegram_ml(prev4, prev3, prev2, prev1, tag))
        return p

    # ── Hibrit aday üretici ──────────────────────────────────────────────────

    def get_candidates(self, word: str) -> List[Tuple[str, float]]:
        """Bilinen sözcük → tablo; OOV → heuristik."""
        w = word.lower()
        if w in self.word_table:
            best = self.word_table[w]
            cands: List[Tuple[str, float]] = [(best, 0.95)]
            for tag, wt in heuristic_candidates_weighted(word):
                if tag != best:
                    cands.append((tag, wt * 0.3))
            return cands
        return heuristic_candidates_weighted(word)

    # ── Greedy decode (tam N-gram bağlamı) ──────────────────────────────────

    def decode_greedy(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Tam geçmiş bağlamıyla soldan-sağa greedy decode."""
        context: List[str] = [BOS1, BOS2]
        results = []
        for tok in tokens:
            cands = self.get_candidates(tok)
            best_tag = max(
                cands, key=lambda x: x[1] * self.score(context, x[0])
            )[0]
            context.append(best_tag)
            results.append((tok, best_tag))
        return results

    # ── Viterbi decode (bigram durumu; 4/5-gram için yaklaşık) ──────────────

    def decode_viterbi(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Bigram-durumlu Viterbi; hibrit adaylar + N-gram skor."""
        n = len(tokens)
        if n == 0:
            return []

        cands_w = [self.get_candidates(tok) for tok in tokens]
        viterbi: Dict[Tuple[str, str], float] = {}
        backptr: List[Dict] = [{}]

        for t0, e0 in cands_w[0]:
            lp = (math.log(self.score([BOS1, BOS2], t0) + 1e-300)
                  + math.log(e0 + 1e-300))
            key = (BOS2, t0)
            if key not in viterbi or lp > viterbi[key]:
                viterbi[key] = lp
                backptr[0][key] = BOS1

        for i in range(1, n):
            new_vit: Dict[Tuple[str, str], float] = {}
            new_bp:  Dict[Tuple[str, str], str]   = {}
            for t_curr, emit in cands_w[i]:
                e_lp = math.log(emit + 1e-300)
                for (t_pp, t_prev), prev_score in viterbi.items():
                    lp = (prev_score
                          + math.log(
                              self.score([t_pp, t_prev], t_curr) + 1e-300)
                          + e_lp)
                    key = (t_prev, t_curr)
                    if key not in new_vit or lp > new_vit[key]:
                        new_vit[key] = lp
                        new_bp[key]  = t_pp
            viterbi = new_vit
            backptr.append(new_bp)

        if not viterbi:
            return [(tok, cands_w[i][0][0]) for i, tok in enumerate(tokens)]

        best_key = max(viterbi, key=lambda k: viterbi[k])
        tags_rev: List[str] = []
        key = best_key
        for i in range(n - 1, -1, -1):
            _, t_curr = key
            tags_rev.append(t_curr)
            t_pp      = backptr[i][key]
            t_prev, _ = key
            key       = (t_pp, t_prev)
        return list(zip(tokens, reversed(tags_rev)))


# ─── Orkestratör: tüm N-gram modelleri kapsayan ensemble ─────────────────────

class OrchestratorLM:
    """
    Birden fazla NgramLM / UnigramLM modelini orkestre eder.

    Her token için:
      1. Tüm modellerden aday tag'lerin birleşimi alınır.
      2. Her modelin ham skorları [0, 1] aralığına normalize edilir.
      3. Normalize skorlar toplanır (oylama).
      4. En yüksek toplam skoru olan tag seçilir.

    Hem uzun mesafe bağlamı (yüksek-mertebe N-gram) hem kelime tablosu
    (unigram / hibrit) bilgisi, normalleştirme sonrası eşit hakla katılır;
    birden fazla modelin anlaştığı seçenek güçlenir.
    """

    def __init__(self, models: list):
        self.models    = models
        self.max_order = max((getattr(m, 'max_order', 1) for m in models),
                             default=1)
        # NgramLM arayüzü ile uyumluluk için boş word_table
        self.word_table: dict = {}

    # ── Yardımcı: tek modelden (tag → raw_score) sözlüğü ────────────────────

    def _raw_scores(self, model, word: str,
                    context: List[str]) -> Dict[str, float]:
        """Bir modelden {tag: raw_score} döndürür."""
        # Duck-type: UnigramLM has word_table but not counts/score()
        has_score = callable(getattr(model, 'score', None))
        is_ngram  = isinstance(model, NgramLM)

        if not has_score and not is_ngram:
            # UnigramLM benzeri: sadece word_table + heuristik
            wt = getattr(model, 'word_table', {})
            w  = word.lower()
            if w in wt:
                return {wt[w]: 1.0}
            cands = heuristic_candidates_weighted(word)
            max_w = max((e for _, e in cands), default=1.0)
            return {t: e / max_w for t, e in cands} if max_w > 0 else \
                   {t: e for t, e in cands}

        if is_ngram:
            cands = model.get_candidates(word)
            return {tag: emit * model.score(context, tag)
                    for tag, emit in cands}

        return {}

    # ── Greedy decode ────────────────────────────────────────────────────────

    def decode_greedy(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Paylaşımlı bağlamla orkestrasyon.
        Kazanan tag → bir sonraki pozisyonun bağlamına girer.
        """
        context: List[str] = [BOS1, BOS2]
        results: List[Tuple[str, str]] = []

        for tok in tokens:
            # Her modelden ham skorlar
            model_raw: List[Dict[str, float]] = [
                self._raw_scores(m, tok, context) for m in self.models
            ]

            # Normalize et [0, 1]
            norm_scores: List[Dict[str, float]] = []
            for raw in model_raw:
                if not raw:
                    norm_scores.append({})
                    continue
                mx = max(raw.values())
                norm_scores.append(
                    {t: s / mx for t, s in raw.items()} if mx > 0 else raw
                )

            # Tüm tag'lerin birleşimini topla
            all_tags: set = set()
            for ns in norm_scores:
                all_tags.update(ns.keys())

            tag_agg: Dict[str, float] = {
                tag: sum(ns.get(tag, 0.0) for ns in norm_scores)
                for tag in all_tags
            }

            best_tag = max(tag_agg, key=tag_agg.__getitem__) \
                if tag_agg else "NONE"
            context.append(best_tag)
            results.append((tok, best_tag))

        return results

    # ── Viterbi decode (bigram durumu, tüm modellerden ortak skor) ──────────

    def decode_viterbi(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Bigram-durumlu Viterbi; geçiş skoru = tüm modellerin
        normalleştirilmiş skorlarının toplamı.
        """
        n = len(tokens)
        if n == 0:
            return []

        # Adayları ve emisyonları en zengin model (ngram5h veya unigram) ile al
        def cands_for(tok: str) -> List[Tuple[str, float]]:
            all_c: Dict[str, float] = {}
            for m in self.models:
                if isinstance(m, NgramLM):
                    for t, w in m.get_candidates(tok):
                        if w > all_c.get(t, -1):
                            all_c[t] = w
            if not all_c:
                for t, w in heuristic_candidates_weighted(tok):
                    all_c[t] = w
            return list(all_c.items())

        def orch_score(context: List[str], tag: str) -> float:
            # Sadece n-gram geçiş olasılığı; emisyon cands_w'den ayrı gelir
            total = 0.0
            count = 0
            for m in self.models:
                if isinstance(m, NgramLM):
                    total += m.score(context, tag)
                    count += 1
            return total / max(count, 1)

        cands_w = [cands_for(tok) for tok in tokens]
        viterbi: Dict[Tuple[str, str], float] = {}
        backptr: List[Dict] = [{}]

        for t0, e0 in cands_w[0]:
            lp = (math.log(orch_score([BOS1, BOS2], t0) + 1e-300)
                  + math.log(e0 + 1e-300))
            key = (BOS2, t0)
            if key not in viterbi or lp > viterbi[key]:
                viterbi[key] = lp
                backptr[0][key] = BOS1

        for i in range(1, n):
            new_vit: Dict[Tuple[str, str], float] = {}
            new_bp:  Dict[Tuple[str, str], str]   = {}
            for t_curr, emit in cands_w[i]:
                e_lp = math.log(emit + 1e-300)
                for (t_pp, t_prev), prev_score in viterbi.items():
                    lp = (prev_score
                          + math.log(orch_score([t_pp, t_prev], t_curr) + 1e-300)
                          + e_lp)
                    key = (t_prev, t_curr)
                    if key not in new_vit or lp > new_vit[key]:
                        new_vit[key] = lp
                        new_bp[key]  = t_pp
            viterbi = new_vit
            backptr.append(new_bp)

        if not viterbi:
            return [(tok, cands_w[i][0][0]) for i, tok in enumerate(tokens)]

        best_key = max(viterbi, key=lambda k: viterbi[k])
        tags_rev: List[str] = []
        key = best_key
        for i in range(n - 1, -1, -1):
            _, t_curr = key
            tags_rev.append(t_curr)
            t_pp      = backptr[i][key]
            t_prev, _ = key
            key       = (t_pp, t_prev)
        return list(zip(tokens, reversed(tags_rev)))

# ─── CharNgramEmission: karakter suffix n-gram tabanlı OOV emisyon modeli ─────

class CharNgramEmission:
    """
    P(tag | word) ≈ Σ_n  n² · P(tag | word[-n:])   (n = min_n..max_n)

    Eğitim verisinden öğrenir; elle yazılmış suffix kurallarının yerini alır.
    OOV Türkçe sözcükler için data-driven aday üretici.

    Tasarım kararları:
    · Type-based sayım (token değil) → sık kelimeler istatistiği ezmez
    · NONE alfabetik OOV'larda dışlanır
    · Gerçek olasılık dağılımı döner (toplam ≈ 1)
    · min_count=3: güvenilmez suffix istatistiklerini filtreler
    """

    _NON_ALPHA_RE = re.compile(
        r"^[^\wğüşıöçĞÜŞİÖÇ]+$|^\d[\d.,]*$", re.UNICODE
    )

    def __init__(self, min_n: int = 2, max_n: int = 6, min_count: int = 3):
        self.min_n     = min_n
        self.max_n     = max_n
        self.min_count = min_count
        # suffix → {tag: type_count}
        self._counts: Dict[str, Counter] = defaultdict(Counter)

    # ── Eğitim ──────────────────────────────────────────────────────────────

    def fit(self, paths: List[Path], weight: float = 1.0) -> "CharNgramEmission":
        """
        Eğitim dosyalarındaki (word, tag) çiftlerinden suffix sayımları öğrenir.
        Type-based: her (word, tag) çifti yalnızca bir kez sayılır.
        weight: bu dosya grubunun sayımlara katkı ağırlığı (örn. BOUN için 3.0).
        """
        seen: set = set()
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.startswith("#") or line == "":
                        continue
                    parts = line.split("\t")
                    if len(parts) < 6 or "-" in parts[0] or "." in parts[0]:
                        continue
                    tag  = parse_feats(parts[5], reduced=False)
                    if not tag or tag == "NONE":
                        continue
                    word = parts[1].lower()
                    tag  = _fix_yor_aspect(word, tag)
                    key  = (word, tag, id(path.parent))  # per-path dedup group
                    if key in seen:
                        continue
                    seen.add(key)
                    for n in range(self.min_n, self.max_n + 1):
                        if len(word) >= n:
                            self._counts[word[-n:]][tag] += weight
        return self

    # ── Aday üretici ────────────────────────────────────────────────────────

    _NOUN_FALLBACK = "Case=Nom|Number=Sing|Person=3"

    def candidates(self, word: str, top_k: int = 12) -> List[Tuple[str, float]]:
        """
        (tag, prob) listesi döner; toplam ≈ 1.0 (normalize edilmiş dağılım).
        Alfabetik OOV'larda NONE dahil edilmez.
        """
        if self._NON_ALPHA_RE.match(word):
            return [("NONE", 1.0)]

        w      = word.lower()
        scores: Dict[str, float] = {}

        for n in range(self.min_n, self.max_n + 1):
            if len(w) < n:
                continue
            suffix = w[-n:]
            cnt    = self._counts.get(suffix)
            if cnt is None:
                continue
            total = sum(cnt.values())
            if total < self.min_count:
                continue
            wt = float(n * n)           # n² → uzun suffix daha güvenilir
            for tag, c in cnt.most_common(20):
                scores[tag] = scores.get(tag, 0.0) + wt * (c / total)

        if not scores:
            # Suffix eşleşmesi yoksa isim-Nom fallback (Türkçe yalın isim default)
            return [(self._NOUN_FALLBACK, 1.0)]

        # Aspect/Mood/Voice suffix boost: her özellik için en ilk eşleşen kural uygulanır
        boosted_feats: set = set()
        for pattern, feat_kv, boost in _AMV_BOOST_RULES:
            feat_key = feat_kv.split("=")[0]
            if feat_key in boosted_feats:
                continue
            if pattern.search(w):
                for tag in list(scores.keys()):
                    if feat_kv in set(tag.split("|")):
                        scores[tag] *= boost
                boosted_feats.add(feat_key)

        # Olasılık dağılımına normalize et (sum → 1)
        total_score = sum(scores.values())
        result = [
            (tag, s / total_score)
            for tag, s in sorted(scores.items(), key=lambda x: -x[1])
        ]
        return result[:top_k]


# ─── WordTagEmission: kelime → etiket olasılık tablosu ───────────────────────

class WordTagEmission:
    """
    P(FEATS | word) = (1 - β) · P_mle(FEATS | word)  +  β · P_backoff(FEATS | word)

    β(word) = n0 / (n0 + count(word))   [Witten-Bell benzeri karıştırma]
        · Çok gözlemlenen sözcük → β küçük → MLE baskın
        · Nadir sözcük            → β büyük → backoff baskın
        · OOV                     → saf backoff

    Backoff öncelik sırası:
      1. CharNgramEmission (varsa) — veriден öğrenilmiş suffix dağılımı
      2. heuristic_candidates_weighted() — elle yazılmış kural fallback
         (fonksiyon kelimeleri + zamir exact-match KoRunur)
    """

    def __init__(self, n0: float = 3.0):
        self._n0: float = n0
        self._counts: Dict[str, Counter] = defaultdict(Counter)   # word_lower → Counter(tag→count)
        self.char_ngram: Optional["CharNgramEmission"] = None     # run_hybrid() tarafından set edilir

    # ── Eğitim ──────────────────────────────────────────────────────────────

    def fit(self, conllu_paths: List[Path]) -> "WordTagEmission":
        for path in conllu_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.startswith("#") or line == "":
                        continue
                    parts = line.split("\t")
                    if len(parts) < 6 or "-" in parts[0] or "." in parts[0]:
                        continue
                    feats = parse_feats(parts[5], reduced=False)
                    feats = _fix_yor_aspect(parts[1], feats)
                    self._counts[parts[1].lower()][feats] += 1
        return self

    # ── Olasılık ve aday üretici ─────────────────────────────────────────────

    # regex for tokens that are purely punctuation/digits (no Turkish letters)
    _NON_ALPHA_RE = re.compile(r"^[^\wğüşıöçĞÜŞİÖÇ]+$|^\d[\d.,]*$", re.UNICODE)

    def _backoff_candidates(self, word: str) -> List[Tuple[str, float]]:
        """
        OOV / nadir kelime için backoff aday listesi döndürür.

        Fonksiyon kelimeleri ve zamir exact-match kuralları (heuristic içinde)
        her zaman öncelik alır. Bunların dışında CharNgramEmission varsa onu,
        yoksa suffix kurallarını kullanır.
        """
        w = word.lower()
        # Fonksiyon kelimesi / zamir → heuristic exact-match zorunlu
        if w in _FUNCTION_WORDS or w in _PRONOUN_FEATS:
            return heuristic_candidates_weighted(word)

        # Alfabetik OOV → CharNgramEmission tercih edilir
        char_ng = getattr(self, "char_ngram", None)
        if char_ng is not None:
            return char_ng.candidates(word)

        # Fallback: elle yazılmış suffix kuralları
        return heuristic_candidates_weighted(word)

    def candidates(self, word: str) -> List[Tuple[str, float]]:
        """
        (tag, prob) listesi döndürür.
        Bilinen sözcük → MLE + backoff karışımı (NONE dahil).
        OOV noktalama/rakam → [("NONE", 1.0)].
        OOV diğer → saf backoff.
        """
        w = word.lower()
        if w not in self._counts:
            if self._NON_ALPHA_RE.match(word):
                return [("NONE", 1.0)]
            return self._backoff_candidates(word)

        dist  = self._counts[w]
        count = sum(dist.values())
        beta  = self._n0 / (self._n0 + count)   # backoff ağırlığı

        # Backoff bileşeni; olasılık dağılımına normalize et
        heur_raw = self._backoff_candidates(word)
        h_total  = sum(wt for _, wt in heur_raw) or 1.0
        heur     = {t: wt / h_total for t, wt in heur_raw}

        # Birleştir: MLE etiketleri (NONE dahil) + backoff etiketleri
        all_tags: set = set(dist.keys()) | set(heur.keys())
        result: List[Tuple[str, float]] = []
        for tag in all_tags:
            p_mle  = dist.get(tag, 0) / count
            p_h    = heur.get(tag, 0.0)
            p      = (1 - beta) * p_mle + beta * p_h
            if p > 0:
                result.append((tag, p))
        return result


# ─── HybridLM: emisyon × N-gram geçiş modeli ─────────────────────────────────

class HybridLM:
    """
    score(word, tag, ctx) = P_emit(word,tag) · P_ngram(tag|ctx)^w_trans
                                              · P_upos(upos(tag)|upos(ctx))^w_upos

    Gerekçe:
      · P_emit  : kelime-özel dağılım (güvenilir; bilinen sözcüklerde çok yoğun)
      · P_ngram : FEATS komşuluğundan bağlam sinyali (1667 sınıf; seyrek)
      · P_upos  : UPOS komşuluğundan kaba bağlam sinyali (17 sınıf; yoğun)
      w_trans / w_upos < 1 → sinyali ölçekler, gürültüyü azaltır
    """

    def __init__(self,
                 emission:      WordTagEmission,
                 ngram:         "NgramLM",
                 w_trans:       float = 0.2,
                 upos_ngram:    Optional["NgramLM"] = None,
                 w_upos:        float = 0.0,
                 feats_to_upos: Optional[Dict[str, str]] = None,
                 w_trans_oov:   Optional[float] = None):
        self.emission      = emission
        self.ngram         = ngram
        self.w_trans       = w_trans
        # OOV token'lar için ayrı geçiş ağırlığı (None → w_trans ile aynı)
        self.w_trans_oov:  float = w_trans_oov if w_trans_oov is not None else w_trans
        self.upos_ngram    = upos_ngram
        self.w_upos        = w_upos
        self.feats_to_upos: Dict[str, str] = feats_to_upos or {}
        self.max_order     = ngram.max_order
        self.word_table: dict = {}   # NgramLM uyumluluk

    # ── Skor ────────────────────────────────────────────────────────────────

    def _ngram_trans_score(self, context: List[str], tag: str) -> float:
        """Altın n-gram ile (varsa) pseudo n-gram olasılık karışımı."""
        gold = self.ngram.score(context, tag)
        pseudo: Optional["NgramLM"] = getattr(self, "pseudo_ngram", None)
        w_pseudo: float = getattr(self, "w_pseudo", 0.0)
        if pseudo is None or w_pseudo <= 0.0:
            return gold
        return (1.0 - w_pseudo) * gold + w_pseudo * pseudo.score(context, tag)

    def _score(self, tag: str, emit_prob: float, context: List[str],
               w_eff: Optional[float] = None) -> float:
        """w_eff: etkin w_trans (None → self.w_trans). OOV için farklı ağırlık iletilir."""
        wt  = self.w_trans if w_eff is None else w_eff
        ng  = self._ngram_trans_score(context, tag)
        if ng <= 0:
            return emit_prob
        score = emit_prob * (ng ** wt)
        if self.w_upos > 0 and self.upos_ngram is not None:
            upos     = self.feats_to_upos.get(tag, "_")
            upos_ctx = [self.feats_to_upos.get(f, "_") for f in context]
            ung = self.upos_ngram.score(upos_ctx, upos)
            if ung > 0:
                score *= ung ** self.w_upos
        return score

    # ── Greedy decode (tam N-gram bağlamı) ──────────────────────────────────

    def decode_greedy(self, tokens: List[str]) -> List[Tuple[str, str]]:
        context: List[str] = [BOS1, BOS2]
        results: List[Tuple[str, str]] = []
        for tok in tokens:
            is_oov  = tok.lower() not in self.emission._counts
            w_eff   = self.w_trans_oov if is_oov else self.w_trans
            cands   = self.emission.candidates(tok)
            best_tag = max(cands,
                           key=lambda x: self._score(x[0], x[1], context, w_eff))[0]
            context.append(best_tag)
            results.append((tok, best_tag))
        return results

    # ── Viterbi decode (ayarlanabilir durum uzayı) ───────────────────────────

    #: Viterbi'de kaç önceki etiketi durum olarak tut (varsayılan 3 → 4-gram skor)
    viterbi_state_size: int = 3

    def decode_viterbi(self, tokens: List[str],
                       state_size: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Genel yüksek mertebe Viterbi.

        state_size: durum tuple boyutu (önceki etiket sayısı).
          2 → bigram durum / trigram skor  (eski varsayılan)
          3 → trigram durum / 4-gram skor  (yeni varsayılan)
          4 → 4-gram durum / 5-gram skor   (tam kapasite)

        score(adım) = log P_emit(word, tag)
                    + w_trans * log P_ngram(tag | prev_state_tags...)
        """
        if state_size is None:
            state_size = self.viterbi_state_size

        n = len(tokens)
        if n == 0:
            return []

        cands_w  = [self.emission.candidates(tok) for tok in tokens]
        # OOV listesi: her token için bağlam ağırlığı ayrı
        wt_list  = [
            self.w_trans_oov if tok.lower() not in self.emission._counts else self.w_trans
            for tok in tokens
        ]

        # Başlangıç BOS bağlamı: son eleman BOS2, öncesi BOS1 ile dolduruluyor
        # state_size uzunluğunda tuple oluşturmak için state_size-1 BOS gerekiyor
        bos_prefix: tuple = tuple([BOS1] * (state_size - 2) + [BOS2])  # uzunluk state_size-1

        # viterbi: {state_tuple (uzunluk state_size): log_score}
        # backptr[i]: {new_state: prev_state}  (adım i'deki geçiş)
        viterbi: Dict[tuple, float] = {}
        backptr: List[Dict[tuple, Optional[tuple]]] = [{}]

        # ── İlk token ───────────────────────────────────────────────────────
        _use_upos = self.w_upos > 0 and self.upos_ngram is not None
        wt0 = wt_list[0]
        for t0, e0 in cands_w[0]:
            if not e0:
                continue
            ng_s  = self._ngram_trans_score(list(bos_prefix), t0)
            lp    = math.log(e0) + wt0 * math.log(max(ng_s, 1e-300))
            if _use_upos:
                upos0    = self.feats_to_upos.get(t0, "_")
                upos_ctx = [self.feats_to_upos.get(f, "_") for f in bos_prefix]
                ung = self.upos_ngram.score(upos_ctx, upos0)  # type: ignore[union-attr]
                if ung > 0:
                    lp += self.w_upos * math.log(max(ung, 1e-300))
            state = bos_prefix + (t0,)          # uzunluk state_size
            if state not in viterbi or lp > viterbi[state]:
                viterbi[state] = lp
                backptr[0][state] = None         # ilk token için önceki durum yok

        # ── Kalan tokenlar ───────────────────────────────────────────────────
        for i in range(1, n):
            new_vit: Dict[tuple, float] = {}
            new_bp:  Dict[tuple, Optional[tuple]] = {}
            wti = wt_list[i]
            for t_curr, e_curr in cands_w[i]:
                if not e_curr:
                    continue
                e_lp = math.log(e_curr)
                for prev_state, prev_score in viterbi.items():
                    # Önceki durum zaten state_size uzunluğunda; tamamı bağlam
                    ng_s  = self._ngram_trans_score(list(prev_state), t_curr)
                    lp    = prev_score + e_lp + wti * math.log(max(ng_s, 1e-300))
                    if _use_upos:
                        upos_c   = self.feats_to_upos.get(t_curr, "_")
                        upos_ctx = [self.feats_to_upos.get(f, "_") for f in prev_state]
                        ung = self.upos_ngram.score(upos_ctx, upos_c)  # type: ignore[union-attr]
                        if ung > 0:
                            lp += self.w_upos * math.log(max(ung, 1e-300))
                    # Yeni durum: pencere kaydır
                    new_state = prev_state[1:] + (t_curr,)
                    if new_state not in new_vit or lp > new_vit[new_state]:
                        new_vit[new_state] = lp
                        new_bp[new_state]  = prev_state
            if not new_vit:     # tüm yollar tükendiyse greedy'ye düş
                remaining = tokens[i:]
                greedy_rest = self.decode_greedy(remaining)
                partial = [(tokens[j], backptr[j].keys().__iter__().__next__()[-1]
                            if backptr[j] else tokens[j]) for j in range(i)]
                # traceback ne kadar toplandıysa onu kullan, kalanı greedy ekle
                break
            viterbi = new_vit
            backptr.append(new_bp)
        else:
            pass  # döngü normal bitti

        if not viterbi:
            # Tüm tokenlar için greedy yedek
            return self.decode_greedy(tokens)

        # ── Geri izleme ──────────────────────────────────────────────────────
        best_state = max(viterbi, key=lambda k: viterbi[k])
        tags_rev: List[str] = []
        state: Optional[tuple] = best_state
        for i in range(n - 1, -1, -1):
            tags_rev.append(state[-1])          # type: ignore[index]
            prev = backptr[i].get(state)        # type: ignore[arg-type]
            if prev is None:
                break
            state = prev

        return list(zip(tokens, reversed(tags_rev)))



def heuristic_candidates_weighted(word: str) -> List[Tuple[str, float]]:
    """
    Her sözcük için (etiket, emisyon_ağırlığı) çiftleri döndürür.
    Ağırlık: 1.0 = kesin fonksiyon sözcüğü / zamir
             0.9 = uzun ek eşleşmesi (≥4)
             0.7 = orta ek (3)
             0.5 = kısa ek (2)
             0.3 = çok kısa ek (1)
             0.1 = varsayılan Nom fallback
    """
    w = word.lower()

    # 1. Fonksiyon sözcüğü → yalnızca NONE
    if w in _FUNCTION_WORDS:
        return [("NONE", 1.0)]

    # 2. Zamir yalın → birincil aday + suffix alternatifleri
    if w in _PRONOUN_FEATS:
        pron_tag = _PRONOUN_FEATS[w]
        result: List[Tuple[str, float]] = [(pron_tag, 0.9)]
        # Kısa suffix eşleşmeleri ek aday olarak ekle
        for pattern, tags in _SUFFIX_RULES:
            alts = pattern.split("|")
            ml = max((len(a) for a in alts if w.endswith(a)), default=0)
            if ml > 0:
                for t in tags:
                    if t != pron_tag and not any(t == x for x, _ in result):
                        result.append((t, 0.3))
                break  # kısa suffix için tek kural yeter
        return result[:5]

    # 3. Suffix kuralları
    matches: List[Tuple[int, List[str]]] = []
    for pattern, tags in _SUFFIX_RULES:
        alts = pattern.split("|")
        ml = max((len(a) for a in alts if w.endswith(a)), default=0)
        if ml > 0:
            matches.append((ml, tags))

    if not matches:
        return [("Case=Nom|Number=Sing|Person=3", 0.1)]

    max_len = max(ml for ml, _ in matches)

    if max_len >= 4:
        weight = 0.9
    elif max_len == 3:
        weight = 0.7
    elif max_len == 2:
        weight = 0.5
    else:
        weight = 0.3

    # Sadece en uzun eşleşen kuraldaki adayları al
    cands: List[Tuple[str, float]] = []
    for ml, tags in matches:
        if ml == max_len:
            for t in tags:
                if not any(t == x for x, _ in cands):
                    cands.append((t, weight))

    # Kısa eşleşmelerde (≤2) alternatif Nom/POSS de ekle
    if max_len <= 2:
        nom  = "Case=Nom|Number=Sing|Person=3"
        poss = "Case=Nom|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3"
        for fallback in (nom, poss):
            if not any(fallback == x for x, _ in cands):
                cands.append((fallback, 0.1))

    return cands[:5]


def heuristic_candidates(word: str) -> List[str]:
    """
    Sözcük son ekine göre olası FEATS etiket listesi döndürür.
    (geriye dönük uyumluluk için - sadece etiket listesi)
    """
    return [tag for tag, _ in heuristic_candidates_weighted(word)]



def lemmatize(word: str, feats_tag: str) -> str:
    """
    Kural tabanlı Türkçe lemmatizasyon.
    feats_tag: parse_feats() çıktısı (örn. 'Case=Loc|Number=Sing|Person=3')
    """
    w = word.lower()

    # Fonksiyon sözcükleri → kendisi
    if w in _FUNCTION_WORDS:
        return w

    # Zamir çekim biçimleri → kök zamir
    if w in _PRONOUN_LEMMA:
        return _PRONOUN_LEMMA[w]
    if w in _PRONOUN_FEATS:
        return w

    # NONE etiketi → sözcük değişmez
    if feats_tag == "NONE":
        return w

    # Suffix stripping (en uzun eşleşen ek)
    stem = w
    for suffix, strip_len in _LEMMA_SUFFIX_STRIP:
        if w.endswith(suffix) and len(w) - strip_len >= 2:
            stem = w[:-strip_len]
            break

    # Sessiz ünsüz sertleşmesi (Türkçe sözcük sonu)
    is_verbal = any(k in feats_tag for k in
                    ("Mood=", "Tense=", "Aspect=", "VerbForm=", "Evident="))
    if stem and stem[-1] in _CONSONANT_DEVOICE:
        stem = stem[:-1] + _CONSONANT_DEVOICE[stem[-1]]
    elif stem and stem[-1] == "d" and is_verbal:
        stem = stem[:-1] + "t"

    # Fiil köklerinde ünlü düşmesi onarımı: başlıyor→başl→başla, söylüyor→söyl→söyle
    if is_verbal and len(stem) >= 2:
        vowels = "aeıioöuü"
        if stem[-1] not in vowels and stem[-2] not in vowels:
            last_vowel = next((c for c in reversed(stem) if c in vowels), "")
            if last_vowel in "aıou":
                stem += "a"
            elif last_vowel in "eiöü":
                stem += "e"

    return stem if stem else w


# ─── UPOS tahmini ─────────────────────────────────────────────────────────────

def upos_from_feats_word(word: str, feats: str) -> str:
    """
    Sözcük biçimi + FEATS etiketinden UD UPOS kategorisi tahmin eder.
    """
    w = word.lower()
    # Noktalama
    if w in _PUNCT_CHARS or (len(w) == 1 and not w.isalnum()):
        return "PUNCT"
    # Sözcük tabanlı kategori
    if w in _CCONJ_WORDS:
        return "CCONJ"
    if w in _ADP_WORDS:
        return "ADP"
    if w in _PART_WORDS:
        return "PART"
    if w in _ADV_WORDS:
        return "ADV"
    if w in _DET_WORDS:
        return "DET"
    if w in _PRON_WORDS or w in _PRONOUN_FEATS or w in _PRONOUN_LEMMA:
        return "PRON"

    if feats == "NONE":
        return "X"

    f = {k: v for part in feats.split("|")
         for k, _, v in [part.partition("=")]}

    if "Mood" in f:
        return "VERB"
    if "Aspect" in f and "Tense" in f:
        return "VERB"
    if "Evident" in f and "Tense" in f:
        return "VERB"
    vf = f.get("VerbForm", "")
    if vf == "Conv":
        return "VERB"
    if vf == "Part":
        return "ADJ"   # participial → sıfat işlevi
    if vf == "Vnoun":
        return "NOUN"
    if "Case" in f:
        return "NOUN"
    if "NumType" in f:
        return "NUM"
    return "X"


# ─── Basit kural tabanlı bağımlılık parseri ───────────────────────────────────

def dependency_parse(tokens: List[str],
                     feats_list: List[str]) -> List[Tuple[int, str]]:
    """
    Basit kural tabanlı Türkçe bağımlılık parseri.

    tokens    : sözcük listesi
    feats_list: parse_feats() çıktıları (FEATS string'leri)
    Döner     : [(head_idx_1indexed, deprel), ...]
                head_idx = 0  → root
    """
    n = len(tokens)
    if n == 0:
        return []

    upos = [upos_from_feats_word(tok, feat)
            for tok, feat in zip(tokens, feats_list)]

    heads   = [0] * n
    deprels = ["dep"] * n

    # Kök bul: son VERB, yoksa son NOUN, yoksa son token
    root_idx = n - 1
    for i in range(n - 1, -1, -1):
        if upos[i] == "VERB":
            root_idx = i
            break
    else:
        for i in range(n - 1, -1, -1):
            if upos[i] == "NOUN":
                root_idx = i
                break

    heads[root_idx]   = 0
    deprels[root_idx] = "root"

    for i in range(n):
        if i == root_idx:
            continue
        up   = upos[i]
        feat = feats_list[i]
        w    = tokens[i].lower()

        if up == "PUNCT":
            heads[i]   = root_idx + 1
            deprels[i] = "punct"

        elif up == "CCONJ":
            # cc: bir sonraki içerik sözcüğüne bağlan
            nxt = next((j for j in range(i + 1, n)
                        if upos[j] not in ("CCONJ", "PUNCT")), root_idx)
            heads[i]   = nxt + 1
            deprels[i] = "cc"

        elif up == "ADP":
            # case: önceki isme bağlan
            prv = next((j for j in range(i - 1, -1, -1)
                        if upos[j] in ("NOUN", "PRON", "DET", "NUM")),
                       root_idx)
            heads[i]   = prv + 1
            deprels[i] = "case"

        elif up == "DET":
            nxt_n = next((j for j in range(i + 1, min(i + 4, n))
                          if upos[j] in ("NOUN", "ADJ", "NUM")), root_idx)
            heads[i]   = nxt_n + 1
            deprels[i] = "det"

        elif up == "PRON":
            if "Case=Nom" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "nsubj"
            elif "Case=Acc" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "obj"
            elif "Case=Dat" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "iobj"
            else:
                heads[i]   = root_idx + 1
                deprels[i] = "obl"

        elif up in ("ADV", "PART", "X"):
            heads[i]   = root_idx + 1
            deprels[i] = "advmod"

        elif up == "ADJ":
            # acl:relcl — participial → sonraki isme bağlan
            nxt_n = next((j for j in range(i + 1, min(i + 5, n))
                          if upos[j] in ("NOUN", "PRON")), None)
            if nxt_n is not None:
                heads[i]   = nxt_n + 1
                deprels[i] = "acl:relcl"
            else:
                heads[i]   = root_idx + 1
                deprels[i] = "acl"

        elif up == "NUM":
            nxt_n = next((j for j in range(i + 1, min(i + 3, n))
                          if upos[j] in ("NOUN",)), root_idx)
            heads[i]   = nxt_n + 1
            deprels[i] = "nummod"

        elif up == "NOUN":
            if feat == "NONE":
                heads[i]   = root_idx + 1
                deprels[i] = "dep"
            elif "Case=Gen" in feat:
                poss_j = next(
                    (j for j in range(i + 1, min(i + 5, n))
                     if "psor" in feats_list[j]
                     or "Person[psor]" in feats_list[j]),
                    None)
                if poss_j is not None:
                    heads[i]   = poss_j + 1
                    deprels[i] = "nmod:poss"
                else:
                    heads[i]   = root_idx + 1
                    deprels[i] = "nmod"
            elif "Case=Nom" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "nsubj" if i < root_idx else "dep"
            elif "Case=Acc" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "obj"
            elif "Case=Dat" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "obl"
            elif "Case=Abl" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "obl"
            elif "Case=Loc" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "obl"
            elif "Case=Ins" in feat:
                heads[i]   = root_idx + 1
                deprels[i] = "obl"
            else:
                heads[i]   = root_idx + 1
                deprels[i] = "dep"

        else:
            heads[i]   = root_idx + 1
            deprels[i] = "dep"

    return list(zip(heads, deprels))



# ─── N-gram model eğitimi ─────────────────────────────────────────────────────

def run_ngram(order: int = 5,
              corpora: Optional[List[str]] = None,
              with_word_table: bool = True) -> "NgramLM":
    """
    Belirtilen N-gram mertebesinde NgramLM eğitir ve kaydeder.

    order          : 2, 3, 4 veya 5
    corpora        : eğitim corpus'ları (None → sadece BOUN)
    with_word_table: True → hibrit aday üretimi (kelime tablosu + heuristik)
    """
    if corpora is None:
        corpora = ["boun"]

    src = "+".join(corpora).upper()
    hybrid_label = "+HYBRID" if with_word_table else ""
    print(f"\n{'#'*55}")
    print(f"  N-gram eğitimi: {order}-gram  ({src}{hybrid_label})")
    print(f"{'#'*55}")

    train_sents = []
    train_path  = None
    for corpus in corpora:
        path = DATA_DIR / f"tr_{corpus}-ud-train.conllu"
        if path.exists():
            sents = parse_conllu(path, reduced=False)
            train_sents += sents
            train_path = path
            print(f"  [{corpus:<8}] train: {len(sents):,} cümle")
        else:
            print(f"  [{corpus:<8}] ATLAIDI (dosya yok: {path})")

    print(f"  N-gram mertebesi : {order}")
    print(f"  Sayımlar hesaplanıyor...", end="", flush=True)
    counts = NgramCountsEx(max_order=order)
    counts.fit(train_sents)
    print(f" ✓  ({len(counts.unigram):,} etiket, "
          f"{len(counts.trigram):,} trigram"
          + (f", {len(counts.fourgram):,} 4-gram" if order >= 4 else "")
          + (f", {len(counts.fivegram):,} 5-gram" if order >= 5 else "")
          + ")")

    word_table: dict = {}
    if with_word_table and train_path:
        from collections import defaultdict as _dd
        freq: dict = _dd(Counter)
        for corpus in corpora:
            path = DATA_DIR / f"tr_{corpus}-ud-train.conllu"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.rstrip("\n")
                    if _line.startswith("#") or _line == "":
                        continue
                    _parts = _line.split("\t")
                    if len(_parts) < 6 or "-" in _parts[0] or "." in _parts[0]:
                        continue
                    _feats = parse_feats(_parts[5], reduced=False)
                    if _feats != "NONE":
                        freq[_parts[1].lower()][_feats] += 1
        word_table = {w: cnt.most_common(1)[0][0] for w, cnt in freq.items()}
        print(f"  Kelime tablosu   : {len(word_table):,} sözcük")

    model = NgramLM(counts, max_order=order,
                    word_table=word_table if with_word_table else {})

    hybrid_suffix = "h" if with_word_table else ""
    name = f"model_ngram{order}{hybrid_suffix}"
    save_model(model, name)
    return model



# ─── HybridLM eğitimi ve w_trans ayarı ──────────────────────────────────────

def run_hybrid(emission_corpora: Optional[List[str]] = None,
               ngram_corpora: Optional[List[str]] = None,
               w_trans: float = 0.2,
               order: int = 5,
               model_name: str = "model_hybrid") -> "HybridLM":
    """
    HybridLM eğitir ve model_hybrid.pkl olarak kaydeder.

      · WordTagEmission  → emission_corpora'dan MLE + Witten-Bell backoff
                           (varsayılan: BOUN — kelime dağılımı domain-specific)
      · NgramLM          → ngram_corpora'dan N-gram geçiş sayımları
                           (varsayılan: tüm 6 treebank — daha az seyreklik)
      · HybridLM         → ikisini w_trans ağırlığıyla birleştirir
    """
    if emission_corpora is None:
        emission_corpora = ["boun"]
    if ngram_corpora is None:
        ngram_corpora = ["boun", "imst", "kenet", "penn", "tourism", "framenet"]

    emit_label  = "+".join(c.upper() for c in emission_corpora)
    ngram_label = "+".join(c.upper() for c in ngram_corpora)
    print(f"\n{'#'*60}")
    print(f"  HybridLM eğitimi")
    print(f"    Emisyon : {emit_label}")
    print(f"    N-gram  : {ngram_label}  (order={order})")
    print(f"    w_trans : {w_trans}")
    print(f"{'#'*60}")

    # ── WordTagEmission (BOUN-only, domain-specific kelime dağılımı) ─────────
    emit_paths: List[Path] = []
    for corpus in emission_corpora:
        path = DATA_DIR / f"tr_{corpus}-ud-train.conllu"
        if path.exists():
            emit_paths.append(path)
            sents_n = sum(1 for l in open(path, encoding="utf-8") if l.strip() == "")
            print(f"  [emit  {corpus:<8}] ~{sents_n:,} cümle")
        else:
            print(f"  [emit  {corpus:<8}] ATLAIDI: {path}")

    print(f"  WordTagEmission eğitiliyor...", end="", flush=True)
    emission = WordTagEmission(n0=3.0)
    emission.fit(emit_paths)
    print(f" {len(emission._counts):,} sözcük kaydedildi")

    # ── CharNgramEmission (OOV suffix backoff; sadece emission corpora) ──────
    # NOT: Diğer treebanklardaki farklı annotation şemaları (özellikle Evident=Nfh
    # için) CharNgramEmission suffix istatistiklerini bozuyor.
    # Örnek: imst/kenet/penn -mış formları için Aspect=Perf|Tense=Pres kullanırken
    # BOUN Evident=Nfh kullanıyor; karıştırma Evident sinyalini %84 → %64'e düşürüyor.
    print(f"  CharNgramEmission eğitiliyor...", end="", flush=True)
    char_ng = CharNgramEmission(min_n=2, max_n=6, min_count=3)
    # Sadece hedef treebank (emit_paths): 3× ağırlık → nadir suffix'ler min_count=3 eşiğini geçer
    char_ng.fit(emit_paths, weight=3.0)
    n_suffixes = sum(len(cnt) for cnt in char_ng._counts.values())
    print(f" {len(char_ng._counts):,} suffix, {n_suffixes:,} (suffix,tag) çifti")
    emission.char_ngram = char_ng

    # ── NgramLM (çoklu treebank, daha az seyreklik) ─────────────────────────
    ngram_sents: List[List[str]] = []
    for corpus in ngram_corpora:
        path = DATA_DIR / f"tr_{corpus}-ud-train.conllu"
        if path.exists():
            sents = parse_conllu(path, reduced=False)
            ngram_sents += sents
            print(f"  [ngram {corpus:<8}] {len(sents):,} cümle")
        else:
            print(f"  [ngram {corpus:<8}] ATLAIDI: {path}")

    print(f"  N-gram sayımları ({len(ngram_sents):,} cümle)...", end="", flush=True)
    counts = NgramCountsEx(max_order=order)
    counts.fit(ngram_sents)
    print(f" {len(counts.unigram):,} etiket, {len(counts.trigram):,} trigram"
          + (f", {len(counts.fourgram):,} 4-gram" if order >= 4 else "")
          + (f", {len(counts.fivegram):,} 5-gram" if order >= 5 else ""))

    ngram = NgramLM(counts, max_order=order, word_table={})

    # ── UposNgramLM (UPOS geçiş sinyali, 17 sınıf) ──────────────────────────
    print(f"  UposNgramLM eğitiliyor...", end="", flush=True)
    upos_sents: List[List[str]] = []
    for path in emit_paths:          # BOUN-only (domain-specific UPOS dağılımı)
        upos_sents += parse_conllu_upos(path)
    upos_counts = NgramCountsEx(max_order=order)
    upos_counts.fit(upos_sents)
    upos_ngram = NgramLM(upos_counts, max_order=order, word_table={})
    print(f" {len(upos_counts.unigram)} UPOS sınıfı, {len(upos_counts.trigram):,} UPOS-trigram")

    print(f"  FEATS→UPOS tablosu oluşturuluyor...", end="", flush=True)
    feats_to_upos = build_feats_to_upos(emit_paths)
    print(f" {len(feats_to_upos):,} eşleme")

    model = HybridLM(emission, ngram, w_trans=w_trans,
                     upos_ngram=upos_ngram, w_upos=0.0,
                     feats_to_upos=feats_to_upos)
    save_model(model, model_name)
    return model


# ─── Multiprocessing worker'ları (modül seviyesinde; Windows spawn için gerekli) ─

_TUNE_MODEL: Optional["HybridLM"] = None
_TUNE_SENTS: Optional[list]       = None


def _init_tune_worker(model_name: str, sents_data: list) -> None:
    """Her worker process başladığında modeli ve veriyi bir kez yükler."""
    global _TUNE_MODEL, _TUNE_SENTS
    _TUNE_MODEL = load_model(model_name)
    _TUNE_SENTS = sents_data


def _eval_tune_combo(args: Tuple[float, int, float]) -> Tuple[float, int, float, float]:
    """
    Tek bir (n0, state_size, w_trans) kombinasyonunu değerlendirir.
    Her worker kendi model kopyasında sırayla çalışır — race condition yok.
    """
    n0, ss, w = args
    m = _TUNE_MODEL
    m.emission._n0       = n0
    m.viterbi_state_size = ss
    m.w_trans            = w
    correct = total = 0
    for sent in _TUNE_SENTS:
        tokens = [t["form"] for t in sent]
        gold   = [t["feats"] for t in sent]
        pairs  = m.decode_viterbi(tokens, state_size=ss)
        pred   = [tag for _, tag in pairs]
        for g, p in zip(gold, pred):
            total   += 1
            correct += (g == p)
    acc = 100.0 * correct / total if total else 0.0
    return (n0, ss, w, acc)


def tune_w_trans(model_name: str = "model_hybrid",
                 n_sents: int = 300,
                 w_values: Optional[List[float]] = None,
                 state_sizes: Optional[List[int]] = None,
                 n0_values: Optional[List[float]] = None,
                 n_workers: Optional[int] = None) -> Tuple[float, int, float]:
    """
    w_trans × viterbi_state_size × emission_n0 grid araması yapar.
    BOUN dev setinin ilk n_sents cümlesini kullanır.
    En yüksek FEATS exact (tüm tokenlar) değerini veren (w_trans, state_size, n0) döner.

    n_workers: paralel worker sayısı (None → cpu_count()).
    n0: Witten-Bell backoff parametresi — emission'ın MLE ↔ heuristik karışım oranını ayarlar.
    """
    import multiprocessing as _mp

    if w_values is None:
        w_values    = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    if state_sizes is None:
        state_sizes = [2, 3, 4]
    if n0_values is None:
        n0_values   = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    if n_workers is None:
        n_workers = _mp.cpu_count()

    dev_path = DATA_DIR / "tr_boun-ud-dev.conllu"
    if not dev_path.exists():
        print("  Ayar için dev dosyası bulunamadı.")
        return 0.2, 3, 3.0

    # Dev altküme yükle
    sents: list = []
    current: List[dict] = []
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if current:
                    sents.append(current)
                    current = []
                    if len(sents) >= n_sents:
                        break
                continue
            parts = line.split("\t")
            if len(parts) < 6 or "-" in parts[0] or "." in parts[0]:
                continue
            feats = parse_feats(parts[5], reduced=False)
            current.append({"form": parts[1], "feats": feats})

    combos = [
        (n0, ss, w)
        for n0 in n0_values
        for ss  in state_sizes
        for w   in w_values
    ]
    total_combos = len(combos)
    print(f"\n  n0 x state_size x w_trans ayarı ({len(sents)} cümle, "
          f"{total_combos} kombinasyon, {n_workers} worker):")
    print(f"  {'n0':>6}  {'state':>6}  {'w_trans':>8}  {'FEATS%':>8}")
    print(f"  {'─'*38}")

    # Multiprocessing: her worker modeli bir kez yükler, combo'ları sırayla işler
    ctx = _mp.get_context("spawn")      # Windows'ta fork yok; spawn güvenli
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_tune_worker,
        initargs=(model_name, sents),
    ) as pool:
        results = pool.map(_eval_tune_combo, combos)

    best_w, best_ss, best_n0, best_acc = 0.2, 3, 3.0, 0.0
    for n0, ss, w, acc in results:
        marker = " <-- en iyi" if acc > best_acc else ""
        print(f"  {n0:>6.1f}  {ss:>6}  {w:>8.3f}  {acc:>7.2f}%{marker}")
        if acc > best_acc:
            best_acc, best_w, best_ss, best_n0 = acc, w, ss, n0

    print(f"\n  En iyi: n0={best_n0}, state_size={best_ss}, w_trans={best_w}  ({best_acc:.2f}%)\n")
    return best_w, best_ss, best_n0


def tune_w_upos(model_name: str = "model_hybrid",
                n_sents: int = 979,
                w_upos_values: Optional[List[float]] = None) -> float:
    """
    w_upos 1D grid araması — diğer parametreler (w_trans, state_size, n0) sabit.
    Modeldeki mevcut w_trans/state_size/n0 değerleri kullanılır.
    En yüksek FEATS exact veren w_upos değerini döner.
    """
    if w_upos_values is None:
        w_upos_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    dev_path = DATA_DIR / "tr_boun-ud-dev.conllu"
    if not dev_path.exists():
        return 0.0

    sents = []
    current: List[dict] = []
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if current:
                    sents.append(current)
                    current = []
                    if len(sents) >= n_sents:
                        break
                continue
            parts = line.split("\t")
            if len(parts) < 6 or "-" in parts[0] or "." in parts[0]:
                continue
            feats = parse_feats(parts[5], reduced=False)
            current.append({"form": parts[1], "feats": feats})

    base_model: HybridLM = load_model(model_name)
    ss = base_model.viterbi_state_size

    if base_model.upos_ngram is None:
        print("  [uyarı] Model UPOS n-gram içermiyor, w_upos ayarı atlanıyor.")
        return 0.0

    print(f"\n  w_upos ayarı ({len(sents)} cümle, state_size={ss},"
          f" w_trans={base_model.w_trans}, n0={base_model.emission._n0}):")
    print(f"  {'w_upos':>8}  {'FEATS%':>8}")
    print(f"  {'─'*20}")

    best_wu, best_acc = 0.0, 0.0
    for wu in w_upos_values:
        base_model.w_upos = wu
        correct = total = 0
        for sent in sents:
            tokens = [t["form"] for t in sent]
            gold   = [t["feats"] for t in sent]
            pairs  = base_model.decode_viterbi(tokens, state_size=ss)
            pred   = [tag for _, tag in pairs]
            for g, p in zip(gold, pred):
                total   += 1
                correct += (g == p)
        acc = 100 * correct / total if total else 0
        marker = " <-- en iyi" if acc > best_acc else ""
        print(f"  {wu:>8.3f}  {acc:>7.2f}%{marker}")
        if acc > best_acc:
            best_acc, best_wu = acc, wu

    print(f"\n  En iyi: w_upos={best_wu}  ({best_acc:.2f}%)\n")
    return best_wu


def tune_w_trans_oov(model_name: str = "model_hybrid",
                     n_sents: int = 979,
                     w_oov_values: Optional[List[float]] = None) -> float:
    """
    w_trans_oov 1D grid araması — OOV token'lar için ayrı geçiş ağırlığı.
    Diğer parametreler (w_trans, state_size, n0, w_upos) modelden alınır.
    """
    if w_oov_values is None:
        w_oov_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    dev_path = DATA_DIR / "tr_boun-ud-dev.conllu"
    if not dev_path.exists():
        return 0.05

    sents = []
    current: List[dict] = []
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if current:
                    sents.append(current)
                    current = []
                    if len(sents) >= n_sents:
                        break
                continue
            parts = line.split("\t")
            if len(parts) < 6 or "-" in parts[0] or "." in parts[0]:
                continue
            feats = parse_feats(parts[5], reduced=False)
            current.append({"form": parts[1], "feats": feats})

    base_model: HybridLM = load_model(model_name)
    ss = base_model.viterbi_state_size

    print(f"\n  w_trans_oov ayarı ({len(sents)} cümle, state_size={ss},"
          f" w_trans={base_model.w_trans}, n0={base_model.emission._n0}):")
    print(f"  {'w_oov':>8}  {'FEATS%':>8}")
    print(f"  {'─'*20}")

    best_wo, best_acc = base_model.w_trans, 0.0
    for wo in w_oov_values:
        base_model.w_trans_oov = wo
        correct = total = 0
        for sent in sents:
            tokens = [t["form"] for t in sent]
            gold   = [t["feats"] for t in sent]
            pairs  = base_model.decode_viterbi(tokens, state_size=ss)
            pred   = [tag for _, tag in pairs]
            for g, p in zip(gold, pred):
                total   += 1
                correct += (g == p)
        acc = 100 * correct / total if total else 0
        marker = " <-- en iyi" if acc > best_acc else ""
        print(f"  {wo:>8.3f}  {acc:>7.2f}%{marker}")
        if acc > best_acc:
            best_acc, best_wo = acc, wo

    print(f"\n  En iyi: w_trans_oov={best_wo}  ({best_acc:.2f}%)\n")
    return best_wo


# ─── Wikipedia self-distillation ─────────────────────────────────────────────

def build_pseudo_ngram(max_sents: int = 100_000,
                       oov_threshold: float = 0.30,
                       order: int = 5,
                       model_name: str = "model_hybrid") -> "NgramLM":
    """
    Türkçe Wikipedia metni üzerinde HybridLM pseudo-etiketlemesi yaparak
    ayrı bir NgramLM oluşturur.

    Filtreler:
      · 4–60 token uzunluk aralığı
      · OOV oranı < oov_threshold (yüksek yabancı-sözcük içerikli cümleler atlanır)
    Pseudo sekanslar [BOS1, BOS2, tag1, ..., EOS] formatında oluşturulur.
    """
    import re as _re

    model: HybridLM = load_model(model_name)
    known_vocab: set = set(model.emission._counts.keys())

    print(f"\n  Wikipedia (tr) yükleniyor...", flush=True)
    from datasets import load_dataset  # type: ignore[import]
    wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train")

    sent_re = _re.compile(r'(?<=[.!?])\s+')
    pseudo_seqs: List[List[str]] = []
    skipped = 0
    processed = 0

    for article in wiki:
        text: str = article["text"]
        for raw in sent_re.split(text):
            tokens = raw.strip().split()
            if len(tokens) < 4 or len(tokens) > 60:
                continue
            oov_n = sum(1 for t in tokens if t.lower() not in known_vocab)
            if oov_n / len(tokens) > oov_threshold:
                skipped += 1
                continue
            pairs = model.decode_viterbi(tokens)
            tags  = [tag for _, tag in pairs]
            pseudo_seqs.append([BOS1, BOS2] + tags + [EOS])
            processed += 1
            if processed % 10_000 == 0:
                print(f"  {processed:,}/{max_sents:,} cümle işlendi "
                      f"({skipped:,} atlandı, yüksek OOV)", flush=True)
            if processed >= max_sents:
                break
        if processed >= max_sents:
            break

    print(f"  Toplam: {processed:,} cümle, {skipped:,} atlandı")
    counts = NgramCountsEx(max_order=order)
    counts.fit(pseudo_seqs)
    print(f"  Pseudo n-gram: {len(counts.unigram):,} etiket, "
          f"{len(counts.trigram):,} trigram, {len(counts.fivegram):,} 5-gram")
    return NgramLM(counts, max_order=order, word_table={})


def tune_w_pseudo(model_name: str = "model_hybrid",
                  n_sents: int = 979,
                  w_values: Optional[List[float]] = None) -> float:
    """
    w_pseudo 1D grid araması — pseudo n-gram karışım ağırlığı.
    Diğer parametreler (w_trans, state_size, n0, w_trans_oov) modelden alınır.
    """
    if w_values is None:
        w_values = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    dev_path = DATA_DIR / "tr_boun-ud-dev.conllu"
    if not dev_path.exists():
        return 0.0

    sents: List[List[dict]] = []
    current: List[dict] = []
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if current:
                    sents.append(current)
                    current = []
                    if len(sents) >= n_sents:
                        break
                continue
            parts = line.split("\t")
            if len(parts) < 6 or "-" in parts[0] or "." in parts[0]:
                continue
            feats = parse_feats(parts[5], reduced=False)
            current.append({"form": parts[1], "feats": feats})
    if current:
        sents.append(current)

    base_model: HybridLM = load_model(model_name)
    ss = base_model.viterbi_state_size

    print(f"\n  w_pseudo ayarı ({len(sents)} cümle, state_size={ss}):")
    print(f"  {'w_pseudo':>10}  {'FEATS%':>8}")
    print(f"  {'─'*22}")

    best_wp, best_acc = 0.0, 0.0
    for wp in w_values:
        base_model.w_pseudo = wp
        correct = total = 0
        for sent in sents:
            tokens = [t["form"] for t in sent]
            gold   = [t["feats"] for t in sent]
            pairs  = base_model.decode_viterbi(tokens, state_size=ss)
            pred   = [tag for _, tag in pairs]
            for g, p in zip(gold, pred):
                total   += 1
                correct += (g == p)
        acc = 100 * correct / total if total else 0
        marker = " <-- en iyi" if acc > best_acc else ""
        print(f"  {wp:>10.3f}  {acc:>7.2f}%{marker}")
        if acc > best_acc:
            best_acc, best_wp = acc, wp

    print(f"\n  En iyi: w_pseudo={best_wp}  ({best_acc:.2f}%)\n")
    base_model.w_pseudo = best_wp
    return best_wp


def interactive_test(model: TrigramLM) -> None:
    """
    Kullanıcı Türkçe cümleler girer.
    Her token için heuristik adaylar üretilir, trigram model rerank eder.
    'çıkış' veya boş satır programı sonlandırır.
    """
    import io, sys
    # Windows'ta stdin encoding sorununu çöz
    if hasattr(sys.stdin, 'buffer'):
        stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    else:
        stdin = sys.stdin

    print(f"\n{'#'*55}")
    print("  İNTERAKTİF UNSEEN CÜMLE TESTİ")
    print("  (Türkçe cümle girin; 'çıkış' veya Enter ile bitirin)")
    print(f"{'#'*55}\n")

    while True:
        try:
            sys.stdout.write("  > ")
            sys.stdout.flush()
            sentence = stdin.readline()
            if sentence == "":  # EOF
                break
            sentence = sentence.strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not sentence or sentence.lower() in ("çıkış", "exit", "quit", "q"):
            break

        tokens = sentence.split()
        context = [BOS1, BOS2]
        print()
        for tok in tokens:
            cands = heuristic_candidates(tok)
            prev2, prev1 = context[-2], context[-1]
            ranked = rerank_candidates(model, (prev2, prev1), cands)
            best_tag, best_score = ranked[0]
            context.append(best_tag)

            print(f"  {tok:<20} → {best_tag}")
            for tag, score in ranked[1:]:
                delta = score / (best_score + 1e-15)
                print(f"  {'':20}   [{delta:.3f}×] {tag}")
        print()

    print("  İnteraktif test tamamlandı.\n")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Türkçe trigram morfoloji modeli eğitici")
    _parser.add_argument("--corpus", default="all",
                         choices=["all", "boun"],
                         help="Eğitim verisi: 'all' (tüm treebank'lar) veya 'boun' (sadece BOUN)")
    _parser.add_argument("--ngram-corpus", default=None,
                         help="NgramLM için ayrı korpus: 'boun', 'all' veya virgülle ayrılmış liste. "
                              "Verilmezse --corpus değeri kullanılır.")
    _parser.add_argument("--order", type=int, default=None,
                         choices=[2, 3, 4, 5],
                         help="N-gram mertebesi (2-5). Verilirse NgramLM eğitilir.")
    _parser.add_argument("--no-hybrid", action="store_true",
                         help="Kelime tablosu olmadan eğit (varsayılan: hibrit açık)")
    _parser.add_argument("--hybrid", action="store_true",
                         help="HybridLM eğit + w_trans otomatik ayarla ve kaydet")
    _parser.add_argument("--w-trans", type=float, default=0.2,
                         help="HybridLM için w_trans başlangıç değeri (varsayılan 0.2)")
    _parser.add_argument("--self-distil", action="store_true",
                         help="Wikipedia pseudo-etiketleme ile self-distillation (model_hybrid.pkl güncellenir)")
    _parser.add_argument("--max-pseudo-sents", type=int, default=100_000,
                         help="Wikipedia'dan kullanılacak maksimum cümle sayısı (varsayılan 100 000)")
    _parser.add_argument("--model-name", type=str, default=None,
                         help="Kaydedilecek model adı (varsayılan: --hybrid → model_hybrid, "
                              "--self-distil → model_hybrid_distil)")
    _args = _parser.parse_args()

    print("UD Turkish treebank'lar indiriliyor (6 kaynak)...")
    download_data()

    if _args.hybrid:
        # ── HybridLM eğitim + n0 × w_trans × state_size grid ayarı ──────────
        _out_model = _args.model_name or "model_hybrid"
        _all_corpora = ["boun", "imst", "kenet", "penn", "tourism", "framenet"]
        emit_corpora = _all_corpora if _args.corpus == "all" else ["boun"]
        # --ngram-corpus: ayrıca belirtilebilir; default olarak --corpus ile aynı davranışı korur
        if _args.ngram_corpus is None:
            ngram_corpora = ["boun"]   # emission ile aynı (önceki davranış)
        elif _args.ngram_corpus == "all":
            ngram_corpora = _all_corpora
        else:
            ngram_corpora = [c.strip() for c in _args.ngram_corpus.split(",")]
        model = run_hybrid(emission_corpora=emit_corpora, ngram_corpora=ngram_corpora,
                           w_trans=_args.w_trans, order=5, model_name=_out_model)
        best_w, best_ss, best_n0 = tune_w_trans(_out_model, n_sents=979)
        model.w_trans = best_w
        model.viterbi_state_size = best_ss
        model.emission._n0 = best_n0
        save_model(model, _out_model)
        best_wu = tune_w_upos(_out_model, n_sents=979)
        model.w_upos = best_wu
        save_model(model, _out_model)
        best_wo = tune_w_trans_oov(_out_model, n_sents=979)
        model.w_trans_oov = best_wo
        save_model(model, _out_model)
        print(f"\n{'='*55}")
        print(f"  Tamamlandı: {_out_model}.pkl")
        print(f"    n0={best_n0}, w_trans={best_w}, w_trans_oov={best_wo}")
        print(f"    state_size={best_ss}, w_upos={best_wu}")
        print(f"{'='*55}\n")

    elif _args.self_distil:
        # ── Wikipedia self-distillation ──────────────────────────────────────
        _out_model = _args.model_name or "model_hybrid_distil"
        import shutil as _shutil, os as _os
        _base = _os.path.join("models", "model_hybrid.pkl")
        _backup = _os.path.join("models", "model_hybrid_base.pkl")
        if not _os.path.exists(_backup):
            _shutil.copy2(_base, _backup)
            print(f"  Yedek: model_hybrid_base.pkl oluşturuldu")
        print(f"\n{'#'*60}")
        print(f"  Self-distillation (Wikipedia pseudo-etiketleme)")
        print(f"  kaynak=model_hybrid  →  çıktı={_out_model}")
        print(f"  max_sents={_args.max_pseudo_sents:,}")
        print(f"{'#'*60}")
        pseudo_ngram = build_pseudo_ngram(
            max_sents=_args.max_pseudo_sents,
            oov_threshold=0.30,
            order=5,
            model_name="model_hybrid",
        )
        sd_model: HybridLM = load_model("model_hybrid")
        sd_model.pseudo_ngram = pseudo_ngram
        sd_model.w_pseudo = 0.0
        save_model(sd_model, _out_model)
        best_wp = tune_w_pseudo(_out_model, n_sents=979)
        sd_model.w_pseudo = best_wp
        save_model(sd_model, _out_model)
        print(f"\n{'='*55}")
        print(f"  Self-distillation tamamlandı: {_out_model}.pkl")
        print(f"    w_pseudo={best_wp}")
        print(f"  Karşılaştırma için:")
        print(f"    python eval.py --model {_out_model.replace('model_', '')} --decode viterbi")
        print(f"    python eval.py --model hybrid --decode viterbi")
        print(f"{'='*55}\n")

    elif _args.order is not None:
        # ── N-gram modeller (NgramLM) ────────────────────────────────────────
        corpora = [_args.corpus] if _args.corpus != "all" else ["boun"]
        with_table = not _args.no_hybrid
        if _args.order == 5 and with_table:
            # Tüm N-gram mertebeleri + hibrit karşılaştırması için toplu eğitim
            for _ord in [2, 3, 4, 5]:
                run_ngram(order=_ord, corpora=corpora, with_word_table=False)
            run_ngram(order=5, corpora=corpora, with_word_table=True)
            print(f"\n{'='*55}")
            print("  Tamamlandı: model_ngram2..5.pkl + model_ngram5h.pkl")
            print(f"{'='*55}\n")
        else:
            run_ngram(order=_args.order, corpora=corpora,
                      with_word_table=with_table)
            print(f"\n{'='*55}")
            print("  Tamamlandı.")
            print(f"{'='*55}\n")

    elif _args.corpus == "boun":
        # ── BOUN-only TrigramLM ──────────────────────────────────────────────
        model_boun, _ = run(reduced=False, corpora=["boun"])
        save_model(model_boun, "model_boun")
        print(f"\n{'='*55}")
        print("  Tamamlandı: model_boun.pkl")
        print(f"{'='*55}\n")
    else:
        # ── Tüm treebank'larla TrigramLM ─────────────────────────────────────
        model_full,    _ = run(reduced=False)
        model_reduced, _ = run(reduced=True)

        combined_full = []
        for corpus, _ in ALL_TREEBANKS:
            path = DATA_DIR / f"tr_{corpus}-ud-train.conllu"
            if path.exists():
                combined_full += parse_conllu(path, reduced=False)
        long_range_mi(combined_full, max_dist=6, min_count=10)
        demo_rerank(model_full, combined_full)

        print(f"\n{'='*55}")
        print("  Model kaydediliyor...")
        save_model(model_full,    "model_full")
        save_model(model_reduced, "model_reduced")
        interactive_test(model_reduced)

        print(f"\n{'='*55}")
        print("  Tamamlandı.")
        print(f"{'='*55}\n")
