"""
crf_morph.py — Per-Feature CRF Türkçe Morfolojik Etiketleyici
=============================================================
Her morfolojik özellik boyutu (Aspect, Mood, Voice, …) için ayrı bir
linear-chain CRF eğitilir. Bu yaklaşım:

  * 1648 tam FEATS etiketi yerine max ~9 etiket/boyut → ~100x hızlı
  * Unseen FEATS kombinasyonu sorununu çözer (her boyut bağımsız)
  * Sequence modeling (CRF geçiş ağırlıkları) korunur

Kullanım:
  python crf_morph.py              # eğit + değerlendir
  python crf_morph.py --tune       # c grid search

eval.py entegrasyonu: FactorizedCRFTagger.decode_viterbi() HybridLM ile aynı
arayüzü sağlar.
"""

import re
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import sklearn_crfsuite

from data.conllu import read_conllu as _read_conllu, parse_feats as _parse_feats
from taggers.ngram import _fix_yor_aspect, DATA_DIR, MODEL_DIR

# ─── Türkçe fonetik sabitleri ─────────────────────────────────────────────────

_VOWELS        = set("aeıioöuüAEIİOÖUÜ")
_FRONT_VOWELS  = set("eiöüEİÖÜ")
_BACK_VOWELS   = set("aıouAIOU")
_ROUNDED       = set("ouöüOUÖÜ")

_MORPHEME_FLAGS: List[Tuple[str, re.Pattern]] = [
    ("yor",   re.compile(r"[ıiuü]yor$",          re.I)),
    ("mış",   re.compile(r"m[ıiuü]ş$",           re.I)),
    ("acak",  re.compile(r"[ae]cak$",             re.I)),
    ("malı",  re.compile(r"m[ae]l[ıi]$",         re.I)),
    ("dı",    re.compile(r"[dt][ıiuü]$",         re.I)),
    ("ıl",    re.compile(r"[ıiuü]l[ıiuü]?$",    re.I)),
    ("ın",    re.compile(r"[ıiuü]n[ıiuü]?$",    re.I)),
    ("tır",   re.compile(r"[dt][ıiuü]r$",        re.I)),
    ("ken",   re.compile(r"ken$",                 re.I)),
    ("arak",  re.compile(r"[ae]r?[ae]k$",         re.I)),
    ("ince",  re.compile(r"[ıi]nce$",             re.I)),
    ("dığı",  re.compile(r"d[ıi][gğ][ıi]$",     re.I)),
    ("acağı", re.compile(r"[ae]c[ae][gğ][ıi]$", re.I)),
    ("mak",   re.compile(r"m[ae]k$",              re.I)),
    ("dır",   re.compile(r"d[ıi]r$",              re.I)),
    ("lar",   re.compile(r"l[ae]r$",              re.I)),
    ("da",    re.compile(r"[dt][ae]$",             re.I)),
    ("dan",   re.compile(r"[dt][ae]n$",            re.I)),
    ("ya",    re.compile(r"[ıiuü]?[iye][ae]$",   re.I)),
    ("yı",    re.compile(r"[ıiuü]$",              re.I)),
    ("sa",    re.compile(r"[ae]$",                re.I)),
]

# ─── Voice: kelime içi (infix) ek kalıpları ───────────────────────────────────
# Türkçe'de ses/çatı ekleri tense/agreement eklerinin ÖNÜNE gelir;
# dolayısıyla kelime sonunda değil, ortasında görünürler.
_VOICE_INFIX_FLAGS: List[Tuple[str, re.Pattern]] = [
    # Edilgen  -ıl-/-il-/-ul-/-ül-  (yazılıyor, sevildi, görüldü)
    ("voi_pass_ifx",  re.compile(r"[ıiuü]l[ıiuü]",          re.I)),
    # Dönüşlü  -ın-/-in-/-un-/-ün-  (yıkanıyor, övünüyor)
    ("voi_rfl_ifx",   re.compile(r"[ıiuü]n[ıiuü]",          re.I)),
    # Karşılıklı -ış-/-iş-/-uş-/-üş- (görüşüyor, anlaşıldı)
    ("voi_rcpr_ifx",  re.compile(r"[ıiuü]ş[ıiuü]",          re.I)),
    # Ettirgen  -dır-/-dir-/-dur-/-dür-/-tır-... (yazdırıyor, pişirdi)
    ("voi_cau_ifx",   re.compile(r"[dt][ıiuü]r[ıiuü]",      re.I)),
    # Ettirgen kısa -t- (sat → sattır, bitir → bitirdi) - sonu ile birlikte
    ("voi_cau_t",     re.compile(r"[^ıiuü]t[ıiuü]r",        re.I)),
    # Ettirgen -er-/-ar- (çıkar, çevirir)
    ("voi_cau_er",    re.compile(r"[ae]r[ıiuü]r$|[ae]rt[ıiuü]$", re.I)),
]

# ─── Mood: kiplik kalıpları ───────────────────────────────────────────────────
_MOOD_FLAGS: List[Tuple[str, re.Pattern]] = [
    # İhtimal (Potential) -ebil-/-abil-  (gelebilir, yapabildi)
    ("moo_pot",       re.compile(r"[ae]bil",                  re.I)),
    # İmkansızlık -eme-/-ama-  (gelemiyor, yapamadı)
    ("moo_neg_pot",   re.compile(r"[ae]m[ae]",                re.I)),
    # Koşul (Conditional) -se/-sa kelime sonunda
    ("moo_cnd",       re.compile(r"s[ae]$",                   re.I)),
    # Koşul geçmiş -seydı/-saydı
    ("moo_cnd_past",  re.compile(r"s[ae][yd][ıiuü]$",        re.I)),
    # Zorunluluk (Necessitative) -meli/-malı
    ("moo_nec",       re.compile(r"m[ae]l[ıi]",               re.I)),
    # Emir 3. tekil -sın/-sin/-sun/-sün
    ("moo_imp3sg",    re.compile(r"s[ıiuü]n$",                re.I)),
    # Emir 3. çoğul -sınlar/-sinler
    ("moo_imp3pl",    re.compile(r"s[ıiuü]nl[ae]r$",          re.I)),
    # İstek (Optative) -eyim/-ayım 1sg, -elim/-alım 1pl
    ("moo_opt1sg",    re.compile(r"[ae]y[ıi]m$",              re.I)),
    ("moo_opt1pl",    re.compile(r"[ae]l[ıi]m$",              re.I)),
    # Şart eki olmayan ama -yor olmayan geniş zaman → Ind varsayılan
    # (bu negatif sinyal; zaten diğerleri yoksa Ind çıkar)
]

# ─── Zamir kökü → PronType eşlemesi ─────────────────────────────────────────
# Türkçe zamirler kapalı bir sınıf; kökleri tanırsak PronType büyük ölçüde çözülür.
# "o/bu/şu/ne" çok kısa → yanlış prefix eşleşmesi olur, direkt form tablosuna alındı.

_PRONOUN_ROOTS: List[Tuple[str, str]] = [
    # uzundan kısaya sıralandı (prefix eşleşmesi için)
    ("hiçbirini", "Neg"),  ("hiçbirine", "Neg"),  ("hiçbirinde","Neg"),
    ("hiçbirinden","Neg"), ("hiçbirleri","Neg"),
    ("birileri",  "Ind"),
    # birbirX → BOUN'da Ind (not Rfl)
    ("birbirlerine","Ind"),("birbirleriyle","Ind"),("birbirlerimize","Ind"),
    ("birbirlerinize","Ind"),("birbirleri","Ind"),("birbirimize","Ind"),
    ("birbirinize","Ind"),("birbirinden","Ind"),("birbirlerine","Ind"),
    ("birbirine",  "Ind"), ("birbirini",  "Ind"), ("birbirinde", "Ind"),
    ("birbirinden","Ind"), ("birbirler",  "Ind"), ("birbirimiz", "Ind"),
    ("birbiriniz", "Ind"), ("birbirle",   "Ind"), ("birbirden",  "Ind"),
    ("hiçbiri",   "Neg"),
    ("hepimizi",  "Ind"),  ("hepimize",  "Ind"),  ("hepimizde", "Ind"),
    ("hepimizden","Ind"),  ("hepinizi",  "Ind"),  ("hepinize",  "Ind"),
    ("hepinizden","Ind"),  ("hepinizde", "Ind"),
    ("herkesin",  "Tot"),  ("herkese",   "Tot"),  ("herkeste",  "Tot"),
    ("herkesten", "Tot"),  ("herkesi",   "Tot"),
    ("kendileri", "Rfl"),  ("kendinizi", "Rfl"),  ("kendinize", "Rfl"),
    ("kendinizde","Rfl"),  ("kendinizden","Rfl"), ("kendimizi", "Rfl"),
    ("kendimize", "Rfl"),  ("kendimizden","Rfl"),
    ("kendisini", "Rfl"),  ("kendisine", "Rfl"),  ("kendisinde","Rfl"),
    ("kendisinden","Rfl"), ("kendiniz",  "Rfl"),  ("kendimiz",  "Rfl"),
    ("kendisi",   "Rfl"),  ("kendini",   "Rfl"),  ("kendine",   "Rfl"),
    ("kendinde",  "Rfl"),  ("kendinden", "Rfl"),
    ("tümünü",    "Tot"),  ("tümüne",    "Tot"),  ("tümünde",   "Tot"),
    ("tümünden",  "Tot"),  ("tümünün",   "Tot"),  ("tümü",      "Tot"),
    ("herhangi",  "Ind"),  ("herşeyi",   "Tot"),  ("herşeye",   "Tot"),
    ("herşeyde",  "Tot"),  ("herşeyden", "Tot"),  ("herşeyin",  "Tot"),
    ("herşey",    "Tot"),
    ("şunlar",    "Dem"),  ("bunlar",    "Dem"),  ("onlar",     "Prs"),
    ("birkaç",    "Ind"),  ("bütünü",    "Tot"),  ("bütüne",    "Tot"),
    ("bütünde",   "Tot"),  ("bütünden",  "Tot"),  ("bütün",     "Tot"),
    ("hangi",     "Int"),  ("herkes",    "Tot"),  ("kimse",     "Neg"),
    # hepsi/hepsini → BOUN'da Ind
    ("hepsini",   "Ind"),  ("hepsine",   "Ind"),  ("hepsinde",  "Ind"),
    ("hepsinden", "Ind"),  ("hepsinin",  "Ind"),  ("hepsi",     "Ind"),
    ("hepimiz",   "Ind"),  ("hepiniz",   "Ind"),
    ("hepim",     "Ind"),  ("hepin",     "Ind"),
    ("bazısı",    "Ind"),  ("bazıları",  "Ind"),  ("bazısını",  "Ind"),
    ("bazı",      "Ind"),  ("birisi",    "Ind"),  ("biri",      "Ind"),
    ("çoğu",      "Ind"),  ("çoğunu",    "Ind"),  ("çoğunun",   "Ind"),
    ("kimisi",    "Ind"),  ("birçoğu",   "Ind"),  ("birçoğunun","Ind"),
    ("tüm",       "Tot"),
    ("kendi",     "Rfl"),
    ("kendim",    "Rfl"),  ("kendin",    "Rfl"),
    ("öbürü",     "Dem"),  ("öteki",     "Dem"),  ("öbürünü",   "Dem"),
    ("böyle",     "Dem"),  ("şöyle",     "Dem"),  ("öyle",      "Dem"),
    ("ben",       "Prs"),  ("sen",       "Prs"),  ("biz",       "Prs"),
    ("siz",       "Prs"),  ("kim",       "Int"),  ("kaç",       "Int"),
    ("hiç",       "Neg"),  ("nasıl",     "Int"),  ("niçin",     "Int"),
    ("niye",      "Int"),  ("hangi",     "Int"),
]

# Kısa ve çok anlamlı zamirler için olası çekim formları
_SHORT_PRON: Dict[str, str] = {}
for _stem, _ptype in [
    ("o",   "Prs"),
    ("bu",  "Dem"),
    ("şu",  "Dem"),
]:
    for _suf in ["", "nu", "na", "nun", "nda", "ndan", "nla", "nlar",
                 "nları", "nlardan", "nların", "nca", "nunla", "nunki",
                 "nunkine", "nunkini", "nundan"]:
        _SHORT_PRON[_stem + _suf] = _ptype
# ben/sen/biz/siz düzensiz çekim formları
for _f in ["bana", "beni", "benim", "bende", "benden", "benle",
           "sana", "seni", "senin", "sende", "senden", "senle",
           "bize",  "bizi",  "bizim",  "bizde",  "bizden",  "bizle",
           "size",  "sizi",  "sizin",  "sizde",  "sizden",  "sizle"]:
    _SHORT_PRON[_f] = "Prs"
# kim → Int; kimi/kimini/kimisi → Ind (BOUN annotation)
for _f in ["kim", "kime", "kimin", "kimde", "kimden", "kimle"]:
    _SHORT_PRON[_f] = "Int"
for _f in ["kimi", "kimini", "kimine", "kiminde", "kiminden",
           "kimisi", "kimisini", "kimisine"]:
    _SHORT_PRON[_f] = "Ind"
# "ne" düzensiz çekim
for _f in ["ne", "neyi", "neye", "neyin", "nede", "neden", "neyle",
           "neler", "neleri", "nelere", "nelerin", "nelerde", "nelerden"]:
    _SHORT_PRON[_f] = "Int"
# Yer zarfı soru/gösterme → PronType=Loc (BOUN corpus)
for _f in ["nerede", "nereye", "nereden", "neresi", "nereyi", "neredeyse",
           "burada", "buraya", "buradan", "burası", "burayı",
           "şurada", "şuraya", "şuradan", "şurası", "şurayı",
           "orada",  "oraya",  "oradan",  "orası",  "orayı"]:
    _SHORT_PRON[_f] = "Loc"
# "her" ve "tüm" exact-match: çok anlamlı kısa kökler
for _f in ["her", "tüm", "tümü", "tümünü", "tümüne", "tümünde", "tümünden", "tümünün"]:
    if _f not in _SHORT_PRON:
        _SHORT_PRON[_f] = "Tot"


def _pronoun_type(wl: str) -> Optional[str]:
    """Sözcüğün zamir türünü döner; yoksa None."""
    if wl in _SHORT_PRON:
        return _SHORT_PRON[wl]
    for root, ptype in _PRONOUN_ROOTS:
        if wl.startswith(root):
            rest = wl[len(root):]
            if not rest:
                return ptype
            # her/tüm gibi çok anlamlı kısa kökler: exact-match only (yukarıda)
            # Diğer kökler: standart Türkçe ek-başı karakter kümesi
            if rest[0] in "aeıioöuüdtnlsyrkvcçmzşğbpfh":
                return ptype
    return None


# predict() override için güvenli exact-match formlar:
# yalnızca belirsizliği olmayan (belirteç/sıfat olarak kullanılmayan) formlar.
# NOT: ben/sen/biz/siz çekim formları (benim/bana/beni...) BOUN corpus'ta
#      PronType annotation tutarsız → false positive riski, buraya alınmadı.
_SAFE_PRON_OVERRIDE: Dict[str, str] = {}
# kim çekim formları: her bağlamda kesinlikle zamir
for _f in ["kim", "kime", "kimin", "kimde", "kimden", "kimle"]:
    _SAFE_PRON_OVERRIDE[_f] = "Int"
# kimi/kimisi → BOUN annotation: Ind (not Int)
for _f in ["kimi", "kimini", "kimine", "kiminde", "kiminden",
           "kimisi", "kimisini", "kimisine"]:
    _SAFE_PRON_OVERRIDE[_f] = "Ind"
# yer zamirleri → Loc
for _f in ["nerede", "nereye", "nereden", "neresi", "nereyi",
           "burada", "buraya", "buradan", "burası", "burayı",
           "şurada", "şuraya", "şuradan", "şurası", "şurayı",
           "orada",  "oraya",  "oradan",  "orası",  "orayı"]:
    _SAFE_PRON_OVERRIDE[_f] = "Loc"


def _pronoun_type_override(wl: str) -> Optional[str]:
    """
    predict() sonrası PronType override için güvenli tespit.

    Sadece _SAFE_PRON_OVERRIDE exact match formları için tahmin döner.
    Kısa/belirsiz base formlar (bu/o/şu/ne/ben/her...) ve prefix match yok
    → corpus annotation tutarsızlığından kaynaklanan false positive önlenir.
    """
    return _SAFE_PRON_OVERRIDE.get(wl)


# ─── Kural-tabanlı ek soyucu ─────────────────────────────────────────────────
# Türkçe morfoloji analizörünün suffix registry'sinden çıkarılmış bilgi.
# Şablon: {A}→a/e  {I}→ı/i/u/ü  {D}→d/t  {C}→c/ç
# (form, etiket, min_kalan_kök_uzunluğu) — uzun ekler önce

_MORPH_SUFFIX_DEFS: List[Tuple[str, str, int]] = [
    # Zaman / Kip
    ("{I}yor",    "PRES",       2),   # şimdiki zaman -ıyor
    ("m{I}ş",    "EVID_PAST",  2),   # duyulan geçmiş -mış
    ("{A}c{A}k", "FUT",        2),   # gelecek -acak/-ecek
    ("{A}bil",   "POT",        2),   # yeterlilik -abil/-ebil
    ("s{I}nl{A}r","IMP3PL",   2),   # emir 3Ç -sınlar
    ("{I}nc{A}", "CONV",       2),   # zarf fiil -ınca
    ("{A}r{A}k", "CONV",       2),   # zarf fiil -arak/-erek
    ("{D}{I}ğ",  "PART",       2),   # sıfat fiil -dığ/-diğ
    ("{A}c{A}ğ", "PART",       2),   # sıfat fiil -acağ/-eceğ
    # Çatı ekleri (VOICE)
    ("{D}{I}r",  "CAU",        2),   # ettirgen / bildirme -dır/-dir/-tır/-tir
    ("l{A}t",    "CAU_LAT",    3),   # ettirgen -lat/-let
    ("{I}l",     "PASS",       3),   # edilgen -ıl/-il/-ul/-ül
    ("{I}ş",     "RCPR",       3),   # işteş -ış/-iş/-uş/-üş
    # Kip
    ("s{A}",     "CND",        2),   # dilek-şart -sa/-se
    ("m{A}l{I}", "NEC",        2),   # gereklilik -malı/-meli
    ("s{I}n",    "IMP3SG",     2),   # emir 3T -sın/-sin
    ("m{A}",     "NEG",        1),   # olumsuzluk -ma/-me
    # İsim hal ekleri
    ("{D}{A}n",  "ABL",        2),   # ayrılma -dan/-den/-tan/-ten
    ("{D}{A}",   "LOC",        2),   # bulunma -da/-de/-ta/-te
    ("m{A}k",    "INF",        2),   # mastar -mak/-mek
    ("ken",      "CONV_KEN",   2),   # zarf fiil -ken
    ("l{A}r",    "PLUR",       2),   # çoğul / kişi-3Ç
    # Zaman (geç sırala: kısa, çok amaçlı ekler)
    ("{D}{I}",   "PAST",       2),   # görülen geçmiş -dı/-di/-du/-dü
    ("{I}r",     "AOR",        3),   # geniş zaman -ır/-ir/-ur/-ür (gelebilir→bil+ir)
    ("{A}r",     "AOR",        3),   # geniş zaman -ar/-er (gel+er)
]


def _expand_suffix_tmpl(tmpl: str) -> List[str]:
    """Şablon değişkenlerini tüm olasılıklara açar."""
    results = [tmpl]
    for ph, opts in [("{A}", "ae"), ("{I}", "ıiuü"), ("{D}", "dt"), ("{C}", "cç")]:
        expanded = []
        for r in results:
            if ph in r:
                for o in opts:
                    expanded.append(r.replace(ph, o))
            else:
                expanded.append(r)
        results = expanded
    return results


# Şablonları önceden aç, uzun ekler önce gelecek şekilde sırala
_MORPH_SUFFIX_FORMS: List[Tuple[str, str, int]] = sorted(
    [
        (form, label, min_stem)
        for tmpl, label, min_stem in _MORPH_SUFFIX_DEFS
        for form in _expand_suffix_tmpl(tmpl)
    ],
    key=lambda x: -len(x[0]),
)


def _strip_morph_labels(word: str, max_strips: int = 5) -> List[str]:
    """
    Sözcükten greedy olarak ekleri soyar, etiket listesi döner.

    Örnek:
      yazılıyor  → ['PRES', 'PASS']
      giderse    → ['CND']
      gelebilir  → ['POT']
    """
    wl = word.lower()
    labels: List[str] = []
    for _ in range(max_strips):
        for form, label, min_stem in _MORPH_SUFFIX_FORMS:
            if wl.endswith(form) and len(wl) - len(form) >= min_stem:
                labels.append(label)
                wl = wl[: -len(form)]
                break
        else:
            break  # hiçbir ek eşleşmedi
    return labels

NONE_LABEL = "NONE"


def _last_vowel(word: str) -> str:
    for ch in reversed(word):
        if ch in _VOWELS:
            return ch.lower()
    return ""


def _vowel_class(word: str) -> str:
    lv = _last_vowel(word)
    if lv in _FRONT_VOWELS: return "front"
    if lv in _BACK_VOWELS:  return "back"
    return "none"


def _rounded_class(word: str) -> str:
    return "rounded" if _last_vowel(word) in _ROUNDED else "unrounded"


def _len_class(word: str) -> str:
    n = sum(1 for c in word if c in _VOWELS)
    return "mono" if n <= 1 else ("bi" if n == 2 else "multi")


# ─── Özellik çıkarımı ─────────────────────────────────────────────────────────

def _word_feats(word: str, prefix: str) -> Dict[str, object]:
    wl = word.lower()
    d: Dict[str, object] = {}
    d[f"{prefix}isupper"]      = word.isupper()
    d[f"{prefix}istitle"]      = word.istitle()
    d[f"{prefix}isdigit"]      = word.isdigit()
    d[f"{prefix}hasapos"]      = "'" in word
    d[f"{prefix}ispunct"]      = not any(c.isalpha() or c.isdigit() for c in word)
    d[f"{prefix}len_class"]    = _len_class(word)
    # Suffix (1-8) ve prefix (1-3)
    for n in range(1, 9):
        d[f"{prefix}suf{n}"] = wl[-n:] if len(wl) >= n else wl
    for n in range(1, 4):
        d[f"{prefix}pre{n}"] = wl[:n] if len(wl) >= n else wl
    d[f"{prefix}vowel_class"]   = _vowel_class(word)
    d[f"{prefix}rounded_class"] = _rounded_class(word)
    d[f"{prefix}last_vowel"]    = _last_vowel(word)
    # Genel morfem kalıpları (kelime sonu)
    for flag_name, pattern in _MORPHEME_FLAGS:
        if pattern.search(wl):
            d[f"{prefix}flag.{flag_name}"] = True
    # Voice infix kalıpları (kelime içi çatı ekleri)
    for flag_name, pattern in _VOICE_INFIX_FLAGS:
        if pattern.search(wl):
            d[f"{prefix}flag.{flag_name}"] = True
    # Mood kalıpları
    for flag_name, pattern in _MOOD_FLAGS:
        if pattern.search(wl):
            d[f"{prefix}flag.{flag_name}"] = True
    # Kural-tabanlı ek soyucu — ek zinciri etiketleri
    morph_labels = _strip_morph_labels(word)
    for lbl in morph_labels:
        d[f"{prefix}ms.{lbl}"] = True
    # Ek zinciri sırası: hangi ek önce/sonra geldi
    if len(morph_labels) >= 2:
        d[f"{prefix}ms.seq"] = "_".join(morph_labels[:2])
    # Zamir türü
    pron_type = _pronoun_type(wl)
    if pron_type:
        d[f"{prefix}pron.{pron_type}"] = True
    return d


def word2features(sent: List[str], i: int) -> Dict[str, object]:
    features: Dict[str, object] = {"bias": 1.0}
    features.update(_word_feats(sent[i], ""))
    if i > 0:
        features.update(_word_feats(sent[i - 1], "-1:"))
    else:
        features["BOS"] = True
    if i < len(sent) - 1:
        features.update(_word_feats(sent[i + 1], "+1:"))
    else:
        features["EOS"] = True
    return features


def sent2features(sent: List[str]) -> List[Dict]:
    return [word2features(sent, i) for i in range(len(sent))]


# ─── FEATS ayrıştırma ─────────────────────────────────────────────────────────

def feats_to_dict(feats_str: str) -> Dict[str, str]:
    """'Aspect=Prog|Mood=Ind|...' → {'Aspect': 'Prog', 'Mood': 'Ind', ...}"""
    if feats_str in ("_", "NONE", ""):
        return {}
    result = {}
    for kv in feats_str.split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            result[k] = v
    return result


def dict_to_feats(d: Dict[str, str]) -> str:
    """{'Aspect': 'Prog', 'Mood': 'Ind'} → 'Aspect=Prog|Mood=Ind' (sıralı)"""
    pairs = [f"{k}={v}" for k, v in sorted(d.items()) if v != NONE_LABEL]
    return "|".join(pairs) if pairs else "NONE"


# ─── Veri okuma ───────────────────────────────────────────────────────────────

def _load_conllu(path: Path) -> List[Tuple[List[str], List[str]]]:
    """[(token_list, feats_list), ...] döner. _fix_yor_aspect normalizasyonu dahil."""
    sentences = []
    for sent in _read_conllu(path):
        tokens = [tok.form for tok in sent.tokens]
        feats  = []
        for tok in sent.tokens:
            f = _parse_feats(tok.feats_raw, reduced=False)
            f = _fix_yor_aspect(tok.form, f)
            feats.append(f)
        sentences.append((tokens, feats))
    return sentences


# ─── Per-Feature CRF Tagger ───────────────────────────────────────────────────

class FactorizedCRFTagger:
    """
    Her morfolojik özellik boyutu için ayrı CRF eğitir.
    1648 tam etiket yerine max ~9 etiket/CRF → pratik eğitim süresi.

    eval.py arayüzü: decode_viterbi(tokens) → [(word, feats_str), ...]
    """

    def __init__(
        self,
        algorithm: str = "pa",   # "pa" veya "lbfgs"
        c: float = 0.1,          # PA agresiflik katsayısı
        c1: float = 0.1,         # L-BFGS L1 düzenlileştirme
        c2: float = 0.1,         # L-BFGS L2 düzenlileştirme
        max_iterations: int = 50,
        min_feat_count: int = 10,
    ):
        self.algorithm = algorithm
        self.c = c
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.min_feat_count = min_feat_count
        self.crfs: Dict[str, sklearn_crfsuite.CRF] = {}
        self.dimensions: List[str] = []

    def _make_crf(self, dim: str = "") -> sklearn_crfsuite.CRF:
        """Seçili algoritmaya göre CRF nesnesi oluştur."""
        if self.algorithm == "lbfgs":
            # PronType: sparse sınıf, L1 regularization sparse feature'ları eziyor.
            # Daha düşük c1 ile daha iyi öğrenme sağlanır.
            c1 = 0.0 if dim == "PronType" else self.c1
            c2 = 0.01 if dim == "PronType" else self.c2
            return sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                c1=c1,
                c2=c2,
                max_iterations=self.max_iterations,
                all_possible_transitions=True,
            )
        # pa (varsayılan)
        return sklearn_crfsuite.CRF(
            algorithm="pa",
            pa_type=1,
            c=self.c,
            max_iterations=self.max_iterations,
            all_possible_transitions=False,
        )

    # ── Eğitim ──────────────────────────────────────────────────────────────

    def fit(self, train_sentences: List[Tuple[List[str], List[str]]]) -> None:
        """Her boyut için ayrı CRF eğit."""
        print("  [CRF] Özellik vektörleri çıkarılıyor...")
        X = [sent2features(toks) for toks, _ in train_sentences]

        # Hangi boyutların yeterli verisi var?
        dim_counts: Dict[str, int] = defaultdict(int)
        for _, feats_list in train_sentences:
            for feats_str in feats_list:
                for k in feats_to_dict(feats_str):
                    dim_counts[k] += 1

        self.dimensions = sorted(
            k for k, cnt in dim_counts.items() if cnt >= self.min_feat_count
        )
        print(f"  [CRF] {len(self.dimensions)} boyut: {', '.join(self.dimensions)}")

        for dim in self.dimensions:
            y = self._extract_dim_labels(train_sentences, dim)
            unique = set(lbl for seq in y for lbl in seq)
            crf = self._make_crf(dim)
            crf.fit(X, y)
            self.crfs[dim] = crf
            print(f"    {dim:<20} {len(unique)} etiket  ✓")

    def _extract_dim_labels(
        self,
        sentences: List[Tuple[List[str], List[str]]],
        dim: str,
    ) -> List[List[str]]:
        result = []
        for _, feats_list in sentences:
            labels = []
            for feats_str in feats_list:
                d = feats_to_dict(feats_str)
                labels.append(d.get(dim, NONE_LABEL))
            result.append(labels)
        return result

    # ── Tahmin ──────────────────────────────────────────────────────────────

    def predict(self, tokens: List[str]) -> List[str]:
        """Token listesinden FEATS string listesi döner."""
        X = [sent2features(tokens)]
        # Her boyut için tahmin
        dim_preds: Dict[str, List[str]] = {}
        for dim, crf in self.crfs.items():
            dim_preds[dim] = crf.predict(X)[0]

        # Boyutları birleştir
        result = []
        for i, tok in enumerate(tokens):
            d = {dim: dim_preds[dim][i] for dim in self.dimensions
                 if dim_preds[dim][i] != NONE_LABEL}
            # PronType: kural-tabanlı tespit varsa CRF tahminini geçersiz kıl.
            # Zamir sınıfı kapalı; kural kapsamamız ~%93 → daha güvenilir.
            pron_rule = _pronoun_type_override(tok.lower())
            if pron_rule:
                d["PronType"] = pron_rule
            elif "PronType" in d and not pron_rule:
                # CRF yanlış pozitif veriyorsa temizle
                del d["PronType"]
            result.append(dict_to_feats(d))
        return result

    def decode_viterbi(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """eval.py uyumlu arayüz."""
        return list(zip(tokens, self.predict(tokens)))

    def decode_greedy(self, tokens: List[str]) -> List[Tuple[str, str]]:
        return self.decode_viterbi(tokens)

    # ── Kaydet / Yükle ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        MODEL_DIR.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [kayıt] {path}  ({path.stat().st_size / 1024:.0f} KB)")

    @classmethod
    def load(cls, path: Path) -> "FactorizedCRFTagger":
        import taggers.crf as _crf_module

        class _Unpickler(pickle.Unpickler):
            """__main__ ile kaydedilmiş pkl'leri taggers.crf'e yönlendir."""
            def find_class(self, module: str, name: str):
                if module == "__main__":
                    module = "taggers.crf"
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return _Unpickler(f).load()


# ─── Değerlendirme ────────────────────────────────────────────────────────────

def evaluate_crf(model: FactorizedCRFTagger,
                 sentences: List[Tuple[List[str], List[str]]],
                 max_sents: Optional[int] = None) -> Dict:
    """FEATS exact + per-feature accuracy döner."""
    data = sentences[:max_sents] if max_sents else sentences
    feats_ok = feats_total = 0
    per_feat: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for tokens, gold_feats in data:
        pred_feats = model.predict(tokens)
        for gold, pred in zip(gold_feats, pred_feats):
            feats_total += 1
            if gold == pred:
                feats_ok += 1
            gold_pairs = set(gold.split("|")) if gold not in ("_", "NONE") else set()
            pred_pairs = set(pred.split("|")) if pred not in ("_", "NONE") else set()
            for kv in gold_pairs:
                k = kv.split("=")[0]
                per_feat[k]["total"] += 1
                if kv in pred_pairs:
                    per_feat[k]["correct"] += 1

    return {
        "feats_exact": feats_ok / feats_total if feats_total else 0,
        "per_feature": {
            k: v["correct"] / v["total"]
            for k, v in per_feat.items() if v["total"] >= 5
        },
    }


def print_results(res: Dict, label: str = "CRF") -> None:
    print(f"\n  [{label}] FEATS exact: {res['feats_exact']*100:.2f}%")
    print("  Per-feature accuracy:")
    for feat, acc in sorted(res["per_feature"].items(), key=lambda x: -x[1]):
        print(f"    {feat:<20} {acc*100:.2f}%")


# ─── c grid search ────────────────────────────────────────────────────────────

def tune_crf(train_sents, dev_sents,
             c_grid=(0.01, 0.05, 0.1, 0.25, 0.5),
             max_iterations: int = 30) -> float:
    best_acc, best_c = -1.0, 0.1
    print(f"\n[tune] {len(c_grid)} c değeri test ediliyor...")
    for c in c_grid:
        tagger = FactorizedCRFTagger(c=c, max_iterations=max_iterations)
        tagger.fit(train_sents)
        res = evaluate_crf(tagger, dev_sents)
        acc = res["feats_exact"]
        print(f"  c={c:.4f}  → FEATS={acc*100:.2f}%")
        if acc > best_acc:
            best_acc, best_c = acc, c
    print(f"  → En iyi: c={best_c}  FEATS={best_acc*100:.2f}%")
    return best_c


# ─── Ana akış ─────────────────────────────────────────────────────────────────

def run_crf(
    tune: bool = False,
    max_iter: int = 100,
    algo: str = "lbfgs",
    c1: float = 0.1,
    c2: float = 0.1,
    c: float = 0.1,
) -> FactorizedCRFTagger:
    train_path = DATA_DIR / "tr_boun-ud-train.conllu"
    dev_path   = DATA_DIR / "tr_boun-ud-dev.conllu"

    print("[CRF] Veri yükleniyor...")
    train_sents = _load_conllu(train_path)
    dev_sents   = _load_conllu(dev_path)
    print(f"  Train: {len(train_sents)} cümle | Dev: {len(dev_sents)} cümle")

    if tune and algo == "pa":
        c = tune_crf(train_sents, dev_sents, max_iterations=max(10, max_iter // 5))

    if algo == "lbfgs":
        print(f"\n[CRF] Per-feature eğitim  (lbfgs  c1={c1}  c2={c2}  max_iter={max_iter})...")
    else:
        print(f"\n[CRF] Per-feature eğitim  (pa  c={c}  max_iter={max_iter})...")

    tagger = FactorizedCRFTagger(
        algorithm=algo, c=c, c1=c1, c2=c2, max_iterations=max_iter
    )
    tagger.fit(train_sents)

    print("\n[CRF] Dev değerlendirmesi...")
    res = evaluate_crf(tagger, dev_sents)
    print_results(res)

    save_name = f"model_crf_{algo}.pkl"
    save_path  = MODEL_DIR / save_name
    tagger.save(save_path)
    return tagger


# ─── Stacked CRF ─────────────────────────────────────────────────────────────

def sent2features_stacked(
    sent: List[str],
    base_preds: List[str],
    exclude_dims: Optional[set] = None,
) -> List[Dict]:
    """word2features + HybridLM soft prediction features (bağlam dahil).

    exclude_dims: HybridLM'in güvenilmez olduğu boyutları hariç tut.
    Varsayılan olarak Voice ve Mood hariç tutulur — bu boyutlarda HybridLM
    data leakage nedeniyle CRF'in kendi öğrendiklerini eziyor.
    """
    if exclude_dims is None:
        exclude_dims = {"Voice", "Mood", "PronType"}
    base_dicts = [
        {k: v for k, v in feats_to_dict(p).items() if k not in exclude_dims}
        for p in base_preds
    ]
    result = []
    for i in range(len(sent)):
        feats = word2features(sent, i)
        for k, v in base_dicts[i].items():
            feats[f"base_{k}"] = v
        if i > 0:
            for k, v in base_dicts[i - 1].items():
                feats[f"-1:base_{k}"] = v
        if i < len(sent) - 1:
            for k, v in base_dicts[i + 1].items():
                feats[f"+1:base_{k}"] = v
        result.append(feats)
    return result


class StackedCRFTagger(FactorizedCRFTagger):
    """
    HybridLM tahminlerini CRF feature olarak kullanan yığınlı model.
    Baz modelin per-token FEATS tahminleri (±1 bağlam dahil) CRF'e ek
    özellik olarak verilir; CRF bu sinyali düzeltmeyi/güçlendirmeyi öğrenir.
    """

    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model

    def _base_preds(self, tokens: List[str]) -> List[str]:
        return [feats for _, feats in self.base_model.decode_viterbi(tokens)]

    def fit(self, train_sentences: List[Tuple[List[str], List[str]]]) -> None:
        print("  [Stacked] HybridLM tahminleri hesaplanıyor...")
        X = []
        for toks, _ in train_sentences:
            bp = self._base_preds(toks)
            X.append(sent2features_stacked(toks, bp))

        dim_counts: Dict[str, int] = defaultdict(int)
        for _, feats_list in train_sentences:
            for feats_str in feats_list:
                for k in feats_to_dict(feats_str):
                    dim_counts[k] += 1

        self.dimensions = sorted(
            k for k, cnt in dim_counts.items() if cnt >= self.min_feat_count
        )
        print(f"  [CRF] {len(self.dimensions)} boyut: {', '.join(self.dimensions)}")

        for dim in self.dimensions:
            y = self._extract_dim_labels(train_sentences, dim)
            unique = set(lbl for seq in y for lbl in seq)
            crf = self._make_crf(dim)
            crf.fit(X, y)
            self.crfs[dim] = crf
            print(f"    {dim:<20} {len(unique)} etiket  ✓")

    def predict(self, tokens: List[str]) -> List[str]:
        bp = self._base_preds(tokens)
        X = [sent2features_stacked(tokens, bp)]
        dim_preds: Dict[str, List[str]] = {}
        for dim, crf in self.crfs.items():
            dim_preds[dim] = crf.predict(X)[0]
        result = []
        for i, tok in enumerate(tokens):
            d = {dim: dim_preds[dim][i] for dim in self.dimensions
                 if dim_preds[dim][i] != NONE_LABEL}
            # PronType: kural-tabanlı tespit varsa CRF tahminini geçersiz kıl.
            pron_rule = _pronoun_type_override(tok.lower())
            if pron_rule:
                d["PronType"] = pron_rule
            elif "PronType" in d and not pron_rule:
                del d["PronType"]
            result.append(dict_to_feats(d))
        return result


def run_stacked(
    max_iter: int = 100,
    algo: str = "lbfgs",
    c1: float = 0.1,
    c2: float = 0.1,
    c: float = 0.1,
) -> "StackedCRFTagger":
    train_path = DATA_DIR / "tr_boun-ud-train.conllu"
    dev_path   = DATA_DIR / "tr_boun-ud-dev.conllu"

    print("[Stacked CRF] Veri yükleniyor...")
    train_sents = _load_conllu(train_path)
    dev_sents   = _load_conllu(dev_path)
    print(f"  Train: {len(train_sents)} cümle | Dev: {len(dev_sents)} cümle")

    print("[Stacked CRF] HybridLM baz model yükleniyor...")
    from taggers.ngram import load_model as _load_hybrid
    base_model = _load_hybrid("model_hybrid")

    if algo == "lbfgs":
        print(f"\n[Stacked CRF] Per-feature eğitim  (lbfgs  c1={c1}  c2={c2}  max_iter={max_iter})...")
    else:
        print(f"\n[Stacked CRF] Per-feature eğitim  (pa  c={c}  max_iter={max_iter})...")

    tagger = StackedCRFTagger(
        base_model=base_model, algorithm=algo, c=c, c1=c1, c2=c2, max_iterations=max_iter
    )
    tagger.fit(train_sents)

    print("\n[Stacked CRF] Dev değerlendirmesi...")
    res = evaluate_crf(tagger, dev_sents)
    print_results(res, label="Stacked CRF")

    save_name = f"model_stacked_crf_{algo}.pkl"
    save_path  = MODEL_DIR / save_name
    tagger.save(save_path)
    return tagger


# ─── Ensemble Tagger ──────────────────────────────────────────────────────────

class EnsembleTagger:
    """
    HybridLM + StackedCRF per-dimension ensemble.

    Karar kuralı:
      - Anlaşma  → o değeri kullan (yüksek güven)
      - Anlaşmazlık → PREFER_HYBRID boyutlarında HybridLM, kalanında CRF
    """

    PREFER_HYBRID: set = {"Voice"}

    def __init__(self, hybrid_model, crf_model):
        self.hybrid = hybrid_model
        self.crf = crf_model

    def predict(self, tokens: List[str]) -> List[str]:
        hybrid_pairs = self.hybrid.decode_viterbi(tokens)
        crf_feats_list = self.crf.predict(tokens)

        result = []
        for (_, h_feats), c_feats in zip(hybrid_pairs, crf_feats_list):
            h_d = feats_to_dict(h_feats)
            c_d = feats_to_dict(c_feats)
            all_dims = set(h_d.keys()) | set(c_d.keys())
            merged = {}
            for dim in all_dims:
                h_val = h_d.get(dim)
                c_val = c_d.get(dim)
                if h_val == c_val:
                    if h_val is not None:
                        merged[dim] = h_val
                elif h_val is None:
                    # Sadece CRF görüyor → konservatif: HybridLM'in sessizliğine güven
                    pass
                elif c_val is None:
                    # Sadece HybridLM görüyor → HybridLM'i al
                    merged[dim] = h_val
                else:
                    # Her ikisi de görüyor ama anlaşamıyor → per-dimension kazanan
                    if dim in self.PREFER_HYBRID:
                        merged[dim] = h_val
                    else:
                        merged[dim] = c_val
            result.append(dict_to_feats(merged))
        return result

    def decode_viterbi(self, tokens: List[str]) -> List[Tuple[str, str]]:
        return list(zip(tokens, self.predict(tokens)))

    def decode_greedy(self, tokens: List[str]) -> List[Tuple[str, str]]:
        return self.decode_viterbi(tokens)


def run_ensemble() -> "EnsembleTagger":
    dev_path     = DATA_DIR / "tr_boun-ud-dev.conllu"
    hybrid_path  = MODEL_DIR / "model_hybrid.pkl"
    stacked_path = MODEL_DIR / "model_stacked_crf.pkl"

    print("[Ensemble] Modeller yükleniyor...")
    from taggers.ngram import load_model as _load_hybrid
    hybrid  = _load_hybrid("model_hybrid")
    stacked = FactorizedCRFTagger.load(stacked_path)

    ensemble = EnsembleTagger(hybrid_model=hybrid, crf_model=stacked)

    print("[Ensemble] Dev değerlendirmesi...")
    dev_sents = _load_conllu(dev_path)
    res = evaluate_crf(ensemble, dev_sents)
    print_results(res, label="Ensemble")

    # Karşılaştırma için HybridLM tek başına
    class _HWrap:
        def __init__(self, m): self.m = m
        def predict(self, t): return [f for _, f in self.m.decode_viterbi(t)]
    res_h = evaluate_crf(_HWrap(hybrid), dev_sents)
    print_results(res_h, label="HybridLM baseline")
    return ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Feature CRF Türkçe Morfoloji Tagger")
    parser.add_argument("--stacked",  action="store_true",
                        help="Stacked model: HybridLM tahminlerini feature olarak kullan")
    parser.add_argument("--ensemble", action="store_true",
                        help="Ensemble: HybridLM + StackedCRF per-dimension birleştir")
    parser.add_argument("--tune",     action="store_true", help="c grid search yap (sadece pa)")
    parser.add_argument("--algo",     type=str, default="lbfgs", choices=["pa", "lbfgs"],
                        help="CRF algoritması: 'lbfgs' (L1+L2, önerilen) veya 'pa' (default: lbfgs)")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Maksimum iterasyon sayısı (default: 100)")
    parser.add_argument("--c",        type=float, default=0.1,
                        help="PA agresiflik katsayısı (default: 0.1)")
    parser.add_argument("--c1",       type=float, default=0.1,
                        help="L-BFGS L1 düzenlileştirme (default: 0.1)")
    parser.add_argument("--c2",       type=float, default=0.1,
                        help="L-BFGS L2 düzenlileştirme (default: 0.1)")
    args = parser.parse_args()

    kw = dict(algo=args.algo, max_iter=args.max_iter, c=args.c, c1=args.c1, c2=args.c2)
    if args.ensemble:
        run_ensemble()
    elif args.stacked:
        run_stacked(**kw)
    else:
        run_crf(tune=args.tune, **kw)


