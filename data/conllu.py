"""
data/conllu.py — CoNLL-U dosya okuyucusu
=========================================
Tek, yetkili CoNLL-U ayrıştırıcısı.

Diğer modüller (trigram_morph, dep_parser, eval) buradan import eder;
kendi yerel okuyucularını tutmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

# ─── FEATS sabitleri ──────────────────────────────────────────────────────────

REDUCED_KEYS: Set[str] = {"Case", "Number", "Person", "Tense", "VerbForm", "Mood", "Poss"}


# ─── Veri modeli ──────────────────────────────────────────────────────────────

@dataclass
class ConlluToken:
    form:     str
    lemma:    str
    upos:     str
    feats_raw: str   # CoNLL-U'dan ham string ("_" yoksa)
    head:     int
    deprel:   str


@dataclass
class ConlluSentence:
    text:   str               # "# text = ..." satırından (yoksa "")
    tokens: List[ConlluToken]


# ─── FEATS ayrıştırma ─────────────────────────────────────────────────────────

def parse_feats(feats_str: str, reduced: bool = False) -> str:
    """
    CoNLL-U FEATS sütununu kanonik sıralı string etikete dönüştürür.
    reduced=True → yalnızca REDUCED_KEYS özelliklerini saklar.
    """
    if feats_str in ("_", ""):
        return "NONE"
    parsed: dict[str, str] = {}
    for pair in feats_str.split("|"):
        k, _, v = pair.partition("=")
        parsed[k] = v
    if reduced:
        parsed = {k: v for k, v in parsed.items() if k in REDUCED_KEYS}
    if not parsed:
        return "NONE"
    return "|".join(f"{k}={v}" for k, v in sorted(parsed.items()))


# ─── Çekirdek okuyucu ─────────────────────────────────────────────────────────

def read_conllu(path: Path) -> List[ConlluSentence]:
    """
    CoNLL-U dosyasını okur.
    Multi-word token satırları (örn. "1-2") ve boş düğümler (".") atlanır.
    Her cümle bir ConlluSentence nesnesi olarak döner.
    """
    sentences: List[ConlluSentence] = []
    current_tokens: List[ConlluToken] = []
    current_text: str = ""

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("# text ="):
                current_text = line[8:].strip()
                continue
            if line.startswith("#"):
                continue
            if line == "":
                if current_tokens:
                    sentences.append(ConlluSentence(current_text, current_tokens))
                current_tokens = []
                current_text = ""
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            tid = parts[0]
            if "-" in tid or "." in tid:
                continue
            head = int(parts[6]) if parts[6].isdigit() else 0
            current_tokens.append(ConlluToken(
                form      = parts[1],
                lemma     = parts[2],
                upos      = parts[3],
                feats_raw = parts[5],
                head      = head,
                deprel    = parts[7],
            ))

    if current_tokens:
        sentences.append(ConlluSentence(current_text, current_tokens))

    return sentences
