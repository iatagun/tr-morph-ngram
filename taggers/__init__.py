"""
taggers — Türkçe morfolojik etiketleyici implementasyonları.

Eski pkl dosyalarıyla geriye dönük uyumluluk için module remap kaydı.
"""
import sys
import importlib

# Eski module adlarıyla kaydedilmiş pkl'ler için remap
_REMAP = {
    "trigram_morph": "taggers.ngram",
    "unigram_morph": "taggers.unigram",
    "crf_morph":     "taggers.crf",
    "dep_parser":    "taggers.dep",
}

for _old, _new in _REMAP.items():
    if _old not in sys.modules:
        try:
            sys.modules[_old] = importlib.import_module(_new)
        except Exception:
            pass
