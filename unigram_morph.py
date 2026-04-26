"""
unigram_morph.py — Türkçe Morfolojik Unigram Model
===================================================
Sözcük-frekansı tabanlı morfoloji modeli.
  • Eğitim verisinden word.lower() → en sık FEATS tablosu oluşturur.
  • OOV sözcükler için heuristik suffix kurallarına döner.

Kullanım (eğitim):
  python unigram_morph.py
  → models/model_unigram.pkl
"""

import pickle
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import List

from trigram_morph import (
    parse_feats,
    heuristic_candidates_weighted,
    DATA_DIR,
    MODEL_DIR,
)


# ─── UnigramLM sınıfı ─────────────────────────────────────────────────────────

class UnigramLM:
    """Sözcük-frekansı tabanlı morfoloji modeli.

    word_table : {word_lower: feats_str}  – eğitim verisindeki en sık etiket
    """

    def __init__(self):
        self.word_table: dict = {}

    # ── Eğitim ──────────────────────────────────────────────────────────────

    def fit(self, conllu_paths: List[Path]) -> "UnigramLM":
        """CoNLL-U dosyalarından kelime→etiket frekans tablosu oluşturur."""
        freq: dict = defaultdict(Counter)
        for path in conllu_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.startswith("#") or line == "":
                        continue
                    parts = line.split("\t")
                    if len(parts) < 6:
                        continue
                    if "-" in parts[0] or "." in parts[0]:
                        continue
                    feats_raw = parts[5]
                    if feats_raw == "_":
                        continue
                    feats = parse_feats(feats_raw, reduced=False)
                    if feats != "NONE":
                        freq[parts[1].lower()][feats] += 1
        self.word_table = {
            w: cnt.most_common(1)[0][0] for w, cnt in freq.items()
        }
        return self

    # ── Tahmin ──────────────────────────────────────────────────────────────

    def predict(self, tokens: List[str]) -> List[str]:
        """Token listesi → FEATS listesi.

        Bilinen sözcük → tablodaki en sık etiket.
        OOV → heuristik suffix kuralı (en yüksek ağırlıklı aday).
        """
        preds = []
        for tok in tokens:
            w = tok.lower()
            if w in self.word_table:
                preds.append(self.word_table[w])
            else:
                cands = heuristic_candidates_weighted(tok)
                preds.append(max(cands, key=lambda x: x[1])[0])
        return preds

    # ── Kaydet / Yükle ──────────────────────────────────────────────────────

    def save(self, name: str) -> Path:
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_kb = path.stat().st_size / 1024
        print(f"  [kayıt] {path}  ({size_kb:.0f} KB)")
        return path

    @staticmethod
    def load(name: str) -> "UnigramLM":
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {path}\n"
                f"Önce 'python unigram_morph.py' çalıştırın."
            )

        import unigram_morph as _um

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, classname):
                if module == "__main__":
                    module = "unigram_morph"
                return super().find_class(module, classname)

        with open(path, "rb") as f:
            return _Unpickler(f).load()


# ─── Ana giriş: eğitim ve kayıt ──────────────────────────────────────────────

def main():
    train_path = DATA_DIR / "tr_boun-ud-train.conllu"
    if not train_path.exists():
        print(f"\n  HATA: {train_path} bulunamadı.")
        print("  Önce 'python trigram_morph.py' çalıştırarak veriyi indirin.\n")
        sys.exit(1)

    print("\n  Unigram modeli eğitiliyor (BOUN train) ...")
    model = UnigramLM()
    model.fit([train_path])
    print(f"  {len(model.word_table):,} eşsiz sözcük kaydedildi.")

    model.save("model_unigram")
    print("  Tamamlandı.\n")


if __name__ == "__main__":
    main()
