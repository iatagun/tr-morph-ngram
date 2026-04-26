"""
dep_parser.py — Arc-Eager Transition-Based Dependency Parser
=============================================================
Averaged perceptron, arc-eager transitions, static oracle.
Projective-only (non-projective training trees skipped).

Kullanım:
  python dep_parser.py                       # BOUN train → models/model_dep.pkl
  python dep_parser.py --pred-feats          # HybridLM tahmin FEATS ile eğit
  python dep_parser.py --eval dev            # dev değerlendirmesi
  python dep_parser.py --eval test           # test değerlendirmesi
"""

import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.conllu import read_conllu as _read_conllu

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

# ─── Geçiş sabitleri ──────────────────────────────────────────────────────────

SHIFT      = "SHIFT"
REDUCE     = "REDUCE"
LEFT_ARC   = "LEFT-ARC"
RIGHT_ARC  = "RIGHT-ARC"

# ─── Morfolojik öznitelik anahtarları ─────────────────────────────────────────

MORPH_ATTRS = [
    "Case", "Number", "Person", "Tense", "Mood",
    "VerbForm", "Polarity", "Voice", "Aspect",
    "Number[psor]", "Person[psor]",
]


def decompose_feats(feats_str: str) -> Dict[str, str]:
    """'Case=Acc|Number=Sing' → {'Case':'Acc','Number':'Sing'}"""
    if not feats_str or feats_str in ("NONE", "_"):
        return {}
    result: Dict[str, str] = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, _, v = part.partition("=")
            result[k] = v
    return result


# ─── Token ───────────────────────────────────────────────────────────────────

class Token:
    __slots__ = ("form", "upos", "morph", "gold_head", "gold_deprel")

    def __init__(self, form: str, upos: str, feats_str: str,
                 gold_head: int = -1, gold_deprel: str = "dep"):
        self.form        = form.lower()
        self.upos        = upos
        self.morph       = decompose_feats(feats_str)
        self.gold_head   = gold_head    # 0 = root
        self.gold_deprel = gold_deprel


# ─── Projektiflik kontrolü ────────────────────────────────────────────────────

def is_projective(heads: List[int]) -> bool:
    """heads[i] = token (i+1)'in head'i (0=ROOT). Çapraz yay yoksa True."""
    n = len(heads)
    for i in range(n):
        h_i = heads[i]
        lo_i, hi_i = sorted([i + 1, h_i]) if h_i != 0 else (0, i + 1)
        for j in range(n):
            if i == j:
                continue
            h_j = heads[j]
            lo_j, hi_j = sorted([j + 1, h_j]) if h_j != 0 else (0, j + 1)
            if lo_i < lo_j < hi_i < hi_j:
                return False
            if lo_j < lo_i < hi_j < hi_i:
                return False
    return True


# ─── Ayrıştırma durumu ────────────────────────────────────────────────────────

class ParseState:
    __slots__ = ("n", "stack", "buffer", "heads", "deprels")

    def __init__(self, n: int):
        self.n        = n
        self.stack    = [0]                   # ROOT = 0
        self.buffer   = list(range(1, n + 1)) # 1-tabanlı token indeksleri
        self.heads    = [-1] * (n + 1)        # heads[i]: token i'nin head'i
        self.deprels  = [""] * (n + 1)

    @property
    def is_terminal(self) -> bool:
        return not self.buffer

    def can_left_arc(self, strict: bool = False) -> bool:
        """strict=True: single-head invariant zorla (sadece head atanmamış s0)."""
        base = (len(self.stack) >= 1 and self.stack[-1] != 0
                and len(self.buffer) >= 1)
        if strict:
            return base and self.heads[self.stack[-1]] == -1
        return base

    def can_right_arc(self) -> bool:
        return len(self.stack) >= 1 and len(self.buffer) >= 1

    def can_reduce(self) -> bool:
        return (len(self.stack) >= 1 and self.stack[-1] != 0
                and self.heads[self.stack[-1]] != -1)

    def can_shift(self) -> bool:
        return len(self.buffer) >= 1

    def apply_left_arc(self, deprel: str) -> None:
        """buffer[0] → stack[-1]: stack top'ın head'i buffer head'i olur."""
        s0 = self.stack.pop()
        b0 = self.buffer[0]
        self.heads[s0]   = b0
        self.deprels[s0] = deprel

    def apply_right_arc(self, deprel: str) -> None:
        """stack[-1] → buffer[0]: stack top, buffer head'in head'i olur."""
        s0 = self.stack[-1]
        b0 = self.buffer.pop(0)
        self.heads[b0]   = s0
        self.deprels[b0] = deprel
        self.stack.append(b0)

    def apply_shift(self) -> None:
        self.stack.append(self.buffer.pop(0))

    def apply_reduce(self) -> None:
        self.stack.pop()


def apply_transition(state: ParseState, action: str, deprel: str) -> None:
    if action == LEFT_ARC:
        state.apply_left_arc(deprel)
    elif action == RIGHT_ARC:
        state.apply_right_arc(deprel)
    elif action == SHIFT:
        state.apply_shift()
    elif action == REDUCE:
        state.apply_reduce()


def is_valid(state: ParseState, action: str,
             strict: bool = False) -> bool:
    if action == SHIFT:     return state.can_shift()
    if action == LEFT_ARC:  return state.can_left_arc(strict=strict)
    if action == RIGHT_ARC: return state.can_right_arc()
    if action == REDUCE:    return state.can_reduce()
    return False


# ─── Statik oracle ────────────────────────────────────────────────────────────

def static_oracle(state: ParseState,
                  tokens: List[Token]) -> Tuple[str, str]:
    """
    Arc-eager statik oracle.
    Öncelik: LEFT-ARC > RIGHT-ARC > REDUCE > SHIFT
    """
    s0 = state.stack[-1] if state.stack else -1
    b0 = state.buffer[0] if state.buffer else -1

    # LEFT-ARC: b0, s0'ın gold head'i mi?
    if s0 > 0 and b0 > 0 and state.can_left_arc(strict=False):
        if tokens[s0 - 1].gold_head == b0:
            return LEFT_ARC, tokens[s0 - 1].gold_deprel

    # RIGHT-ARC: s0, b0'ın gold head'i mi?
    if s0 >= 0 and b0 > 0 and state.can_right_arc():
        if tokens[b0 - 1].gold_head == s0:
            return RIGHT_ARC, tokens[b0 - 1].gold_deprel

    # REDUCE: s0'ın head'i atanmış ve buffer'da kalan gold bağımlısı yok
    if state.can_reduce():
        pending = any(
            0 < j <= state.n and tokens[j - 1].gold_head == s0
            for j in state.buffer
        )
        if not pending:
            return REDUCE, ""

    # SHIFT
    return SHIFT, ""


# ─── Öznitelik çıkarımı ──────────────────────────────────────────────────────

def _tok_feats(tok: Optional[Token], prefix: str) -> List[str]:
    """Tek tokendan öznitelik listesi."""
    if tok is None:
        return [f"{prefix}.FORM=_NULL_", f"{prefix}.UPOS=_NULL_"]
    feats = [
        f"{prefix}.FORM={tok.form}",
        f"{prefix}.UPOS={tok.upos}",
    ]
    for attr in MORPH_ATTRS:
        v = tok.morph.get(attr, "_")
        if v != "_":
            feats.append(f"{prefix}.{attr}={v}")
    return feats


def extract_features(state: ParseState, tokens: List[Token]) -> List[str]:
    """Mevcut durum için öznitelik vektörü."""
    n = state.n

    def get(idx: int) -> Optional[Token]:
        return tokens[idx - 1] if 0 < idx <= n else None

    s0i = state.stack[-1] if state.stack else 0
    s1i = state.stack[-2] if len(state.stack) >= 2 else 0
    b0i = state.buffer[0] if state.buffer else 0
    b1i = state.buffer[1] if len(state.buffer) >= 2 else 0

    s0, s1, b0, b1 = get(s0i), get(s1i), get(b0i), get(b1i)

    # s0/b0'ın sol/sağ çocukları: closest sol, rightmost sağ
    lc_s0: Optional[Token] = None
    rc_s0: Optional[Token] = None
    lc_b0: Optional[Token] = None
    for j in range(1, n + 1):
        if state.heads[j] == s0i:
            if j < s0i:
                lc_s0 = get(j)   # keeps overwriting → closest left child
            elif j > s0i:
                rc_s0 = get(j)   # keeps overwriting → rightmost right child
        if state.heads[j] == b0i and j < b0i:
            lc_b0 = get(j)       # closest left child of b0

    f: List[str] = []
    f += _tok_feats(s0, "s0")
    f += _tok_feats(s1, "s1")
    f += _tok_feats(b0, "b0")
    f += _tok_feats(b1, "b1")
    f += _tok_feats(lc_s0, "lc_s0")
    f += _tok_feats(rc_s0, "rc_s0")
    f += _tok_feats(lc_b0, "lc_b0")

    # İkili öznitelikler
    s0_upos = s0.upos if s0 else "_NULL_"
    b0_upos = b0.upos if b0 else "_NULL_"
    s0_form = s0.form if s0 else "_NULL_"
    b0_form = b0.form if b0 else "_NULL_"
    s1_upos = s1.upos if s1 else "_NULL_"
    b1_upos = b1.upos if b1 else "_NULL_"
    s0_case = s0.morph.get("Case", "_") if s0 else "_"
    b0_case = b0.morph.get("Case", "_") if b0 else "_"
    s0_vf   = s0.morph.get("VerbForm", "_") if s0 else "_"
    b0_vf   = b0.morph.get("VerbForm", "_") if b0 else "_"

    f.append(f"bi:s0u+b0u={s0_upos}+{b0_upos}")
    f.append(f"bi:s0f+b0u={s0_form}+{b0_upos}")
    f.append(f"bi:s0u+b0f={s0_upos}+{b0_form}")
    f.append(f"bi:s1u+s0u={s1_upos}+{s0_upos}")
    f.append(f"bi:b0u+b1u={b0_upos}+{b1_upos}")
    f.append(f"bi:s0c+b0u={s0_case}+{b0_upos}")
    f.append(f"bi:s0u+b0c={s0_upos}+{b0_case}")
    f.append(f"bi:s0vf+b0vf={s0_vf}+{b0_vf}")

    # Mesafe (s0 ile b0 arasındaki fark, 0-9 arası)
    if s0i > 0 and b0i > 0:
        f.append(f"dist={min(abs(b0i - s0i), 9)}")

    return f


# ─── Averaged Perceptron ──────────────────────────────────────────────────────

class AveragedPerceptron:
    """Çok sınıflı averaged perceptron."""

    def __init__(self) -> None:
        self.weights: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        self._totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        self._ts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        self._t: int = 0

    def predict(self, features: List[str]) -> str:
        scores = self.score(features)
        return max(scores, key=scores.__getitem__) if scores else SHIFT

    def _catchup(self, feat: str, cls: str) -> None:
        self._totals[feat][cls] += (
            (self._t - self._ts[feat][cls]) * self.weights[feat][cls])
        self._ts[feat][cls] = self._t

    def update(self, truth: str, guess: str, features: List[str]) -> None:
        self._t += 1
        if truth == guess:
            return
        for feat in features:
            self._catchup(feat, truth)
            self._catchup(feat, guess)
            self.weights[feat][truth] += 1.0
            self.weights[feat][guess] -= 1.0

    def finalize(self) -> None:
        """Ağırlıkları ortala — eğitim sonunda çağır."""
        T = max(self._t, 1)
        for feat in self.weights:
            for cls in self.weights[feat]:
                self._catchup(feat, cls)
                self.weights[feat][cls] = self._totals[feat][cls] / T
        self._totals.clear()
        self._ts.clear()
        # Lambda defaultdict pickle'lanamaz → tümünü regular dict'e çevir
        self.weights = {f: dict(m) for f, m in self.weights.items()}
        self._totals = {}
        self._ts = {}

    def score(self, features: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for feat in features:
            cls_map = self.weights.get(feat)
            if cls_map:
                for cls, w in cls_map.items():
                    scores[cls] += w
        return scores
def transition_cost(state: ParseState, tokens: List[Token],
                    action: str) -> int:
    """
    Verilen geçişin kaybettireceği gold arc sayısını döner.
    (Goldberg & Nivre 2012 dynamic oracle maliyet fonksiyonu)
    """
    s0i = state.stack[-1] if state.stack else 0
    b0i = state.buffer[0] if state.buffer else 0
    cost = 0

    if action == LEFT_ARC:
        # s0 yeni head alıyor (b0). Gold head farklıysa kayıp.
        if s0i > 0 and tokens[s0i - 1].gold_head != b0i:
            cost += 1
        # s0 stack'ten çıkıyor → buffer'daki gold bağımlıları kaybolur
        if s0i > 0:
            cost += sum(1 for j in state.buffer
                        if tokens[j - 1].gold_head == s0i)

    elif action == RIGHT_ARC:
        # b0 yeni head alıyor (s0). Gold head farklıysa kayıp.
        if b0i > 0 and tokens[b0i - 1].gold_head != s0i:
            cost += 1
        # b0 buffer'dan stack'e geçiyor → stack token'larının b0'ı
        # head yapacağı LEFT-ARC imkânı ortadan kalkar
        if b0i > 0:
            cost += sum(1 for j in state.stack
                        if j > 0 and state.heads[j] == -1
                        and tokens[j - 1].gold_head == b0i)

    elif action == REDUCE:
        # s0 kalıcı olarak çıkarılıyor
        if s0i > 0:
            # Henüz head atanmamışsa ve gold head buffer'daysa kayıp
            if state.heads[s0i] == -1:
                if tokens[s0i - 1].gold_head in state.buffer:
                    cost += 1
            # s0'ın hedeflediği gold bağımlılar buffer'daysa kayıp
            cost += sum(1 for j in state.buffer
                        if tokens[j - 1].gold_head == s0i)

    elif action == SHIFT:
        # b0 stack'e geçiyor → b0 ile stack token'ları arasındaki
        # her iki yönlü arc (RIGHT-ARC ve LEFT-ARC) artık mümkün değil
        if b0i > 0:
            for j in state.stack:
                if j <= 0:
                    continue
                # j → b0 arci: RIGHT-ARC imkânı kayboluyor
                if state.heads[b0i] == -1 and tokens[b0i - 1].gold_head == j:
                    cost += 1
                # b0 → j arci: LEFT-ARC imkânı kayboluyor
                if state.heads[j] == -1 and tokens[j - 1].gold_head == b0i:
                    cost += 1

    return cost


def dynamic_oracle(state: ParseState,
                   tokens: List[Token]) -> set:
    """
    Mevcut durumda minimum maliyetli geçiş etiketlerini döner.
    Döner: set of label strings (örn. {"LEFT-ARC:nsubj"})
    """
    s0i = state.stack[-1] if state.stack else 0
    b0i = state.buffer[0] if state.buffer else 0

    valid_actions = [a for a in (LEFT_ARC, RIGHT_ARC, REDUCE, SHIFT)
                     if is_valid(state, a, strict=False)]

    costs: Dict[str, int] = {}
    for act in valid_actions:
        c = transition_cost(state, tokens, act)
        if act == LEFT_ARC and s0i > 0:
            lbl = _label(act, tokens[s0i - 1].gold_deprel)
        elif act == RIGHT_ARC and b0i > 0:
            lbl = _label(act, tokens[b0i - 1].gold_deprel)
        else:
            lbl = act
        costs[lbl] = c

    if not costs:
        return {SHIFT}

    min_cost = min(costs.values())
    return {lbl for lbl, c in costs.items() if c == min_cost}




def _label(action: str, deprel: str) -> str:
    return f"{action}:{deprel}" if deprel else action


def train_parser(train_data: List[List[Token]],
                 n_iter: int = 10,
                 use_dynamic: bool = False) -> AveragedPerceptron:
    """
    train_data: Her elemanı Token listesi olan cümle listesi.
    n_iter: Eğitim geçiş sayısı.
    use_dynamic: True → dynamic oracle + model exploration.
    """
    perceptron = AveragedPerceptron()
    indices = list(range(len(train_data)))

    for it in range(n_iter):
        random.shuffle(indices)
        errors = 0
        for idx in indices:
            tokens = train_data[idx]
            state  = ParseState(len(tokens))
            while not state.is_terminal:
                feats = extract_features(state, tokens)

                if use_dynamic:
                    oracle_set = dynamic_oracle(state, tokens)
                    # En yüksek skorlu GEÇERLİ tahmin
                    scores = perceptron.score(feats)
                    guess = SHIFT
                    for lbl, _ in sorted(scores.items(), key=lambda x: -x[1]):
                        if ":" in lbl:
                            act, _ = lbl.split(":", 1)
                        else:
                            act = lbl
                        if is_valid(state, act, strict=False):
                            guess = lbl
                            break
                    # Oracle ile karşılaştır
                    if guess not in oracle_set:
                        oracle_label = next(iter(oracle_set))
                        perceptron.update(oracle_label, guess, feats)
                        errors += 1
                    # Modelin tahminiyle ilerle (exploration)
                    if ":" in guess:
                        g_act, g_dep = guess.split(":", 1)
                    else:
                        g_act, g_dep = guess, ""
                    if is_valid(state, g_act, strict=False):
                        apply_transition(state, g_act, g_dep)
                    else:
                        apply_transition(state, SHIFT, "")
                    # Exploration sonrası buffer boşaldıysa stack'i temizle
                    if not state.buffer:
                        while state.can_reduce():
                            state.apply_reduce()
                else:
                    action, deprel = static_oracle(state, tokens)
                    label  = _label(action, deprel)
                    guess  = perceptron.predict(feats)
                    perceptron.update(label, guess, feats)
                    if guess != label:
                        errors += 1
                    apply_transition(state, action, deprel)

        print(f"  Iter {it + 1:2d}: {errors:,} hata")

    perceptron.finalize()
    return perceptron


# ─── Ayrıştırma ──────────────────────────────────────────────────────────────

def parse_tokens(perceptron: AveragedPerceptron,
                 tokens: List[Token]) -> List[Tuple[int, str]]:
    """
    tokens listesini ayrıştır.
    Döner: [(head_1indexed, deprel), ...] — her token için.
    """
    state = ParseState(len(tokens))

    while not state.is_terminal:
        feats  = extract_features(state, tokens)
        scores = perceptron.score(feats)

        best_action = SHIFT
        best_deprel = ""

        # En yüksek skorlu geçerli eylemi seç
        for label, _ in sorted(scores.items(), key=lambda x: -x[1]):
            if ":" in label:
                act, dep = label.split(":", 1)
            else:
                act, dep = label, ""
            if is_valid(state, act):
                best_action = act
                best_deprel = dep
                break

        # Hiç geçerli eylem skorlanmadıysa fallback
        if not is_valid(state, best_action):
            if state.can_shift():
                best_action, best_deprel = SHIFT, ""
            elif state.can_reduce():
                best_action, best_deprel = REDUCE, ""
            else:
                break

        apply_transition(state, best_action, best_deprel)

    # Head atanmamış tokenları ROOT'a bağla
    result: List[Tuple[int, str]] = []
    for i in range(1, state.n + 1):
        h = state.heads[i]
        d = state.deprels[i]
        if h == -1:
            h = 0
            d = "root"
        result.append((h, d))
    return result


def parse_sentence(perceptron: AveragedPerceptron,
                   forms: List[str],
                   feats_list: List[str],
                   upos_list: List[str]) -> List[Tuple[int, str]]:
    """
    Dışarıdan kullanım için: (forms, feats, upos) → [(head, deprel), ...]
    eval.py ve test.py bu fonksiyonu çağırır.
    """
    tokens = [Token(f, u, feat)
              for f, u, feat in zip(forms, upos_list, feats_list)]
    return parse_tokens(perceptron, tokens)


# ─── CoNLL-U okuyucu ─────────────────────────────────────────────────────────

def read_conllu(path: Path,
                skip_nonproj: bool = True) -> List[List[Token]]:
    """
    CoNLL-U dosyasını okur, her cümleyi Token listesi olarak döner.
    skip_nonproj=True ise non-projektif cümleler atlanır.
    data.conllu.read_conllu'ya delegate eder; Token dönüşümü burada yapılır.
    """
    sentences: List[List[Token]] = []
    for sent in _read_conllu(path):
        heads = [tok.head for tok in sent.tokens]
        if skip_nonproj and not is_projective(heads):
            continue
        tokens = [
            Token(tok.form, tok.upos, tok.feats_raw, tok.head, tok.deprel)
            for tok in sent.tokens
        ]
        sentences.append(tokens)
    return sentences


# ─── Kaydet / yükle ──────────────────────────────────────────────────────────

def save_parser(perceptron: AveragedPerceptron,
                name: str = "model_dep") -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(perceptron, f)
    size_kb = path.stat().st_size // 1024
    print(f"  [kayıt] {path}  ({size_kb} KB)")


class _DepUnpickler(pickle.Unpickler):
    """Model __main__ olarak kaydedilmişse sınıfı bu modüle yönlendir."""
    def find_class(self, module, name):
        if name == "AveragedPerceptron":
            return AveragedPerceptron
        return super().find_class(module, name)


def load_parser(name: str = "model_dep") -> AveragedPerceptron:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return _DepUnpickler(f).load()


# ─── Değerlendirme ───────────────────────────────────────────────────────────

def evaluate_parser(perceptron: AveragedPerceptron,
                    test_data: List[List[Token]]) -> Tuple[float, float]:
    """
    UAS ve LAS döner (punct hariç).
    """
    uas_correct = uas_total = 0
    las_correct = las_total = 0

    for tokens in test_data:
        result = parse_tokens(perceptron, tokens)
        for i, (pred_head, pred_dep) in enumerate(result):
            tok = tokens[i]
            if tok.upos == "PUNCT":
                continue
            uas_total += 1
            las_total += 1
            if pred_head == tok.gold_head:
                uas_correct += 1
                if pred_dep == tok.gold_deprel:
                    las_correct += 1

    uas = 100 * uas_correct / max(uas_total, 1)
    las = 100 * las_correct / max(las_total, 1)
    return uas, las


def replace_feats_with_predictions(sentences: List[List[Token]],
                                   model) -> List[List[Token]]:
    """
    Her cümlenin token FEATS'ini HybridLM tahminleriyle değiştirir.
    HEAD/DEPREL gold kalır; sadece morph alanı güncellenir.
    """
    from trigram_morph import upos_from_feats_word
    updated = []
    forms_batch = [[t.form for t in sent] for sent in sentences]

    for i, (sent, forms) in enumerate(zip(sentences, forms_batch)):
        if i % 1000 == 0:
            print(f"    {i:,}/{len(sentences):,} cümle işlendi...", end="\r")
        try:
            pairs = model.decode_viterbi(forms)
            pred_feats = [tag for _, tag in pairs]
        except Exception:
            # Hata olursa gold FEATS koru
            updated.append(sent)
            continue
        new_sent = []
        for tok, pf in zip(sent, pred_feats):
            new_tok = Token(tok.form, upos_from_feats_word(tok.form, pf), pf,
                            tok.gold_head, tok.gold_deprel)
            new_sent.append(new_tok)
        updated.append(new_sent)
    print(f"    {len(sentences):,}/{len(sentences):,} cümle işlendi.   ")
    return updated


# ─── Ana giriş ───────────────────────────────────────────────────────────────

ALL_CORPORA = ["boun", "framenet", "imst", "kenet", "penn", "tourism"]

if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Arc-Eager Dependency Parser")
    _parser.add_argument("--iter",  type=int, default=10,
                         help="Eğitim geçiş sayısı (varsayılan: 10)")
    _parser.add_argument("--eval",  choices=["dev", "test"], default=None,
                         help="Sadece değerlendir (eğitim yok)")
    _parser.add_argument("--model", default="model_dep",
                         help="Model adı (varsayılan: model_dep)")
    _parser.add_argument("--pred-feats", action="store_true",
                         help="Eğitimde gold FEATS yerine HybridLM tahminlerini kullan")
    _parser.add_argument("--dynamic", action="store_true",
                         help="Dynamic oracle + model exploration kullan")
    _parser.add_argument("--corpus", default="boun",
                         help="Eğitim korpusu: 'boun', 'framenet', 'imst', 'kenet', "
                              "'penn', 'tourism' veya virgülle ayrılmış liste / 'all'")
    _args = _parser.parse_args()

    dev_path   = DATA_DIR / "tr_boun-ud-dev.conllu"
    test_path  = DATA_DIR / "tr_boun-ud-test.conllu"

    # Korpus seçimi
    if _args.corpus == "all":
        _corpus_list = ALL_CORPORA
    else:
        _corpus_list = [c.strip() for c in _args.corpus.split(",")]

    if _args.eval:
        print(f"  Model yükleniyor: {_args.model}...")
        _perc = load_parser(_args.model)
        _eval_path = dev_path if _args.eval == "dev" else test_path
        print(f"  {_args.eval} seti değerlendiriliyor...")
        _data = read_conllu(_eval_path, skip_nonproj=False)
        _uas, _las = evaluate_parser(_perc, _data)
        print(f"\n  {'=' * 40}")
        print(f"  Split  : {_args.eval}")
        print(f"  UAS    : {_uas:.2f}%  (punct hariç)")
        print(f"  LAS    : {_las:.2f}%  (punct hariç)")
        print(f"  {'=' * 40}\n")

    else:
        # Eğitim verisi — seçilen korpusları birleştir
        _train = []
        _total_skip = 0
        for _corp in _corpus_list:
            _corp_path = DATA_DIR / f"tr_{_corp}-ud-train.conllu"
            if not _corp_path.exists():
                print(f"  UYARI: {_corp_path.name} bulunamadı, atlanıyor.")
                continue
            _raw  = read_conllu(_corp_path, skip_nonproj=False)
            _proj = read_conllu(_corp_path, skip_nonproj=True)
            _total_skip += len(_raw) - len(_proj)
            _train.extend(_proj)
            print(f"  {_corp:<12} {len(_proj):,} projektif cümle")
        print(f"  {'─'*35}")
        print(f"  Toplam: {len(_train):,} cümle  ({_total_skip} non-projektif atlandı)")

        print(f"\n  Eğitim başlıyor ({_args.iter} geçiş)...")
        if _args.pred_feats:
            print("  HybridLM modeli yükleniyor...")
            from trigram_morph import load_model as _load_hybrid
            _hybrid = _load_hybrid("model_hybrid")
            print("  Eğitim FEATS'leri HybridLM ile tahmin ediliyor...")
            _train = replace_feats_with_predictions(_train, _hybrid)
            print("  Tahmin tamamlandı.")
        _perc = train_parser(_train, n_iter=_args.iter,
                             use_dynamic=_args.dynamic)
        save_parser(_perc, _args.model)

        print(f"\n  Dev seti değerlendiriliyor...")
        _dev  = read_conllu(dev_path, skip_nonproj=False)
        _uas, _las = evaluate_parser(_perc, _dev)
        print(f"\n  {'=' * 40}")
        print(f"  UAS (dev): {_uas:.2f}%  (punct hariç)")
        print(f"  LAS (dev): {_las:.2f}%  (punct hariç)")
        print(f"  {'=' * 40}\n")
