from taggers.ngram import load_model
from taggers import dep as _dep_parser
import os

_DEP_MODELS = {"model_dep_pred_new40", "model_dep_pred", "model_dep"}

for name in ["model_hybrid", "model_dep_pred_new40"]:
    path = f"models/{name}.pkl"
    if not os.path.exists(path):
        continue
    if name in _DEP_MODELS:
        m = _dep_parser.load_parser(name)
    else:
        m = load_model(name)
    mtype = type(m).__name__
    print(f"=== {name} ({mtype}) ===")
    for attr in ["w_trans","w_trans_oov","w_upos","viterbi_state_size","max_order"]:
        val = getattr(m, attr, "YOK")
        print(f"  {attr}: {val}")
    if hasattr(m, "emission"):
        e = m.emission
        n0 = e._n0
        vocab = len(e._counts)
        cng = getattr(e, "char_ngram", None)
        print(f"  emission.n0={n0}  vocab={vocab:,}")
        if cng:
            nsuf = len(cng._counts)
            topk = cng.candidates.__defaults__[0]
            print(f"  char_ngram: {nsuf:,} suffix, top_k={topk}")
    if hasattr(m, "ngram"):
        ng = m.ngram
        print(f"  ngram max_order={ng.max_order}")
    print()
