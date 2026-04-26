"""Mine BOUN training data for suffix → FEATS mappings."""
from pathlib import Path
from collections import defaultdict, Counter

EXTRA_SUFFIXES = [
    "dılar","diler","tılar","tiler","dular","düler","tular","tüler",
    "dım","dim","dum","düm","tım","tim","tum","tüm",
    "dın","din","dun","dün","tın","tin","tun","tün",
    "dı","di","du","dü","tı","ti","tu","tü",
    "mışsın","mişsin","muşsun","müşsün",
    "mışım","mişim","muşum","müşüm",
    "sın","sin","sun","sün",
    "sınlar","sinler","sunlar","sünler",
    "ıyorum","iyorum","uyorum","üyorum",
    "ıyorsun","iyorsun","uyorsun","üyorsun",
    "medim","madım","medin","madın",
]

def extract_suffix_feats(conllu_path, suffix_list):
    results = defaultdict(Counter)
    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            if "-" in parts[0] or "." in parts[0]:
                continue
            form = parts[1].lower()
            feats = parts[5] if parts[5] != "_" else "NONE"
            for suf in suffix_list:
                if form.endswith(suf):
                    results[suf][feats] += 1
                    break
    return results

path = Path("data/tr_boun-ud-train.conllu")
suffixes = [
    "mışlar","mişler","muşlar","müşler",
    "mışsın","mişsin","muşsun","müşsün",
    "mışım","mişim","muşum","müşüm",
    "mış","miş","muş","müş",
    "ecekler","acaklar","eceksin","acaksın","eceğim","acağım",
    "ecek","acak",
    "meyecek","mayacak",
    "ıyordu","iyordu","uyordu","üyordu",
    "miyordu","mıyordu","muyordu","müyordu",
    "ıyor","iyor","uyor","üyor",
    "arım","erim","ırım","irim","urum","ürüm",
    "arsın","ersin","ırsın","irsin","ursun","ürsün",
    "arlar","erler","ırlar","irler","urlar","ürler",
    "maz","mez",
    "ar","er",
    "arak","erek",
    "ınca","ince","unca","ünce",
    "ıp","ip","up","üp",
    "madan","meden",
    "mak","mek",
    "meli","malı",
    "ıldı","ildi","uldu","üldü",
    "ılıyor","iliyor",
    "se","sa",
    "dığı","diği","duğu","düğü",
    "tığı","tiği","tuğu","tüğü",
]
results = extract_suffix_feats(path, suffixes)
for suf in suffixes:
    cnt = results.get(suf)
    if not cnt:
        continue
    total = sum(cnt.values())
    if total < 5:
        continue
    print(f"\n=== -{suf} === (toplam: {total})")
    for feats, n in cnt.most_common(5):
        print(f"  {n:5d}  {feats}")
