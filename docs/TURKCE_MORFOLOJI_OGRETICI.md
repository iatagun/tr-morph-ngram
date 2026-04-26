# Türkçe Morfoloji Tagger — Kapsamlı Öğretici

## İçindekiler
1. [Proje Nedir?](#1-proje-nedir)
2. [Neden Zor?](#2-neden-zor)
3. [Veri Seti: UD CoNLL-U](#3-veri-seti-ud-conll-u)
4. [Temel Kavramlar](#4-temel-kavramlar)
5. [Modeller: HybridLM](#5-modeller-hybridlm)
6. [Modeller: Dependency Parser](#6-modeller-dependency-parser)
7. [Eğitim Süreci ve Parametre Ayarı](#7-eğitim-süreci-ve-parametre-ayarı)
8. [Performans Tablosu](#8-performans-tablosu)
9. [Neyi Denedik, Ne Öğrendik?](#9-neyi-denedik-ne-öğrendik)
10. [Dosya Rehberi](#10-dosya-rehberi)

---

## 1. Proje Nedir?

Türkçe cümlelerdeki her sözcüğün **morfolojik özelliklerini** (FEATS) otomatik olarak tahmin eden bir sistem.

```
Girdi:  "1936 yılında öğretmenler grevdeydi."
Çıktı:
  1936        → Case=Nom|Number=Sing|Person=3
  yılında     → Case=Loc|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3
  öğretmenler → Case=Nom|Number=Plur|Person=3
  grevdeydi   → Aspect=Perf|Mood=Ind|Number=Plur|Person=3|Polarity=Pos|Tense=Past|VerbForm=Fin
```

Her etiket (tag) birden fazla özelliği birlikte kodlar:
- **Case** (Durum): Nom, Acc, Dat, Loc, Abl, Gen, Ins — ismin aldığı hal
- **Number** (Sayı): Sing (tekil), Plur (çoğul)
- **Person** (Kişi): 1, 2, 3
- **Tense** (Zaman): Pres, Past, Fut
- **Aspect** (Görünüş): Imp (süreç), Perf (bitmiş), Hab (alışkanlık)
- **Voice** (Çatı): Act, Pass, Caus, Rfl
- **Mood** (Kip): Ind (haber), Cnd (koşul), Imp (emir), Opt (istek)
- ve daha fazlası...

---

## 2. Neden Zor?

### Türkçe soneksel (agglutinative) bir dil
Tek bir sözcük, bir köke birden fazla ek eklenerek oluşur:

```
ev          → ev (kök)
evde        → ev + de (Loc)
evdeydi     → ev + de + ydi (Past)
evdeydiniz  → ev + de + ydı + nız (Past + 2.Plur)
```

Bu yüzden:
- Kelime biçimleri çok çeşitli → kelime tablosu küçük kalır, OOV (bilinmeyen kelime) oranı yüksek
- Aynı ek farklı anlamlar taşıyabilir → bağlam olmadan kesin yorum güç

### Etiket uzayı büyük
BOUN treebank'ta **1,667 farklı FEATS kombinasyonu** var. Bu 1,667-sınıflı bir sınıflandırma problemi — BERT gibi derin modeller bile zorlanır.

### OOV sorunu
Eğitim verisinde görülmemiş kelimeler test setinin ~%15-20'sini oluşturur. Bunlar için sadece kelime gövde/suffix ipuçlarına dayanmak gerekir.

---

## 3. Veri Seti: UD CoNLL-U

**Universal Dependencies (UD)** projesi, 100+ dil için standart formatta ağaçbankalar (treebank) içerir.

### CoNLL-U Formatı
Her satır bir sözcük:
```
# sent_id = ins_833
# text = 1936 yılındayız.
1   1936    1936   NUM  Year  Case=Nom|Number=Sing|Person=3    2  nmod:poss  _  _
2   yılında yıl    NOUN _     Case=Loc|Number=Sing|...         0  root       _  _
3   yız     null   AUX  Zero  Number=Plur|Person=1|Tense=Pres  2  cop        _  _
```

| Sütun | İçerik | Örnek |
|-------|--------|-------|
| 1 | Token no | 1 |
| 2 | Yüzey biçim | "yılında" |
| 3 | Lemma (kök) | "yıl" |
| 4 | UPOS (kaba POS) | NOUN, VERB, ADJ... |
| 5 | XPOS (dile özgü POS) | _ |
| **6** | **FEATS** | **Case=Loc\|Number=Sing\|...** |
| 7 | Bağımlılık başı | 0 (kök) |
| 8 | Bağımlılık ilişkisi | root, nsubj, obj... |

### Kullandığımız Treebank'lar

| Treebank | Cümle | Alan | Not |
|----------|-------|------|-----|
| BOUN | ~7,800 | Genel/Akademik | **Ana hedef** — etiketler bu şemaya göre |
| IMST | ~3,435 | Haber/Edebi | Farklı annotation konvansiyonları |
| Kenet | ~15,400 | Tarihî | Çok farklı FEATS stringleri |
| Penn | ~14,850 | Penn çevirisi | Farklı FEATS stringleri |
| Tourism | ~15,500 | Turizm rehberi | Kısmi annotation |
| FrameNet | ~2,300 | FrameNet | Sınırlı çeşitlilik |

> **Kritik uyarı:** Her treebank farklı FEATS string kullanır. "Case=Loc" BOUN'da bir string iken Kenet'te farklı olabilir. Bu yüzden emission (sözcük→etiket) modeli **her zaman BOUN-only**, n-gram geçişleri ise tüm treebank'lardan faydalanabilir.

---

## 4. Temel Kavramlar

### 4.1 Emisyon Olasılığı: P(kelime | etiket)

"Bu kelimeyi gördüğümde, hangi etiketi taşıma ihtimali var?"

**MLE (Maximum Likelihood Estimation):** Eğitim verisinde sayarak tahmin:
```
P(Case=Nom | "ev") = count("ev", Case=Nom) / count("ev")
```

Sorun: Eğitim setinde az görülen kelimeler için güvenilmez.

**Witten-Bell Smoothing:** Nadir/OOV kelimeler için backoff:
```
β(kelime) = n0 / (n0 + count(kelime))
P(tag | kelime) = (1-β) × P_MLE + β × P_backoff
```
- `count` büyükse → β küçük → MLE'ye güven
- `count` küçükse/0 → β büyük → backoff'a güven
- **n0 parametresi**: backoff ağırlığının eşiği — tune edilir

### 4.2 N-gram Geçiş Olasılığı: P(etiket_t | etiket_{t-1}, etiket_{t-2})

"Önceki iki etiket buysa, bu etiketin gelme olasılığı ne?"

**N-gram:** Art arda n öğenin birlikte görülme sayımı
- Unigram (n=1): P(tag)
- Bigram (n=2): P(tag | prev)
- Trigram (n=3): P(tag | prev2, prev1)
- 5-gram (n=5): P(tag | prev4, prev3, prev2, prev1)

**Jelinek-Mercer Smoothing:** Geri düşme (backoff) stratejisi:
```
P_smooth(t | h5,h4,h3,h2,h1) =
    λ5 × P(t|h5,h4,h3,h2,h1)   ← 5-gram
  + λ4 × P(t|h4,h3,h2,h1)      ← 4-gram
  + λ3 × P(t|h3,h2,h1)         ← trigram
  + λ2 × P(t|h2,h1)            ← bigram
  + λ1 × P(t)                  ← unigram
  + λ0 × (1/|V|)               ← uniform
```

Seyrek n-gram'larda kısa bağlam daha güvenilir.

### 4.3 Viterbi Algoritması

N-gram geçişlerini düşününce: "Her adımda en iyi kombinasyonu bulmak için tüm yolları denememiz gerekir mi?"

Hayır — **Viterbi** dinamik programlama ile O(T × S²) karmaşıklıkta optimal yolu bulur (T=cümle uzunluğu, S=aday etiket sayısı).

```
cümle:    "öğretmenler  grevdeydi"
adım 1:   [Tag_A: 0.7, Tag_B: 0.2, Tag_C: 0.1]   # 3 aday
adım 2:   [Tag_X: ?, Tag_Y: ?]                      # önceki adımla birlikte
          Tag_A→Tag_X: 0.7 × 0.8 = 0.56 ✓
          Tag_B→Tag_X: 0.2 × 0.9 = 0.18
          → en iyi yol: Tag_A → Tag_X
```

`state_size` parametresi: n-gram bağlamı için kaç önceki etiketi takip ettiğimiz (state_size=2 → bigram context for Viterbi beam).

**Greedy decode:** Her adımda sadece en iyi etiketi seç (hızlı ama optimal değil).
**Viterbi decode:** Global optimal yolu bul (yavaş ama daha doğru).

### 4.4 CharNgramEmission (Karakter N-gram Suffix Modeli)

OOV kelimeler için son kurtarıcı:

```
"öğretmenlere" → suffix'ler: "e", "re", "ere", "lere", "nlere", "enlere"
                              ↓
            eğitimden: "-lere" çoğul, datif anlamına gelir
                              ↓
            P(Case=Dat|Number=Plur | "-lere") → yüksek olasılık
```

- **min_n=2, max_n=6**: 2-6 karakter uzunluğundaki suffix'ler
- **Ağırlık n²**: Uzun suffix daha güvenilir (6²=36, 2²=4)
- **Type-based**: Her (kelime, etiket) çifti bir kez sayılır — sık kelimeler istatistiği ezmez
- **top_k=12**: En olası 12 aday döndür (Viterbi'nin seçmesi için)

### 4.5 UPOS N-gram

FEATS 1,667 sınıflı → çok seyrek. UPOS sadece 17 sınıflı (NOUN, VERB, ADJ...) → çok denser.

Hafif bir UPOS geçiş sinyali eklemek bazen yardımcı olur:
```
score = P_emit × P_feats_ngram^w_trans × P_upos_ngram^w_upos
```

Sonuç: `w_upos=0.0` — yani hiç katkı sağlamadı. FEATS zaten UPOS bilgisini içeriyor.

---

## 5. Modeller: HybridLM

### Mimarı

```
           ┌─────────────────────────────┐
  kelime → │    WordTagEmission          │ → P(tag | kelime)
           │    ├─ MLE (bilinen kelime)  │
           │    └─ Backoff               │
           │        ├─ CharNgramEmission │  (OOV için)
           │        └─ Heuristic rules   │  (son kurtarıcı)
           └─────────────────────────────┘
                          ×
           ┌─────────────────────────────┐
  bağlam → │    NgramLM (5-gram FEATS)   │ → P(tag | önceki etiketler)^w_trans
           └─────────────────────────────┘
                          ×
           ┌─────────────────────────────┐
  bağlam → │    UposNgramLM (3-gram)     │ → P(upos | önceki upos)^w_upos
           └─────────────────────────────┘
                          ↓
                    Viterbi Decode
                          ↓
               En olası etiket dizisi
```

### Skor Formülü

```python
score(kelime, tag, context) = (
    P_emit(tag | kelime)
    × P_ngram(tag | context[-2:])  ^ w_trans
    × P_upos(upos(tag) | upos_ctx) ^ w_upos
)
```

Logaritmik formda (numeric underflow'u önlemek için):
```python
log_score = log(P_emit) + w_trans × log(P_ngram) + w_upos × log(P_upos)
```

### Optimize Edilen Parametreler

| Parametre | Değer | Anlamı |
|-----------|-------|--------|
| `n0` | 1.0 | Witten-Bell backoff eşiği |
| `w_trans` | 0.1 | N-gram geçiş ağırlığı |
| `w_trans_oov` | 0.05 | OOV token'lar için ayrı w_trans |
| `w_upos` | 0.0 | UPOS geçiş ağırlığı (devre dışı) |
| `state_size` | 2 | Viterbi bağlam derinliği |
| `max_order` | 5 | N-gram mertebesi (5-gram) |
| `top_k` | 12 | CharNgram'dan kaç aday |

### Eğitim Kaynakları

| Bileşen | Eğitim Verisi |
|---------|---------------|
| WordTagEmission | **BOUN-only** (29,234 sözcük) |
| CharNgramEmission | **BOUN-only 3× + diğer 1×** (34,394 suffix) |
| NgramLM | **Tüm 6 treebank** (59,247 cümle, 85,329 trigram) |

---

## 6. Modeller: Dependency Parser

### Bağımlılık Parsing Nedir?

```
"Öğretmenler kitabı aldı."

     aldı
    /    \
öğretmenler  kitabı
(nsubj)     (obj)
```

Her token için iki şeyi tahmin ediyoruz:
1. **Baş (head):** Bu token hangi token'a bağlı?
2. **İlişki (relation):** Bu bağ ne tür? (nsubj, obj, det, nmod...)

**UAS (Unlabeled Attachment Score):** Sadece baş doğru mu? (ilişki yanlış olsa da tamam)
**LAS (Labeled Attachment Score):** Hem baş hem ilişki doğru mu?

### Arc-Eager Algoritması

Cümleyi soldan sağa gezerek bir **yığın (stack)** ve **kuyruk (buffer)** ile işler:

```
Başlangıç:  Stack: [ROOT]  Buffer: [öğretmenler, kitabı, aldı]

Adımlar:
  SHIFT    → Stack: [ROOT, öğretmenler]  Buffer: [kitabı, aldı]
  SHIFT    → Stack: [ROOT, öğretmenler, kitabı]  Buffer: [aldı]
  LEFT-ARC → kitabı ← aldı  (obj)
             Stack: [ROOT, öğretmenler]  Buffer: [aldı]
  RIGHT-ARC → öğretmenler → aldı  (nsubj)
  ...
```

4 eylem: `SHIFT`, `REDUCE`, `LEFT-ARC(rel)`, `RIGHT-ARC(rel)`

### Averaged Perceptron

Parser bir **yapısal sınıflandırıcı**: Her parser durumunda hangi eylemi yapmalı?

**Perceptron:** Lineer sınıflandırıcı — özellik vektörü × ağırlık vektörü

**Averaged Perceptron:** Eğitim boyunca ağırlıkların ortalamasını al → daha stabil, overfitting'e dayanıklı

**Özellikler (features):**
- Stack'teki ve buffer'daki token'ların kelimesi, UPOS, FEATS
- Aralarındaki bağımlılık ilişkileri
- Karakter n-gram prefix/suffix'leri
- ...toplam ~yüzbinlerce sparse feature

### Predicted FEATS Neden Önemli?

**Hata:** `python dep_parser.py --eval dev` → **gold FEATS** kullanır → ~60% UAS (misleading)
**Doğru:** `python eval.py --dep-model model_dep_pred_new40` → **predicted FEATS** kullanır → ~67% UAS

Gerçek kullanımda FEATS'i bilmiyoruz — HybridLM ile tahmin edilen FEATS dep parser'a girdi olarak verilir. Bu yüzden "predicted FEATS ile dep parser" değerlendirmesi gerçek performansı yansıtır.

---

## 7. Eğitim Süreci ve Parametre Ayarı

### HybridLM Eğitimi

```bash
python trigram_morph.py --hybrid --corpus boun --ngram-corpus all --model-name MODEL_ADI
```

Sırasıyla:
1. WordTagEmission eğit (BOUN-only)
2. CharNgramEmission eğit (BOUN-biased)
3. NgramLM eğit (tüm treebank'lar)
4. `tune_w_trans` → grid search: n0 × state_size × w_trans (126 kombinasyon)
5. `tune_w_upos` → grid search: w_upos
6. `tune_w_trans_oov` → grid search: w_trans_oov

### Grid Search Nasıl Çalışır?

```python
best_score = 0
for n0 in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    for state_size in [2, 3, 4]:
        for w_trans in [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]:
            model.n0 = n0
            model.state_size = state_size
            model.w_trans = w_trans
            score = evaluate(model, dev_set, greedy_decode)
            if score > best_score:
                best_score = score
                best_params = (n0, state_size, w_trans)
```

Dev seti (979 cümle) üzerinde **greedy decode** ile hızlı değerlendirme, sonra en iyi parametrelerle **Viterbi decode** ile final eval.

### Dep Parser Eğitimi

```bash
python dep_parser.py --pred-feats --iter 40 --model MODEL_ADI --corpus boun
```

- `--pred-feats`: HybridLM ile tahmin edilmiş FEATS kullan (gerçekçi eğitim)
- `--iter 40`: 40 epoch perceptron eğitimi
- Eğitim sırasında her epoch'ta dev UAS raporlanır

---

## 8. Performans Tablosu

### Mevcut En İyi Modeller

| Model | Dosya | Boyut |
|-------|-------|-------|
| HybridLM | `model_hybrid.pkl` | 27 MB |
| Dep Parser | `model_dep_pred_new40.pkl` | 18 MB |

### Metrik Sonuçları (BOUN dev seti, 979 cümle)

| Metrik | Skor | Notlar |
|--------|------|--------|
| **FEATS exact match** | **79.29%** | Tüm özelliklerin birebir eşleşmesi |
| FEATS partial | 60.15% | En az bir özellik doğru |
| UPOS | 68.04% | 17-sınıflı POS tagging |
| LEMMA | 70.45% | Kök bulma |
| **Dep UAS** | **67.78%** | Bağımlılık başı doğruluğu |
| Dep LAS | 57.84% | Hem baş hem ilişki doğruluğu |

### Özellik Bazında FEATS Performansı

| Özellik | Skor | Yorum |
|---------|------|-------|
| Person | 92.73% | Kolayca öğrenilir |
| Number | 92.40% | Kolayca öğrenilir |
| Case | 89.78% | Suffix'ten tahmin edilebilir |
| Person[psor] | 92.18% | İyelik kişisi |
| Number[psor] | 89.87% | İyelik sayısı |
| Polarity | 82.27% | "-me/-ma" bilgisi |
| Tense | 82.58% | Zaman ekleri belirgin |
| Aspect | 56.66% | "-yor/-miş/-acak" belirsiz |
| VerbForm | 76.63% | Fiil biçimi |
| Mood | 54.59% | Kip karmaşık |
| Voice | 52.93% | Çatı belirsiz |
| PronType | 12.50% | Zamir tipi — kural tabanlı yetersiz |

---

## 9. Neyi Denedik, Ne Öğrendik?

### ✅ İşe Yarayanlar

| Yaklaşım | Etki | Neden İşe Yaradı |
|----------|------|-----------------|
| CharNgramEmission (BOUN-only) | +5.56% FEATS | OOV kelimeler için data-driven suffix istatistikleri |
| 5-gram NgramLM | Base | Daha uzun bağlam, daha az seyreklik |
| Çoklu treebank N-gram | +0.05% | Geçiş sayımları zenginleşti |
| w_trans_oov = 0.05 | +0.12% | OOV token'lar için daha az geçiş baskısı |
| Predicted FEATS ile dep eğitimi | Kritik | Domain uyumu |
| Dynamic oracle | Test | Daha esnek hata kurtarma |

### ❌ İşe Yaramayanlar

| Yaklaşım | Sonuç | Neden Çalışmadı |
|----------|-------|-----------------|
| Tüm treebank emission | 66-69% | Annotation şemaları uyumsuz |
| Self-distillation (Wikipedia) | w_pseudo=0.0 | Domain mismatch (Wikipedia ≠ BOUN akademik) |
| BOUN-biased CharNgram + extra | -1.27% | Diğer treebank suffix'leri annotation farkı yarattı |
| w_upos > 0 | Nötr/kötü | FEATS zaten UPOS bilgisini içeriyor, çift sayım |

### 🔑 Kritik Dersler

1. **Annotation tutarlılığı her şeyden önce:** Tahmin hedefi BOUN annotation şeması — diğer treebank'lar sadece yardımcı.
2. **Emission ve N-gram farklı şeyler:** Emission domain'e bağlıdır (BOUN-only), N-gram geçişleri daha genel (tüm treebank'lar faydalı).
3. **Self-distil ancak kaliteli pseudo-label ile işe yarar:** Wikipedia etiketsiz, tahmin gürültülüydü.
4. **Dep parser değerlendirmesinde dikkat:** `dep_parser.py --eval` ≠ `eval.py` — ikincisi gerçek senaryoyu ölçer.

---

## 10. Dosya Rehberi

| Dosya | Ne Yapar |
|-------|----------|
| `trigram_morph.py` | Ana model: HybridLM, NgramLM, CharNgramEmission eğitim+decode |
| `dep_parser.py` | Arc-eager bağımlılık parser'ı |
| `eval.py` | Tam pipeline değerlendirme (FEATS + UPOS + LEMMA + Dep) |
| `mine_suffixes.py` | Suffix kural madenciliği (araştırma aracı) |
| `models/model_hybrid.pkl` | Mevcut en iyi HybridLM (79.29% FEATS) |
| `models/model_dep_pred_new40.pkl` | Mevcut en iyi dep parser (67.78% UAS) |
| `data/tr_boun-ud-*.conllu` | BOUN treebank (train/dev/test) |

### Temel Komutlar

```bash
# Eğitim (sıfırdan)
python trigram_morph.py --hybrid --corpus boun --ngram-corpus all

# Değerlendirme
python eval.py --model hybrid --decode viterbi
python eval.py --model hybrid --decode viterbi --dep-model model_dep_pred_new40

# Dep parser eğitimi
python dep_parser.py --pred-feats --iter 40 --corpus boun --model model_dep_v3

# Self-distillation (Wikipedia)
python trigram_morph.py --self-distil --max-pseudo-sents 100000
```

---

## Sonraki Adımlar (Öneriler)

### Kısa Vadeli
- **Aspect/Mood/Voice için özel suffix kuralları** (PronType gibi `_FUNCTION_WORDS` benzeri) → +1-2% FEATS beklentisi
- **Dep parser feature engineering** → daha fazla prefix/suffix feature → +1-2% UAS beklentisi

### Orta Vadeli
- **CRF (Conditional Random Field):** HybridLM'i CRF ile değiştirmek, tüm cümleyi birlikte optimize eder
- **Semi-supervised:** Domain'e yakın Türkçe metin (gazete/akademik makale) ile self-distil

### Uzun Vadeli
- **Karakter-level LSTM/Transformer:** Türkçe gibi agglutinative dillerde karakter modelleri güçlü
- **mBERT / XLM-R fine-tune:** Türkçe UD üzerinde BERT fine-tune → 90%+ FEATS mümkün

---

*Hazırlayan: Proje geliştirme süreci boyunca öğrenilen dersler, Nisan 2026*
