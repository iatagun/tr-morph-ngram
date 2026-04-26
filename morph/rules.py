"""
morph/rules.py — Türkçe morfolojik kural tabloları
====================================================
trigram_morph.py'den ayrılmış; tüm sabit kural tabloları burada tutulur.
Bu modül bir veri kaynağıdır — fonksiyon içermez, yalnızca sabitler.
"""
from __future__ import annotations
import re
from typing import Dict, List, Set, Tuple


_AMV_BOOST_RULES: List[Tuple[re.Pattern, str, float]] = [
    # Aspect=Prog: -(ı/i/u/ü)yor- şimdiki zaman eki (çok güvenilir)
    # gidiyor, gidiyorum, gidiyordu, gidiyorken, ...
    (re.compile(r"[ıiuü]yor"),                  "Aspect=Prog",  3.0),

    # Aspect=Prosp: gelecek zaman eki -ecek/-acak (çok güvenilir)
    # gidecek, gidecekler, gideceğim, ...
    (re.compile(r"(ecek|acak)"),                "Aspect=Prosp", 3.0),

    # Mood=Cnd: koşul kipi -(y)se/-(y)sa + kişi ekleri
    # gitsek, gitsen, gitseydi, gitseler, gidiyorsa, ...
    (re.compile(r"(seydi|saydı|seniz|sanız"
                r"|seler|salar|sek|sem|sam|yse|ysa)$"),
                                                 "Mood=Cnd",   3.0),

    # Mood=Nec: gereklilik kipi -meli/-malı  (gitmeliyim, gitmelisin, ...)
    (re.compile(r"m(eli|alı)"),                  "Mood=Nec",   3.0),

    # Mood=Imp: emir kipi -sın/-sin/-sun/-sün (gitsin, gelsin, ...)
    (re.compile(r"(sınlar|sinler|sunlar|sünler"
                r"|sın|sin|sun|sün)$"),           "Mood=Imp",   2.5),

    # Voice=Pass: edilgen eki -(ı/i/u/ü)l- + zaman/kip eki
    # kırıldı, kırılıyor, kırılmış, kırılan, kırılmak, ...
    (re.compile(r"[ıiuü]l(d[ıiuü]|[ıiuü]yor|m[ıiuü]ş|[ıiuü]r"
                r"|an|en|mak|mek|[ıiuü]nca)"),   "Voice=Pass", 3.0),

    # Voice=Cau: ettirgen eki -(dır/tır/dir/tir/dur/tur)- uzun formlar
    # (kırdır-, yaptır-, getirt-, ...)
    (re.compile(r"[dtDT][ıiuü]r(d[ıiuü]|[ıiuü]yor|m[ıiuü]ş"
                r"|[ıiuü]r|mak|mek|an|en)"),     "Voice=Cau",  2.5),

    # VerbForm=Vnoun: isim-fiil eki -mak/-mek (çok güvenilir)
    # gitmek, gelmek, yazmak, ...
    (re.compile(r"(mak|mek)$"),                  "VerbForm=Vnoun", 3.0),

    # VerbForm=Conv: zarf-fiil eki -arak/-erek (güvenilir)
    # giderek, bakarak, koşarak, ...
    (re.compile(r"(arak|erek)$"),                "VerbForm=Conv", 3.0),

    # Aspect=Prog: -makta/-mekte şimdiki zaman devamlılık eki
    # bulunmaktadır, yapmaktadır, gelmekte, sürmektedir, ...
    (re.compile(r"(makta|mekte)"),               "Aspect=Prog",   3.0),

    # Mood=Imp: nazik/resmi 2. çoğul emir eki -ınız/-iniz/-unuz/-ünüz
    # yapınız, gidiniz, alınız, söyleyiniz, ...
    (re.compile(r"[ıiuü]n[ıiuü]z$"),            "Mood=Imp",      2.0),

    # Evident=Nfh: nakli geçmiş eki -mış/-miş/-muş/-müş (çok güvenilir)
    # gitmiş, gelmiş, yazmış, duymuş, ...
    # Not: -mış VerbForm=Part için de kullanılır; 2.0 ılımlı bir artış.
    (re.compile(r"m[ıiuü]ş"),                    "Evident=Nfh",  2.0),

    # VerbForm=Part: -mış/-miş partizip kullanımı için dengeleyici boost
    # (Evident=Nfh ile birlikte; her iki yorumu güçlendirir)
    (re.compile(r"m[ıiuü]ş"),                    "VerbForm=Part", 1.5),
]



_FUNCTION_WORDS: set = {
    # Bağlaçlar
    "ve", "ama", "fakat", "veya", "ya", "ki", "çünkü", "ancak", "hem",
    "lakin", "dahi", "ya da", "ne", "veya",
    # Söylem partikülleri (standalone)
    "da", "de", "bile", "ise", "diye",
    # Soru partikülleri
    "mı", "mi", "mu", "mü",
    # Edatlar (hiç çekilmez)
    "için", "gibi", "kadar", "göre", "sonra", "önce", "beri",
    "karşı", "doğru", "itibaren", "rağmen",
    # Olumsuzluk
    "değil",
    # Alıntı
    "diye",
    # Genel
    "evet", "hayır", "tamam", "peki", "işte",
}

# Zamirler: yalın hal → birincil FEATS aday
_PRONOUN_FEATS: Dict[str, str] = {
    "ben":    "Case=Nom|Number=Sing|Person=1",
    "sen":    "Case=Nom|Number=Sing|Person=2",
    "o":      "Case=Nom|Number=Sing|Person=3",
    "biz":    "Case=Nom|Number=Plur|Person=1",
    "siz":    "Case=Nom|Number=Plur|Person=2",
    "onlar":  "Case=Nom|Number=Plur|Person=3",
    "bu":     "Case=Nom|Number=Sing|Person=3",
    "şu":     "Case=Nom|Number=Sing|Person=3",
    "bunlar": "Case=Nom|Number=Plur|Person=3",
    "şunlar": "Case=Nom|Number=Plur|Person=3",
    "kim":    "Case=Nom|Number=Sing|Person=3",
}

# UPOS tayini için sözcük listeleri
_CCONJ_WORDS: set = {"ve", "ama", "fakat", "veya", "ya", "ki", "çünkü",
                     "ancak", "hem", "lakin", "dahi"}
_ADP_WORDS:   set = {"için", "gibi", "kadar", "göre", "sonra", "önce", "beri",
                     "karşı", "doğru", "itibaren", "rağmen", "ile", "olarak",
                     "dair", "üzere", "başka", "dolayı"}
_ADV_WORDS:   set = {"çok", "az", "daha", "en", "hep", "hiç", "zaten",
                     "artık", "sadece", "bile", "nasıl", "neden", "nerede",
                     "yine", "pek", "gayet", "oldukça", "ne", "şimdi",
                     "hemen", "çabuk", "yavaş", "yüksek", "alçak", "gerçekten",
                     "kesinlikle", "muhtemelen", "genellikle", "bazen", "sık",
                     "tabii", "elbette", "hakikaten"}
_DET_WORDS:   set = {"bir", "bu", "şu", "o", "her", "bazı", "tüm",
                     "bütün", "diğer", "hangi", "hiçbir", "herhangi",
                     "birkaç", "birçok"}
_PRON_WORDS:  set = {"ben", "sen", "o", "biz", "siz", "onlar",
                     "bu", "şu", "bunlar", "şunlar", "kim", "ne",
                     "kendisi", "kendileri", "kendi", "hepsi", "biri"}
_PART_WORDS:  set = {"da", "de", "mı", "mi", "mu", "mü", "bile",
                     "ise", "diye", "değil"}
_PUNCT_CHARS: set = {".", ",", ";", ":", "!", "?", "...", "—", "-",
                     '"', "'", "(", ")", "[", "]", "{", "}"}


# BOUN UD treebank sözlüğüne göre — Basit, yaklaşık (gerçek morfolojik analizör değil)
_SUFFIX_RULES: List[Tuple[str, List[str]]] = [
    # Sıra: uzundan kısaya (kod en uzun eşleşmeyi seçer)

    # ── Olumsuz gelecek zaman (-meyecek/-mayacak + kişi ekleri) ──
    ("meyeceğim|mayacağım",
     ["Aspect=Prosp|Number=Sing|Person=1|Polarity=Neg|Tense=Fut"]),
    ("meyeceksin|mayacaksın",
     ["Aspect=Prosp|Number=Sing|Person=2|Polarity=Neg|Tense=Fut"]),
    ("meyecekler|mayacaklar",
     ["Aspect=Prosp|Number=Plur|Person=3|Polarity=Neg|Tense=Fut"]),
    ("meyecek|mayacak",
     ["Aspect=Prosp|Number=Sing|Person=3|Polarity=Neg|Tense=Fut"]),

    # ── Olumsuz şimdiki-geçmiş (-miyordu/-mıyordu + kişi ekleri) ──
    ("miyordum|mıyordum|muyordum|müyordum",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=1|Polarity=Neg|Tense=Past"]),
    ("miyordun|mıyordun|muyordun|müyordun",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=2|Polarity=Neg|Tense=Past"]),
    ("miyordu|mıyordu|muyordu|müyordu",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=3|Polarity=Neg|Tense=Past"]),

    # ── Olumsuz şimdiki zaman (-miyor/-mıyor + kişi ekleri) ──
    ("miyorum|mıyorum|muyorum|müyorum",
     ["Aspect=Prog|Number=Sing|Person=1|Polarity=Neg|Tense=Pres"]),
    ("miyorsun|mıyorsun|muyorsun|müyorsun",
     ["Aspect=Prog|Number=Sing|Person=2|Polarity=Neg|Tense=Pres"]),
    ("miyor|mıyor|muyor|müyor",
     ["Aspect=Prog|Number=Sing|Person=3|Polarity=Neg|Tense=Pres",
      "Aspect=Prog|Polarity=Neg|VerbForm=Part"]),

    # ── Olumsuz geçmiş (-medi/-madı + kişi ekleri) ──
    ("mediler|madılar",
     ["Aspect=Perf|Evident=Fh|Number=Plur|Person=3|Polarity=Neg|Tense=Past"]),
    ("medim|madım",
     ["Aspect=Perf|Evident=Fh|Number=Sing|Person=1|Polarity=Neg|Tense=Past"]),
    ("medin|madın",
     ["Aspect=Perf|Evident=Fh|Number=Sing|Person=2|Polarity=Neg|Tense=Past"]),
    ("medi|madı",
     ["Aspect=Perf|Evident=Fh|Number=Sing|Person=3|Polarity=Neg|Tense=Past"]),

    # ── Olumsuz partizip (-mediği/-madığı vb.) ──
    ("mediği|madığı|muduğu|müdüğü",
     ["Aspect=Perf|Number[psor]=Sing|Person[psor]=3|Polarity=Neg|Tense=Past|VerbForm=Part",
      "Aspect=Perf|Case=Nom|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3|Polarity=Neg|Tense=Past|VerbForm=Part"]),

    # ── Partizip + iyelik (sıfat-fiil: -dığı/-diği/-duğu/-düğü vb.) ──
    ("dığım|diğim|duğum|düğüm|tığım|tiğim|tuğum|tüğüm",
     ["Aspect=Perf|Number[psor]=Sing|Person[psor]=1|Polarity=Pos|Tense=Past|VerbForm=Part"]),
    ("dığın|diğin|duğun|düğün|tığın|tiğin|tuğun|tüğün",
     ["Aspect=Perf|Number[psor]=Sing|Person[psor]=2|Polarity=Pos|Tense=Past|VerbForm=Part"]),
    ("dığı|diği|duğu|düğü|tığı|tiği|tuğu|tüğü",
     ["Aspect=Perf|Number[psor]=Sing|Person[psor]=3|Polarity=Pos|Tense=Past|VerbForm=Part",
      "Aspect=Perf|Case=Nom|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3|Polarity=Pos|Tense=Past|VerbForm=Part"]),

    # ── Çoğul geçmiş zaman (-dılar/-tılar) ──
    ("dılar|diler|tılar|tiler|dular|düler|tular|tüler",
     ["Aspect=Perf|Evident=Fh|Number=Plur|Person=3|Polarity=Pos|Tense=Past"]),

    # ── Şimdiki-geçmiş zaman (-ıyordu/-iyordu + kişi ekleri) ──
    ("ıyordum|iyordum|uyordum|üyordum",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=1|Polarity=Pos|Tense=Past"]),
    ("ıyordun|iyordun|uyordun|üyordun",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=2|Polarity=Pos|Tense=Past"]),
    ("ıyordu|iyordu|uyordu|üyordu",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=3|Polarity=Pos|Tense=Past"]),

    # ── Geçmiş zaman tekil (-dım/-dın/-dı) ──
    ("dım|dim|dum|düm|tım|tim|tum|tüm",
     ["Aspect=Perf|Evident=Fh|Number=Sing|Person=1|Polarity=Pos|Tense=Past"]),
    ("dın|din|dun|dün|tın|tin|tun|tün",
     ["Aspect=Perf|Evident=Fh|Number=Sing|Person=2|Polarity=Pos|Tense=Past"]),
    ("dı|di|du|dü|tı|ti|tu|tü",
     ["Aspect=Perf|Evident=Fh|Number=Sing|Person=3|Polarity=Pos|Tense=Past"]),

    # ── Şimdiki zaman (-ıyorum/-iyorum + kişi ekleri) ──
    ("ıyorum|iyorum|uyorum|üyorum",
     ["Aspect=Prog|Number=Sing|Person=1|Polarity=Pos|Tense=Pres"]),
    ("ıyorsun|iyorsun|uyorsun|üyorsun",
     ["Aspect=Prog|Number=Sing|Person=2|Polarity=Pos|Tense=Pres"]),
    ("ıyor|iyor|uyor|üyor",
     ["Aspect=Prog|Number=Sing|Person=3|Polarity=Pos|Tense=Pres",
      "Aspect=Prog|Polarity=Pos|VerbForm=Part"]),

    # ── Rivayet geçmiş + kişi ekleri (-mış/-miş/-muş/-müş) ──
    ("mışlar|mişler|muşlar|müşler",
     ["Evident=Nfh|Number=Plur|Person=3|Polarity=Pos|Tense=Past"]),
    ("mışsın|mişsin|muşsun|müşsün",
     ["Evident=Nfh|Number=Sing|Person=2|Polarity=Pos|Tense=Past"]),
    ("mışım|mişim|muşum|müşüm",
     ["Evident=Nfh|Number=Sing|Person=1|Polarity=Pos|Tense=Past"]),
    ("mış|miş|muş|müş",
     ["Evident=Nfh|Number=Sing|Person=3|Polarity=Pos|Tense=Past",
      "Aspect=Perf|Polarity=Pos|VerbForm=Part"]),

    # ── Gelecek zaman + kişi ekleri (-ecek/-acak) ──
    ("ecekler|acaklar",
     ["Aspect=Prosp|Number=Plur|Person=3|Polarity=Pos|Tense=Fut"]),
    ("eceğim|acağım",
     ["Aspect=Prosp|Number=Sing|Person=1|Polarity=Pos|Tense=Fut"]),
    ("eceksin|acaksın",
     ["Aspect=Prosp|Number=Sing|Person=2|Polarity=Pos|Tense=Fut"]),
    ("eceği|acağı",
     ["Aspect=Prosp|Number[psor]=Sing|Person[psor]=3|Polarity=Pos|Tense=Fut|VerbForm=Part"]),
    ("ecek|acak",
     ["Aspect=Prosp|Number=Sing|Person=3|Polarity=Pos|Tense=Fut",
      "Aspect=Prosp|Polarity=Pos|Tense=Fut|VerbForm=Part"]),

    # ── Geniş zaman çoğul (-arlar/-erler) ──
    ("arlar|erler|ırlar|irler|urlar|ürler",
     ["Aspect=Hab|Number=Plur|Person=3|Polarity=Pos|Tense=Pres",
      "Case=Nom|Number=Plur|Person=3"]),

    # ── Geniş zaman 1.tekil (-arım/-erim) ──
    ("arım|erim|ırım|irim|urum|ürüm",
     ["Aspect=Hab|Number=Sing|Person=1|Polarity=Pos|Tense=Pres"]),

    # ── Geniş zaman 2.tekil (-arsın/-ersin) ──
    ("arsın|ersin|ırsın|irsin|ursun|ürsün",
     ["Aspect=Hab|Number=Sing|Person=2|Polarity=Pos|Tense=Pres"]),

    # ── Gereklilik kipi (-meli/-malı + kişi ekleri) ──
    ("meliyim|malıyım",
     ["Mood=Nec|Number=Sing|Person=1|Polarity=Pos"]),
    ("melisin|malısın",
     ["Mood=Nec|Number=Sing|Person=2|Polarity=Pos"]),
    ("meliler|malılar",
     ["Mood=Nec|Number=Plur|Person=3|Polarity=Pos"]),
    ("meli|malı",
     ["Mood=Nec|Number=Sing|Person=3|Polarity=Pos"]),

    # ── Mastar (-mak/-mek) ──
    ("mak|mek",
     ["Case=Nom|Number=Sing|Person=3|Polarity=Pos|VerbForm=Vnoun"]),

    # ── Geniş zaman olumsuz (-maz/-mez) ──
    ("maz|mez",
     ["Aspect=Hab|Number=Sing|Person=3|Polarity=Neg|Tense=Pres",
      "Aspect=Hab|Polarity=Neg|VerbForm=Part"]),

    # ── Edilgen geçmiş (-ıldı/-ildi) ──
    ("ıldı|ildi|uldu|üldü",
     ["Aspect=Perf|Evident=Fh|Number=Sing|Person=3|Polarity=Pos|Tense=Past|Voice=Pass"]),

    # ── Zarf-fiil (-arak/-erek) ──
    ("arak|erek",
     ["Polarity=Pos|VerbForm=Conv",
      "Aspect=Prog|Polarity=Pos|VerbForm=Conv"]),

    # ── Zarf-fiil (-madan/-meden) ──
    ("madan|meden",
     ["Case=Abl|Number=Sing|Person=3|Polarity=Pos|VerbForm=Conv"]),

    # ── Zarf-fiil (-ınca/-ince/-unca/-ünce) ──
    ("ınca|ince|unca|ünce",
     ["Polarity=Pos|VerbForm=Conv",
      "Case=Nom|Number=Sing|Person=3"]),

    # ── Zarf-fiil (-dıkça/-dikçe vb.) ──
    ("dıkça|dikçe|dukça|dükçe|tıkça|tikçe|tukça|tükçe",
     ["Polarity=Pos|VerbForm=Conv"]),

    # ── Zarf-fiil (-ıp/-ip) ──
    ("ıp|ip|up|üp",
     ["Polarity=Pos|VerbForm=Conv"]),

    # ── Yeterlilik kipi (-ebilir/-abilir + kişi ekleri) ── Mood=Pot
    ("ebilirsiniz|abilirsiniz",
     ["Aspect=Hab|Mood=Pot|Number=Plur|Person=2|Polarity=Pos|Tense=Pres"]),
    ("ebilirler|abilirler",
     ["Aspect=Hab|Mood=Pot|Number=Plur|Person=3|Polarity=Pos|Tense=Pres"]),
    ("ebiliriz|abiliriz",
     ["Aspect=Hab|Mood=Pot|Number=Plur|Person=1|Polarity=Pos|Tense=Pres"]),
    ("ebilirsin|abilirsin",
     ["Aspect=Hab|Mood=Pot|Number=Sing|Person=2|Polarity=Pos|Tense=Pres"]),
    ("ebilirim|abilirim",
     ["Aspect=Hab|Mood=Pot|Number=Sing|Person=1|Polarity=Pos|Tense=Pres"]),
    ("ebilmek|abilmek",
     ["Case=Nom|Mood=Pot|Number=Sing|Person=3|Polarity=Pos|VerbForm=Vnoun"]),
    ("ebilir|abilir",
     ["Aspect=Hab|Mood=Pot|Number=Sing|Person=3|Polarity=Pos|Tense=Pres"]),
    ("emez|amaz",
     ["Aspect=Hab|Mood=Pot|Number=Sing|Person=3|Polarity=Neg|Tense=Pres",
      "Aspect=Hab|Mood=Pot|Polarity=Neg|VerbForm=Part"]),

    # ── Ettirgen uzun formlar (-tırmak/-dırmak) ── Voice=Cau
    ("tırmak|dırmak|tirmek|dirmek|turmak|durmak|türmek|dürmek",
     ["Case=Nom|Number=Sing|Person=3|Polarity=Pos|VerbForm=Vnoun|Voice=Cau"]),
    ("tıran|dıran|tiren|diren|turan|duran|türen|düren",
     ["Polarity=Pos|Tense=Pres|VerbForm=Part|Voice=Cau"]),
    ("tırın|dırın|tirin|dirin|turun|durun|türün|dürün",
     ["Mood=Imp|Number=Plur|Person=2|Polarity=Pos|Voice=Cau"]),
    ("tırarak|dırarak|tiretek|diretek",
     ["Polarity=Pos|VerbForm=Conv|Voice=Cau"]),

    # ── Edilgen - geniş formlar ── Voice=Pass
    ("ıldığında|ildiğinde|ulduğunda|üldüğünde",
     ["Aspect=Perf|Case=Loc|Evident=Fh|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3|Polarity=Pos|Tense=Past|VerbForm=Part|Voice=Pass"]),
    ("ılmaktadır|ilmektedir",
     ["Aspect=Prog|Mood=Gen|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Vnoun|Voice=Pass"]),

    # ── Şimdiki zaman -makta/-mekte aktif formları (Aspect=Prog) ──
    ("maktadırlar|mektedirler",
     ["Aspect=Prog|Mood=Gen|Number=Plur|Person=3|Polarity=Pos|Tense=Pres"]),
    ("maktadır|mektedir",
     ["Aspect=Prog|Mood=Gen|Number=Sing|Person=3|Polarity=Pos|Tense=Pres"]),
    ("maktaydım|mekteydim",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=1|Polarity=Pos|Tense=Past"]),
    ("maktaydın|mekteydin",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=2|Polarity=Pos|Tense=Past"]),
    ("maktaydı|mekteydi",
     ["Aspect=Prog|Evident=Fh|Number=Sing|Person=3|Polarity=Pos|Tense=Past"]),
    ("maktayım|mekteyim",
     ["Aspect=Prog|Number=Sing|Person=1|Polarity=Pos|Tense=Pres"]),
    ("maktasın|mekteysin",
     ["Aspect=Prog|Number=Sing|Person=2|Polarity=Pos|Tense=Pres"]),
    ("maktalar|mekteler",
     ["Aspect=Prog|Number=Plur|Person=3|Polarity=Pos|Tense=Pres"]),
    ("makta|mekte",
     ["Aspect=Prog|Number=Sing|Person=3|Polarity=Pos|Tense=Pres",
      "Case=Loc|Number=Sing|Person=3|Polarity=Pos|VerbForm=Vnoun"]),
    ("ılanlar|ilenler|ulanlar|ülenler",
     ["Case=Nom|Number=Plur|Person=3|Polarity=Pos|VerbForm=Part|Voice=Pass"]),
    ("ılanın|ilenin|ulanın|ülenin",
     ["Case=Gen|Number=Sing|Person=3|Polarity=Pos|VerbForm=Part|Voice=Pass"]),
    ("ılmış|ilmiş|ulmuş|ülmüş",
     ["Aspect=Perf|Polarity=Pos|VerbForm=Part|Voice=Pass"]),
    ("ılan|ilen|ulan|ülen",
     ["Polarity=Pos|Tense=Pres|VerbForm=Part|Voice=Pass"]),
    ("ılır|ilir|ulur|ülür",
     ["Aspect=Hab|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|Voice=Pass"]),
    ("ılmak|ilmek|ulmak|ülmek",
     ["Case=Nom|Number=Sing|Person=3|Polarity=Pos|VerbForm=Vnoun|Voice=Pass"]),
    ("ılarak|ilerek|ularak|ülerek",
     ["Polarity=Pos|VerbForm=Conv|Voice=Pass"]),
    ("ınan|inen|unan|ünen",
     ["Polarity=Pos|Tense=Pres|VerbForm=Part|Voice=Rfl"]),

    # ── Koşul kipi (-sa/-se + kişi ekleri) ── Mood=Cnd / Mood=Des
    ("sanız|seniz|sanız|sunuz|sünüz",
     ["Mood=Cnd|Number=Plur|Person=2|Polarity=Pos"]),
    ("salar|seler",
     ["Mood=Cnd|Number=Plur|Person=3|Polarity=Pos"]),
    ("sak|sek",
     ["Mood=Cnd|Number=Plur|Person=1|Polarity=Pos"]),
    ("sam|sem",
     ["Mood=Cnd|Number=Sing|Person=1|Polarity=Pos"]),
    ("san|sen",
     ["Mood=Cnd|Number=Sing|Person=2|Polarity=Pos"]),
    ("ysa|yse",
     ["Aspect=Perf|Mood=Cnd|Number=Sing|Person=3|Tense=Pres",
      "Mood=Des|Number=Sing|Person=3|Polarity=Pos"]),
    ("sa|se",
     ["Mood=Cnd|Number=Sing|Person=3|Polarity=Pos",
      "Mood=Des|Number=Sing|Person=3|Polarity=Pos"]),

    # ── İstek kipi (-alım/-elim, -ayım/-eyim) ── Mood=Opt
    ("alım|elim",
     ["Mood=Opt|Number=Plur|Person=1|Polarity=Pos"]),
    ("ayım|eyim",
     ["Mood=Opt|Number=Sing|Person=1|Polarity=Pos"]),

    # ── Geniş zaman -dır/-dir (Mood=Gen, evidential copula) ──
    ("dırlar|dirler|durlar|dürler|tırlar|tirler|turlar|türler",
     ["Aspect=Perf|Mood=Gen|Number=Plur|Person=3|Tense=Pres"]),
    ("dır|dir|dur|dür|tır|tir|tur|tür",
     ["Aspect=Perf|Mood=Gen|Number=Sing|Person=3|Tense=Pres"]),

    # ── Emir kipi 3.tekil/çoğul (-sın/-sin) ──
    ("sınlar|sinler|sunlar|sünler",
     ["Mood=Imp|Number=Plur|Person=3|Polarity=Pos"]),
    ("sın|sin|sun|sün",
     ["Mood=Imp|Number=Sing|Person=3|Polarity=Pos"]),

    # ── Emir kipi 2. çoğul (yın/yin/yun/yün; ınız/iniz/unuz/ünüz) ──
    # pişirin, yapın, bırakın → ın (2ch); doğrayın, söyleyin → yın (3ch)
    # yapınız, gidiniz, alınız → ınız (4ch, formal/polite)
    ("yınız|yiniz|yunuz|yünüz",
     ["Mood=Imp|Number=Plur|Person=2|Polarity=Pos"]),
    ("ınız|iniz|unuz|ünüz",
     ["Mood=Imp|Number=Plur|Person=2|Polarity=Pos",
      "Case=Gen|Number=Sing|Person=3"]),
    ("yın|yin|yun|yün",
     ["Mood=Imp|Number=Plur|Person=2|Polarity=Pos"]),

    # ── Çoğul isim (-ler/-lar) ──
    ("ler|lar",
     ["Case=Nom|Number=Plur|Person=3"]),
    # ── Tamlayan eki (-nın/-nin/-ın/-in) ve 2. çoğul emir (-ın/-in) ──
    ("nın|nin|nun|nün",
     ["Case=Gen|Number=Sing|Person=3",
      "Case=Gen|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3"]),
    ("ın|in|un|ün",
     ["Case=Gen|Number=Sing|Person=3",
      "Case=Gen|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3",
      "Mood=Imp|Number=Plur|Person=2|Polarity=Pos"]),
    # ── Bulunma (-nda/-nde/-da/-de/-ta/-te) ──
    ("nda|nde|da|de|ta|te",
     ["Case=Loc|Number=Sing|Person=3"]),
    # ── Ayrılma (-ndan/-nden/-dan/-den/-tan/-ten) ──
    ("ndan|nden|dan|den|tan|ten",
     ["Case=Abl|Number=Sing|Person=3"]),
    # ── Yönelme (-na/-ne/-a/-e) ──
    ("na|ne|a|e",
     ["Case=Dat|Number=Sing|Person=3"]),
    # ── İyelik 3.tekil (-sı/-si/-su/-sü) ──
    ("sı|si|su|sü",
     ["Case=Nom|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3"]),
    # ── Belirtme (-nı/-ni/-yı/-yi/-ı/-i) ──
    ("nı|ni|nu|nü|yı|yi|yu|yü|ı|i|u|ü",
     ["Case=Acc|Number=Sing|Person=3"]),
    # ── Araç eki (-yla/-yle/-la/-le) ──
    ("yla|yle|la|le",
     ["Case=Ins|Number=Sing|Person=3"]),
]

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



_LEMMA_SUFFIX_STRIP: List[Tuple[str, int]] = sorted([
    # 3sg POSS + CASE kombinasyonları (compound)
    ("sından", 6), ("sinden", 6), ("sundan", 6), ("sünden", 6),
    ("sında",  5), ("sinde",  5), ("sunda",  5), ("sünde",  5),
    ("sına",   4), ("sine",   4), ("suna",   4), ("süne",   4),
    ("sını",   4), ("sini",   4), ("sunu",   4), ("sünü",   4),
    ("sının",  5), ("sinin",  5), ("sunun",  5), ("sünün",  5),
    # 3sg POSS + suffix (vowel-initial POSS)
    ("ından",  5), ("inden",  5), ("undan",  5), ("ünden",  5),
    ("ında",   4), ("inde",   4), ("unda",   4), ("ünde",   4),
    ("ına",    3), ("ine",    3), ("una",    3), ("üne",    3),
    ("ını",    3), ("ini",    3), ("unu",    3), ("ünü",    3),
    ("ının",   4), ("inin",   4), ("unun",   4), ("ünün",   4),
    ("ıyla",   4), ("iyle",   4), ("uyla",   4), ("üyle",   4),
    # Plural + CASE
    ("lardan", 6), ("lerden", 6),
    ("larda",  5), ("lerde",  5),
    ("larla",  5), ("lerle",  5),
    ("larca",  5), ("lerce",  5),
    ("lara",   4), ("lere",   4),
    ("ları",   4), ("leri",   4),
    ("larca",  5), ("lerce",  5),
    ("ların",  4), ("lerin",  4),
    # Verb: negative progressive
    ("miyordum",  8), ("mıyordum",  8), ("muyordum",  8), ("müyordum",  8),
    ("miyorum",   7), ("mıyorum",   7), ("muyorum",   7), ("müyorum",   7),
    ("miyorsun",  8), ("mıyorsun",  8), ("muyorsun",  8), ("müyorsun",  8),
    ("miyoruz",   7), ("mıyoruz",   7), ("muyoruz",   7), ("müyoruz",   7),
    ("miyorlar",  9), ("mıyorlar",  9),
    ("miyor",     5), ("mıyor",     5), ("muyor",     5), ("müyor",     5),
    # Verb: negative past
    ("mediler",   7), ("madılar",   7),
    ("medim",     5), ("madım",     5),
    ("medin",     5), ("madın",     5),
    ("medi",      4), ("madı",      4),
    # Verb: progressive
    ("ıyordum",   7), ("iyordum",   7), ("uyordum",   7), ("üyordum",   7),
    ("ıyorum",    6), ("iyorum",    6), ("uyorum",    6), ("üyorum",    6),
    ("ıyorsun",   7), ("iyorsun",   7), ("uyorsun",   7), ("üyorsun",   7),
    ("ıyoruz",    6), ("iyoruz",    6), ("uyoruz",    6), ("üyoruz",    6),
    ("ıyor",      4), ("iyor",      4), ("uyor",      4), ("üyor",      4),
    # Verb: past
    ("dılar",  5), ("diler",  5), ("tılar",  5), ("tiler",  5),
    ("dular",  5), ("düler",  5), ("tular",  5), ("tüler",  5),
    ("dım",    3), ("dim",    3), ("dum",    3), ("düm",    3),
    ("tım",    3), ("tim",    3), ("tum",    3), ("tüm",    3),
    ("dın",    3), ("din",    3), ("dun",    3), ("dün",    3),
    ("tın",    3), ("tin",    3), ("tun",    3), ("tün",    3),
    ("dı",     2), ("di",     2), ("du",     2), ("dü",     2),
    ("tı",     2), ("ti",     2), ("tu",     2), ("tü",     2),
    # Participial
    ("dığım",  5), ("diğim",  5), ("duğum",  5), ("düğüm",  5),
    ("tığım",  5), ("tiğim",  5), ("tuğum",  5), ("tüğüm",  5),
    ("dığın",  5), ("diğin",  5), ("duğun",  5), ("düğün",  5),
    ("tığın",  5), ("tiğin",  5), ("tuğun",  5), ("tüğün",  5),
    ("dığı",   4), ("diği",   4), ("duğu",   4), ("düğü",   4),
    ("tığı",   4), ("tiği",   4), ("tuğu",   4), ("tüğü",   4),
    # Ablative
    ("ndan",   4), ("nden",   4),
    ("dan",    3), ("den",    3), ("tan",    3), ("ten",    3),
    # Locative
    ("nda",    3), ("nde",    3),
    ("da",     2), ("de",     2), ("ta",     2), ("te",     2),
    # Genitive
    ("nın",    3), ("nin",    3), ("nun",    3), ("nün",    3),
    ("ın",     2), ("in",     2), ("un",     2), ("ün",     2),
    # Dative
    ("na",     2), ("ne",     2), ("ya",     2), ("ye",     2),
    ("a",      1), ("e",      1),
    # Plural
    ("lar",    3), ("ler",    3),
    # Possessive 3sg
    ("sı",     2), ("si",     2), ("su",     2), ("sü",     2),
    # Accusative
    ("nı",     2), ("ni",     2), ("nu",     2), ("nü",     2),
    ("yı",     2), ("yi",     2), ("yu",     2), ("yü",     2),
    ("ı",      1), ("i",      1), ("u",      1), ("ü",      1),
    # Instrumental
    ("yla",    3), ("yle",    3), ("la",     2), ("le",     2),
    # Verb: past progressive (missing entries)
    ("ıyordu",    6), ("iyordu",    6), ("uyordu",    6), ("üyordu",    6),
    ("miyordu",   7), ("mıyordu",   7), ("muyordu",   7), ("müyordu",   7),
    ("ıyordun",   7), ("iyordun",   7), ("uyordun",   7), ("üyordun",   7),
    ("miyordun",  8), ("mıyordun",  8), ("muyordun",  8), ("müyordun",  8),
    # Verb: evidential past (-mış/-miş/-muş/-müş)
    ("mışlardı",  8), ("mişlerdi",  8),
    ("mışlar",    6), ("mişler",    6), ("muşlar",    6), ("müşler",    6),
    ("mışsın",    6), ("mişsin",    6), ("muşsun",    6), ("müşsün",    6),
    ("mışım",     5), ("mişim",     5), ("muşum",     5), ("müşüm",     5),
    ("mış",       3), ("miş",       3), ("muş",       3), ("müş",       3),
    # Verb: future (-ecek/-acak)
    ("meyecekler", 10), ("mayacaklar", 10),
    ("meyeceğim",   9), ("mayacağım",   9),
    ("meyeceksin",  10), ("mayacaksın", 10),
    ("meyecek",     7), ("mayacak",     7),
    ("ecekler",     7), ("acaklar",     7),
    ("eceğim",      6), ("acağım",      6),
    ("eceksin",     7), ("acaksın",     7),
    ("eceği",       5), ("acağı",       5),
    ("ecek",        4), ("acak",        4),
    # Verb: aorist / negative aorist
    ("arlar",  5), ("erler",  5), ("ırlar",  5), ("irler",  5),
    ("arım",   4), ("erim",   4), ("ırım",   4), ("irim",   4),
    ("arsın",  5), ("ersin",  5), ("ırsın",  5), ("irsin",  5),
    ("maz",    3), ("mez",    3),
    # Verb: obligative (-meli/-malı)
    ("meliyim", 7), ("malıyım", 7),
    ("melisin", 7), ("malısın", 7),
    ("meliler", 7), ("malılar", 7),
    ("meli",    4), ("malı",    4),
    # Verb: infinitive/vnoun (-mak/-mek)
    ("mak",    3), ("mek",    3),
    # Verb: passive past (-ıldı/-ildi)
    ("ıldı",   4), ("ildi",   4), ("uldu",   4), ("üldü",   4),
    # Verb: converbs
    ("madan",  5), ("meden",  5),
    ("arak",   4), ("erek",   4),
    ("ınca",   4), ("ince",   4), ("unca",   4), ("ünce",   4),
    ("dıkça",  5), ("dikçe",  5), ("dukça",  5), ("dükçe",  5),
    ("tıkça",  5), ("tikçe",  5), ("tukça",  5), ("tükçe",  5),
    ("ıp",     2), ("ip",     2), ("up",     2), ("üp",     2),
    # Verb: imperative
    ("sınlar", 6), ("sinler", 6), ("sunlar", 6), ("sünler", 6),
    ("sın",    3), ("sin",    3), ("sun",    3), ("sün",    3),
], key=lambda x: -len(x[0]))


_CONSONANT_DEVOICE: Dict[str, str] = {
    "b": "p", "c": "ç", "g": "k", "ğ": "k"
    # "d" → "t" yalnızca fiil kökleri için (aşağıda koşullu)
}

# Zamir biçimleri → kökü
_PRONOUN_LEMMA: Dict[str, str] = {
    "benim": "ben", "bana": "ben", "beni": "ben",
    "bende": "ben", "benden": "ben", "benimle": "ben",
    "senin": "sen", "sana": "sen", "seni": "sen",
    "sende": "sen", "senden": "sen",
    "onun": "o",  "ona": "o",   "onu": "o",
    "onda": "o",  "ondan": "o", "onunla": "o",
    "bizim": "biz", "bize": "biz", "bizi": "biz",
    "bizde": "biz", "bizden": "biz",
    "sizin": "siz", "size": "siz", "sizi": "siz",
    "onların": "onlar", "onlara": "onlar",
    "onları": "onlar",  "onlarda": "onlar",
    "bunun": "bu", "buna": "bu", "bunu": "bu",
    "bunda": "bu", "bundan": "bu",
    "şunun": "şu", "şuna": "şu", "şunu": "şu",
}
