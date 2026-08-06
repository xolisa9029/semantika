# Unigram Tokenizer - Huquqiy Matnlar Uchun

Bu loyiha huquqiy sohaga tegishli matnlar to'plamidan foydalanib **Unigram Tokenizer** algoritmi asosida tokenizer yaratadi va uni test qilish imkonini beradi.

## Fayllar tuzilishi

- `unigram_tokenizer.py` - Tokenizerni o'qitish va saqlash uchun asosiy skript
- `test_tokenizer.py` - Saqlangan tokenizerni test qilish uchun skript
- `tokenizer_vocab.json` - O'qitilgan tokenizer lug'ati (3000 ta token)

## Ishlatish

### 1. Tokenizerni o'qitish

```bash
python3 unigram_tokenizer.py
```

Bu skript:
- `/workspace` papkasidagi `.txt` fayllardan huquqiy matnlarni yuklaydi
- Unigram algoritmi yordamida eng ko'p ishlatiladigan tokenlarni ajratib oladi
- 3000 ta tokendan iborat lug'at yaratadi
- Lug'atni `tokenizer_vocab.json` fayliga saqlaydi
- Bir nechta test matnlari orqali tokenizerni sinovdan o'tkazadi

### 2. Tokenizerni test qilish

```bash
python3 test_tokenizer.py
```

Bu skript:
- Saqlangan `tokenizer_vocab.json` faylidan lug'atni yuklaydi
- Bir nechta test matnlari orqali tokenizerni sinovdan o'tkazadi
- Interaktiv rejimda istalgan matnni tokenlarga ajratish imkonini beradi

## Tokenizer xususiyatlari

- **Unigram algoritmi**: Matndan eng ko'p ishlatiladigan substring'larni (tokenlarni) ajratib oladi
- **Greedy tokenizatsiya**: Matnni tokenlarga ajratishda eng uzun mos keluvchi tokenni tanlaydi
- **Lug'at hajmi**: 3000 ta token (sozlanishi mumkin)
- **Minimal chastota**: Har bir token kamida 5 marta uchrashi kerak (sozlanishi mumkin)

## Natijalar

O'qitish natijasida quyidagi kabi eng ko'p ishlatiladigan tokenlar ajratib olindi:

| # | Token | Chastota |
|---|-------|----------|
| 1 | a | 1,077,042 |
| 2 | i | 1,004,026 |
| 3 | n | 479,157 |
| 4 | o | 450,788 |
| 5 | l | 439,879 |
| ... | ... | ... |
| 17 | la | 180,609 |
| 19 | ar | 173,937 |
| 20 | an | 155,003 |
| 28 | lar | 97,506 |
| 50 | ish | 58,361 |

## Misol

```python
from unigram_tokenizer import UnigramTokenizer

# Tokenizerni yaratish
tokenizer = UnigramTokenizer(vocab_size=3000, min_frequency=5)

# Matnlarni yuklash va o'qitish
texts = ["Huquqiy hujjat", "Sud qarori", ...]
tokenizer.train(texts)

# Tokenlarga ajratish
tokens = tokenizer.tokenize("Dushanba kuni sud qarori e'lon qilindi.")
print(tokens)
# Natija: ['dushanba', ' ', 'kuni', ' ', 'sud', ' ', 'qaror', 'i', ...]

# Token ID'larni olish
token_ids = tokenizer.tokenize("Matn", return_ids=True)
print(token_ids)
# Natija: [123, 456, ...]

# Detokenizatsiya
text = tokenizer.detokenize(token_ids)
print(text)
# Natija: "matn"

# Lug'atni saqlash/yuklash
tokenizer.save_vocab('vocab.json')
tokenizer.load_vocab('vocab.json')
```

## Talablar

- Python 3.6+
- Hech qanday tashqi kutubxona talab qilinmaydi (faqat standart kutubxonalar)

## Muallif

Huquqiy matnlar to'plami asosida yaratilgan.
