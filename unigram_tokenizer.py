#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unigram Tokenizer - Huquqiy matnlar uchun tokenizer yaratish va test qilish
Optimizatsiya qilingan versiya - katta hajmli ma'lumotlar uchun
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Iterator
import math


class UnigramTokenizer:
    """
    Unigram Tokenizer implementatsiyasi.
    Bu tokenizer matnni eng ko'p ishlatiladigan tokenlarga ajratadi.
    """
    
    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2):
        """
        Tokenizerni ishga tushirish.
        
        Args:
            vocab_size: Lug'at hajmi (nechta token saqlash)
            min_frequency: Minimal chastota (shu chastotadan kam bo'lsa olinmaydi)
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab: Dict[str, int] = {}  # token -> id
        self.id_to_token: Dict[int, str] = {}  # id -> token
        self.token_frequencies: Counter = Counter()
        self.is_trained = False
    
    def _preprocess_text(self, text: str) -> str:
        """Matnni oldindan ishlash."""
        # Kichik harflarga o'tkazish
        text = text.lower()
        return text
    
    def _extract_substrings_from_word(self, word: str, max_length: int = 10) -> List[str]:
        """Bitta so'zdan barcha mumkin bo'lgan substring'larni ajratib olish."""
        tokens = []
        for i in range(len(word)):
            for j in range(i + 1, min(i + max_length, len(word) + 1)):
                tokens.append(word[i:j])
        return tokens
    
    def _stream_substrings(self, texts: List[str], batch_size: int = 1000) -> Iterator[str]:
        """Matnlardan substring'larni generator sifatida olish."""
        for text in texts:
            text = self._preprocess_text(text)
            words = re.findall(r'\w+', text, flags=re.UNICODE)
            
            for word in words:
                # Har bir so'zdan substring'larni olish
                for i in range(len(word)):
                    for j in range(i + 1, min(i + 10, len(word) + 1)):
                        yield word[i:j]
            
            # Bo'sh joy va punctuatsiyani ham qo'shish
            spaces = re.findall(r'\s+', text, flags=re.UNICODE)
            for space in spaces:
                if len(space.strip()) == 0:
                    yield ' '
    
    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        Tokenizerni o'qitish.
        
        Args:
            texts: O'qitish uchun matnlar ro'yxati
            verbose: Jarayon haqida ma'lumot chiqarish
        """
        if verbose:
            print(f"O'qitish boshlandi... {len(texts)} ta matn")
        
        # Generator orqali substring'larni hisoblash (xotirani tejash)
        if verbose:
            print("Substring'larni ajratib olish...")
        
        # Batchlarda ishlash
        batch_size = 5000
        total_processed = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for substring in self._stream_substrings(batch):
                self.token_frequencies[substring] += 1
            total_processed += len(batch)
            if verbose and total_processed % 50000 == 0:
                print(f"  {total_processed}/{len(texts)} matn ishlandi...")
        
        # Bo'sh joy tokenini qo'shish
        self.token_frequencies[' '] = sum(1 for t in self.token_frequencies if t == ' ')
        
        # Minimal chastotadan yuqori bo'lgan tokenlarni tanlash
        filtered_tokens = [
            (token, freq) for token, freq in self.token_frequencies.items()
            if freq >= self.min_frequency and len(token.strip()) > 0
        ]
        
        # Eng ko'p ishlatiladigan tokenlarni tanlash
        filtered_tokens.sort(key=lambda x: x[1], reverse=True)
        top_tokens = filtered_tokens[:self.vocab_size]
        
        # Lug'atni yaratish
        self.vocab = {token: idx for idx, (token, _) in enumerate(top_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        self.is_trained = True
        
        if verbose:
            print(f"O'qitish tugallandi!")
            print(f"Lug'at hajmi: {len(self.vocab)} ta token")
            print(f"\nEng ko'p ishlatiladigan 20 ta token:")
            for token, freq in top_tokens[:20]:
                print(f"  '{token}': {freq} marta")
    
    def tokenize(self, text: str, return_ids: bool = False) -> List:
        """
        Matnni tokenlarga ajratish.
        
        Args:
            text: Tokenlarga ajratiladigan matn
            return_ids: Agar True bo'lsa, token ID'larini qaytaradi
            
        Returns:
            Tokenlar ro'yxati yoki token ID'lari ro'yxati
        """
        if not self.is_trained:
            raise ValueError("Tokenizer hali o'qitilmagan! train() metodini chaqiring.")
        
        # Matnni oldindan ishlash
        text = self._preprocess_text(text)
        
        # Greedy algoritm bilan tokenlarga ajratish
        tokens = []
        i = 0
        
        while i < len(text):
            # Eng uzun mos keluvchi tokenni topish
            matched = False
            for length in range(min(10, len(text) - i), 0, -1):
                substring = text[i:i + length]
                if substring in self.vocab:
                    if return_ids:
                        tokens.append(self.vocab[substring])
                    else:
                        tokens.append(substring)
                    i += length
                    matched = True
                    break
            
            # Agar hech qanday token mos kelmasa, bir belgini olish
            if not matched:
                char = text[i]
                if return_ids:
                    # Noma'lum belgi uchun -1
                    tokens.append(-1)
                else:
                    tokens.append(char)
                i += 1
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Token ID'lardan matnni qayta tiklash.
        
        Args:
            token_ids: Token ID'lari ro'yxati
            
        Returns:
            Qayta tiklangan matn
        """
        if not self.is_trained:
            raise ValueError("Tokenizer hali o'qitilmagan!")
        
        tokens = []
        for token_id in token_ids:
            if token_id == -1:
                continue  # Noma'lum tokenlarni o'tkazib yuborish
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(f"<UNK:{token_id}>")
        
        return "".join(tokens)
    
    def save_vocab(self, filepath: str) -> None:
        """Lug'atni faylga saqlash."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
                'min_frequency': self.min_frequency
            }, f, ensure_ascii=False, indent=2)
        print(f"Lug'at {filepath} fayliga saqlandi.")
    
    def load_vocab(self, filepath: str) -> None:
        """Lug'atni fayldan yuklash."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.vocab_size = data['vocab_size']
            self.min_frequency = data['min_frequency']
            self.id_to_token = {idx: token for token, idx in self.vocab.items()}
            self.is_trained = True
        print(f"Lug'at {filepath} faylidan yuklandi.")


def load_legal_texts(directory: str = '/workspace', sample_size: Optional[int] = None) -> List[str]:
    """Huquqiy matnlarni yuklash."""
    import glob
    import os
    
    texts = []
    
    # Barcha .txt fayllarni topish
    txt_files = sorted(glob.glob(os.path.join(directory, '*.txt')))
    
    for filepath in txt_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # JSON formatidagi satrlarni parse qilish
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if line.startswith('{'):
                        try:
                            # Oxiridagi vergulni olib tashlash
                            if line.endswith(','):
                                line = line[:-1]
                            data = json.loads(line)
                            # 'text' maydonini olish
                            if 'text' in data and data['text']:
                                texts.append(data['text'])
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Xatolik {filepath}: {e}")
        
        # Agar sample_size belgilangan bo'lsa va yetarli matn yuklangan bo'lsa, to'xtash
        if sample_size and len(texts) >= sample_size:
            texts = texts[:sample_size]
            break
    
    if sample_size:
        print(f"Jami {len(texts)} ta matn yuklandi (namuna: {sample_size}).")
    else:
        print(f"Jami {len(texts)} ta matn yuklandi.")
    return texts


def main():
    """Asosiy funksiya."""
    print("=" * 60)
    print("UNIGRAM TOKENIZER - HUQUQIY MATNLAR UCHUN")
    print("=" * 60)
    
    # 1. Matnlarni yuklash (birinchi 50000 ta matndan namuna)
    print("\n1. Matnlarni yuklash...")
    texts = load_legal_texts('/workspace', sample_size=50000)
    
    if len(texts) == 0:
        print("Xatolik: Hech qanday matn topilmadi!")
        return
    
    # 2. Tokenizerni yaratish va o'qitish
    print("\n2. Tokenizerni o'qitish...")
    tokenizer = UnigramTokenizer(vocab_size=3000, min_frequency=5)
    tokenizer.train(texts, verbose=True)
    
    # 3. Lug'atni saqlash
    print("\n3. Lug'atni saqlash...")
    tokenizer.save_vocab('/workspace/tokenizer_vocab.json')
    
    # 4. Test qilish
    print("\n4. Tokenizerni test qilish...")
    print("-" * 60)
    
    # Test matnlari
    test_texts = [
        "Dushanba kuni sud qarori e'lon qilindi.",
        "Avstraliya aksiyalari 6,6 foizga tushdi.",
        "Janubiy Koreyaning yadroviy elchisi muzokaralar olib bordi.",
        "Facebookning yangi ko'rinishi foydalanuvchilarga yoqdi."
    ]
    
    for i, test_text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"  Asl matn: {test_text}")
        
        # Tokenlarga ajratish
        tokens = tokenizer.tokenize(test_text)
        print(f"  Tokenlar: {' | '.join(tokens[:30])}{'...' if len(tokens) > 30 else ''}")
        
        # Token ID'larni olish
        token_ids = tokenizer.tokenize(test_text, return_ids=True)
        print(f"  Token ID'lar: {token_ids[:30]}{'...' if len(token_ids) > 30 else ''}")
        
        # Detokenizatsiya
        reconstructed = tokenizer.detokenize(token_ids)
        print(f"  Qayta tiklangan: {reconstructed}")
    
    # 5. Statistika
    print("\n" + "=" * 60)
    print("STATISTIKA")
    print("=" * 60)
    print(f"Jami o'qitilgan matnlar: {len(texts)}")
    print(f"Lug'at hajmi: {len(tokenizer.vocab)} ta token")
    print(f"Minimal chastota: {tokenizer.min_frequency}")
    
    print("\nEng ko'p ishlatiladigan 50 ta token:")
    sorted_tokens = sorted(tokenizer.token_frequencies.items(), key=lambda x: x[1], reverse=True)[:50]
    for idx, (token, freq) in enumerate(sorted_tokens, 1):
        if token in tokenizer.vocab:
            print(f"  {idx:2d}. '{token}' - {freq} marta")
    
    print("\n" + "=" * 60)
    print("Tokenizer muvaffaqiyatli yaratildi va test qilindi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
