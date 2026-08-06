#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizerni test qilish uchun alohida skript.
Bu skript saqlangan tokenizer_vocab.json faylidan foydalanadi.
"""

import json
import re
from typing import List, Dict


class UnigramTokenizer:
    """Saqlangan lug'atdan foydalanadigan Tokenizer."""
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.is_trained = False
    
    def load_vocab(self, filepath: str) -> None:
        """Lug'atni fayldan yuklash."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.id_to_token = {idx: token for token, idx in self.vocab.items()}
            self.is_trained = True
        print(f"Lug'at {filepath} faylidan yuklandi.")
        print(f"Lug'at hajmi: {len(self.vocab)} ta token")
    
    def tokenize(self, text: str, return_ids: bool = False) -> List:
        """Matnni tokenlarga ajratish."""
        if not self.is_trained:
            raise ValueError("Tokenizer hali o'qitilmagan!")
        
        # Matnni oldindan ishlash (kichik harflarga)
        text = text.lower()
        
        # Greedy algoritm bilan tokenlarga ajratish
        tokens = []
        i = 0
        
        while i < len(text):
            matched = False
            # Eng uzun mos keluvchi tokenni topish (max 10 belgi)
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
                    tokens.append(-1)  # Noma'lum belgi
                else:
                    tokens.append(char)
                i += 1
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Token ID'lardan matnni qayta tiklash."""
        if not self.is_trained:
            raise ValueError("Tokenizer hali o'qitilmagan!")
        
        tokens = []
        for token_id in token_ids:
            if token_id == -1:
                continue
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(f"<UNK:{token_id}>")
        
        return "".join(tokens)


def main():
    """Test qilish."""
    print("=" * 60)
    print("TOKENIZERNI TEST QILISH")
    print("=" * 60)
    
    # Tokenizerni yuklash
    tokenizer = UnigramTokenizer()
    tokenizer.load_vocab('/workspace/tokenizer_vocab.json')
    
    # Test matnlari
    test_texts = [
        "Dushanba kuni sud qarori e'lon qilindi.",
        "Avstraliya aksiyalari 6,6 foizga tushdi.",
        "Janubiy Koreyaning yadroviy elchisi muzokaralar olib bordi.",
        "Facebookning yangi ko'rinishi foydalanuvchilarga yoqdi.",
        "Prezident Abdulaziz Buteflika raisligidagi Jazoir vazirlar mahkamasi.",
        "Yevropa ittifoqi yetakchilari favqulodda sammitda."
    ]
    
    print("\n" + "-" * 60)
    for i, test_text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"  Asl matn: {test_text}")
        
        # Tokenlarga ajratish
        tokens = tokenizer.tokenize(test_text)
        print(f"  Tokenlar ({len(tokens)} ta): {' | '.join(tokens[:20])}{'...' if len(tokens) > 20 else ''}")
        
        # Token ID'larni olish
        token_ids = tokenizer.tokenize(test_text, return_ids=True)
        print(f"  Token ID'lar: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        
        # Detokenizatsiya
        reconstructed = tokenizer.detokenize(token_ids)
        print(f"  Qayta tiklangan: {reconstructed}")
        
        # Solishtirish
        original_lower = test_text.lower().replace(' ', '')
        reconstructed_no_space = reconstructed.replace(' ', '')
        match = "✓" if original_lower == reconstructed_no_space else "✗"
        print(f"  Moslik: {match}")
    
    print("\n" + "=" * 60)
    print("Test yakunlandi!")
    print("=" * 60)
    
    # Interaktiv rejim
    print("\nInteraktiv rejim (to'xtatish uchun 'quit' deb yozing):")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nMatn kiriting: ").strip()
            if user_input.lower() in ['quit', 'exit', 'chiqu']:
                print("Xayr!")
                break
            
            if not user_input:
                continue
            
            # Tokenlarga ajratish
            tokens = tokenizer.tokenize(user_input)
            token_ids = tokenizer.tokenize(user_input, return_ids=True)
            
            print(f"  Tokenlar: {tokens}")
            print(f"  Token ID'lar: {token_ids}")
            print(f"  Qayta tiklangan: {tokenizer.detokenize(token_ids)}")
            
        except KeyboardInterrupt:
            print("\n\nTo'xtatildi.")
            break
        except EOFError:
            print("\n\nTo'xtatildi.")
            break


if __name__ == "__main__":
    main()
