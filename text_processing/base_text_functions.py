import logging
from collections import Counter
import re
import string
from commons.constants.constants import *
from text_processing.lemmatization import nlp

# Ритмичность (количество слогов)
def syllable_count(word):
    vowels = "аеёиоуыэюя"
    return sum(1 for char in word.lower() if char in vowels)

def sentence_rhythm(tokens):
    return sum(syllable_count(token) for token in tokens)

# Аллитерации
def alliteration_score(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    first_letters = [token[0] for token in tokens]
    letter_counts = Counter(first_letters)
    return max(letter_counts.values()) if letter_counts else 0

# Контекстуальность (метафоры)
def contains_metaphor(text):
    metaphors = re.findall(r"\b(как|словно|подобно)\b", text, re.IGNORECASE)
    return len(metaphors) > 0

def emotional_score(tokens):
    return sum(emotional_words.get(token.lower(), 0) for token in tokens)

# Частота частей речи
def pos_frequency(pos_tags):
    pos_counts = Counter(pos_tags)
    return {pos: count for pos, count in pos_counts.items()}

# Лексическое разнообразие
def lexical_diversity(tokens):
    unique_words = len(set(tokens))
    total_words = len(tokens)
    return unique_words / total_words if total_words > 0 else 0

# --- Функция подсчёта слов ---
def count_words(text):
    return len(str(text).split())

# Очистка текстов
def clean_text_1(text):
    if not text or not isinstance(text, str):  # Проверка на None и тип данных
        return ""
    text = text.strip()  # Удаление лишних пробелов
    return text

# Обработка текстов через spaCy с прогресс-баром
def preprocess_text(text):
    try:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]
        return tokens, lemmas, pos_tags
    except Exception as e:
        logging.error(f"Ошибка при обработке текста: {text}")
        logging.error(f"Исключение: {e}")
        return [], [], []