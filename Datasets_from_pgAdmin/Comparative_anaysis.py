from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import spacy
from project_db.model_3 import *
import pandas as pd
from collections import Counter
import re
import logging
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

# Подключение к PostgreSQL
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
Session = sessionmaker(bind=engine)
session = Session()

# Извлечение всех предложений
sentences_query = session.query(Sentences.Sentence_text, DicTexts.Text_Author, DicTexts.Text_genre).\
    join(DicTexts, Sentences.TextID == DicTexts.TextID).all()

# Преобразование в DataFrame
sentences_df = pd.DataFrame(sentences_query, columns=["Sentence", "Author", "Genre"])

# Очистка текстов
def clean_text(text):
    if not text or not isinstance(text, str):  # Проверка на None и тип данных
        return ""
    text = text.strip()  # Удаление лишних пробелов
    return text

sentences_df["Sentence"] = sentences_df["Sentence"].apply(clean_text)
sentences_df = sentences_df[sentences_df["Sentence"] != ""]

# Ограничение длины предложений
sentences_df["Sentence"] = sentences_df["Sentence"].apply(lambda x: x[:500])

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

# Использование nlp.pipe с прогресс-баром
texts = sentences_df["Sentence"].tolist()
docs = []

logging.info("Начинается обработка текстов через spaCy...")
for doc in tqdm(nlp.pipe(texts, batch_size=50), total=len(texts), desc="Обработка текстов"):
    docs.append(doc)

# Извлечение токенов, лемм и POS-тегов
sentences_df["Tokens"] = [[token.text for token in doc] for doc in docs]
sentences_df["Lemmas"] = [[token.lemma_ for token in doc] for doc in docs]
sentences_df["POS"] = [[token.pos_ for token in doc] for doc in docs]

# Лексическое разнообразие
def lexical_diversity(tokens):
    unique_words = len(set(tokens))
    total_words = len(tokens)
    return unique_words / total_words if total_words > 0 else 0

logging.info("Вычисление лексического разнообразия...")
sentences_df["Lexical_Diversity"] = sentences_df["Tokens"].apply(lexical_diversity)

# Сложность предложений (количество слов)
logging.info("Вычисление сложности предложений...")
sentences_df["Sentence_Length"] = sentences_df["Tokens"].apply(len)

# Частота частей речи
def pos_frequency(pos_tags):
    pos_counts = Counter(pos_tags)
    return {pos: count for pos, count in pos_counts.items()}

logging.info("Вычисление частоты частей речи...")
sentences_df["POS_Frequency"] = sentences_df["POS"].apply(pos_frequency)

# Эмоциональная окраска (пример словаря эмоций)
emotional_words = {"сладкий": 1, "горький": -1, "приторный": -1}

def emotional_score(tokens):
    return sum(emotional_words.get(token.lower(), 0) for token in tokens)

logging.info("Вычисление эмоциональной окраски...")
sentences_df["Emotional_Score"] = sentences_df["Tokens"].apply(emotional_score)

# Контекстуальность (метафоры)
def contains_metaphor(text):
    metaphors = re.findall(r"\b(как|словно|подобно)\b", text, re.IGNORECASE)
    return len(metaphors) > 0

logging.info("Поиск метафор...")
sentences_df["Contains_Metaphor"] = sentences_df["Sentence"].apply(contains_metaphor)

# Аллитерации
def alliteration_score(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    first_letters = [token[0] for token in tokens]
    letter_counts = Counter(first_letters)
    return max(letter_counts.values()) if letter_counts else 0

logging.info("Вычисление аллитераций...")
sentences_df["Alliteration_Score"] = sentences_df["Sentence"].apply(alliteration_score)

# Ритмичность (количество слогов)
def syllable_count(word):
    vowels = "аеёиоуыэюя"
    return sum(1 for char in word.lower() if char in vowels)

def sentence_rhythm(tokens):
    return sum(syllable_count(token) for token in tokens)

logging.info("Вычисление ритмичности...")
sentences_df["Rhythm_Score"] = sentences_df["Tokens"].apply(sentence_rhythm)

# Агрегация по авторам с прогресс-баром
logging.info("Агрегация данных по авторам...")
author_stats = sentences_df.groupby("Author").agg({
    "Lexical_Diversity": "mean",
    "Sentence_Length": "mean",
    "Emotional_Score": "sum",
    "Contains_Metaphor": "sum",
    "Alliteration_Score": "mean",
    "Rhythm_Score": "mean"
}).reset_index()

print(author_stats)