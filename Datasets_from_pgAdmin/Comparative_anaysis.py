from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import spacy
from project_db.db3_from_csv import *  # Импорт моделей БД
import pandas as pd
from collections import Counter
import re
import logging
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

# Подключение к PostgreSQL
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
Session = sessionmaker(bind=engine)
session = Session()

# Список ключевых слов для фильтрации предложений
keywords = [
    "духи", "одеколон", "аромат", "букет", "вонь", "запах", "перегар", "смрад", "парфюм", "душок",
    "благоухание", "благовоние", "зловоние", "запашок", "фимиам", "миазм", "амбре", "амбра", "пригарь",
    "тухлятина", "испарение", "дуновение", "ладан", "скверна", "дым", "навоз", "дерьмо",
    "веять", "вонять", "благоухать", "попахивать", "разить", "смердеть", "чадить", "чад", "пахнуть"
]

# Формирование безопасного SQL-запроса с параметрами
query_conditions = " OR ".join([f"Sentence_text ILIKE :kw{idx}" for idx in range(len(keywords))])
query = text(f'''
    SELECT Sentences.Sentence_text, DicTexts.Text_Author, DicTexts.Text_genre
    FROM Sentences
    JOIN DicTexts ON Sentences.TextID = DicTexts.TextID
    WHERE {query_conditions}
''')

params = {f"kw{idx}": f"%{word}%" for idx, word in enumerate(keywords)}
sentences_query = session.execute(query, params).fetchall()

# Проверка на пустой результат
if not sentences_query:
    logging.warning("SQL-запрос не вернул данных!")

# Преобразование в DataFrame
sentences_df = pd.DataFrame(sentences_query, columns=["Sentence", "Author", "Genre"])

# Очистка текстов
sentences_df["Sentence"] = sentences_df["Sentence"].astype(str).str.strip()
sentences_df = sentences_df[sentences_df["Sentence"] != ""]

# Ограничение длины предложений
sentences_df["Sentence"] = sentences_df["Sentence"].apply(lambda x: x[:500])

# Обработка текстов через spaCy с прогресс-баром
texts = sentences_df["Sentence"].dropna().tolist()
docs = list(tqdm(nlp.pipe(texts, batch_size=50), total=len(texts), desc="Обработка текстов"))

# Извлечение токенов, лемм и POS-тегов
sentences_df["Tokens"] = [[token.text for token in doc] for doc in docs]
sentences_df["Lemmas"] = [[token.lemma_ for token in doc] for doc in docs]
sentences_df["POS"] = [[token.pos_ for token in doc] for doc in docs]

# Функция вычисления лексического разнообразия
def lexical_diversity(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0

sentences_df["Lexical_Diversity"] = sentences_df["Tokens"].apply(lexical_diversity)

# Длина предложений
sentences_df["Sentence_Length"] = sentences_df["Tokens"].apply(len)

# Частота частей речи
def pos_frequency(pos_tags):
    return dict(Counter(pos_tags)) if pos_tags else {}

sentences_df["POS_Frequency"] = sentences_df["POS"].apply(pos_frequency)

# Эмоциональная окраска
emotional_words = {"сладкий": 1, "горький": -1, "приторный": -1}

def emotional_score(tokens):
    return sum(emotional_words.get(token.lower(), 0) for token in tokens)

sentences_df["Emotional_Score"] = sentences_df["Tokens"].apply(emotional_score)

# Метафоры
def contains_metaphor(text):
    return bool(re.search(r"\b(как|словно|подобно)\b", text, re.IGNORECASE))

sentences_df["Contains_Metaphor"] = sentences_df["Sentence"].apply(contains_metaphor)

# Аллитерация
def alliteration_score(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    first_letters = [token[0] for token in tokens if token]
    return max(Counter(first_letters).values()) if first_letters else 0

sentences_df["Alliteration_Score"] = sentences_df["Sentence"].apply(alliteration_score)

# Ритмичность (по слогам)
def syllable_count(word):
    return sum(1 for char in word.lower() if char in "аеёиоуыэюя")

def sentence_rhythm(tokens):
    return sum(syllable_count(token) for token in tokens)

sentences_df["Rhythm_Score"] = sentences_df["Tokens"].apply(sentence_rhythm)

# Агрегация по авторам
sentences_df = sentences_df.dropna(subset=["Author"])
author_stats = sentences_df.groupby("Author").agg({
    "Lexical_Diversity": "mean",
    "Sentence_Length": "mean",
    "Emotional_Score": "sum",
    "Contains_Metaphor": "sum",
    "Alliteration_Score": "mean",
    "Rhythm_Score": "mean"
}).reset_index()

print(author_stats)