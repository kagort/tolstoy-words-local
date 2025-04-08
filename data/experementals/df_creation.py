import pandas as pd
import spacy
from collections import Counter
import re
import logging
from tqdm import tqdm  # Для прогресс-бара
import os
import pandas as pd

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")


# Определение среды выполнения
if os.getenv("STREAMLIT_CLOUD"):  # Переменная окружения для Streamlit Cloud
    file_path = "Datasets_from_pgAdmin/sentences_with_tokens.csv"  # Относительный путь для Streamlit Cloud
else:
    file_path = r"C:\Users\User\PycharmProjects\Tolstoy_words_local\Datasets_from_pgAdmin\sentences_with_tokens.csv"  # Локальный путь

# Проверка существования файла
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Файл не найден: {file_path}")

# Загрузка данных
df = pd.read_csv(file_path)
print("Данные успешно загружены!")

# Загрузка данных из CSV файла
# logging.info("Загрузка данных из CSV файла...")
# file_path = r"C:\Users\User\PycharmProjects\Tolstoy_words_local\Datasets_from_pgAdmin\sentences_with_tokens.csv"
# df = pd.read_csv(file_path)

# Фильтрация только ольфакторных предложений
olfactory_keywords = [
    "духи", "одеколон", "аромат", "букет", "вонь", "запах", "перегар", "смрад",
    "парфюм", "душок", "благовоние", "благоухание", "зловоние", "запашок",
    "фимиам", "миазм", "амбре", "амбра", "пригарь", "тухлятина", "испарение",
    "дуновение", "ладан", "скверна", "дым", "навоз", "дерьмо", "веять",
    "вонять", "благоухать", "попахивать", "разить", "смердеть", "чадить",
    "чад", "пахнуть"
]

df["Is_Olfactory"] = df["Sentence"].apply(lambda x: any(word in x.lower() for word in olfactory_keywords))
olfactory_sentences_df = df[df["Is_Olfactory"]].copy()

# Очистка текстов
def clean_text(text):
    if not text or not isinstance(text, str):  # Проверка на None и тип данных
        return ""
    text = text.strip()  # Удаление лишних пробелов
    return text

olfactory_sentences_df["Sentence"] = olfactory_sentences_df["Sentence"].apply(clean_text)
olfactory_sentences_df = olfactory_sentences_df[olfactory_sentences_df["Sentence"] != ""]

# Ограничение длины предложений
olfactory_sentences_df["Sentence"] = olfactory_sentences_df["Sentence"].apply(lambda x: x[:500])

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
texts = olfactory_sentences_df["Sentence"].tolist()
docs = []

logging.info("Начинается обработка текстов через spaCy...")
for doc in tqdm(nlp.pipe(texts, batch_size=50), total=len(texts), desc="Обработка текстов"):
    docs.append(doc)

# Извлечение токенов, лемм и POS-тегов
olfactory_sentences_df["Tokens"] = [[token.text for token in doc] for doc in docs]
olfactory_sentences_df["Lemmas"] = [[token.lemma_ for token in doc] for doc in docs]
olfactory_sentences_df["POS"] = [[token.pos_ for token in doc] for doc in docs]

# Лексическое разнообразие
def lexical_diversity(tokens):
    unique_words = len(set(tokens))
    total_words = len(tokens)
    return unique_words / total_words if total_words > 0 else 0

logging.info("Вычисление лексического разнообразия...")
olfactory_sentences_df["Lexical_Diversity"] = olfactory_sentences_df["Tokens"].apply(lexical_diversity)

# Сложность предложений (количество слов)
logging.info("Вычисление сложности предложений...")
olfactory_sentences_df["Sentence_Length"] = olfactory_sentences_df["Tokens"].apply(len)

# Частота частей речи
def pos_frequency(pos_tags):
    pos_counts = Counter(pos_tags)
    return {pos: count for pos, count in pos_counts.items()}

logging.info("Вычисление частоты частей речи...")
olfactory_sentences_df["POS_Frequency"] = olfactory_sentences_df["POS"].apply(pos_frequency)

# Эмоциональная окраска (пример словаря эмоций)
emotional_words = {"сладкий": 1, "горький": -1, "приторный": -1}

def emotional_score(tokens):
    return sum(emotional_words.get(token.lower(), 0) for token in tokens)

logging.info("Вычисление эмоциональной окраски...")
olfactory_sentences_df["Emotional_Score"] = olfactory_sentences_df["Tokens"].apply(emotional_score)

# Контекстуальность (метафоры)
def contains_metaphor(text):
    metaphors = re.findall(r"\b(как|словно|подобно)\b", text, re.IGNORECASE)
    return len(metaphors) > 0

logging.info("Поиск метафор...")
olfactory_sentences_df["Contains_Metaphor"] = olfactory_sentences_df["Sentence"].apply(contains_metaphor)

# Аллитерации
def alliteration_score(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    first_letters = [token[0] for token in tokens]
    letter_counts = Counter(first_letters)
    return max(letter_counts.values()) if letter_counts else 0

logging.info("Вычисление аллитераций...")
olfactory_sentences_df["Alliteration_Score"] = olfactory_sentences_df["Sentence"].apply(alliteration_score)

# Ритмичность (количество слогов)
def syllable_count(word):
    vowels = "аеёиоуыэюя"
    return sum(1 for char in word.lower() if char in vowels)

def sentence_rhythm(tokens):
    return sum(syllable_count(token) for token in tokens)

logging.info("Вычисление ритмичности...")
olfactory_sentences_df["Rhythm_Score"] = olfactory_sentences_df["Tokens"].apply(sentence_rhythm)

# Агрегация по авторам с прогресс-баром
logging.info("Агрегация данных по авторам...")
author_stats = olfactory_sentences_df.groupby("Author").agg({
    "Lexical_Diversity": "mean",
    "Sentence_Length": "mean",
    "Emotional_Score": "sum",
    "Contains_Metaphor": "sum",
    "Alliteration_Score": "mean",
    "Rhythm_Score": "mean"
}).reset_index()

print(author_stats)


