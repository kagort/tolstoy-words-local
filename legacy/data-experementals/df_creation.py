import spacy
from tqdm import tqdm  # Для прогресс-бара
import os
import pandas as pd
from text_processing.base_text_functions import *

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")


# Определение среды выполнения
if os.getenv("STREAMLIT_CLOUD"):  # Переменная окружения для Streamlit Cloud
    file_path = sentences_with_tokens_streamlit_cloud
else:
    file_path = sentences_with_tokens_local_path  # Локальный путь

# Проверка существования файла
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Файл не найден: {file_path}")

# Загрузка данных
df = pd.read_csv(file_path)
print("Данные успешно загружены!")

# Фильтрация только ольфакторных предложений
df["Is_Olfactory"] = df["Sentence"].apply(lambda x: any(word in x.lower() for word in olfactory_keywords))
olfactory_sentences_df = df[df["Is_Olfactory"]].copy()

olfactory_sentences_df["Sentence"] = olfactory_sentences_df["Sentence"].apply(clean_text_1)
olfactory_sentences_df = olfactory_sentences_df[olfactory_sentences_df["Sentence"] != ""]

# Ограничение длины предложений
olfactory_sentences_df["Sentence"] = olfactory_sentences_df["Sentence"].apply(lambda x: x[:500])


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

logging.info("Вычисление лексического разнообразия...")
olfactory_sentences_df["Lexical_Diversity"] = olfactory_sentences_df["Tokens"].apply(lexical_diversity)

# Сложность предложений (количество слов)
logging.info("Вычисление сложности предложений...")
olfactory_sentences_df["Sentence_Length"] = olfactory_sentences_df["Tokens"].apply(len)

logging.info("Вычисление частоты частей речи...")
olfactory_sentences_df["POS_Frequency"] = olfactory_sentences_df["POS"].apply(pos_frequency)


logging.info("Вычисление эмоциональной окраски...")
olfactory_sentences_df["Emotional_Score"] = olfactory_sentences_df["Tokens"].apply(emotional_score)

logging.info("Поиск метафор...")
olfactory_sentences_df["Contains_Metaphor"] = olfactory_sentences_df["Sentence"].apply(contains_metaphor)

logging.info("Вычисление аллитераций...")
olfactory_sentences_df["Alliteration_Score"] = olfactory_sentences_df["Sentence"].apply(alliteration_score)

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


