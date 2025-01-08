import pandas as pd
import nltk
import spacy
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
import logging
from project_db.db import *

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация NLTK и SpaCy
nltk.download('punkt')

# Загрузка моделей
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()

# Функция очистки текста
def clean_text(text):
    return text.replace('\x00', '').strip()


# Ввод данных
text_title = input("Введите название текста: ")
file_path = input("Введите путь к файлу текста: ")

# Проверка наличия текста в базе
existing_text = session.query(DicTexts).filter_by(TextTitle=text_title).first()

if existing_text:
    print(f"Текст '{text_title}' уже существует в базе с ID {existing_text.TextID}.")
    text_id = existing_text.TextID
else:
    # Загрузка текста
    with open(file_path, 'r', encoding='utf-8') as file:
        text = clean_text(file.read())

    # Добавление текста в базу
    text_entry = DicTexts(TextTitle=text_title)
    session.add(text_entry)
    session.flush()

    text_id = text_entry.TextID  # Присвоение ID

    # Токенизация текста
    sentences = sent_tokenize(text, language='russian')

    # Добавление предложений
    for idx, sent in enumerate(sentences):
        session.add(Sentences(Sentence_index=idx + 1, Sentence_text=sent, TextID=text_id))

    session.commit()

# Поиск слова
search_word = input("Введите слово для поиска: ").strip().lower()
parsed_word = morph.parse(search_word)[0]
forms = {form.word for form in parsed_word.lexeme}
lemma = parsed_word.normal_form

# Фильтрация предложений
filtered_sentences = session.query(Sentences).filter(
    Sentences.TextID == text_id,
    Sentences.Sentence_text.ilike(f"%{search_word}%")
).all()

# Обработка окружения
pos_data = defaultdict(list)
for sentence in tqdm(filtered_sentences, desc="Анализ окружения"):
    doc = nlp(sentence.Sentence_text)
    for token in doc:
        if token.lemma_ == lemma:
            if token.head.pos_ == "VERB" and token.head != token:
                pos_data['VERB_HEAD'].append((token.head.text, sentence.SentenceID))
            for child in token.children:
                pos_data[child.pos_].append((child.text, sentence.SentenceID))

# Добавление данных в базу
token_cache = {}
for pos, words in tqdm(pos_data.items(), desc="Добавление токенов"):
    for word, sentence_id in words:
        # Проверка существующего токена
        existing_token = session.query(TokenID).filter_by(Token_text=word).first()
        if existing_token:
            token_cache[word] = existing_token.TokenID
        else:
            token_entry = TokenID(Token_text=word)
            session.add(token_entry)
            session.flush()
            token_cache[word] = token_entry.TokenID

        # Добавляем слово
        word_entry = Words(
            word=word,
            Part_of_speech=pos,
            TextID=text_id,
            TokenID=token_cache[word]
        )
        session.add(word_entry)
        session.flush()

        # Добавляем связь с предложением
        session.add(CrossWordsSentences(WordID=word_entry.WordID, SentenceID=sentence_id))

session.commit()
session.close()

print("Данные успешно добавлены в базу данных.")