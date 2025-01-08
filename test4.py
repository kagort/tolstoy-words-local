import pandas as pd
import nltk
import spacy
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
from project_db.db2 import *

import logging
logging.basicConfig(level=logging.INFO)

nltk.download('punkt')
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()


def clean_text(text):
    return text.replace('\x00', '').strip()


# Ввод данных
text_title = input("Введите название текста: ")
file_path = input("Введите путь к файлу текста: ")

# Проверка текста
existing_text = session.query(DicTexts).filter_by(TextTitle=text_title).first()

if existing_text:
    print(f"Текст '{text_title}' уже существует в базе с ID {existing_text.TextID}.")
    text_id = existing_text.TextID
else:
    # Загрузка текста
    with open(file_path, 'r', encoding='utf-8') as file:
        text = clean_text(file.read())

    # Добавление текста
    text_entry = DicTexts(TextTitle=text_title)
    session.add(text_entry)
    session.flush()

    text_id = text_entry.TextID

    # Токенизация предложений
    sentences = sent_tokenize(text, language='russian')

    for idx, sent in enumerate(sentences):
        session.add(Sentences(Sentence_index=idx + 1, Sentence_text=sent, TextID=text_id))

    session.commit()

# Поиск слова
search_word = input("Введите слово для поиска: ").strip().lower()
parsed_word = morph.parse(search_word)[0]
lemma = parsed_word.normal_form

# Добавление в tokenid
existing_token = session.query(TokenID).filter_by(Token_text=lemma).first()
if not existing_token:
    token_entry = TokenID(Token_text=lemma)
    session.add(token_entry)
    session.flush()
    token_id = token_entry.TokenID
else:
    token_id = existing_token.TokenID

# Фильтрация предложений
filtered_sentences = session.query(Sentences).filter(
    Sentences.TextID == text_id,
    Sentences.Sentence_text.ilike(f"%{search_word}%")
).all()

# Обработка окружения
for sentence in tqdm(filtered_sentences, desc="Анализ окружения"):
    doc = nlp(sentence.Sentence_text)
    for token in doc:
        if token.lemma_ == lemma:
            for child in token.children:
                # Добавляем в words
                word_entry = Words(
                    word=child.text,
                    Part_of_speech=child.pos_,
                    SentenceID=sentence.SentenceID
                )
                session.add(word_entry)

session.commit()
session.close()

print("Данные успешно добавлены в базу данных.")
