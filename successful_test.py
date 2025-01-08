import pandas as pd
import nltk
import spacy
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from pymorphy3 import MorphAnalyzer

# Инициализация NLTK и SpaCy
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from project_db.db import *

# Загрузка моделей
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()
#
# # Определение базы данных
# Base = declarative_base()
#
# # Таблицы по вашей схеме
# class DicTexts(Base):
#     __tablename__ = 'DicTexts'
#     TextID = Column(Integer, primary_key=True, autoincrement=True)
#     TextTitle = Column(String(255))
#
# class Sentences(Base):
#     __tablename__ = 'Senteces'
#     SentenceID = Column(Integer, primary_key=True, autoincrement=True)
#     Sentence_index = Column(Integer)
#     Sentence_text = Column(Text)
#     TextID = Column(Integer, ForeignKey('DicTexts.TextID'))
#
# class TokenID(Base):
#     __tablename__ = 'TokenID'
#     TokenID = Column(Integer, primary_key=True, autoincrement=True)
#     Token_text = Column(String(255))
#
# class Words(Base):
#     __tablename__ = 'Words'
#     WordID = Column(Integer, primary_key=True, autoincrement=True)
#     word = Column(String(255))
#     Part_of_speech = Column(String(50))
#     TextID = Column(Integer, ForeignKey('DicTexts.TextID'))
#     TokenID = Column(Integer, ForeignKey('TokenID.TokenID'))
#
# class CrossWordsSentences(Base):
#     __tablename__ = 'Cross_words_sentences'
#     WordID = Column(Integer, ForeignKey('Words.WordID'), primary_key=True)
#     SentenceID = Column(Integer, ForeignKey('Senteces.SentenceID'), primary_key=True)
#
# # Подключение к базе данных
# DATABASE_URL = 'postgresql://avnadmin:AVNS_DptIPNFJ8aKMx-GvN-V@test-project-lp35-test.d.aivencloud.com:11150/defaultdb?sslmode=require'
# engine = create_engine(DATABASE_URL)
#
# Session = sessionmaker(bind=engine)
# session = Session()
Base.metadata.create_all(engine)
# Загрузка текста
with open("voina_i_mir.txt", 'r', encoding='utf-8') as file:
    text = file.read()

# Добавление текста в базу
text_entry = DicTexts(TextTitle='Война и мир')
session.add(text_entry)
session.flush()

# Токенизация текста
sentences = sent_tokenize(text, language='russian')

# Ввод слова для поиска
search_word = input("Введите слово для поиска: ").strip().lower()

# Лемматизация
parsed_word = morph.parse(search_word)[0]
forms = {form.word for form in parsed_word.lexeme}
lemma = parsed_word.normal_form

# Фильтрация предложений
filtered_sentences = []
for idx, sent in enumerate(sentences):
    if any(form in sent.lower() for form in forms):
        filtered_sentences.append((idx + 1, sent))

# Добавление предложений в БД
sentence_objects = []
for idx, sent in filtered_sentences:
    sentence_entry = Sentences(Sentence_index=idx, Sentence_text=sent, TextID=text_entry.TextID)
    session.add(sentence_entry)
    session.flush()
    sentence_objects.append(sentence_entry)

# Обработка окружения
pos_data = defaultdict(list)
for sentence_entry in tqdm(sentence_objects, desc="Анализ окружения"):
    doc = nlp(sentence_entry.Sentence_text)
    for token in doc:
        if token.lemma_ == lemma:
            if token.head.pos_ == "VERB" and token.head != token:
                pos_data['VERB_HEAD'].append((token.head.text, sentence_entry.SentenceID))
            for child in token.children:
                pos_data[child.pos_].append((child.text, sentence_entry.SentenceID))

# Добавление результатов в БД
token_cache = {}
for pos, words in pos_data.items():
    for word, sentence_id in words:
        # Добавляем токен
        if word not in token_cache:
            token_entry = TokenID(Token_text=word)
            session.add(token_entry)
            session.flush()
            token_cache[word] = token_entry.TokenID

        # Добавляем слово
        word_entry = Words(
            word=word,
            Part_of_speech=pos,
            TextID=text_entry.TextID,
            TokenID=token_cache[word]
        )
        session.add(word_entry)
        session.flush()

        # Добавляем связь
        session.add(CrossWordsSentences(WordID=word_entry.WordID, SentenceID=sentence_id))

# Сохранение данных и закрытие сессии
session.commit()
session.close()

print("Данные успешно добавлены в базу данных.")