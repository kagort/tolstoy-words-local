import pandas as pd
import nltk
import spacy
from collections import defaultdict, Counter
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
from project_db.db2 import *  # Импортируем обновленную схему базы данных

import logging
logging.basicConfig(level=logging.INFO)  # Журналирование logging.DEBUG

# Инициализация библиотек
nltk.download('punkt')
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()

def clean_text(text):
    return text.replace('\x00', '').strip()

# Ввод данных
text_title = input("Введите название текста: ")
file_path = input("Введите путь к файлу текста: ")

# Проверка существующего текста
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

    text_id = text_entry.TextID

    # Токенизация предложений и их добавление в базу
    sentences = sent_tokenize(text, language='russian')

    sentence_map = {}  # Словарь для хранения соответствий текста и ID предложений
    for sent in sentences:
        sentence_entry = Sentences(Sentence_text=sent, TextID=text_id)
        session.add(sentence_entry)
        session.flush()  # Получаем SentenceID
        sentence_map[sent] = sentence_entry.SentenceID

    session.commit()

# Поиск слова
search_word = input("Введите слово для поиска: ").strip().lower()
parsed_word = morph.parse(search_word)[0]
lemma = parsed_word.normal_form

# Добавление леммы в TokenID
existing_token = session.query(TokenID).filter_by(Token_text=lemma).first()
if not existing_token:
    token_entry = TokenID(Token_text=lemma)
    session.add(token_entry)
    session.flush()
    token_id = token_entry.TokenID
else:
    token_id = existing_token.TokenID

# Фильтрация предложений, содержащих слово
filtered_sentences = session.query(Sentences).filter(
    Sentences.TextID == text_id,
    Sentences.Sentence_text.ilike(f"%{search_word}%")
).all()

# Подсчет частотности зависимых слов
dependent_word_frequencies = Counter()

# Анализ окружения слов в отфильтрованных предложениях
for sentence in tqdm(filtered_sentences, desc="Анализ окружения"):
    doc = nlp(sentence.Sentence_text)
    for token in doc:
        if token.lemma_ == lemma:
            for child in token.children:
                # Подсчет частотности для зависимых слов
                dependent_word_frequencies[child.lemma_] += 1

                # Ограничение длины текста слова
                word_text = child.text[:255] if len(child.text) > 255 else child.text

                # Проверка наличия слова в Words
                word_entry = session.query(Words).filter_by(
                    Word_text=word_text,
                    Part_of_speech=child.pos_,
                    SentenceID=sentence.SentenceID
                ).first()

                if not word_entry:
                    word_entry = Words(
                        Word_text=word_text,
                        Part_of_speech=child.pos_,
                        SentenceID=sentence.SentenceID,
                        Frequency=1  # Начальная частотность
                    )
                    session.add(word_entry)
                    session.flush()
                else:
                    word_entry.Frequency += 1

                # Добавляем связь между Words и Sentences
                session.execute(word_sentence_association.insert().values(
                    WordID=word_entry.WordID,
                    SentenceID=sentence.SentenceID
                ))

# Завершение работы
try:
    session.commit()
except Exception as e:
    session.rollback()
    print(f"Ошибка при сохранении данных: {e}")
finally:
    session.close()

# Вывод частотности зависимых слов
print("Частотность зависимых слов в отфильтрованных предложениях:")
for word, freq in dependent_word_frequencies.most_common():
    print(f"{word}: {freq}")
