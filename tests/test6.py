import pandas as pd
import nltk
import spacy
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
from project_db.db3_from_csv import *  # Импортируем обновленную схему базы данных
import string  # Для работы с пунктуацией

import logging
logging.basicConfig(level=logging.INFO)

# Инициализация библиотек
nltk.download('punkt')
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()

def clean_text(text):
    return text.replace('\x00', '').strip()

def remove_punctuation(word):
    """Удаляет пунктуацию из строки."""
    return word.translate(str.maketrans('', '', string.punctuation))

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

    # Токенизация предложений и добавление в базу
    sentences = sent_tokenize(text, language='russian')

    for sent in sentences:
        sentence_entry = Sentences(Sentence_text=sent, TextID=text_id)  # Сохраняем текст с пунктуацией
        session.add(sentence_entry)
        session.flush()

    session.commit()

# Поиск слова
search_word = input("Введите слово для поиска: ").strip().lower()
parsed_word = morph.parse(search_word)[0]
lemma = parsed_word.normal_form

# Проверяем существование токена с учётом TextID
existing_token = session.query(TokenID).filter_by(Token_text=lemma, TextID=text_id).first()

if not existing_token:
    # Если токен для данного текста отсутствует, создаём новую запись
    token_entry = TokenID(Token_text=lemma, TextID=text_id)
    session.add(token_entry)
    session.flush()  # Сохраняем в базе и получаем TokenID
    token_id = token_entry.TokenID
else:
    # Если токен уже существует, используем его ID
    token_id = existing_token.TokenID

# Фильтрация предложений, содержащих слово
filtered_sentences = session.query(Sentences).filter(
    Sentences.TextID == text_id,
    Sentences.Sentence_text.ilike(f"%{search_word}%")
).all()

# Схема анализа зависимых слов
pos_data = {
    "ADJ": defaultdict(list),       # Прилагательные
    "NOUN": defaultdict(list),      # Существительные с падежом
    "VERB_HEAD": defaultdict(list), # Глаголы, от которых слово зависит
    "VERB_CHILD": defaultdict(list), # Глаголы, зависящие от слова
    "ADV": defaultdict(list),       # Наречия
    "PRON": defaultdict(list),      # Местоимения
    "OTHER": defaultdict(list)      # Прочее
}

# Анализ окружения слов в отфильтрованных предложениях
for sentence in tqdm(filtered_sentences, desc="Анализ окружения"):
    doc = nlp(sentence.Sentence_text)  # Текст с пунктуацией сохраняется
    for token in doc:
        if token.lemma_ == lemma:
            # Глагол, от которого зависит слово
            if token.head.pos_ == "VERB" and token.head != token:
                pos_data["VERB_HEAD"][remove_punctuation(token.head.lemma_)].append(sentence.SentenceID)

            # Зависимые дети слова
            for child in token.children:
                cleaned_word = remove_punctuation(child.lemma_)
                if cleaned_word:  # Пропускаем пустые строки (пунктуация)
                    if child.pos_ == "VERB":
                        pos_data["VERB_CHILD"][cleaned_word].append(sentence.SentenceID)
                    elif child.pos_ == "NOUN":
                        case = child.morph.get("Case")
                        case_label = f"{cleaned_word} ({case[0] if case else 'Неизвестный падеж'})"
                        pos_data["NOUN"][case_label].append(sentence.SentenceID)
                    elif child.pos_ == "ADJ":
                        pos_data["ADJ"][cleaned_word].append(sentence.SentenceID)
                    elif child.pos_ == "ADV":
                        pos_data["ADV"][cleaned_word].append(sentence.SentenceID)
                    elif child.pos_ == "PRON":
                        pos_data["PRON"][cleaned_word].append(sentence.SentenceID)
                    else:
                        pos_data["OTHER"][cleaned_word].append(sentence.SentenceID)

# Подсчет общей частотности по частям речи
pos_frequencies = {pos: sum(len(ids) for ids in words.values()) for pos, words in pos_data.items()}

# Вывод данных анализа
print("\nРезультаты анализа зависимых слов (без пунктуации):")
for pos, words in pos_data.items():
    print(f"\nЧасть речи: {pos} (Частотность: {pos_frequencies[pos]})")  # Вывод частотности
    for word, sentence_ids in words.items():
        print(f"{word} ({len(sentence_ids)}): {', '.join(map(str, sentence_ids))}")

# Сохранение данных в базу
parsed_word = morph.parse(search_word)[0]
lemma = parsed_word.normal_form

# Проверяем, есть ли токен в базе
token_entry = session.query(TokenID).filter_by(Token_text=lemma, TextID=text_id).first()

if not token_entry:
    # Создаём токен для поискового слова
    token_entry = TokenID(Token_text=lemma, TextID=text_id)
    session.add(token_entry)
    session.flush()

# Получаем TokenID
token_id = token_entry.TokenID

# Обрабатываем слова из анализа
for pos, words in pos_data.items():
    for word, sentence_ids in words.items():
        # Проверяем или добавляем слово
        word_entry = session.query(Words).filter_by(
            Word_text=word,
            Part_of_speech=pos,
            TextID=text_id
        ).first()

        if not word_entry:
            # Добавляем новое слово с общим TokenID
            word_entry = Words(
                Word_text=word,
                Part_of_speech=pos,
                Frequency=len(sentence_ids),
                TextID=text_id,
                TokenID=token_id  # Присваиваем токен, связанный с поисковым словом
            )
            session.add(word_entry)
            session.flush()

        # Добавление связей с предложениями
        for sentence_id in sentence_ids:
            row = Cross(
                WordID=word_entry.WordID,
                SentenceID=sentence_id,
                TextID=text_id
            )
            session.add(row)



try:
    session.commit()
except Exception as e:
    session.rollback()
    print(f"Ошибка при сохранении данных: {e}")
finally:
    session.close()
