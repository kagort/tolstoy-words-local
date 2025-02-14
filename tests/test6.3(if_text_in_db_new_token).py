import pandas as pd
import nltk
import spacy
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
from project_db.model_3 import *  # Импорт базы
import string  # Для работы с пунктуацией

import logging
logging.basicConfig(level=logging.INFO)

# Инициализация библиотек
nltk.download('punkt')
nlp = spacy.load("ru_core_news_sm")

morph = MorphAnalyzer()

def clean_text(text):
    return text.replace('\x00', '').strip()
    session.close()

# Функция для удаления пунктуации
def remove_punctuation(word):
    """Удаляет пунктуацию из строки."""
    return word.translate(str.maketrans('', '', string.punctuation))

# ... (все предыдущие импорты и инициализации остаются без изменений)

# Функция для удаления пунктуации
def remove_punctuation(word):
    """Удаляет пунктуацию из строки."""
    return word.translate(str.maketrans('', '', string.punctuation))

# Функция для отображения списка текстов из базы данных
def display_texts():
    texts = session.query(DicTexts).all()
    if not texts:
        print("В базе данных нет ни одного текста.")
        return None

    print("Доступные тексты в базе данных:")
    for text in texts:
        print(f"ID: {text.TextID}, Название: {text.TextTitle}, Автор: {text.Text_Author}, Год: {text.Text_year_creation}")

    try:
        selected_id = int(input("Введите ID текста для анализа: "))
        selected_text = session.query(DicTexts).filter_by(TextID=selected_id).first()
        if not selected_text:
            print(f"Текст с ID {selected_id} не найден.")
            return None
        return selected_text
    except ValueError:
        print("Ошибка: Введите корректный ID текста.")
        return None

# Основной код программы
print("Успешное подключение к базе данных!")

# Спрашиваем пользователя, существует ли текст в базе данных
exists_in_db = input("Существует ли текст в базе данных? (да/нет): ").strip().lower()

if exists_in_db == "да":
    # Пользователь выбирает текст из базы данных
    selected_text = display_texts()
    if not selected_text:
        print("Анализ пропущен.")
        exit()

    text_id = selected_text.TextID
    text_title = selected_text.TextTitle
    print(f"Выбран текст: '{text_title}' с ID {text_id}.")
else:
    # Загрузка нового текста
    text_title = input("Введите название текста: ").strip()
    text_author = input("Введите ФИО автора в формате: Имя Отчество Фамилия: ").strip()
    try:
        text_year_creation = int(input("Введите год создания произведения в формате: 0000: "))
    except ValueError:
        print("Ошибка: Год создания должен быть числом.")
        exit()
    text_genre = input("Введите наименование жанра: ").strip()

    # Проверка существующего текста
    existing_text = session.query(DicTexts).filter_by(TextTitle=text_title).first()

    if existing_text:
        print(f"Текст '{text_title}' уже существует в базе с ID {existing_text.TextID}.")
        text_id = existing_text.TextID
    else:
        file_path = input("Введите путь к файлу текста: ").strip()
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = clean_text(file.read())
        except FileNotFoundError:
            print(f"Файл по пути {file_path} не найден.")
            exit()

        # Добавление нового текста в базу
        text_entry = DicTexts(
            TextTitle=text_title,
            Text_Author=text_author,
            Text_year_creation=text_year_creation,
            Text_genre=text_genre
        )
        session.add(text_entry)
        session.flush()  # Получаем TextID
        text_id = text_entry.TextID

        # Токенизация предложений и добавление в базу
        sentences = sent_tokenize(text, language='russian')
        for sent in sentences:
            sentence_entry = Sentences(Sentence_text=sent, TextID=text_id)
            session.add(sentence_entry)
        session.commit()

# Поиск слова
search_word = input("Введите слово для поиска: ").strip().lower()
if not search_word:
    print("Ошибка: Слово для поиска не может быть пустым.")
    exit()

lemma = morph.parse(search_word)[0].normal_form

# Проверка существования токена для данного текста
existing_token = session.query(TokenID).filter_by(Token_text=lemma, TextID=text_id).first()

if existing_token:
    print(f"Токен '{search_word}' уже существует для текста '{text_title}'. Анализ пропущен.")
    session.close()
    exit()

# Создание новой записи для токена
token_entry = TokenID(Token_text=lemma, TextID=text_id, Token_count=0)
session.add(token_entry)
session.flush()  # Получаем TokenID
token_id = token_entry.TokenID

# Фильтрация предложений, содержащих слово
filtered_sentences = session.query(Sentences).filter(
    Sentences.TextID == text_id,
    Sentences.Sentence_text.ilike(f"%{search_word}%")
).all()

if not filtered_sentences:
    print(f"Слово '{search_word}' не найдено в тексте '{text_title}'.")
    session.close()
    exit()

# Инициализация данных для анализа зависимостей
pos_data = {
    "ADJ": defaultdict(list),
    "NOUN": defaultdict(list),
    "VERB_HEAD": defaultdict(list),
    "VERB_CHILD": defaultdict(list),
    "ADV": defaultdict(list),
    "PRON": defaultdict(list),
    "OTHER": defaultdict(list)
}

# Счетчик количества вхождений токена
total_occurrences = 0
for sentence in tqdm(filtered_sentences, desc="Анализ предложений"):
    doc = nlp(sentence.Sentence_text)
    occurrences_in_sentence = sum(1 for token in doc if token.lemma_ == lemma)
    total_occurrences += occurrences_in_sentence  # Увеличиваем счетчик

    # Анализ зависимостей токена в предложении
    for token in doc:
        if token.lemma_ == lemma:
            if token.head.pos_ == "VERB" and token.head != token:
                pos_data["VERB_HEAD"][remove_punctuation(token.head.lemma_)].append(sentence.SentenceID)
            for child in token.children:
                cleaned_word = remove_punctuation(child.lemma_)
                if cleaned_word:
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

# Обновление Token_count
if total_occurrences > 0:
    token_entry.Token_count = total_occurrences

# Сохранение результатов анализа в базу
for pos, words in pos_data.items():
    for word, sentence_ids in words.items():
        word_entry = Words(
            Word_text=word,
            Part_of_speech=pos,
            Frequency=len(sentence_ids),
            TextID=text_id,
            TokenID=token_id
        )
        session.add(word_entry)
        session.flush()  # Получаем WordID

        # Добавление связей с предложениями
        for sentence_id in sentence_ids:
            row = Cross(
                WordID=word_entry.WordID,
                SentenceID=sentence_id,
                TextID=text_id,
                TokenID=token_id
            )
            session.add(row)

try:
    session.commit()
    print(f"Анализ токена '{search_word}' для текста '{text_title}' успешно завершен.")
except Exception as e:
    session.rollback()
    print(f"Ошибка при сохранении данных: {e}")
finally:
    session.close()