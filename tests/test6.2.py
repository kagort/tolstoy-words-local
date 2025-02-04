import pandas as pd
import nltk
import spacy
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
from project_db.model_3 import *  # Импортируем обновленную схему базы данных
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
text_author = input("Введите ФИО автора в формате: Имя Отчество Фамилия: ")
try:
    text_year_creation = int(input("Введите год создания произведения в формате: 0000: "))
except ValueError:
    print("Ошибка: Год создания должен быть числом.")
    exit()
text_genre = input("Введите наименование жанра: ")
file_path = input("Введите путь к файлу текста: ")

# Проверка существующего текста
existing_text = session.query(DicTexts).filter_by(TextTitle=text_title).first()

if existing_text:
    print(f"Текст '{text_title}' уже существует в базе с ID {existing_text.TextID}.")
    text_id = existing_text.TextID
else:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = clean_text(file.read())
    except FileNotFoundError:
        print(f"Файл по пути {file_path} не найден.")
        exit()

    # Добавление текста в базу
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
parsed_word = morph.parse(search_word)[0]
lemma = parsed_word.normal_form

# Получаем или создаем запись в TokenID
existing_token = session.query(TokenID).filter_by(Token_text=lemma, TextID=text_id).first()

if not existing_token:
    token_entry = TokenID(Token_text=lemma, TextID=text_id, Token_count=0)
    session.add(token_entry)
    session.flush()  # Получаем TokenID
    token_id = token_entry.TokenID
    existing_token = token_entry  # Чтобы использовать далее в коде
else:
    token_id = existing_token.TokenID

# Фильтрация предложений, содержащих слово
filtered_sentences = session.query(Sentences).filter(
    Sentences.TextID == text_id,
    Sentences.Sentence_text.ilike(f"%{search_word}%")
).all()

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
    total_occurrences += occurrences_in_sentence  # Увеличиваем общий счетчик

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
    existing_token.Token_count += total_occurrences

# Сохранение результатов анализа в базу
for pos, words in pos_data.items():
    for word, sentence_ids in words.items():
        word_entry = session.query(Words).filter_by(
            Word_text=word,
            Part_of_speech=pos,
            TextID=text_id
        ).first()

        if not word_entry:
            word_entry = Words(
                Word_text=word,
                Part_of_speech=pos,
                Frequency=len(sentence_ids),
                TextID=text_id,
                TokenID=token_id
            )
            session.add(word_entry)
            session.flush()  # Получаем WordID

        # Добавление связей с предложениями (теперь с TokenID)
        for sentence_id in sentence_ids:
            row = Cross(
                WordID=word_entry.WordID,
                SentenceID=sentence_id,
                TextID=text_id,
                TokenID=token_id  # Теперь заполняем TokenID
            )
            session.add(row)

try:
    session.commit()
except Exception as e:
    session.rollback()
    print(f"Ошибка при сохранении данных: {e}")
finally:
    session.close()
