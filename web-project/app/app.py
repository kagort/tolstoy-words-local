from flask import Flask, render_template, request, redirect, url_for, jsonify
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, scoped_session
from collections import defaultdict
import nltk
import spacy
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
import string
import logging
from os import environ
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)

from database.model_3 import *

# Инициализация библиотек
nltk.download('punkt', quiet=True)
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Создание Flask-приложения
app = Flask(__name__)

# Подключение к PostgreSQL
DB_USER = environ.get('DB_USER', 'postgres')
DB_PASSWORD = environ.get('DB_PASSWORD', 'ouganda77')
DB_HOST = environ.get('DB_HOST', 'localhost')
DB_PORT = environ.get('DB_PORT', '5432')
DB_NAME = environ.get('DB_NAME', 'tolstoy_words_csv')

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
Session = scoped_session(sessionmaker(bind=engine))
session = Session()

# Глобальная переменная для хранения прогресса
progress = 0

# Удаление пунктуации
def remove_punctuation(word):
    return word.translate(str.maketrans('', '', string.punctuation))

# Очистка текста
def clean_text(text):
    return text.replace('\x00', '').strip()

# Главная страница
@app.route('/')
def index():
    try:
        texts = session.query(DicTexts).all()
        if not texts:
            return redirect(url_for('add_text'))
        numbered_texts = list(enumerate(texts, start=1))
        return render_template('index.html', texts=numbered_texts)
    except Exception as e:
        logging.error(f"Ошибка при получении текстов: {e}")
        return "Ошибка при загрузке текстов.", 500

# Добавление нового текста
def save_text_to_db(text_title, text_author, text_year_creation, text_genre, text):
    try:
        text_entry = DicTexts(
            TextTitle=text_title,
            Text_Author=text_author,
            Text_year_creation=text_year_creation,
            Text_genre=text_genre
        )
        session.add(text_entry)
        session.flush()
        text_id = text_entry.TextID

        sentences = sent_tokenize(text, language='russian')
        for sent in sentences:
            sentence_entry = Sentences(Sentence_text=sent, TextID=text_id)
            session.add(sentence_entry)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logging.error(f"Ошибка при сохранении текста: {e}")
        return False

@app.route('/add_text', methods=['GET', 'POST'])
def add_text():
    if request.method == 'POST':
        text_title = request.form['text_title'].strip()
        text_author = request.form['text_author'].strip()
        try:
            text_year_creation = int(request.form['text_year_creation'])
        except ValueError:
            return "Ошибка: Год создания должен быть числом."
        text_genre = request.form['text_genre'].strip()
        file_path = request.form['file_path'].strip()

        if session.query(DicTexts).filter_by(TextTitle=text_title).first():
            return f"Текст '{text_title}' уже существует в базе."

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = clean_text(file.read())
        except FileNotFoundError:
            return f"Файл по пути {file_path} не найден."

        if save_text_to_db(text_title, text_author, text_year_creation, text_genre, text):
            return redirect(url_for('index'))
        else:
            return "Ошибка при сохранении текста.", 500

    return render_template('add_text.html')

# Получение текущего прогресса
@app.route('/progress')
def get_progress():
    global progress
    return jsonify({'progress': progress})

# Анализ слов
def analyze_word_in_text(text_id, search_word):
    lemma = morph.parse(search_word)[0].normal_form
    existing_token = session.query(TokenID).filter_by(Token_text=lemma, TextID=text_id).first()
    if existing_token:
        return f"Токен '{search_word}' уже существует для текста ID {text_id}."

    token_entry = TokenID(Token_text=lemma, TextID=text_id, Token_count=0)
    session.add(token_entry)
    session.flush()
    token_id = token_entry.TokenID

    filtered_sentences = session.query(Sentences).filter(
        Sentences.TextID == text_id,
        Sentences.Sentence_text.ilike(f"%{search_word}%")
    ).all()

    if not filtered_sentences:
        return f"Слово '{search_word}' не найдено в тексте ID {text_id}."

    pos_data = defaultdict(lambda: defaultdict(list))
    total_occurrences = 0

    for sentence in filtered_sentences:
        doc = nlp(sentence.Sentence_text)
        occurrences_in_sentence = sum(1 for token in doc if token.lemma_ == lemma)
        total_occurrences += occurrences_in_sentence

        for token in doc:
            if token.lemma_ == lemma:
                if token.head.pos_ == "VERB" and token.head != token:
                    pos_data["VERB_HEAD"][remove_punctuation(token.head.lemma_)].append(sentence.SentenceID)
                for child in token.children:
                    cleaned_word = remove_punctuation(child.lemma_)
                    if cleaned_word:
                        pos_data[child.pos_][cleaned_word].append(sentence.SentenceID)

    if total_occurrences > 0:
        token_entry.Token_count = total_occurrences

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
            session.flush()

            for sentence_id in sentence_ids:
                row = Cross(
                    WordID=word_entry.WordID,
                    SentenceID=sentence_id,
                    TextID=text_id,
                    TokenID=token_id
                )
                session.add(row)

    session.commit()
    return f"Анализ токена '{search_word}' успешно завершен для текста ID {text_id}."

@app.route('/analyze_word', methods=['GET', 'POST'])
def analyze_word():
    if request.method == 'GET':
        try:
            texts = session.query(DicTexts).all()
            return render_template('analyze_word.html', texts=texts)
        except Exception as e:
            logging.error(f"Ошибка при получении текстов: {e}")
            return "Ошибка при загрузке текстов.", 500

    elif request.method == 'POST':
        global progress
        try:
            data = request.get_json()
            text_ids = data.get('text_ids', [])
            search_words_input = data.get('search_words', '').strip().lower()

            if not search_words_input:
                return jsonify({'error': 'Список слов для анализа не может быть пустым.'}), 400

            search_words = [word.strip() for word in search_words_input.split(",") if word.strip()]
            results = []
            total_steps = len(text_ids) * len(search_words)
            current_step = 0

            for text_id in text_ids:
                for search_word in search_words:
                    result = analyze_word_in_text(text_id, search_word)
                    results.append(result)

                    current_step += 1
                    progress = int((current_step / total_steps) * 100)

            progress = 0
            return jsonify({'message': 'Анализ завершен успешно.', 'results': results})
        except Exception as e:
            progress = 0
            logging.error(f"Ошибка при анализе слов: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

