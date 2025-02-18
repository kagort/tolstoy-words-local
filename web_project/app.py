from flask import Flask, render_template, request, redirect, url_for, jsonify
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from project_db.model_3 import *
from collections import defaultdict
import nltk
import spacy
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import sent_tokenize
import string
import logging

# Инициализация библиотек
nltk.download('punkt')
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Создание Flask-приложения
app = Flask(__name__)

# Подключение к PostgreSQL
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
Base.metadata.bind = engine
Session = sessionmaker(bind=engine)
session = Session()

# Удаление пунктуации
def remove_punctuation(word):
    return word.translate(str.maketrans('', '', string.punctuation))

# Очистка текста
def clean_text(text):
    return text.replace('\x00', '').strip()

# Главная страница
@app.route('/')
def index():
    texts = session.query(DicTexts).all()
    if not texts:
        # Если текстов нет, перенаправляем на форму добавления текста
        return redirect(url_for('add_text'))
    return render_template('index.html', texts=texts)

# Добавление нового текста
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

        # Проверка существующего текста
        existing_text = session.query(DicTexts).filter_by(TextTitle=text_title).first()
        if existing_text:
            return f"Текст '{text_title}' уже существует в базе."

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = clean_text(file.read())
        except FileNotFoundError:
            return f"Файл по пути {file_path} не найден."

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
        return redirect(url_for('index'))

    return render_template('add_text.html')

# Поиск и анализ слова
@app.route('/analyze_word', methods=['GET', 'POST'])
def analyze_word():
    if request.method == 'POST':
        text_id = int(request.form['text_id'])
        search_word = request.form['search_word'].strip().lower()

        if not search_word:
            return "Ошибка: Слово для поиска не может быть пустым."

        lemma = morph.parse(search_word)[0].normal_form

        # Проверка существования токена для данного текста
        existing_token = session.query(TokenID).filter_by(Token_text=lemma, TextID=text_id).first()
        if existing_token:
            return f"Токен '{search_word}' уже существует для выбранного текста."

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
            return f"Слово '{search_word}' не найдено в тексте."

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
            return f"Анализ токена '{search_word}' успешно завершен."
        except Exception as e:
            session.rollback()
            return f"Ошибка при сохранении данных: {e}"

    texts = session.query(DicTexts).all()
    return render_template('analyze_word.html', texts=texts)

if __name__ == '__main__':
    app.run(debug=True)