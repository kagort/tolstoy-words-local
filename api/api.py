import logging
from collections import defaultdict
from nltk.tokenize import sent_tokenize

from flask import Flask, render_template, request, redirect, url_for, jsonify
from sqlalchemy.orm import sessionmaker, scoped_session, Session

from database.model import (
    DicTexts, Sentences, TokenID, Words, Cross
)
from text_processing.lemmatization import *

morph = MorphAnalyzer()
nlp = spacy.load("ru_core_news_sm")

def initialization():
    # Инициализация NLP-библиотек
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TextAnalysisAPI:
    session = Session()

    def __init__(self, engine):
        # Настраиваем SQLAlchemy
        self.engine   = engine
        self.session  = scoped_session(sessionmaker(bind=self.engine))
        self.progress = 0

        initialization()

        # Создаём Flask-приложение
        self.app = Flask(__name__)
        self._register_routes()

    def _register_routes(self):
        app = self.app

        @app.route('/')
        def index():
            try:
                texts = self.session.query(DicTexts).all()
                if not texts:
                    return redirect(url_for('add_text'))
                numbered = list(enumerate(texts, start=1))
                return render_template('index.html', texts=numbered)
            except Exception as e:
                logging.error(f"Ошибка при получении текстов: {e}")
                return "Ошибка при загрузке текстов.", 500
            finally:
                self.session.close()

        @app.route('/add_text', methods=['GET', 'POST'])
        def add_text():
            if request.method == 'GET':
                return render_template('add_text.html')

            # POST
            title  = request.form['text_title'].strip()
            author = request.form['text_author'].strip()
            try:
                year = int(request.form['text_year_creation'])
            except ValueError:
                return "Ошибка: год создания должен быть числом.", 400
            genre    = request.form['text_genre'].strip()
            filepath = request.form['file_path'].strip()

            try:
                if self.session.query(DicTexts).filter_by(TextTitle=title).first():
                    return f"Текст '{title}' уже есть в базе.", 400

                try:
                    raw = open(filepath, 'r', encoding='utf-8').read()
                except FileNotFoundError:
                    return f"Файл не найден: {filepath}", 404

                text = clean_text(raw)
                self._save_text_to_db(title, author, year, genre, text)
                return redirect(url_for('index'))
            except Exception as e:
                self.session.rollback()
                logging.error(f"Ошибка при сохранении текста: {e}")
                return "Ошибка при сохранении текста.", 500
            finally:
                self.session.close()

        @app.route('/progress')
        def get_progress():
            return jsonify({'progress': self.progress})

        @app.route('/analyze_word', methods=['GET', 'POST'])
        def analyze_word():
            if request.method == 'GET':
                try:
                    texts = self.session.query(DicTexts).all()
                    return render_template('analyze_word.html', texts=texts)
                except Exception as e:
                    logging.error(f"Ошибка при загрузке текстов: {e}")
                    return "Ошибка при загрузке текстов.", 500
                finally:
                    self.session.close()

            # POST JSON
            data = request.get_json()
            text_ids = data.get('text_ids', [])
            words_raw = data.get('search_words', '').strip().lower()
            if not words_raw:
                return jsonify({'error': 'Список слов не может быть пустым.'}), 400

            search_words = [w.strip() for w in words_raw.split(',') if w.strip()]
            results = []
            total = len(text_ids) * len(search_words)
            step = 0

            try:
                for tid in text_ids:
                    for w in search_words:
                        msg = self._analyze_word_in_text(tid, w)
                        results.append(msg)
                        step += 1
                        self.progress = int(step / total * 100)
                return jsonify({'message': 'Анализ завершён.', 'results': results})
            except Exception as e:
                self.session.rollback()
                logging.error(f"Ошибка при анализе слов: {e}")
                return jsonify({'error': str(e)}), 500
            finally:
                self.progress = 0
                self.session.close()

    def _save_text_to_db(self, title, author, year, genre, text):
        entry = DicTexts(
            TextTitle=title,
            Text_Author=author,
            Text_year_creation=year,
            Text_genre=genre
        )
        self.session.add(entry)
        self.session.flush()
        tid = entry.TextID

        for sent in sent_tokenize(text, language='russian'):
            self.session.add(Sentences(Sentence_text=sent, TextID=tid))

        self.session.commit()

    def _analyze_word_in_text(self, text_id, search_word):
        lemma = morph.parse(search_word)[0].normal_form
        # проверяем дублирование
        if self.session.query(TokenID).filter_by(Token_text=lemma, TextID=text_id).first():
            return f"Токен '{search_word}' уже анализировался для текста {text_id}."

        token = TokenID(Token_text=lemma, TextID=text_id, Token_count=0)
        self.session.add(token)
        self.session.flush()
        token_id = token.TokenID

        sents = self.session.query(Sentences).filter(
            Sentences.TextID==text_id,
            Sentences.Sentence_text.ilike(f'%{search_word}%')
        ).all()
        if not sents:
            return f"Слово '{search_word}' не найдено в тексте {text_id}."

        pos_data = defaultdict(lambda: defaultdict(list))
        total = 0

        for s in sents:
            doc = nlp(s.Sentence_text)
            occ = sum(1 for t in doc if t.lemma_ == lemma)
            total += occ

            for t in doc:
                if t.lemma_ == lemma:
                    # родственники
                    if t.head.pos_ == "VERB" and t.head != t:
                        pos_data["VERB_HEAD"][remove_punctuation(t.head.lemma_)].append(s.SentenceID)
                    for c in t.children:
                        cw = remove_punctuation(c.lemma_)
                        if cw:
                            pos_data[c.pos_][cw].append(s.SentenceID)

        token.Token_count = total
        for pos, words in pos_data.items():
            for w, sids in words.items():
                we = Words(
                    Word_text=w,
                    Part_of_speech=pos,
                    Frequency=len(sids),
                    TextID=text_id,
                    TokenID=token_id
                )
                self.session.add(we)
                self.session.flush()
                for sid in sids:
                    self.session.add(Cross(
                        WordID=we.WordID,
                        SentenceID=sid,
                        TextID=text_id,
                        TokenID=token_id
                    ))
        self.session.commit()
        return f"Токен '{search_word}' проанализирован для текста {text_id}."
