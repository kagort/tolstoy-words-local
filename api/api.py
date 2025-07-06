import logging
import re
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from sqlalchemy import or_

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

        doc = nlp(text)
        for sent in doc.sents:
            self.session.add(Sentences(Sentence_text=sent.text.strip(), TextID=tid))

        self.session.commit()

    def _analyze_word_in_text(self, text_id, search_word):
        parsed = morph.parse(search_word)[0]
        lemma = parsed.normal_form
        if self.session.query(TokenID).filter_by(Token_text=lemma, TextID=text_id).first():
            return f"Токен '{search_word}' уже анализировался для текста {text_id}."

        lexeme_forms = {v.word for v in parsed.lexeme}
        filters = [
            Sentences.Sentence_text.ilike(f"%{form}%")
            for form in lexeme_forms
        ]
        sents = (
            self.session.query(Sentences)
            .filter(
                Sentences.TextID == text_id,
                or_(*filters)
            )
            .all()
        )
        if not sents:
            return f"Слово '{search_word}' (лемма '{lemma}') не найдено в тексте {text_id}."

        # 3) Создаем TokenID
        token = TokenID(Token_text=lemma, TextID=text_id, Token_count=0)
        self.session.add(token)
        self.session.flush()
        token_id = token.TokenID

        if not hasattr(self, "_nlp_optimized"):
            for pipe in ("ner", "textcat", "entity_linker"):
                if pipe in nlp.pipe_names:
                    nlp.disable_pipe(pipe)
            self._nlp_optimized = True

        total = 0
        pos_data = defaultdict(lambda: defaultdict(set))
        docs = nlp.pipe((s.Sentence_text for s in sents), batch_size=64)

        for sent_obj, doc in zip(sents, docs):
            occ = sum(1 for t in doc if t.lemma_ == lemma)
            if occ == 0:
                continue
            total += occ

            for t in doc:
                if t.lemma_ != lemma:
                    continue
                if t.head.pos_ == "VERB" and t.head != t:
                    head = remove_punctuation(t.head.lemma_)
                    pos_data["VERB_HEAD"][head].add(sent_obj.SentenceID)
                for c in t.children:
                    child = remove_punctuation(c.lemma_)
                    if child:
                        pos_data[c.pos_][child].add(sent_obj.SentenceID)

        if total == 0:
            self.session.rollback()
            return f"Слово '{search_word}' не найдено в тексте {text_id}."

        token.Token_count = total

        words = []
        crosses = []
        for pos, forms in pos_data.items():
            for form, sids in forms.items():
                w = Words(
                    Word_text=form,
                    Part_of_speech=pos,
                    Frequency=len(sids),
                    TextID=text_id,
                    TokenID=token_id
                )
                words.append((w, sids))

        self.session.add_all(w for w, _ in words)
        self.session.flush()

        for w, sids in words:
            for sid in sids:
                crosses.append(Cross(
                    WordID=w.WordID,
                    SentenceID=sid,
                    TextID=text_id,
                    TokenID=token_id
                ))

        self.session.bulk_save_objects(crosses)
        self.session.commit()

        return (
            f"Токен '{search_word}' (лемма '{lemma}') проанализирован: "
            f"{total} вхождений, {len(pos_data)} зависимых слов."
        )
