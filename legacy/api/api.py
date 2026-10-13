import logging
from collections import defaultdict
from sqlalchemy import select, func, literal_column, true, bindparam
from sqlalchemy.sql import lateral
from sqlalchemy.orm import aliased
from collections import defaultdict
import re
import pymorphy3
from sqlalchemy.orm import aliased
from sqlalchemy.sql import lateral
from sqlalchemy import or_
from sqlalchemy import join

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
        parses = morph.parse(search_word)
        parsed = parses[0]
        lemma = parsed.normal_form
        pos = parsed.tag.POS  # исходная часть речи

        # фильтр словоформ по исходной части речи
        if pos in ("INFN"):  # если токен - глагол в начальной форме, то берем INFN и главгольные словоформы
            allowed_pos = ("INFN", "VERB")
        else:
            allowed_pos = (pos,)

        forms_main = {wf.word.lower() for p in parses if p.tag.POS in allowed_pos
                      for wf in p.lexeme
                      if wf.tag.POS in allowed_pos} or {lemma}

        punctuation = r'\s+|[!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~]+'
        start_punct = rf'(?:^|{punctuation})'
        end_punct = rf'(?:$|{punctuation})'
        core = '|'.join(re.escape(w) for w in sorted(forms_main, key=len, reverse=True))
        sql_pattern = rf'(?:' + '|'.join(f'{start_punct}{re.escape(w)}{end_punct}'
                                         for w in sorted(forms_main, key=len, reverse=True)) + r')'

        token_pattern = rf'^(?:{core})$'
        rx_word = re.compile(token_pattern, flags=re.IGNORECASE)

        token = (self.session.query(TokenID)
                 .filter_by(Token_text=lemma, TextID=text_id)
                 .first())
        if not token:
            token = TokenID(Token_text=lemma, TextID=text_id, Token_count=0)
            self.session.add(token)
            self.session.flush()
        token_id = token.TokenID

        s = aliased(Sentences)
        rm_lat = lateral(
            func.regexp_matches(
                func.lower(s.Sentence_text),
                bindparam('pat'),  # передаём sql_pattern
                literal_column("'g'")
            )
        ).alias('rm')

        matches_per_sent = (
            select(s.SentenceID.label('sid'), func.count().label('cnt'))
            .select_from(join(s, rm_lat, true()))  # CROSS JOIN LATERAL
            .where(s.TextID == text_id)
            .group_by(s.SentenceID)
        ).subquery()

        total = self.session.execute(
            select(func.coalesce(func.sum(matches_per_sent.c.cnt), 0)).params(pat=sql_pattern)
        ).scalar_one()

        if total == 0:
            token.Token_count = 0
            self.session.commit()
            return (f"Слово '{search_word}' (лемма '{lemma}') не найдено в тексте {text_id}.")

        sid_list = self.session.execute(
            select(matches_per_sent.c.sid).params(pat=sql_pattern)
        ).scalars().all()

        sents = (self.session.query(Sentences)
                 .filter(Sentences.TextID == text_id,
                         Sentences.SentenceID.in_(sid_list))
                 .all())

        if not hasattr(self, "_nlp_optimized"):
            for pipe in ("ner", "textcat", "entity_linker"):
                if pipe in nlp.pipe_names:
                    nlp.disable_pipe(pipe)
            self._nlp_optimized = True

        from collections import defaultdict
        pos_data = defaultdict(lambda: defaultdict(set))

        docs = nlp.pipe((sent.Sentence_text for sent in sents), batch_size=64)
        for sent_obj, doc in zip(sents, docs):
            for t in doc:
                tt = t.text.strip()
                if not tt:
                    continue
                if not rx_word.search(tt.lower()):
                    continue

                if t.head.pos_ == "VERB" and t.head != t:
                    head = remove_punctuation(t.head.lemma_)
                    if head:
                        pos_data["VERB_HEAD"][head].add(sent_obj.SentenceID)

                for c in t.children:
                    child = remove_punctuation(c.lemma_)
                    if child:
                        pos_data[c.pos_][child].add(sent_obj.SentenceID)

        existing_words = (self.session.query(Words)
                          .filter_by(TextID=text_id, TokenID=token_id)
                          .all())
        word_by_key = {(w.Word_text, w.Part_of_speech): w for w in existing_words}

        existing_cross = (self.session.query(Cross.WordID, Cross.SentenceID)
                          .filter(Cross.TextID == text_id, Cross.TokenID == token_id)
                          .all())
        crosses_by_wid = defaultdict(set)
        for wid, sid in existing_cross:
            crosses_by_wid[wid].add(sid)

        new_words = []
        for pos, forms_map in pos_data.items():
            for form, sids in forms_map.items():
                key = (form, pos)
                w = word_by_key.get(key)
                if w is None:
                    w = Words(
                        Word_text=form,
                        Part_of_speech=pos,
                        Frequency=0,
                        TextID=text_id,
                        TokenID=token_id
                    )
                    self.session.add(w)
                    new_words.append(w)
                    word_by_key[key] = w

        if new_words:
            self.session.flush()

        to_insert = []
        for (form, pos), w in word_by_key.items():
            sids_new = pos_data.get(pos, {}).get(form, set())
            if not sids_new:
                continue
            existing_sids = crosses_by_wid.get(w.WordID, set())
            add_sids = sids_new - existing_sids
            for sid in add_sids:
                to_insert.append(Cross(
                    WordID=w.WordID,
                    SentenceID=sid,
                    TextID=text_id,
                    TokenID=token_id
                ))
            w.Frequency = len(existing_sids | sids_new)

        if to_insert:
            self.session.bulk_save_objects(to_insert)

        token.Token_count = int(total)
        self.session.commit()

        return (f"Токен '{search_word}' (лемма '{lemma}') проанализирован: "
                f"{total} вхождений, {len(pos_data)} зависимых слов.")
