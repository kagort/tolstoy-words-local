import logging
from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy.orm import sessionmaker
from pymorphy3 import MorphAnalyzer
from database.db3_from_csv import *
from database.model_3 import *

app = Flask(__name__)

# Настройка базы данных
Session = sessionmaker(bind=engine)
session = Session()

# Инициализация морфологического анализатора
morph = MorphAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def search_sentences():
    """Форма поиска предложений по токену."""
    text_titles = session.query(DicTexts.TextID, DicTexts.TextTitle).all()
    text_id = None
    token_text = None
    if request.method == 'POST':
        text_id = request.form.get('text_id')
        token_text = request.form.get('token_text')

    if request.method == 'GET':
        text_id = request.args.get('text_id')
        token_text = request.args.get('token_text')
    if text_id and token_text:

        # Приводим слово к нормальной форме
        normal_form = morph.parse(token_text)[0].normal_form

        # Получаем токен
        token = session.query(TokenID).filter_by(TextID=text_id, Token_text=normal_form).first()
        if not token:
            return render_template('index0.html', error="Токен не найден.", text_titles=text_titles)

        # Получаем все пересечения токена из таблицы Cross
        cross_entries = session.query(Cross).filter_by(TokenID=token.TokenID).all()

        # Получаем SentenceID и CrossID
        sentence_word_pairs = [(cross.SentenceID, cross.CrossID) for cross in cross_entries]

        # Загружаем предложения
        sentence_dict = {s.SentenceID: s.Sentence_text for s in session.query(Sentences).filter(
            Sentences.SentenceID.in_([s_id for s_id, _ in sentence_word_pairs])
        ).all()}

        # Подготавливаем список с подсветкой токена
        highlighted_sentences = []
        for sentence_id, cross_id in sentence_word_pairs:
            sentence_text = sentence_dict.get(sentence_id, "Текст не найден")
            highlighted_text = sentence_text.replace(
                token_text, f"<span class='highlight'>{token_text}</span>"
            )
            highlighted_sentences.append({
                'sentence_id': sentence_id,
                'cross_id': cross_id,
                'text': highlighted_text
            })

        return render_template(
            'index0.html',
            sentences=highlighted_sentences,
            text_id=text_id,
            token_text=token_text,
            text_titles=text_titles
        )

    return render_template('index0.html', text_titles=text_titles)

@app.route('/delete', methods=['POST'])
def delete_sentences():
    """Удаление конкретных вхождений токена."""
    selected_entries = request.form.getlist('selected_sentences')
    text_id = request.form.get('text_id')
    token_text = request.form.get('token_text')

    if not selected_entries or not text_id or not token_text:
        logging.info('redirect')
        return redirect(url_for('search_sentences'))
    logging.info('normal')
    # Приводим слово к нормальной форме
    normal_form = morph.parse(token_text)[0].normal_form

    # Получаем токен
    token = session.query(TokenID).filter_by(TextID=text_id, Token_text=normal_form).first()
    if not token:
        return redirect(url_for('search_sentences'))

    deleted_occurrences = 0

    for cross_id in selected_entries:
        try:
            cross_id = int(cross_id)
        except ValueError:
            continue  # Пропускаем некорректные значения

        # Удаляем конкретное вхождение из Cross
        cross_entry = session.query(Cross).filter_by(CrossID=cross_id).first()
        if cross_entry:
            session.delete(cross_entry)
            deleted_occurrences += 1

    # Обновляем Token_count
    if deleted_occurrences > 0:
        token.Token_count = max(0, token.Token_count - deleted_occurrences)

    session.commit()
    logging.info("reload")
    # Перезагружаем страницу
    return redirect(url_for('search_sentences', text_id=text_id, token_text=token_text))

if __name__ == '__main__':
    app.run(debug=True)
