from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy.orm import sessionmaker
from pymorphy3 import MorphAnalyzer
from project_db.db3_from_csv import *
from project_db.model_3 import *

app = Flask(__name__)

# Настройка базы данных
Session = sessionmaker(bind=engine)
session = Session()

# Инициализация морфологического анализатора
morph = MorphAnalyzer()

@app.route('/', methods=['GET'])
def search_sentences():
    """Форма поиска предложений по токену (учитывает все формы слова)."""
    text_titles = session.query(DicTexts.TextID, DicTexts.TextTitle).all()

    if request.args.get('text_id'):
        text_id = request.args.get('text_id')
        token_text = request.args.get('token_text')

        # Приводим слово к нормальной форме
        normal_form = morph.parse(token_text)[0].normal_form

        # Получаем токен из базы
        token = session.query(TokenID).filter_by(TextID=text_id, Token_text=normal_form).first()
        if not token:
            return render_template('index.html', error="Токен не найден.", text_titles=text_titles)

        # Получаем предложения, содержащие любые формы токена
        sentences = session.query(Sentences).filter(
            Sentences.TextID == text_id,
            Sentences.Sentence_text.ilike(f"%{normal_form}%")  # Ищем слово в любом склонении
        ).all()

        highlighted_sentences = []
        for sentence in sentences:
            highlighted_text = sentence.Sentence_text
            # Подсвечиваем все формы слова в тексте
            for form in morph.parse(token_text):
                highlighted_text = highlighted_text.replace(
                    form.word, f"<span class='highlight'>{form.word}</span>"
                )

            highlighted_sentences.append({'id': sentence.SentenceID, 'text': highlighted_text})

        return render_template(
            'index.html',
            sentences=highlighted_sentences,
            text_id=text_id,
            token_text=token_text,
            text_titles=text_titles
        )

    return render_template('index.html', text_titles=text_titles)

@app.route('/delete', methods=['POST'])
def delete_sentences():
    """Удаление связей токена с предложениями и обновление списка."""
    selected_sentences = request.form.getlist('selected_sentences')
    text_id = request.form.get('text_id')
    token_text = request.form.get('token_text')

    if not selected_sentences or not text_id or not token_text:
        return redirect(url_for('search_sentences'))

    # Приводим слово к нормальной форме
    normal_form = morph.parse(token_text)[0].normal_form

    # Получаем токен
    token = session.query(TokenID).filter_by(TextID=text_id, Token_text=normal_form).first()
    if not token:
        return redirect(url_for('search_sentences'))

    deleted_occurrences = 0

    for sentence_id in selected_sentences:
        # Удаляем связи в таблице Cross
        cross_entries = session.query(Cross).filter_by(SentenceID=sentence_id, TextID=text_id).all()
        deleted_occurrences += len(cross_entries)

        for entry in cross_entries:
            session.delete(entry)

    # Обновляем Token_count
    if deleted_occurrences > 0:
        token.Token_count = max(0, token.Token_count - deleted_occurrences)

    session.commit()

    # Обновляем список предложений после удаления
    return redirect(url_for('search_sentences', text_id=text_id, token_text=token_text))

if __name__ == '__main__':
    app.run(debug=True)
