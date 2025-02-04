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
    """Вывод всех случаев вхождения токена без повторов."""
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

        # Получаем все вхождения токена из Cross
        cross_entries = session.query(Cross).filter_by(TokenID=token.TokenID).all()

        # Создаем структуру для уникальных вхождений
        sentence_word_pairs = set()  # Используем `set`, чтобы исключить дубли

        # Заполняем `set` только уникальными комбинациями `SentenceID + WordID`
        for cross in cross_entries:
            sentence_word_pairs.add((cross.SentenceID, cross.WordID))

        # Получаем тексты предложений по уникальным SentenceID
        sentence_dict = {s.SentenceID: s.Sentence_text for s in session.query(Sentences).filter(
            Sentences.SentenceID.in_([s_id for s_id, _ in sentence_word_pairs])
        ).all()}

        # Формируем список без повторов
        highlighted_sentences = []
        for sentence_id, word_id in sorted(sentence_word_pairs):
            sentence_text = sentence_dict.get(sentence_id, "Текст не найден")
            highlighted_text = sentence_text.replace(
                token_text, f"<span class='highlight'>{token_text}</span>"
            )

            highlighted_sentences.append({
                'sentence_id': sentence_id,
                'word_id': word_id,
                'text': highlighted_text
            })

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
    """Удаление конкретных вхождений токена."""
    selected_entries = request.form.getlist('selected_sentences')
    text_id = request.form.get('text_id')
    token_text = request.form.get('token_text')

    if not selected_entries or not text_id or not token_text:
        return redirect(url_for('search_sentences'))

    # Приводим слово к нормальной форме
    normal_form = morph.parse(token_text)[0].normal_form

    # Получаем токен
    token = session.query(TokenID).filter_by(TextID=text_id, Token_text=normal_form).first()
    if not token:
        return redirect(url_for('search_sentences'))

    deleted_occurrences = 0

    for entry in selected_entries:
        sentence_id, word_id = map(int, entry.split("_"))  # Разделяем "SentenceID_WordID"

        # Удаляем конкретное вхождение из Cross
        cross_entry = session.query(Cross).filter_by(SentenceID=sentence_id, WordID=word_id,
                                                     TokenID=token.TokenID).first()
        if cross_entry:
            session.delete(cross_entry)
            deleted_occurrences += 1

    # Обновляем Token_count
    if deleted_occurrences > 0:
        token.Token_count = max(0, token.Token_count - deleted_occurrences)

    session.commit()

    # Обновляем список предложений после удаления
    return redirect(url_for('search_sentences', text_id=text_id, token_text=token_text))


if __name__ == '__main__':
    app.run(debug=True)
