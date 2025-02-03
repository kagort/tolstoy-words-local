from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy.orm import sessionmaker
from project_db.db3_from_csv import *
from project_db.model_3 import *

app = Flask(__name__)

# Настройка базы данных
Session = sessionmaker(bind=engine)
session = Session()

@app.route('/', methods=['GET', 'POST'])
def search_sentences():
    if request.method == 'POST':
        text_id = request.form.get('text_id')
        token_text = request.form.get('token_text')

        # Получение токена
        token = session.query(TokenID).filter_by(TextID=text_id, Token_text=token_text).first()
        if not token:
            return render_template('index0.html', error="Токен не найден.")

        # Получение предложений с вхождениями токена
        sentences = session.query(Sentences).filter(
            Sentences.TextID == text_id,
            Sentences.Sentence_text.ilike(f"%{token_text}%")
        ).all()

        # Обработка предложений: выделение токена
        highlighted_sentences = []
        for sentence in sentences:
            highlighted_text = sentence.Sentence_text.replace(
                token_text,
                f"<span class='highlight'>{token_text}</span>"
            )
            highlighted_sentences.append({
                'id': sentence.SentenceID,
                'text': highlighted_text
            })

        return render_template(
            'index0.html',
            sentences=highlighted_sentences,
            text_id=text_id,
            token_text=token_text
        )

    return render_template('index0.html')

@app.route('/delete', methods=['POST'])
def delete_sentences():
    selected_sentences = request.form.getlist('selected_sentences')

    for sentence_id in selected_sentences:
        sentence = session.query(Sentences).filter_by(SentenceID=sentence_id).first()
        if sentence:
            session.delete(sentence)

    session.commit()
    return redirect(url_for('search_sentences'))

if __name__ == '__main__':
    app.run(debug=True)