import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database.db3_from_csv import *
from database.model_3 import *
# Инициализация базы данных

engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
db_session = scoped_session(sessionmaker(bind=engine))
session = Session()

# Инициализация Dash
app = dash.Dash(__name__)
app.title = "Лингвистический анализ"

# Получаем список текстов
texts = session.query(DicTexts).all()
text_options = [{"label": f"{t.TextTitle} ({t.Text_Author})", "value": t.TextID} for t in texts]

# Макет
app.layout = html.Div([
    html.H1("Интерфейс для лингвистического анализа"),

    html.Label("Выберите текст:"),
    dcc.Dropdown(id="text-dropdown", options=text_options, value=text_options[0]['value'] if text_options else None),

    html.Br(),
    html.H2("Токены в тексте:"),
    dcc.Graph(id="token-bar"),

    html.Br(),
    html.H2("Слова, связанные с токеном:"),
    dash_table.DataTable(id="word-table", columns=[
        {"name": "Слово", "id": "Word_text"},
        {"name": "Часть речи", "id": "Part_of_speech"},
        {"name": "Частота", "id": "Frequency"}
    ], page_size=10),

    html.Br(),
    html.H2("Предложения с выбранным словом:"),
    html.Div(id="sentence-output")
])

# Коллбек: от текста к токенам
@app.callback(
    Output("token-bar", "figure"),
    Input("text-dropdown", "value")
)
def update_token_bar(text_id):
    tokens = session.query(TokenID).filter_by(TextID=text_id).all()
    df = pd.DataFrame([{"Token": t.Token_text, "Count": t.Token_count, "TokenID": t.TokenID} for t in tokens])
    fig = {
        "data": [{"x": df["Token"], "y": df["Count"], "type": "bar", "customdata": df["TokenID"], "hoverinfo": "x+y"}],
        "layout": {"clickmode": "event+select", "title": "Частотность токенов"}
    }
    return fig

# Коллбек: от токена к словам
@app.callback(
    Output("word-table", "data"),
    Input("token-bar", "clickData"),
    State("text-dropdown", "value")
)
def update_words(clickData, text_id):
    if not clickData:
        return []

    token_id = clickData["points"][0]["customdata"]
    words = session.query(Words).filter_by(TextID=text_id, TokenID=token_id).all()
    return [{"Word_text": w.Word_text, "Part_of_speech": w.Part_of_speech, "Frequency": w.Frequency} for w in words]

# Коллбек: от слова к предложениям
@app.callback(
    Output("sentence-output", "children"),
    Input("word-table", "active_cell"),
    State("word-table", "data"),
    State("text-dropdown", "value")
)
def update_sentences(cell, rows, text_id):
    if not cell:
        return "Выберите слово для просмотра предложений."
    word_text = rows[cell["row"]]["Word_text"]

    word_obj = session.query(Words).filter_by(Word_text=word_text, TextID=text_id).first()
    if not word_obj:
        return "Слово не найдено."

    crosses = session.query(Cross).filter_by(WordID=word_obj.WordID).all()
    sentence_ids = [c.SentenceID for c in crosses]
    sentences = session.query(Sentences).filter(Sentences.SentenceID.in_(sentence_ids)).all()

    return html.Ul([html.Li(s.Sentence_text) for s in sentences])

# Запуск
if __name__ == "__main__":
    app.run(debug=True)
