import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Загрузка данных
file_path = r"C:\Users\User\PycharmProjects\Tolstoy_words_local\Datasets_from_pgAdmin\data.csv"
data = pd.read_csv(file_path, delimiter=",", quotechar='"', encoding="utf-8")

# Функция для безопасного преобразования JSON

def safe_json_parse(x):
    try:
        return json.loads(x.replace("'", "\"")) if isinstance(x, str) else {}
    except json.JSONDecodeError:
        return {}

# Преобразование JSON-полей
data["top_tokens"] = data["top_tokens"].apply(safe_json_parse)
data["top_pos_stats"] = data["top_pos_stats"].apply(safe_json_parse)

# Инициализация Dash приложения
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Анализ текстовых данных"),
    dcc.Graph(id="general-stats"),
    dcc.Graph(id="top-tokens"),
    dcc.Graph(id="pos-distribution"),
    dcc.Graph(id="author-timeline"),
    dcc.Interval(
        id="interval-component", interval=1000, n_intervals=0  # Обновление раз в секунду
    )
])

@app.callback(
    Output("general-stats", "figure"),
    Input("interval-component", "n_intervals")
)
def update_general_stats(_):
    total_sentences = data["total_sentences"].sum()
    token_sentences = data["token_sentences"].sum()
    avg_token_percent = data["token_sentence_percent"].mean()
    avg_words_per_sentence = data["avg_words_per_token_sentence"].mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Все предложения", "С токенами"], y=[total_sentences, token_sentences]))
    fig.add_trace(go.Scatter(x=["Средний %", "Среднее кол-во слов"], y=[avg_token_percent, avg_words_per_sentence], mode="lines+markers"))
    fig.update_layout(title="Общая статистика")
    return fig

@app.callback(
    Output("top-tokens", "figure"),
    Input("interval-component", "n_intervals")
)
def update_top_tokens(_):
    all_tokens = [item for sublist in data["top_tokens"] for item in sublist]
    tokens_df = pd.DataFrame(all_tokens).groupby("token")["count"].sum().reset_index().nlargest(10, "count")
    return px.bar(tokens_df, x="token", y="count", title="Топ-10 токенов")

@app.callback(
    Output("pos-distribution", "figure"),
    Input("interval-component", "n_intervals")
)
def update_pos_distribution(_):
    all_pos = [
        {"part_of_speech": pos["part_of_speech"], "frequency": pos["frequency"]}
        for pos_dict in data["top_pos_stats"] for key, stats in pos_dict.items() for pos in stats
    ]
    pos_df = pd.DataFrame(all_pos).groupby("part_of_speech")["frequency"].sum().reset_index()
    return px.pie(pos_df, names="part_of_speech", values="frequency", title="Распределение частей речи")

@app.callback(
    Output("author-timeline", "figure"),
    Input("interval-component", "n_intervals")
)
def update_author_timeline(_):
    return px.timeline(data, x_start="Text_year_creation", x_end="Text_year_creation", y="Text_Author", color="TextTitle", title="Таймлайн по годам")

if __name__ == "__main__":
    app.run_server(debug=True)
