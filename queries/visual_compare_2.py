import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from collections import defaultdict

# Загрузка данных
df = pd.read_csv('analytical_table.csv')

# Дополнительные вычисления:
# 1. Самый частотный токен в каждом тексте
top_tokens = (
    df.groupby('TextTitle')['TokenTotalCount']
    .idxmax()
    .apply(lambda x: df.loc[x, 'TokenText'])
    .reset_index(name='TopToken')
)

# 2. Рейтинг текстов по общему количеству вхождений токенов
text_ratings = (
    df.groupby('TextTitle')['TokenTotalCount']
    .sum()
    .reset_index(name='TotalTokens')
    .sort_values('TotalTokens', ascending=False)
    .reset_index(drop=True)
    .assign(Rank=lambda x: x['TotalTokens'].rank(ascending=False, method='first').astype(int))
)

# 3. Количество уникальных частей речи для каждого (TextTitle, TokenText)
pos_counts = (
    df.groupby(['TextTitle', 'TokenText'])['PartOfSpeech']
    .nunique()
    .reset_index(name='UniquePOSCount')
)

# 4. Токен с максимальным количеством частей речи в тексте
max_pos_tokens = (
    pos_counts.groupby('TextTitle')
    .apply(lambda x: x.nlargest(1, 'UniquePOSCount'))
    .reset_index(drop=True)
)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Анализ текстов", style={'text-align': 'center'}),

    # SunBurst-диаграмма (текст -> токен -> часть речи)
    dcc.Graph(
        id='sunburst',
        figure=px.sunburst(
            df,
            path=['TextTitle', 'TokenText', 'PartOfSpeech'],
            values='Frequency',
            color='PartOfSpeech',
            title='Иерархия частотности: Текст → Токен → Часть речи'
        )
    ),

    # Выбор текста для детального анализа
    dcc.Dropdown(
        id='text-selector',
        options=[{'label': text, 'value': text} for text in df['TextTitle'].unique()],
        value=df['TextTitle'].unique()[0],
        style={'width': '50%', 'margin': '20px auto'}
    ),

    # Раздел для вывода метрик
    html.Div([
        html.Div(id='top-token', style={'margin': '20px'}),
        html.Div(id='text-rating', style={'margin': '20px'}),
        html.Div(id='max-pos-token', style={'margin': '20px'})
    ], style={'display': 'flex', 'gap': '20px'})
])

# Исправленный коллбэк
@app.callback(
    [Output('top-token', 'children'),
     Output('text-rating', 'children'),
     Output('max-pos-token', 'children')],
    [Input('text-selector', 'value')]
)
def update_metrics(selected_text):
    # 1. Самый частотный токен
    top_token = top_tokens[top_tokens['TextTitle'] == selected_text]['TopToken'].values[0]

    # 2. Рейтинг текста
    rating_row = text_ratings[text_ratings['TextTitle'] == selected_text]
    text_rank = rating_row['Rank'].values[0]
    total_tokens = rating_row['TotalTokens'].values[0]

    # 3. Токен с наибольшим количеством частей речи
    max_pos_row = max_pos_tokens[max_pos_tokens['TextTitle'] == selected_text]
    token_with_max_pos = max_pos_row['TokenText'].values[0]
    pos_count = max_pos_row['UniquePOSCount'].values[0]

    return (
        html.Div([
            html.H3("Самый частотный токен:"),
            html.P(
                f"{top_token} (Частота: {df[(df['TextTitle'] == selected_text) & (df['TokenText'] == top_token)]['TokenTotalCount'].values[0]})")
        ]),

        html.Div([
            html.H3("Рейтинг текста:"),
            html.P(f"Место: {text_rank} (Всего токенов: {total_tokens})")
        ]),

        html.Div([
            html.H3("Токен с наибольшим количеством частей речи:"),
            html.P(f"{token_with_max_pos} (Части речи: {pos_count})")
        ])
    )

if __name__ == '__main__':
    app.run_server(debug=True)