
from table_test import get_token_data, target_token_ids
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Функция для сбора данных о всех текстах и токенах
def collect_data():
    data = []
    text_titles = {}  # Для хранения названий текстов по TextID

    for text_id in [1, 2, 4, 5, 8, 9]:  # Обрабатываем оба текста
        for token_id in target_token_ids:
            result = get_token_data(text_id, token_id)

            if result is None:
                continue  # Пропускаем, если данных нет

            # Сохраняем название текста
            text_title = result['token_data'].Text_Title
            text_titles[text_id] = text_title

            # Добавляем данные в список
            data.append({
                'TextID': text_id,
                'TextTitle': text_title,
                'Token': result['token_data'].Token_Text,
                'TotalSentences': result['total_sentences'],
                'TokenCount': result['token_data'].Token_Total_Count,
                'DependentWords': sum(word.Total_Frequency for word in result['dependent_words']),
                'SentencesWithToken': result['token_data'].Sentence_Count,
                'PartsOfSpeech': {word.Part_of_speech: word.Total_Frequency for word in result['dependent_words']}
            })

    return data, text_titles


# Функция для создания дашборда
def create_dashboard(data):
    # Преобразуем данные в DataFrame
    df = pd.DataFrame(data)

    # График 1: Сравнение общего количества предложений в текстах
    fig_total_sentences = px.bar(
        df.groupby('TextTitle', as_index=False).agg({'TotalSentences': 'sum'}),
        x='TextTitle',
        y='TotalSentences',
        title='Общее количество предложений в текстах',
        labels={'TotalSentences': 'Количество предложений', 'TextTitle': 'Текст'}
    )

    # График 2: Сравнение частоты вхождений токенов
    fig_token_count = px.bar(
        df,
        x='Token',
        y='TokenCount',
        color='TextTitle',
        barmode='group',
        title='Частота вхождений токенов в текстах',
        labels={'TokenCount': 'Количество вхождений', 'Token': 'Токен'}
    )
    fig_token_count.update_xaxes(tickangle=45)

    # График 3: Сравнение зависимых слов по частям речи
    pos_data = []
    for row in data:
        for part_of_speech, frequency in row['PartsOfSpeech'].items():
            pos_data.append({
                'TextTitle': row['TextTitle'],
                'Token': row['Token'],
                'PartOfSpeech': part_of_speech,
                'Frequency': frequency
            })

    pos_df = pd.DataFrame(pos_data)
    fig_parts_of_speech = px.bar(
        pos_df,
        x='Token',
        y='Frequency',
        color='PartOfSpeech',
        facet_col='TextTitle',
        title='Зависимые слова по частям речи',
        labels={'Frequency': 'Частота', 'Token': 'Токен', 'PartOfSpeech': 'Часть речи'}
    )
    fig_parts_of_speech.update_xaxes(tickangle=45)

    # Создание дашборда
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Сравнение текстов", style={'text-align': 'center'}),

        # График 1
        dcc.Graph(figure=fig_total_sentences),

        # График 2
        dcc.Graph(figure=fig_token_count),

        # График 3
        dcc.Graph(figure=fig_parts_of_speech),
    ])

    return app


# Основная функция
if __name__ == "__main__":
    print("Сбор данных...")
    data, _ = collect_data()

    if not data:
        print("Нет данных для визуализации.")
    else:
        print("Создание дашборда...")
        app = create_dashboard(data)
        app.run_server(debug=True)