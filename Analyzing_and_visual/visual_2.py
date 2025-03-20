import pandas as pd
import plotly.express as px

# Загрузка данных из CSV-файла
file_path = '2.csv'  # Укажите путь к вашему CSV-файлу
df = pd.read_csv(file_path)

# Переименование столбцов для удобства (если необходимо)
df.columns = [
    "Title", "Author", "Year", "Genre", "KeyWord", "DependentWord", "PartOfSpeech", "Frequency"
]

# Фильтрация данных: выбираем только строки с ключевым словом "запах"
df = df[df['KeyWord'] == 'запах']

# Группировка данных по авторам, частям речи и зависимым словам
grouped = (
    df.groupby(['Author', 'PartOfSpeech', 'DependentWord'])['Frequency']
    .sum()
    .reset_index()
)

# Создаем столбец с текстом для всплывающих подсказок
grouped['Tooltip'] = grouped.apply(
    lambda row: f"{row['DependentWord']} ({row['Frequency']})", axis=1
)

# Группируем данные по авторам и частям речи
pos_grouped = (
    grouped.groupby(['Author', 'PartOfSpeech'])
    .agg({'Frequency': 'sum', 'Tooltip': lambda x: '<br>'.join(x)})
    .reset_index()
)

# Сортируем авторов по общей частотности в порядке убывания
author_total_freq = pos_grouped.groupby('Author')['Frequency'].sum().reset_index()
author_total_freq = author_total_freq.sort_values(by='Frequency', ascending=False)
sorted_authors = author_total_freq['Author'].tolist()

# Добавляем порядковый номер автора для сортировки
pos_grouped['AuthorOrder'] = pos_grouped['Author'].apply(sorted_authors.index)

# Создание интерактивного графика
fig = px.bar(
    pos_grouped,
    x='Author',
    y='Frequency',
    color='PartOfSpeech',
    hover_data={'Tooltip': True},
    title='Частотность зависимых слов для ключевого слова "запах" по авторам',
    labels={'Frequency': 'Частотность', 'Author': 'Автор'},
    category_orders={'Author': sorted_authors},  # Сортировка авторов
    text='Frequency'  # Отображение значений над столбцами
)

# Настройка внешнего вида графика
fig.update_traces(
    hovertemplate=(
        '<b>Часть речи:</b> %{customdata[0]}<br>' +
        '<b>Частотность:</b> %{y}<br>' +
        '<b>Слова:</b><br>%{customdata[1]}'
    ),
    customdata=pos_grouped[['PartOfSpeech', 'Tooltip']]
)
fig.update_layout(
    xaxis_title='Автор',
    yaxis_title='Частотность',
    legend_title='Части речи',
    hoverlabel=dict(bgcolor="white", font_size=12),
    xaxis={'categoryorder': 'array', 'categoryarray': sorted_authors}
)

# Отображение графика
fig.show()