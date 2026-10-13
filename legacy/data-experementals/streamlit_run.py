import streamlit as st
import pandas as pd
from df_creation import olfactory_sentences_df

# Загрузка данных из DataFrame
def load_data():
    # Здесь предполагается, что `olfactory_sentences_df` уже создан и содержит все данные
    global olfactory_sentences_df
    return olfactory_sentences_df

# Загрузка данных
data = load_data()

# Преобразование сложных типов данных в строки (опционально)
# Если вы хотите сохранить сложные типы данных, можно пропустить этот шаг
data["Tokens"] = data["Tokens"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
data["Lemmas"] = data["Lemmas"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
data["POS"] = data["POS"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
data["POS_Frequency"] = data["POS_Frequency"].apply(str)

# Заголовок приложения


# Фильтр по авторам
st.subheader("Фильтр по авторам")
selected_authors = st.multiselect(
    "Выберите автора:",
    options=data['Author'].unique(),
    default=None
)

# Фильтрация данных
if selected_authors:
    filtered_data = data[data['Author'].isin(selected_authors)]
else:
    filtered_data = data

# Отображение таблицы
st.subheader("Таблица с результатами анализа")
st.dataframe(filtered_data, use_container_width=True)

# Дополнительные метрики (опционально)
st.subheader("Общая статистика")
st.write(f"Общее количество предложений: {len(filtered_data)}")
st.write(f"Уникальные авторы: {', '.join(filtered_data['Author'].unique())}")

# Интерактивный график (опционально)
st.subheader("График лексического разнообразия")
if not filtered_data.empty:
    st.bar_chart(filtered_data.groupby("Author")["Lexical_Diversity"].mean())