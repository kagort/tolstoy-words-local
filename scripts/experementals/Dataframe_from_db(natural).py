import pandas as pd
from sqlalchemy import func, select, distinct
from itertools import count
from database.model import *

from commons.config.db_config import engine_natural
from text_processing.base_text_functions import count_words
from text_processing.dataframe_functions import expand_pos_column


# --- SQL-запросы ---
def build_queries():
    token_freq_query = (
        select(
            TokenID.Token_text,
            func.sum(TokenID.Token_count).label('total_token_count')
        )
        .group_by(TokenID.Token_text)
    )

    # sentence_count
    sentence_count_query = (
        select(
            TokenID.Token_text,
            func.count(distinct(Cross.SentenceID)).label('sentence_count')
        )
        .join(Cross, Cross.TokenID == TokenID.TokenID)  # <- outer!
        .group_by(TokenID.Token_text)
    )

    # dependent words
    dependent_word_count_query = (
        select(
            TokenID.Token_text,
            func.count(distinct(Words.Word_text)).label('dependent_word_count')
        )
        .join(Cross, Cross.TokenID == TokenID.TokenID)
        .join(Words, Words.WordID == Cross.WordID)
        .group_by(TokenID.Token_text)
    )

    # POS
    pos_dependency_query = (
        select(
            TokenID.Token_text,
            Words.Part_of_speech.label('dependent_pos'),
            func.count(distinct(Words.Word_text)).label('pos_count')
        )
        .join(Cross, Cross.TokenID == TokenID.TokenID)
        .join(Words, Words.WordID == Cross.WordID)
        .group_by(TokenID.Token_text, Words.Part_of_speech)
    )

    # sentences (без дублей)
    sentence_with_words_query = (
        select(
            TokenID.Token_text,
            Sentences.Sentence_text
        )
        .join(Cross, Cross.TokenID == TokenID.TokenID)
        .join(Sentences, Sentences.SentenceID == Cross.SentenceID)
        .distinct(TokenID.Token_text, Sentences.SentenceID)
    )

    return {
        'token_freq': token_freq_query,
        'sentence_count': sentence_count_query,
        'dependent_word_count': dependent_word_count_query,
        'pos_dependency': pos_dependency_query,
        'sentence_words': sentence_with_words_query
    }


# --- Загрузка данных из БД ---
def load_data(engine, queries):
    with engine.connect() as conn:
        df_tokens = pd.DataFrame(conn.execute(queries['token_freq']).fetchall())
        df_sentences = pd.DataFrame(conn.execute(queries['sentence_count']).fetchall())
        df_dependents = pd.DataFrame(conn.execute(queries['dependent_word_count']).fetchall())
        df_pos = pd.DataFrame(conn.execute(queries['pos_dependency']).fetchall())
        df_sentence_words = pd.DataFrame(conn.execute(queries['sentence_words']).fetchall())

    return {
        'tokens': df_tokens,
        'sentences': df_sentences,
        'dependents': df_dependents,
        'pos': df_pos,
        'sentence_words': df_sentence_words
    }


# --- Основная функция ---
def main():
    print("Запуск анализа токенов...")

    # Построение запросов
    queries = build_queries()

    # Подключение к базе и загрузка данных
    data = load_data(engine_natural, queries)

    # --- Подсчёт количества слов в предложениях ---
    df_sw = data['sentence_words']
    df_sw.columns = ['Token_text', 'Sentence_text']

    # Нормализуем регистр перед подсчётом
    df_sw['word_count'] = df_sw['Sentence_text'].str.lower().apply(count_words)

    word_stats = (
        df_sw.groupby('Token_text')['word_count']
        .agg(['mean', 'median'])
        .rename(columns={'mean': 'avg_word_count', 'median': 'median_word_count'})
        .reset_index()
    )

    # --- Обработка основных метрик ---
    df_tokens = data['tokens']
    df_tokens.columns = ['Token_text', 'total_token_count']

    df_sentences = data['sentences']
    df_sentences.columns = ['Token_text', 'sentence_count']

    df_dependents = data['dependents']
    df_dependents.columns = ['Token_text', 'dependent_word_count']

    df_pos = data['pos']
    df_pos.columns = ['Token_text', 'dependent_pos', 'pos_count']

    # --- Форматируем зависимости по частям речи ---
    def format_pos(group):
        return ', '.join([f"{row['dependent_pos']}: {row['pos_count']}" for _, row in group.iterrows()])

    df_pos_grouped = df_pos.groupby("Token_text", group_keys=False).apply(format_pos).reset_index(name="POS_Dependencies")

    # --- Объединение всех DF ---
    df_final = df_tokens.merge(df_sentences, on='Token_text', how='left')
    df_final = df_final.merge(df_dependents, on='Token_text', how='left')
    df_final = df_final.merge(df_pos_grouped, on='Token_text', how='left')
    df_final = df_final.merge(word_stats, on='Token_text', how='left')

    # --- Приведение типов ---
    df_final['sentence_count'] = df_final['sentence_count'].fillna(0).astype(int)
    df_final['dependent_word_count'] = df_final['dependent_word_count'].fillna(0).astype(int)
    df_final['avg_word_count'] = df_final['avg_word_count'].fillna(0).round(2)
    df_final['median_word_count'] = df_final['median_word_count'].fillna(0).astype(int)

    # --- Группировка по Token_text (итоговая агрегация) ---
    grouped = df_final.sort_values(by='total_token_count', ascending=False).reset_index(drop=True)

    # --- Присвоение новых уникальных ID ---
    id_gen = count(start=1)
    grouped.insert(0, 'TokenID', [next(id_gen) for _ in range(len(grouped))])

    # --- Расширяем POS в отдельные столбцы ---
    expanded_df = expand_pos_column(grouped, 'POS_Dependencies')

    # Удаляем старый столбец POS_Dependencies
    expanded_df.drop(columns=['POS_Dependencies'], inplace=True, errors='ignore')

    # --- Вывод и сохранение ---
    result = expanded_df

    print("\nТОП 20 токенов:")
    print(result.head(20))

    # Сохраняем в Excel, чтобы избежать интерпретации чисел как дат
    result.to_excel("merged_tokens_expanded_natural_new.xlsx", index=False, engine='openpyxl')
    print("\nРезультат с развернутыми POS сохранён в файл: merged_tokens_expanded.xlsx")


# --- Точка входа ---
if __name__ == "__main__":
    main()
