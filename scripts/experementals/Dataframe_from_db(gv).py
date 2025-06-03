import pandas as pd
from sqlalchemy import func, select
from database.model_3   import *

# 1. Общая частотность токенов по корпусу
token_freq_query = (
    select(
        TokenID.TokenID,
        TokenID.Token_text,
        func.sum(TokenID.Token_count).label('total_token_count')
    )
    .group_by(TokenID.TokenID, TokenID.Token_text)
)

# 2. Количество предложений на каждый токен
sentence_count_query = (
    select(
        Cross.TokenID,
        func.count(func.distinct(Cross.SentenceID)).label('sentence_count')
    )
    .group_by(Cross.TokenID)
)

# 3. Виды грамматических связей (POS зависимых слов) + их частота
pos_dependency_query = (
    select(
        Cross.TokenID,
        Words.Part_of_speech.label('dependent_pos'),
        func.count(Words.WordID).label('pos_count')
    )
    .join(Words, Words.WordID == Cross.WordID)
    .group_by(Cross.TokenID, Words.Part_of_speech)
)

# 4. Общее количество зависимых слов по каждому токену
dependent_word_count_query = (
    select(
        Cross.TokenID,
        func.count(Words.WordID).label('dependent_word_count')
    )
    .outerjoin(Words, Words.WordID == Cross.WordID)
    .group_by(Cross.TokenID)
)

# Выполняем все запросы
with engine.connect() as conn:
    df_tokens = pd.DataFrame(conn.execute(token_freq_query).fetchall())
    df_sentences = pd.DataFrame(conn.execute(sentence_count_query).fetchall())
    df_dependents = pd.DataFrame(conn.execute(dependent_word_count_query).fetchall())
    df_pos = pd.DataFrame(conn.execute(pos_dependency_query).fetchall())

# Переименовываем столбцы для удобства
df_tokens.columns = ["TokenID", "Token_text", "total_token_count"]
df_sentences.columns = ["TokenID", "sentence_count"]
df_dependents.columns = ["TokenID", "dependent_word_count"]
df_pos.columns = ["TokenID", "Dependent_POS", "POS_Count"]

# Функция для объединения POS зависимостей в строку
def format_pos(group):
    return ', '.join([f"{row['Dependent_POS']}: {row['POS_Count']}" for _, row in group.iterrows()])

# Группируем зависимости по TokenID
df_pos_grouped = df_pos.groupby("TokenID", group_keys=False).apply(format_pos).reset_index(name="POS_Dependencies")

# Объединяем всё в один датафрейм
df_final = df_tokens.merge(df_sentences, on="TokenID", how="left")
df_final = df_final.merge(df_dependents, on="TokenID", how="left")
df_final = df_final.merge(df_pos_grouped, on="TokenID", how="left")

# Замена NaN на 0 и приведение к целым числам
df_final["sentence_count"] = df_final["sentence_count"].fillna(0).astype(int)
df_final["dependent_word_count"] = df_final["dependent_word_count"].fillna(0).astype(int)

# Показываем результат
print(df_final.head())

# Сохраняем в CSV (если нужно)
df_final.to_csv("token_dependency_analysis.csv", index=False, encoding='utf-8-sig')

import pandas as pd
from itertools import count






# Убедимся, что строки не NaN
df_final['POS_Dependencies'] = df_final['POS_Dependencies'].fillna('')

# Создаём новую группировку по Token_text
grouped = df_final.groupby('Token_text', as_index=False).agg({
    'total_token_count': 'sum',
    'sentence_count': 'sum',
    'dependent_word_count': 'sum',
    'POS_Dependencies': lambda x: '|'.join([dep for dep in x if dep])
})

# Очищаем и объединяем POS зависимости, суммируя частотности
from collections import defaultdict


def merge_pos_dependencies(row):
    if not row['POS_Dependencies']:
        return ''

    pos_counter = defaultdict(int)
    deps = row['POS_Dependencies'].split('|')

    for dep_str in deps:
        for item in dep_str.split(', '):
            if ': ' in item:
                pos, cnt = item.split(': ')
                pos_counter[pos] += int(cnt)

    # Формируем строку
    return ', '.join(f"{pos}: {cnt}" for pos, cnt in sorted(pos_counter.items()))


# Применяем к каждой строке
grouped['POS_Dependencies'] = grouped.apply(merge_pos_dependencies, axis=1)

# Присваиваем новые уникальные ID
id_gen = count(start=1)
grouped.insert(0, 'New_TokenID', [next(id_gen) for _ in range(len(grouped))])

# Переименовываем столбцы
grouped.rename(columns={'New_TokenID': 'TokenID'}, inplace=True)

# Показываем результат
print(grouped[['TokenID', 'Token_text', 'total_token_count', 'sentence_count', 'dependent_word_count',
               'POS_Dependencies']].head(15))

# Сохранение в CSV (опционально)
grouped.to_csv("merged_tokens.csv", index=False, encoding='utf-8-sig')