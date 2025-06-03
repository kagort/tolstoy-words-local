import pandas as pd
from sqlalchemy import func, select
from database.model_3 import TokenID, Words, Sentences, Cross, engine
from collections import defaultdict
from itertools import count

# --- 1. Общая частотность токенов по корпусу ---
token_freq_query = (
    select(
        TokenID.TokenID,
        TokenID.Token_text,
        func.sum(TokenID.Token_count).label('total_token_count')
    )
    .group_by(TokenID.TokenID, TokenID.Token_text)
)

# --- 2. Количество предложений на каждый токен ---
sentence_count_query = (
    select(
        Cross.TokenID,
        func.count(func.distinct(Cross.SentenceID)).label('sentence_count')
    )
    .group_by(Cross.TokenID)
)

# --- 3. Количество зависимых слов по каждому токену ---
dependent_word_count_query = (
    select(
        Cross.TokenID,
        func.count(Words.WordID).label('dependent_word_count')
    )
    .outerjoin(Words, Words.WordID == Cross.WordID)
    .group_by(Cross.TokenID)
)

# --- 4. Виды грамматических связей (POS зависимых слов) + их частота ---
pos_dependency_query = (
    select(
        Cross.TokenID,
        Words.Part_of_speech.label('dependent_pos'),
        func.count(Words.WordID).label('pos_count')
    )
    .join(Words, Words.WordID == Cross.WordID)
    .group_by(Cross.TokenID, Words.Part_of_speech)
)

# --- 5. Длина каждого предложения (количество слов в нём) ---
sentence_word_count_query = (
    select(
        Cross.SentenceID,
        func.count(Words.WordID).label("word_count")
    )
    .join(Words, Words.WordID == Cross.WordID)
    .group_by(Cross.SentenceID)
)

# --- Выполняем все запросы ---
with engine.connect() as conn:
    df_tokens = pd.DataFrame(conn.execute(token_freq_query).fetchall())
    df_sentences = pd.DataFrame(conn.execute(sentence_count_query).fetchall())
    df_dependents = pd.DataFrame(conn.execute(dependent_word_count_query).fetchall())
    df_pos = pd.DataFrame(conn.execute(pos_dependency_query).fetchall())
    df_sentence_lengths = pd.DataFrame(conn.execute(sentence_word_count_query).fetchall())

# --- Переименовываем столбцы для удобства ---
df_tokens.columns = ["TokenID", "Token_text", "total_token_count"]
df_sentences.columns = ["TokenID", "sentence_count"]
df_dependents.columns = ["TokenID", "dependent_word_count"]
df_pos.columns = ["TokenID", "Dependent_POS", "POS_Count"]
df_sentence_lengths.columns = ["SentenceID", "word_count"]

# --- Убедимся, что word_count > 0 (фильтруем пустые предложения) ---
df_sentence_lengths = df_sentence_lengths[df_sentence_lengths['word_count'] > 0]

# --- Формируем зависимости POS ---
def format_pos(group):
    return ', '.join([f"{row['Dependent_POS']}: {row['POS_Count']}" for _, row in group.iterrows()])

df_pos_grouped = df_pos.groupby("TokenID", group_keys=False).apply(format_pos).reset_index(name="POS_Dependencies")

# --- Объединяем всё в один датафрейм ---
df_final = df_tokens.merge(df_sentences, on="TokenID", how="left")
df_final = df_final.merge(df_dependents, on="TokenID", how="left")
df_final = df_final.merge(df_pos_grouped, on="TokenID", how="left")

# --- Добавляем SentenceID в df_final ---
# Получаем список всех пар (TokenID, SentenceID)
with engine.connect() as conn:
    df_cross = pd.DataFrame(conn.execute(select(Cross.TokenID, Cross.SentenceID)).fetchall())
df_cross.columns = ['TokenID', 'SentenceID']

# --- Объединяем с длинами предложений ---
df_token_sentences = df_cross.merge(df_sentence_lengths, on='SentenceID', how='inner')

# --- Фильтруем: убираем предложения с нулевой длиной ---
df_token_sentences = df_token_sentences[df_token_sentences['word_count'] > 0]

# --- Считаем статистику по предложениям для каждого токена ---
sentence_stats = (
    df_token_sentences.groupby('TokenID')['word_count']
    .agg(['mean', 'median'])
    .rename(columns={'mean': 'avg_sentence_length', 'median': 'median_sentence_length'})
    .round(2)
)

# --- Объединяем статистику обратно с основным датафреймом ---
df_final = df_final.merge(sentence_stats, on='TokenID', how='left')

# --- Замена NaN и приведение типов ---
df_final["sentence_count"] = df_final["sentence_count"].fillna(0).astype(int)
df_final["dependent_word_count"] = df_final["dependent_word_count"].fillna(0).astype(int)
df_final["avg_sentence_length"] = df_final["avg_sentence_length"].fillna(0).astype(float)
df_final["median_sentence_length"] = df_final["median_sentence_length"].fillna(0).astype(float)

# --- Группировка по Token_text ---
# Убедимся, что строки не NaN
df_final['POS_Dependencies'] = df_final['POS_Dependencies'].fillna('')

# Создаём новую группировку по Token_text
grouped = df_final.groupby('Token_text', as_index=False).agg({
    'total_token_count': 'sum',
    'sentence_count': 'sum',
    'dependent_word_count': 'sum',
    'avg_sentence_length': 'mean',
    'median_sentence_length': 'median',
    'POS_Dependencies': lambda x: '|'.join([dep for dep in x if dep])
})

# --- Очистка и объединение POS зависимостей ---
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
    return ', '.join(f"{pos}: {cnt}" for pos, cnt in sorted(pos_counter.items()))

grouped['POS_Dependencies'] = grouped.apply(merge_pos_dependencies, axis=1)

# --- Присваиваем новые уникальные ID ---
id_gen = count(start=1)
grouped.insert(0, 'TokenID', [next(id_gen) for _ in range(len(grouped))])

# --- Показываем результат ---
print(grouped[['TokenID', 'Token_text', 'total_token_count', 'sentence_count',
               'dependent_word_count', 'avg_sentence_length', 'median_sentence_length',
               'POS_Dependencies']].head(15))

# --- Сохраняем в CSV ---
grouped.to_csv("merged_tokens_with_sentence_stats.csv", index=False, encoding='utf-8-sig')