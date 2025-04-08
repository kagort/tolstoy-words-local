import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Подключение к базе данных
conn = psycopg2.connect(
    dbname="tolstoy_words_csv",
    user="postgres",
    password="ouganda77",
    host="localhost",
    port="5432"
)

# SQL-запрос для извлечения данных
query = """
WITH text_metrics AS (
    -- Общие метрики по текстам: общее количество предложений, с токенами, процент
    SELECT
        t."TextID",
        t."TextTitle",
        COUNT(DISTINCT s."SentenceID") AS total_sentences,
        COUNT(DISTINCT CASE WHEN tk."Token_count" > 0 THEN s."SentenceID" END) AS token_sentences,
        ROUND(
            (COUNT(DISTINCT CASE WHEN tk."Token_count" > 0 THEN s."SentenceID" END) * 100.0 /
            NULLIF(COUNT(DISTINCT s."SentenceID"), 0)),
            2
        ) AS token_sentence_percent
    FROM public.dictexts t
    LEFT JOIN public.sentences s ON t."TextID" = s."TextID"
    LEFT JOIN public."cross" c ON s."SentenceID" = c."SentenceID"
    LEFT JOIN public.tokenid tk ON c."TokenID" = tk."TokenID" AND t."TextID" = tk."TextID"
    GROUP BY t."TextID", t."TextTitle"
),
avg_words AS (
    -- Среднее количество слов в предложениях с токенами
    SELECT
        token_sentences."TextID",
        ROUND(AVG(word_counts.word_count), 2) AS avg_words_per_token_sentence
    FROM (
        SELECT DISTINCT
            s."TextID",
            s."SentenceID"
        FROM public.sentences s
        JOIN public."cross" c ON s."SentenceID" = c."SentenceID"
        JOIN public.tokenid tk ON c."TokenID" = tk."TokenID"
        WHERE tk."Token_count" > 0
    ) AS token_sentences
    CROSS JOIN LATERAL (
        SELECT COUNT(*) AS word_count
        FROM public."cross" c
        WHERE c."SentenceID" = token_sentences."SentenceID"
    ) AS word_counts
    GROUP BY token_sentences."TextID"
)
-- Финальный вывод
SELECT
    tm."TextID",
    tm."TextTitle",
    tm.total_sentences,
    tm.token_sentences,
    aw.avg_words_per_token_sentence
FROM text_metrics tm
LEFT JOIN avg_words aw ON tm."TextID" = aw."TextID"
ORDER BY tm.token_sentences DESC;
"""

# Загрузка данных в DataFrame
df = pd.read_sql(query, conn)

# Закрытие соединения
conn.close()

# Построение диаграммы рассеяния
plt.figure(figsize=(10, 6))

# Данные для графика
x = df['token_sentences']  # Количество предложений с токенами
y = df['avg_words_per_token_sentence']  # Средняя длина предложений с токенами
texts = df['TextTitle']  # Названия текстов для аннотаций

# Создание точек
plt.scatter(x, y, color='green', s=100, alpha=0.7)

# Добавление аннотаций для каждой точки
for i, txt in enumerate(texts):
    plt.annotate(txt, (x[i], y[i]), fontsize=8, ha='right')

# Настройка графика
plt.title("Диаграмма рассеяния: Распределение текстов", fontsize=14)
plt.xlabel("Количество предложений с токенами", fontsize=12)
plt.ylabel("Средняя длина предложений с токенами", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Отображение графика
plt.tight_layout()
plt.show()