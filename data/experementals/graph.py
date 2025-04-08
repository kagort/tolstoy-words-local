import psycopg2
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Подключение к базе данных
conn = psycopg2.connect(
    dbname="tolstoy_words_csv",
    user="postgres",
    password="ouganda77",
    host="localhost",
    port="5432"
)

# Запрос к базе данных
query = """
WITH text_metrics AS (
    -- Общие метрики по текстам: общее количество предложений, с токенами, процент
    SELECT
        t."TextID",
        t."TextTitle",
        t."Text_Author",
        t."Text_year_creation",
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
    GROUP BY t."TextID", t."TextTitle", t."Text_Author", t."Text_year_creation"
),
top_tokens AS (
    -- Топ-5 токенов по каждому тексту
    SELECT
        tk."TextID",
        jsonb_agg(
            jsonb_build_object(
                'token', tk."Token_text",
                'count', tk."Token_count"
            ) ORDER BY tk."Token_count" DESC
        ) AS top_tokens
    FROM public.tokenid tk
    WHERE tk."Token_count" > 0
    GROUP BY tk."TextID"
),
top_pos_per_token AS (
    -- Топ-5 частей речи зависимых слов для каждого токена
    SELECT
        tk."TextID",
        tk."Token_text",
        jsonb_agg(
            jsonb_build_object(
                'part_of_speech', sub.pos,
                'frequency', sub.freq,
                'examples', sub.examples
            ) ORDER BY sub.freq DESC
        ) FILTER (WHERE sub.pos IS NOT NULL) AS top_pos
    FROM public.tokenid tk
    LEFT JOIN LATERAL (
        SELECT
            w."Part_of_speech" AS pos,
            COUNT(*) AS freq,
            ARRAY_AGG(DISTINCT w."Word_text") AS examples
        FROM public.words w
        WHERE w."TokenID" = tk."TokenID"
        GROUP BY w."Part_of_speech"
    ) sub ON true
    WHERE tk."Token_count" > 0
    GROUP BY tk."TextID", tk."Token_text"
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
),
avg_words_for_token_sentence AS (
    -- Среднее количество слов в предложениях, содержащих любой токен
    SELECT
        t."TextID",
        ROUND(AVG(word_counts.word_count), 2) AS avg_words_for_token_sentence
    FROM public.dictexts t
    LEFT JOIN public.sentences s ON t."TextID" = s."TextID"
    LEFT JOIN public."cross" c ON s."SentenceID" = c."SentenceID"
    LEFT JOIN public.tokenid tk ON c."TokenID" = tk."TokenID"
    CROSS JOIN LATERAL (
        SELECT COUNT(*) AS word_count
        FROM public."cross" c
        WHERE c."SentenceID" = s."SentenceID"
    ) AS word_counts
    WHERE tk."Token_count" > 0
    GROUP BY t."TextID"
)
-- Финальный вывод
SELECT
    tm."TextID",
    tm."TextTitle",
    tm."Text_Author",
    tm."Text_year_creation",
    tm.total_sentences,
    tm.token_sentences,
    tm.token_sentence_percent,
    tt.top_tokens,
    aw.avg_words_per_token_sentence,
    awfts.avg_words_for_token_sentence,
    JSONB_OBJECT_AGG(
        'top_pos_' || tt_top."token",
        tppt.top_pos
    ) AS top_pos_stats
FROM text_metrics tm
LEFT JOIN top_tokens tt ON tm."TextID" = tt."TextID"
LEFT JOIN avg_words aw ON tm."TextID" = aw."TextID"
LEFT JOIN avg_words_for_token_sentence awfts ON tm."TextID" = awfts."TextID"
LEFT JOIN LATERAL (
    SELECT
        value->>'token' AS "token"
    FROM jsonb_array_elements(tt.top_tokens)
) tt_top ON true
LEFT JOIN top_pos_per_token tppt 
    ON tm."TextID" = tppt."TextID" AND tt_top."token" = tppt."Token_text"
GROUP BY
    tm."TextID",
    tm."TextTitle",
    tm."Text_Author",
    tm."Text_year_creation",
    tm.total_sentences,
    tm.token_sentences,
    tm.token_sentence_percent,
    tt.top_tokens,
    aw.avg_words_per_token_sentence,
    awfts.avg_words_for_token_sentence
ORDER BY tm."TextID";
"""

# Загрузка данных в DataFrame
df = pd.read_sql(query, conn)

# Закрытие соединения
conn.close()

# Создание графа
G = nx.DiGraph()

# Добавление узлов и ребер
for _, row in df.iterrows():
    text_id = row["TextID"]
    text_title = row["TextTitle"]
    author = row["Text_Author"]
    year = row["Text_year_creation"]
    top_tokens = row["top_tokens"]  # JSONB массив топ-токенов

    # Добавление узла для текста
    G.add_node(text_id, label=f"{text_title} ({author}, {year})", type="text")

    # Добавление узлов и ребер для топ-токенов
    if isinstance(top_tokens, list):
        for token in top_tokens[:5]:  # Берем только топ-5 токенов
            token_text = token["token"]
            token_count = token["count"]

            # Добавление узла для токена
            G.add_node(token_text, label=f"{token_text} (count={token_count})", type="token")

            # Добавление ребра от текста к токену
            G.add_edge(text_id, token_text, label=f"contains")

# Визуализация графа
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, seed=42)  # Расположение узлов
node_colors = {
    "text": "lightblue",
    "token": "orange"
}
colors = [node_colors[G.nodes[node]["type"]] for node in G.nodes]

nx.draw(
    G, pos, with_labels=True, node_size=3000, node_color=colors, font_size=10, font_weight="bold"
)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Граф связей между текстами и топ-токенами")
plt.show()

import plotly.graph_objects as go

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[G.nodes[node]['label'] for node in G.nodes()],
    textposition="bottom center",
    marker=dict(size=10, color='lightblue')
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(showlegend=False, hovermode='closest'))
fig.show()

nx.write_gexf(G, "text_token_graph.gexf")