import pandas as pd
import pymorphy3
from database.model import Base, TokenID, Sentences, Words, Cross
from commons.config.db_config import engine_natural
from text_processing.base_text_functions import count_words
from text_processing.dataframe_functions import expand_pos_column
from sqlalchemy import (
    select, func, distinct, literal_column, true, join, case, Table, MetaData
)
from sqlalchemy.orm import aliased
from sqlalchemy.sql import lateral


def prepare_regex_table(conn):
    import re
    from sqlalchemy import Table, Column, Text, MetaData

    morph = pymorphy3.MorphAnalyzer()

    tokens = [t for (t,) in conn.execute(
        select(TokenID.Token_text).distinct()
    )]

    rows = []
    for lemma in tokens:
        forms = {f.word.lower() for f in morph.parse(lemma)[0].lexeme}
        # Убираем границы слова, чтобы ловить "веять" и "развеять"
        regex = r'(?:' + '|'.join(re.escape(w) for w in forms) + r')'
        rows.append({'token_text': lemma, 'pattern': regex})

    conn.exec_driver_sql("DROP TABLE IF EXISTS tmp_token_regex")
    conn.exec_driver_sql("""
        CREATE TEMP TABLE tmp_token_regex (
            token_text text PRIMARY KEY,
            pattern    text NOT NULL
        ) ON COMMIT PRESERVE ROWS
    """)

    tmp = Table('tmp_token_regex', MetaData(),
                Column('token_text', Text, primary_key=True),
                Column('pattern',    Text))

    conn.execute(tmp.insert(), rows)


def build_queries(conn):
    # уникальные леммы
    tok_subq = (
        select(TokenID.Token_text.label('token_text'))
        .distinct()
        .subquery(name='tok')
    )
    tok = aliased(tok_subq)

    # алиасы
    s  = aliased(Sentences)
    w  = Words
    cr = Cross

    # temp-таблица regex
    tr = Table('tmp_token_regex', MetaData(), autoload_with=conn)

    # lateral-подзапрос на матчи в одном предложении
    rm_lateral = lateral(
        func.regexp_matches(
            func.lower(s.Sentence_text),
            tr.c.pattern,
            literal_column("'g'")
        )
    ).alias('rm')

    # lateral: считаем сколько матчей в каждом предложении
    matches_lateral = (
        select(
            s.SentenceID.label('sid'),
            func.count().label('cnt')
        )
        .select_from(
            join(s, tr, tr.c.token_text == tok.c.token_text)
            .join(rm_lateral, true())
        )
        .group_by(s.SentenceID)
        .lateral()
        .alias('m')
    )

    # total вхождений (сумма всех cnt)
    token_freq_query = (
        select(
            tok.c.token_text.label('Token_text'),
            func.coalesce(func.sum(matches_lateral.c.cnt), 0)
                .label('total_token_count')
        )
        .select_from(tok)
        .outerjoin(matches_lateral, true())
        .group_by(tok.c.token_text)
    )

    # sentence_count = число предложений с хоть одним вхождением
    sentence_count_query = (
        select(
            tok.c.token_text.label('Token_text'),
            func.coalesce(
                func.count(distinct(matches_lateral.c.sid)),
                0
            ).label('sentence_count')
        )
        .select_from(tok)
        .outerjoin(matches_lateral, true())
        .group_by(tok.c.token_text)
    )

    # dependent_word_count — distinct WordID
    tok2id = join(TokenID, tok, TokenID.Token_text == tok.c.token_text)
    dependent_word_count_query = (
        select(
            tok.c.token_text.label('Token_text'),
            func.count(distinct(w.WordID)).label('dependent_word_count')
        )
        .select_from(tok2id)
        .outerjoin(cr, cr.TokenID == TokenID.TokenID)
        .outerjoin(w,  w.WordID  == cr.WordID)
        .group_by(tok.c.token_text)
    )

    # pos_dependency = distinct WordID в разрезе POS
    pos_dependency_query = (
        select(
            tok.c.token_text.label('Token_text'),
            w.Part_of_speech.label('dependent_pos'),
            func.count(distinct(w.WordID)).label('pos_count')
        )
        .select_from(tok2id)
        .outerjoin(cr, cr.TokenID == TokenID.TokenID)
        .outerjoin(w,  w.WordID  == cr.WordID)
        .group_by(tok.c.token_text, w.Part_of_speech)
    )

    # предложения с токенами
    sentence_with_words_query = (
        select(
            tok.c.token_text.label('Token_text'),
            s.Sentence_text
        )
        .select_from(tok2id)
        .join(cr, cr.TokenID == TokenID.TokenID)
        .join(s,  s.SentenceID == cr.SentenceID)
        .distinct(tok.c.token_text, s.SentenceID)
    )

    return {
        'token_freq'          : token_freq_query,
        'sentence_count'      : sentence_count_query,
        'dependent_word_count': dependent_word_count_query,
        'pos_dependency'      : pos_dependency_query,
        'sentence_words'      : sentence_with_words_query
    }


def load_data(conn, queries):
    return {
        name: pd.DataFrame(conn.execute(q).fetchall())
        for name, q in queries.items()
    }


def main():
    print("Запуск анализа токенов…")

    with engine_natural.begin() as conn:
        prepare_regex_table(conn)
        queries = build_queries(conn)
        data = load_data(conn, queries)

    # длины предложений и статистика по ним
    df_sw = data['sentence_words']
    df_sw.columns = ['Token_text', 'Sentence_text']
    df_sw['word_count'] = df_sw['Sentence_text'].str.lower().apply(count_words)

    # <-- НОВОЕ ПОЛЕ: var_word_count
    word_stats = (
        df_sw
        .groupby('Token_text', as_index=False)['word_count']
        .agg(
            avg_word_count='mean',
            median_word_count='median',
            var_word_count='var'  # <-- Вычисляем дисперсию
        )
    )

    # прочие метрики: приводим все dataframes к единым именам
    df_tokens     = data['token_freq']
    df_tokens.columns     = ['Token_text', 'total_token_count']

    df_sentences  = data['sentence_count']
    df_sentences.columns  = ['Token_text', 'sentence_count']

    df_dependents = data['dependent_word_count']
    df_dependents.columns = ['Token_text', 'dependent_word_count']

    df_pos        = data['pos_dependency']
    df_pos.columns        = ['Token_text', 'dependent_pos', 'pos_count']

    df_pos_grouped = (
        df_pos
        .groupby('Token_text', group_keys=False)
        .apply(lambda g: ', '.join(
            f"{row.dependent_pos}: {row.pos_count}"
            for _, row in g.iterrows()
        ))
        .reset_index(name='POS_Dependencies')
    )

    df_final = (
        df_tokens
        .merge(df_sentences,   on='Token_text', how='left')
        .merge(df_dependents,  on='Token_text', how='left')
        .merge(df_pos_grouped, on='Token_text', how='left')
        .merge(word_stats,     on='Token_text', how='left')
    )

    df_final['avg_word_count']       = df_final['avg_word_count'].fillna(0).round(2)
    df_final['var_word_count']       = df_final['var_word_count'].fillna(0).round(2)  # <-- ФОРМАТИРОВАНИЕ
    df_final['median_word_count']    = df_final['median_word_count'].fillna(0).astype(int)
    df_final['sentence_count']       = df_final['sentence_count'].fillna(0).astype(int)
    df_final['dependent_word_count'] = df_final['dependent_word_count'].fillna(0).astype(int)

    df_sorted = df_final.sort_values('total_token_count', ascending=False).reset_index(drop=True)
    df_sorted.insert(0, 'TokenID', range(1, len(df_sorted) + 1))

    expanded = expand_pos_column(df_sorted, 'POS_Dependencies') \
                 .drop(columns='POS_Dependencies', errors='ignore')

    print("\nТОП-20 токенов:")
    print(expanded[['Token_text', 'avg_word_count', 'var_word_count', 'total_token_count']].head(20))  # <-- Вывод

    out_file = "natural_tokens_expanded_var.xlsx"
    expanded.to_excel(out_file, index=False, engine='openpyxl')
    print(f"\nРезультат сохранён в «{out_file}»")


if __name__ == "__main__":
    main()