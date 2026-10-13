import re
import pandas as pd
import pymorphy3
from sqlalchemy import (
    select, func, distinct, literal_column, true, join, Table, MetaData, bindparam
)
from sqlalchemy.orm import aliased
from sqlalchemy.sql import lateral

from database.model import Base, TokenID, Sentences, Words, Cross
from commons.config.db_config import engine_generic
from text_processing.base_text_functions import count_words
from text_processing.dataframe_functions import expand_pos_column


# функция для построения паттерна
def build_sql_boundary_pattern(forms):
    punctuation = r'\s+|[!"#$№%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~]+'
    start = rf'(?:^|{punctuation})'
    end = rf'(?:$|{punctuation})'
    parts = [f'{start}{re.escape(w)}{end}' for w in sorted(forms, key=len, reverse=True)]
    return r'(?:' + '|'.join(parts) + r')'


def prepare_regex_table(conn, text_id=2):
    from sqlalchemy import Table, Column, Text, MetaData

    morph = pymorphy3.MorphAnalyzer()

    # получаем все уникальные токены для TextID = 2
    tokens = [t for (t,) in conn.execute(
        select(TokenID.Token_text)
        .distinct()
        .where(TokenID.TextID == text_id)
    )]

    rows = []
    for lemma in tokens:
        parses = morph.parse(lemma)
        main = parses[0]
        pos = main.tag.POS

        if pos in ("INFN"):  # если токен - глагол в начальной форме, то берем INFN и главгольные словоформы
            allowed_pos = ("INFN", "VERB")
        else:
            allowed_pos = (pos,)

        forms_main = {wf.word.lower() for p in parses if p.tag.POS in allowed_pos
                      for wf in p.lexeme
                      if wf.tag.POS in allowed_pos} or {main.normal_form}

        # строим паттерн для всех форм, попавших в forms main
        regex = build_sql_boundary_pattern(forms_main)

        rows.append({'token_text': lemma, 'pattern': regex})

    # временная таблица с паттернами
    conn.exec_driver_sql("DROP TABLE IF EXISTS tmp_token_regex")
    conn.exec_driver_sql("""
        CREATE TEMP TABLE tmp_token_regex (
            token_text text PRIMARY KEY,
            pattern    text NOT NULL
        ) ON COMMIT PRESERVE ROWS
    """)

    tmp = Table('tmp_token_regex', MetaData(),
                Column('token_text', Text, primary_key=True),
                Column('pattern', Text))
    # сохраняем
    pd.Series([str(r) for r in rows]).to_csv("token_forms_raw.csv", index=False, header=False, encoding="utf-8-sig")
    conn.execute(tmp.insert(), rows)


def build_queries(conn, text_id=2):
    # Берем токены только для TextID = 2
    tok_subq = (
        select(TokenID.Token_text.label('token_text'))
        .distinct()
        .where(TokenID.TextID == text_id)
        .subquery(name='tok')
    )
    tok = aliased(tok_subq)

    s  = aliased(Sentences)
    w  = Words
    cr = Cross

    # временная таблица с regex для всех форм лемм
    tr = Table('tmp_token_regex', MetaData(), autoload_with=conn)

    # LATERAL: все совпадения regex в одном предложении (только для TextID = 2)
    rm_lateral = lateral(
        func.regexp_matches(
            func.lower(s.Sentence_text),
            tr.c.pattern,
            literal_column("'g'")
        )
    ).alias('rm')

    # LATERAL-подзапрос: для каждого предложения считаем число совпадений форм (только для TextID = 2)
    matches_lateral = (
        select(
            s.SentenceID.label('sid'),
            func.count().label('cnt')
        )
        .select_from(
            join(s, tr, tr.c.token_text == tok.c.token_text)
            .join(rm_lateral, true())
        )
        .where(s.TextID == text_id)  # Фильтр по TextID
        .group_by(s.SentenceID)
        .lateral()
        .alias('m')
    )

    # total_token_count: сумма всех совпадений по предложениям
    token_freq_query = (
        select(
            tok.c.token_text.label('Token_text'),
            func.coalesce(func.sum(matches_lateral.c.cnt), 0).label('total_token_count')
        )
        .select_from(tok)
        .outerjoin(matches_lateral, true())
        .group_by(tok.c.token_text)
    )

    # sentence_count: число предложений с хотя бы одним совпадением
    sentence_count_query = (
        select(
            tok.c.token_text.label('Token_text'),
            func.coalesce(func.count(distinct(matches_lateral.c.sid)), 0).label('sentence_count')
        )
        .select_from(tok)
        .outerjoin(matches_lateral, true())
        .group_by(tok.c.token_text)
    )

    # dependent_word_count_query с фильтром по TextID
    dependent_word_count_query = (
        select(
            tok.c.token_text.label('Token_text'),
            func.count(distinct(w.WordID)).label('dependent_word_count')
        )
        .select_from(
            join(tok, TokenID, TokenID.Token_text == tok.c.token_text)
            .outerjoin(cr, cr.TokenID == TokenID.TokenID)
            .outerjoin(w, w.WordID == cr.WordID)
        )
        .where(TokenID.TextID == text_id)  # Фильтр по TextID
        .where(w.TextID == text_id)  # Фильтр по TextID
        .group_by(tok.c.token_text)
    )

    # pos_dependency_query с фильтром по TextID
    pos_dependency_query = (
        select(
            tok.c.token_text.label('Token_text'),
            w.Part_of_speech.label('dependent_pos'),
            func.count(distinct(w.WordID)).label('pos_count')
        )
        .select_from(
            join(tok, TokenID, TokenID.Token_text == tok.c.token_text)
            .outerjoin(cr, cr.TokenID == TokenID.TokenID)
            .outerjoin(w, w.WordID == cr.WordID)
        )
        .where(TokenID.TextID == text_id)  # Фильтр по TextID
        .where(w.TextID == text_id)  # Фильтр по TextID
        .group_by(tok.c.token_text, w.Part_of_speech)
    )

    # sentence_with_words_query с фильтром по TextID
    sentence_with_words_query = (
        select(
            tok.c.token_text.label('Token_text'),
            s.Sentence_text
        )
        .select_from(
            join(tok, TokenID, TokenID.Token_text == tok.c.token_text)
            .join(cr, cr.TokenID == TokenID.TokenID)
            .join(s, s.SentenceID == cr.SentenceID)
        )
        .where(TokenID.TextID == text_id)  # Фильтр по TextID
        .where(s.TextID == text_id)  # Фильтр по TextID
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
    text_id = 6  # Указываем нужный TextID
    print(f'Запуск анализа токенов для TextID = {text_id}')



    with engine_generic.begin() as conn:
        prepare_regex_table(conn, text_id)
        queries = build_queries(conn, text_id)
        data = load_data(conn, queries)

    # длины предложений и статистика по ним
    df_sw = data['sentence_words']
    df_sw.columns = ['Token_text', 'Sentence_text']
    df_sw['word_count'] = df_sw['Sentence_text'].str.lower().apply(count_words)

    word_stats = (
        df_sw
        .groupby('Token_text', as_index=False)['word_count']
        .agg(avg_word_count='mean', median_word_count='median')
    )

    # метрики
    df_tokens = data['token_freq'].rename(
        columns={'Token_text': 'Token_text', 'total_token_count': 'total_token_count'})
    df_sent = data['sentence_count'].rename(columns={'Token_text': 'Token_text', 'sentence_count': 'sentence_count'})
    df_dep = data['dependent_word_count'].rename(
        columns={'Token_text': 'Token_text', 'dependent_word_count': 'dependent_word_count'})
    df_pos = data['pos_dependency'].rename(
        columns={'Token_text': 'Token_text', 'dependent_pos': 'dependent_pos', 'pos_count': 'pos_count'})

    df_pos_grouped = (
        df_pos.groupby('Token_text', group_keys=False)
        .apply(lambda g: ', '.join(f"{row.dependent_pos}: {row.pos_count}" for _, row in g.iterrows()))
        .reset_index(name='POS_Dependencies')
    )

    df_final = (
        df_tokens
        .merge(df_sent, on='Token_text', how='left')
        .merge(df_dep, on='Token_text', how='left')
        .merge(df_pos_grouped, on='Token_text', how='left')
        .merge(word_stats, on='Token_text', how='left')
    )

    df_final['sentence_count'] = df_final['sentence_count'].fillna(0).astype(int)
    df_final['dependent_word_count'] = df_final['dependent_word_count'].fillna(0).astype(int)
    df_final['avg_word_count'] = df_final['avg_word_count'].fillna(0).round(2)
    df_final['median_word_count'] = df_final['median_word_count'].fillna(0).astype(int)

    df_sorted = df_final.sort_values('total_token_count', ascending=False).reset_index(drop=True)
    df_sorted.insert(0, 'TokenID', range(1, len(df_sorted) + 1))

    expanded = expand_pos_column(df_sorted, 'POS_Dependencies').drop(columns='POS_Dependencies', errors='ignore')

    print("\nТОП-20 токенов для TextID = 2:")
    print(expanded.head(20))

    out_file = f"text_{text_id}_tokens_expanded.xlsx"
    expanded.to_excel(out_file, index=False, engine='openpyxl')
    print(f"\nРезультат сохранён в «{out_file}»")


if __name__ == "__main__":
    main()