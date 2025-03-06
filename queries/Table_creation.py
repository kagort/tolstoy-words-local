import pandas as pd
from sqlalchemy import create_engine, func, distinct
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from project_db.model_3 import *
from tqdm import tqdm

engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
Session = sessionmaker(bind=engine)
session = Session()

def main():
    try:
        # 1. Метрики по предложениям
        sentences_cte = (
            session.query(
                DicTexts.TextID,
                DicTexts.TextTitle,
                func.count(Sentences.SentenceID).label('TotalSentences'),
                func.count(distinct(Cross.SentenceID)).label('SentencesWithTokens')
            )
            .outerjoin(Sentences, DicTexts.TextID == Sentences.TextID)
            .outerjoin(Cross, Cross.TextID == DicTexts.TextID)
            .join(TokenID, TokenID.TextID == DicTexts.TextID)
            .filter(TokenID.Token_count != 0)
            .group_by(DicTexts.TextID, DicTexts.TextTitle)
            .subquery()
        )

        # 2. Подсчет количества слов в предложениях с токенами (без агрегации avg)
        words_per_sentence = (
            session.query(
                Cross.TextID,
                Cross.SentenceID,
                func.count(distinct(Words.WordID)).label('WordsCount')
            )
            .join(Words, Cross.WordID == Words.WordID)
            .group_by(Cross.TextID, Cross.SentenceID)
            .subquery()
        )

        # 2.1 Теперь считаем среднее количество слов на предложение
        avg_words = (
            session.query(
                words_per_sentence.c.TextID,
                func.avg(words_per_sentence.c.WordsCount).label('AvgWordsPerTokenSentence')
            )
            .group_by(words_per_sentence.c.TextID)
            .subquery()
        )

        # 3. Топ-5 токенов
        top_tokens = (
            session.query(
                DicTexts.TextID,
                TokenID.Token_text,
                func.sum(TokenID.Token_count).label('TotalTokenCount')
            )
            .join(TokenID, DicTexts.TextID == TokenID.TextID)
            .group_by(DicTexts.TextID, TokenID.Token_text)
            .subquery()
        )

        top_tokens_ranked = (
            session.query(
                top_tokens.c.TextID,
                top_tokens.c.Token_text,
                top_tokens.c.TotalTokenCount,
                func.row_number().over(
                    partition_by=top_tokens.c.TextID,
                    order_by=top_tokens.c.TotalTokenCount.desc()
                ).label('rank')
            )
            .subquery()
        )

        # 4. Части речи
        pos_counts = (
            session.query(
                DicTexts.TextID,
                Words.Part_of_speech,
                func.sum(Words.Frequency).label('TotalFrequency')
            )
            .join(Words, DicTexts.TextID == Words.TextID)
            .join(Cross, Cross.WordID == Words.WordID)
            .group_by(DicTexts.TextID, Words.Part_of_speech)
            .subquery()
        )

        pos_ranked = (
            session.query(
                pos_counts.c.TextID,
                pos_counts.c.Part_of_speech,
                pos_counts.c.TotalFrequency,
                func.row_number().over(
                    partition_by=pos_counts.c.TextID,
                    order_by=pos_counts.c.TotalFrequency.desc()
                ).label('rank')
            )
            .subquery()
        )

        # 5. Формируем основной запрос
        query = (
            session.query(
                DicTexts.TextTitle,
                sentences_cte.c.TotalSentences,
                sentences_cte.c.SentencesWithTokens,
                (sentences_cte.c.SentencesWithTokens / sentences_cte.c.TotalSentences * 100).label('Percentage'),
                avg_words.c.AvgWordsPerTokenSentence
            )
            .join(sentences_cte, DicTexts.TextID == sentences_cte.c.TextID)
            .join(avg_words, DicTexts.TextID == avg_words.c.TextID)
        )

        # 6. Загрузка основных метрик
        with tqdm(total=1, desc="Загрузка метрик") as pbar:
            df = pd.read_sql_query(query.statement, session.bind)
            pbar.update(1)

        # 7. Загрузка топ-токенов
        top_tokens_df = (
            pd.read_sql_query(
                session.query(
                    top_tokens_ranked.c.TextID,
                    top_tokens_ranked.c.Token_text,
                    top_tokens_ranked.c.TotalTokenCount
                )
                .filter(top_tokens_ranked.c.rank <= 5)
                .statement,
                session.bind
            )
        )

        # 8. Пивот топ-токенов
        top_pivot = top_tokens_df.pivot(
            index='TextID',
            columns='rank',
            values=['Token_text', 'TotalTokenCount']
        ).swaplevel(axis=1)

        # 9. Загрузка частей речи
        pos_df = (
            pd.read_sql_query(
                session.query(
                    pos_ranked.c.TextID,
                    pos_ranked.c.Part_of_speech,
                    pos_ranked.c.TotalFrequency
                )
                .filter(pos_ranked.c.rank <= 5)
                .statement,
                session.bind
            )
        )

        # 10. Пивот частей речи
        pos_pivot = pos_df.pivot(
            index='TextID',
            columns='rank',
            values=['Part_of_speech', 'TotalFrequency']
        ).swaplevel(axis=1)

        # 11. Объединение данных
        df = df.merge(top_pivot, left_on='TextID', right_index=True, how='left')
        df = df.merge(pos_pivot, left_on='TextID', right_index=True, how='left')

        # 12. Переименование колонок
        df.columns = [
            'TextTitle', 'TotalSentences', 'SentencesWithTokens', 'Percentage', 'AvgWordsPerTokenSentence',
            *[f"Top{i}_Token" for i in range(1, 6)],
            *[f"Top{i}_Count" for i in range(1, 6)],
            *[f"POS{i}" for i in range(1, 6)],
            *[f"POS{i}_Count" for i in range(1, 6)]
        ]

        # 13. Сохранение
        df.to_csv('analytical_table_v3.csv', index=False)
        print(f"Данные сохранены (строк: {len(df)})")

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        raise e

if __name__ == "__main__":
    main()