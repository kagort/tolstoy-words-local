from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from project_db.model_3 import *

engine = create_engine('your_database_url')
Session = sessionmaker(bind=engine)
session = Session()

# Метрики для каждого текста
query = (
    session.query(
        DicTexts.TextTitle,
        func.count(Sentences.SentenceID).label('TotalSentences'),
        func.count(Cross.SentenceID).label('SentencesWithTokens'),  # Предложения с токенами
        (func.count(Cross.SentenceID) * 100 / func.count(Sentences.SentenceID)).label('Percentage'),
        (
            session.query(TokenID.Token_text, TokenID.Token_count)
            .filter(TokenID.TextID == DicTexts.TextID)
            .order_by(TokenID.Token_count.desc())
            .limit(5)
            .subquery()
        ).label('Top5Tokens')  # Топ-5 токенов по Token_count
    )
    .select_from(DicTexts)
    .join(Sentences, DicTexts.TextID == Sentences.TextID)
    .outerjoin(Cross, Cross.TextID == DicTexts.TextID)
    .group_by(DicTexts.TextTitle)
)

# Получение результатов
results = query.all()

# Преобразование в DataFrame (для удобства)
import pandas as pd
df = pd.read_sql_query(query.statement, session.bind)