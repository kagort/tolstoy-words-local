from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func, desc
from project_db.db3_from_csv import DicTexts, Words, Sentences, Cross, TokenID


engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
Session = sessionmaker(bind=engine)
session = Session()

# SQLAlchemy-запрос для получения топ-10 прилагательных
top_adjectives_token1 = (
    session.query(Words.Word_text, Words.Frequency, TokenID.TokenID)
    .join(TokenID, Words.TokenID == TokenID.TokenID)
    .filter(Words.Part_of_speech == 'ADJ', TokenID.TokenID == 1)
    .order_by(desc(Words.Frequency))
    .limit(10)
    .all()
)

# Для TokenID = 2
top_adjectives_token2 = (
    session.query(Words.Word_text, Words.Frequency, TokenID.TokenID)
    .join(TokenID, Words.TokenID == TokenID.TokenID)
    .filter(Words.Part_of_speech == 'ADJ', TokenID.TokenID == 2)
    .order_by(desc(Words.Frequency))
    .limit(10)
    .all()
)

# Вывод результатов
print("Топ-10 прилагательных для TokenID = 1:")
for word, frequency, token_id in top_adjectives_token1:
    print(f"{word}, {frequency}, TokenID: {token_id}")

print("\nТоп-10 прилагательных для TokenID = 2:")
for word, frequency, token_id in top_adjectives_token2:
    print(f"{word}, {frequency}, TokenID: {token_id}")




