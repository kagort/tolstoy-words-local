from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Table, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session
import logging

# Логирование
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

# Таблицы
class DicTexts(Base):
    __tablename__ = 'dictexts'
    TextID = Column(Integer, primary_key=True, autoincrement=True)
    TextTitle = Column(String(255), unique=True)
    Text_Author = Column(String(255), unique=True)
    Text_year_creation = Column(Integer, nullable=True)
    Text_genre = Column(String(255), nullable=True)

class TokenID(Base):
    __tablename__ = 'tokenid'
    TokenID = Column(Integer, primary_key=True, autoincrement=True)
    Token_text = Column(String(255), nullable=False)
    TextID = Column(Integer, ForeignKey('dictexts.TextID'), nullable=False)
    Token_count = Column(Integer)

    __table_args__ = (
        UniqueConstraint('Token_text', 'TextID', name='token_text_textid_unique'),
    )

class Sentences(Base):
    __tablename__ = 'sentences'
    SentenceID = Column(Integer, primary_key=True, autoincrement=True)
    Sentence_text = Column(Text)
    TextID = Column(Integer, ForeignKey('dictexts.TextID'))

class Words(Base):
    __tablename__ = 'words'
    WordID = Column(Integer, primary_key=True, autoincrement=True)
    TokenID = Column(Integer, ForeignKey('tokenid.TokenID'))
    Word_text = Column(String(255))
    Part_of_speech = Column(String(50))
    Frequency = Column(Integer)
    TextID = Column(Integer, ForeignKey('dictexts.TextID'))

# Промежуточная таблица для связи многие-ко-многим
class Cross(Base):
    __tablename__ = 'cross'
    CrossID = Column(Integer, primary_key=True, autoincrement=True)
    WordID = Column(Integer, ForeignKey('words.WordID'))
    SentenceID = Column(Integer, ForeignKey('sentences.SentenceID'))
    TextID = Column(Integer, ForeignKey('dictexts.TextID'))


# Подключение к базе данных
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
db_session = scoped_session(sessionmaker(bind=engine))

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Проверка подключения
try:
    conn = engine.connect()
    print("Успешное подключение к базе данных!")
    conn.close()
except Exception as e:
    print(f"Ошибка подключения: {e}")

# Создание таблиц
Base.metadata.create_all(engine)