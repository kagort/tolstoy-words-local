from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Определение базы данных
Base = declarative_base()

# Определение таблиц
class DicTexts(Base):
    __tablename__ = 'dictexts'
    TextID = Column(Integer, primary_key=True, autoincrement=True)
    TextTitle = Column(String(255), unique=True)


class Sentences(Base):
    __tablename__ = 'sentences'
    SentenceID = Column(Integer, primary_key=True, autoincrement=True)
    Sentence_index = Column(Integer)
    Sentence_text = Column(Text)
    TextID = Column(Integer, ForeignKey('dictexts.TextID'))


class TokenID(Base):
    __tablename__ = 'tokenid'
    TokenID = Column(Integer, primary_key=True, autoincrement=True)
    Token_text = Column(String(255))


class Words(Base):
    __tablename__ = 'words'
    WordID = Column(Integer, primary_key=True, autoincrement=True)
    word = Column(String(255))
    Part_of_speech = Column(String(50))
    TextID = Column(Integer, ForeignKey('dictexts.TextID'))
    TokenID = Column(Integer, ForeignKey('tokenid.TokenID'))


class CrossWordsSentences(Base):
    __tablename__ = 'cross_words_sentences'
    WordID = Column(Integer, ForeignKey('words.WordID'), primary_key=True)
    SentenceID = Column(Integer, ForeignKey('sentences.SentenceID'), primary_key=True)


# Подключение к базе данных
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words')  # Локальный сервер
db_session = scoped_session(sessionmaker(bind=engine))


Base.query = db_session.query_property()
Session = sessionmaker(bind=engine)
session = Session()

try:
    conn = engine.connect()
    print("Успешное подключение к базе данных!")
    conn.close()
except Exception as e:
    print(f"Ошибка подключения: {e}")

Base.metadata.create_all(engine)
