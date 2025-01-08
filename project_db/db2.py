from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import logging

# Логирование
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

# Таблицы
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


class TokenID(Base):  # Токен-слово пользователя
    __tablename__ = 'tokenid'
    TokenID = Column(Integer, primary_key=True, autoincrement=True)
    Token_text = Column(String(255), unique=True)


class Words(Base):  # Контекстные слова
    __tablename__ = 'words'
    WordID = Column(Integer, primary_key=True, autoincrement=True)
    word = Column(String(255))
    Part_of_speech = Column(String(50))
    SentenceID = Column(Integer, ForeignKey('sentences.SentenceID'))  # Ссылка на предложение

# Промежуточная таблица для связи многие-ко-многим
#word_sentence_association = Table(
    #'word_sentence_association', Base.metadata,
    #Column('WordID', Integer, ForeignKey('words.WordID')),
    #Column('SentenceID', Integer, ForeignKey('sentences.SentenceID'))
#)


# Подключение к базе данных
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words')
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

# Создание таблиц
Base.metadata.create_all(engine)
