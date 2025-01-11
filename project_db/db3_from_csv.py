from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Table
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session
import logging

# Логирование
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

# Таблицы
class Sentences(Base):
    __tablename__ = 'sentences'
    SentenceID = Column(Integer, primary_key=True, autoincrement=True)
    Sentence_text = Column(Text)

class Words(Base):
    __tablename__ = 'words'
    WordID = Column(Integer, primary_key=True, autoincrement=True)
    Word_text = Column(String(255))
    Part_of_speech = Column(String(50))
    Frequency = Column(Integer)

# Промежуточная таблица для связи многие-ко-многим
word_sentence_association = Table(
    'word_sentence_association', Base.metadata,
    Column('WordID', Integer, ForeignKey('words.WordID')),
    Column('SentenceID', Integer, ForeignKey('sentences.SentenceID'))
)

# Подключение к базе данных
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
db_session = scoped_session(sessionmaker(bind=engine))

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
