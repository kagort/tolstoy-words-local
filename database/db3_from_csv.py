from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Table, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session
import logging

# Логирование
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

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

