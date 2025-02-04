import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session


from tests.user_word_context import rows
from tests.user_word_context import sentence_data

# Создание DataFrame
context_df = pd.DataFrame(rows)

# Определение модели SQLAlchemy
Base = declarative_base()

class Sentence(Base):
    __tablename__ = 'sentences'

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text)

class UserWordContext(Base):
    __tablename__ = 'user_word_context'

    id = Column(Integer, primary_key=True, autoincrement=True)
    part_of_speech = Column(String(50))
    word = Column(String(100))
    frequency = Column(Integer)
    sentence_ids = Column(String(255))

class CrossWordsSentences(Base):
    __tablename__ = 'cross_words_sentences'

    word_id = Column(Integer, ForeignKey('zapah_context.id'), primary_key=True)
    sentence_id = Column(Integer, ForeignKey('sentences.id'), primary_key=True)

engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words')  # Локальный сервер
db_session = scoped_session(sessionmaker(bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()

try:
    conn = engine.connect()
    print("Успешное подключение к базе данных!")
    conn.close()
except Exception as e:
    print(f"Ошибка подключения: {e}")

# Добавление предложений в базу данных
sentence_objects = []
for idx, sentence in sentence_data.items():
    sent_obj = Sentence(id=idx, text=sentence)
    sentence_objects.append(sent_obj)
    db_session.add(sent_obj)

# Промежуточный flush для сохранения ID предложений
db_session.flush()

# Добавление контекста в базу данных
context_objects = []
for index, row in context_df.iterrows():
    context = UserWordContext(
        part_of_speech=row['Part of Speech'],
        word=row['Word'],
        frequency=row['Frequency'],
        sentence_ids=row['Sentence ID']
    )
    db_session.add(context)
    context_objects.append(context)

# Промежуточный flush для сохранения ID слов
db_session.flush()

# Добавление связей между словами и предложениями
for context in context_objects:
    sentence_ids = context.sentence_ids.split(', ')
    for sentence_id in sentence_ids:
        db_session.add(CrossWordsSentences(word_id=context.id, sentence_id=int(sentence_id)))

# Добавление связей между словами и предложениями
# Добавление связей между словами и предложениями
added_links = set()  # Множество для хранения добавленных пар

for context in context_objects:
    sentence_ids = context.sentence_ids.split(', ')
    for sentence_id in sentence_ids:
        # Преобразуем в числовой формат
        sentence_id = int(sentence_id)

        # Проверка существования связи
        link = (context.id, sentence_id)

        if link not in added_links:  # Добавляем только уникальные пары
            exists = db_session.query(CrossWordsSentences).filter_by(
                word_id=context.id, sentence_id=sentence_id
            ).first()

            if not exists:  # Проверка в базе данных
                db_session.add(CrossWordsSentences(word_id=context.id, sentence_id=sentence_id))
                added_links.add(link)  # Добавляем в множество

# Финальный коммит
db_session.commit()
db_session.close()