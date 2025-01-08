import pandas as pd
from project_db.db3_from_csv import *

# Загрузка файлов
context_df = pd.read_csv("context_analysis.csv")
sentence_df = pd.read_csv("sentences_with_id.csv")

# Добавляем предложения в базу
sentence_map = {}  # Для хранения ID предложений
for _, row in sentence_df.iterrows():
    sentence = Sentences(SentenceID=row['Sentence ID'], Sentence_text=row['Sentence'])
    session.merge(sentence)  # Используем merge для избежания дубликатов
    session.commit()
    sentence_map[row['Sentence ID']] = sentence.SentenceID

# Добавляем слова и связи с предложениями
for _, row in context_df.iterrows():
    # Добавляем слово в базу
    word = session.query(Words).filter_by(Word_text=row['Word']).first()
    if not word:
        word = Words(
            Word_text=row['Word'],
            Part_of_speech=row['Part of Speech'],
            Frequency=row['Frequency']
        )
        session.add(word)
        session.flush()  # Получаем ID нового слова

    # Добавляем связи с предложениями
    sentence_ids = [int(sid) for sid in row['Sentence ID'].split(", ")]
    for sid in sentence_ids:
        session.execute(word_sentence_association.insert().values(WordID=word.WordID, SentenceID=sentence_map[sid]))

session.commit()
session.close()

print("Данные успешно загружены в базу данных.")
