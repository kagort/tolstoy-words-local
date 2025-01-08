from sqlalchemy import func
from project_db.db3_from_csv import *
from sqlalchemy.orm import aliased

# Ввод части речи для поиска
part_of_speech = input("Введите часть речи (например, 'ADJ'): ").strip().upper()

# Выполняем запрос
results = session.query(
    Words.Word_text,  # Слово
    func.count(Words.Word_text).label('Frequency'),  # Частотность
    func.array_agg(Sentences.Sentence_text).label('Sentences')  # Собираем все предложения
).join(
    word_sentence_association, Words.WordID == word_sentence_association.c.WordID
).join(
    Sentences, word_sentence_association.c.SentenceID == Sentences.SentenceID
).filter(
    Words.Part_of_speech == part_of_speech  # Фильтр по части речи
).group_by(
    Words.Word_text  # Группировка по слову
).order_by(
    func.count(Words.Word_text).desc()  # Сортировка по частотности
).all()

# Вывод результатов
print(f"Топ зависимых слов с частью речи '{part_of_speech}':")
for word, freq, sentences in results:
    print(f"\n{word} (Частотность: {freq}):")
    for sent in sentences:
        print(f" - {sent}")

session.close()
