from project_db.model_3 import *
from sqlalchemy import func

# Определяем интересующие нас TokenID (например, существительные)
target_token_ids = [4, 5]  # Пример: список ID токенов

# Подзапрос для получения зависимых слов и их предложений
subquery = session.query(
    Words.Part_of_speech.label('Part_of_speech'),
    Words.Word_text.label('Word_text'),
    Words.Frequency.label('Frequency'),
    Sentences.Sentence_text.label('Sentence_text'),
    func.row_number().over(
        partition_by=[Words.Part_of_speech],  # Группировка по части речи
        order_by=Words.Frequency.desc()       # Сортировка по частоте
    ).label('rn')  # Номер строки внутри группы
).join(
    Cross, Words.WordID == Cross.WordID
).join(
    Sentences, Cross.SentenceID == Sentences.SentenceID
).filter(
    Words.TokenID.in_(target_token_ids)  # Фильтр по TokenID
).subquery()

# Основной запрос для выборки топ-5 слов
result = session.query(
    subquery.c.Part_of_speech.label('Part_of_speech'),
    subquery.c.Word_text.label('Word_text'),
    subquery.c.Frequency.label('Frequency'),
    subquery.c.Sentence_text.label('Sentence_text')
).filter(
    subquery.c.rn <= 5  # Только топ-5 слов
).all()

# Вывод результатов
for row in result:
    part_of_speech, word_text, frequency, sentence_text = row
    print(f"Часть речи: {part_of_speech}, Слово: {word_text}, Частота: {frequency}, Контекст: {sentence_text}")