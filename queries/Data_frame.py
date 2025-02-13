import pandas as pd
from project_db.model_3 import Words, session
from sqlalchemy import func, desc
from collections import defaultdict
from sqlalchemy.sql import text


# Определяем интересующие нас TokenID
target_token_ids = [1, 2]  # Пример: список ID токенов

# Подзапрос для получения топ-5 слов для каждой части речи
subquery = session.query(
    Words.Part_of_speech.label('Part_of_speech'),
    Words.TextID.label('TextID'),
    Words.Word_text.label('Word_text'),
    Words.Frequency.label('Frequency'),
    func.row_number().over(
        partition_by=[Words.Part_of_speech, Words.TextID],  # Группировка по части речи и TextID
        order_by=Words.Frequency.desc()                     # Сортировка по частоте
    ).label('rn')  # Номер строки внутри группы
).filter(
    Words.TokenID.in_(target_token_ids)  # Фильтр по TokenID
).subquery()

# Основной запрос для выборки топ-5 слов
res_text1 = session.query(
    subquery.c.Part_of_speech.label('Part_of_speech'),
    subquery.c.TextID.label('TextID'),
    subquery.c.Word_text.label('Word_text'),
    subquery.c.Frequency.label('Frequency')
).filter(
    subquery.c.TextID == 1,  # Фильтр по первому тексту
    subquery.c.rn <= 5       # Только топ-5 слов
).all()

res_text2 = session.query(
    subquery.c.Part_of_speech.label('Part_of_speech'),
    subquery.c.TextID.label('TextID'),
    subquery.c.Word_text.label('Word_text'),
    subquery.c.Frequency.label('Frequency')
).filter(
    subquery.c.TextID == 2,  # Фильтр по второму тексту
    subquery.c.rn <= 5       # Только топ-5 слов
).all()


#Объединение результатов
# Создаем словарь для хранения результатов
top_words_table = defaultdict(lambda: {'Text1': [], 'Text2': []})

# Добавляем результаты для TextID = 1
for row in res_text1:
    part_of_speech, text_id, word_text, frequency = row
    top_words_table[part_of_speech]['Text1'].append((word_text, frequency))

# Добавляем результаты для TextID = 2
for row in res_text2:
    part_of_speech, text_id, word_text, frequency = row
    top_words_table[part_of_speech]['Text2'].append((word_text, frequency))

# Вывод

print(f"{'Часть речи':<20} | {'Текст 1 (топ-5)':<40} | {'Текст 2 (топ-5)':<40}")
print("-" * 105)

for part_of_speech, data in top_words_table.items():
    text1_words = ", ".join([f"{word} ({freq})" for word, freq in data['Text1']])
    text2_words = ", ".join([f"{word} ({freq})" for word, freq in data['Text2']])

    print(f"{part_of_speech:<20} | {text1_words:<40} | {text2_words:<40}")


#DataFrame

# Преобразуем словарь в DataFrame
df_data = []
for part_of_speech, data in top_words_table.items():
    text1_words = ", ".join([f"{word} ({freq})" for word, freq in data['Text1']])
    text2_words = ", ".join([f"{word} ({freq})" for word, freq in data['Text2']])
    df_data.append({
        'Part_of_speech': part_of_speech,
        'Text1_Top5': text1_words,
        'Text2_Top5': text2_words
    })

df = pd.DataFrame(df_data)

# Выводим таблицу
print(df)
