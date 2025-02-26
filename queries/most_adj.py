from project_db.model_3 import Words, session

# Запрос: выбираем Word_text и Frequency для прилагательных, отсортированных по Frequency
res = session.query(Words.Word_text, Words.Frequency).filter(
    Words.Part_of_speech == 'ADJ'
).order_by(Words.Frequency.desc()).all()

# Вывод первых нескольких результатов
for word, freq in res[:10]:  # Например, выводим топ-10
    print(word,freq)






