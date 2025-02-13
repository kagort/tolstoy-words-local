from project_db.model_3 import Words, session
from sqlalchemy import func

# Список интересующих нас TokenID
target_token_ids = [1, 2]

# Проверяем, что список не пустой
if not target_token_ids:
    raise ValueError("Список TokenID пуст")

# Запрос для TextID = 1
res_text1 = session.query(
    Words.Part_of_speech.label('Part_of_speech'),
    func.sum(Words.Frequency).label('Total_Frequency')
).filter(
    Words.TextID == 1,
    Words.TokenID.in_(target_token_ids)
).group_by(
    Words.Part_of_speech
).all()

# Запрос для TextID = 2
res_text2 = session.query(
    Words.Part_of_speech.label('Part_of_speech'),
    func.sum(Words.Frequency).label('Total_Frequency')
).filter(
    Words.TextID == 2,
    Words.TokenID.in_(target_token_ids)
).group_by(
    Words.Part_of_speech
).all()

# Вывод результатов
print("Результаты для TextID = 1:")
for row in res_text1:
    print(f"{row.Part_of_speech}, Частота: {row.Total_Frequency}")

print("\nРезультаты для TextID = 2:")
for row in res_text2:
    print(f"{row.Part_of_speech}, Частота: {row.Total_Frequency}")