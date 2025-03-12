from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import logging

# Настройка логирования
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Подключение к PostgreSQL
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
Session = sessionmaker(bind=engine)
session = Session()

# Список ключевых слов
keywords = [
    "духи", "одеколон", "аромат", "букет", "вонь", "запах", "перегар", "смрад", "парфюм", "душок",
    "благоухание", "благовоние", "зловоние", "запашок", "фимиам", "миазм", "амбре", "амбра", "пригарь",
    "тухлятина", "испарение", "дуновение", "ладан", "скверна", "дым", "навоз", "дерьмо",
    "веять", "вонять", "благоухать", "попахивать", "разить", "смердеть", "чадить", "чад", "пахнуть"
]

# Проверка на пустой список ключевых слов
if not keywords:
    raise ValueError("Список ключевых слов не может быть пустым.")

# Формирование безопасного SQL-запроса с параметрами
query_conditions = " OR ".join([f"Sentences.Sentence_text ILIKE :kw{idx}" for idx in range(len(keywords))])
query = text(f'''
    SELECT Sentences.Sentence_text, DicTexts.Text_Author, DicTexts.Text_genre
    FROM Sentences
    JOIN "Cross" ON Sentences.SentenceID = "Cross".SentenceID  -- Исправлено: "Cross"
    JOIN DicTexts ON "Cross".TextID = DicTexts.TextID          -- Исправлено: "Cross"
    WHERE {query_conditions}
''')

# Параметры для запроса
params = {f"kw{idx}": f"%{word}%" for idx, word in enumerate(keywords)}
print("Параметры:", params)

# Выполнение запроса
sentences_query = session.execute(query, params).fetchall()

# Вывод результатов
for row in sentences_query:
    print(row)