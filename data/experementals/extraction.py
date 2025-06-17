from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import logging

from commons.config.db_config import engine_natural
from commons.constants.constants import olfactory_keywords

# Настройка логирования
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Подключение к PostgreSQL
# engine_natural

# Формирование безопасного SQL-запроса с параметрами
query_conditions = " OR ".join([f"Sentences.Sentence_text ILIKE :kw{idx}" for idx in range(len(olfactory_keywords))])
query = text(f'''
    SELECT Sentences.Sentence_text, DicTexts.Text_Author, DicTexts.Text_genre
    FROM Sentences
    JOIN "Cross" ON Sentences.SentenceID = "Cross".SentenceID  -- Исправлено: "Cross"
    JOIN DicTexts ON "Cross".TextID = DicTexts.TextID          -- Исправлено: "Cross"
    WHERE {query_conditions}
''')

# Параметры для запроса
params = {f"kw{idx}": f"%{word}%" for idx, word in enumerate(olfactory_keywords)}
print("Параметры:", params)

# Выполнение запроса
sentences_query = engine_natural.execute(query, params).fetchall()

# Вывод результатов
for row in sentences_query:
    print(row)