from project_db.model_3 import *
from sqlalchemy import func

# Список интересующих нас TokenID
target_token_ids = [1, 2, 4, 5, 8, 9]

# Проверяем, что список не пустой
if not target_token_ids:
    raise ValueError("Список TokenID пуст")

# Функция для получения данных по каждому токену
def get_token_data(text_id, token_id):
    # Подзапрос для подсчета общего количества предложений в тексте
    total_sentences = (
        session.query(func.count(Sentences.SentenceID))
        .filter(Sentences.TextID == text_id)
        .scalar()
    )

    # Основной запрос
    result = (
        session.query(
            DicTexts.TextTitle.label('Text_Title'),  # Название текста
            TokenID.Token_text.label('Token_Text'),  # Текст токена
            TokenID.Token_count.label('Token_Total_Count'),  # Всего вхождений токена
            func.count(func.distinct(Cross.SentenceID)).label('Sentence_Count')  # Количество предложений с токеном
        )
        .select_from(Words)  # Явно указываем, что начинаем с таблицы Words
        .join(TokenID, Words.TokenID == TokenID.TokenID, full=False)  # Присоединяем TokenID
        .join(DicTexts, Words.TextID == DicTexts.TextID, full=False)  # Присоединяем DicTexts
        .outerjoin(
            Cross,
            (Cross.TokenID == Words.TokenID) & (Cross.TextID == Words.TextID),  # Указываем ON-условие
            full=False
        )
        .filter(
            Words.TextID == text_id,  # Фильтр по тексту
            Words.TokenID == token_id  # Фильтр по токену
        )
        .group_by(DicTexts.TextTitle, TokenID.Token_text, TokenID.Token_count)
        .first()
    )

    # Если результат пустой, возвращаем None
    if not result:
        return None

    # Подзапрос для подсчета зависимых слов и их группировка по части речи
    dependent_words = (
        session.query(
            Words.Part_of_speech.label('Part_of_speech'),
            func.sum(Words.Frequency).label('Total_Frequency')
        )
        .select_from(Words)  # Явно указываем, что начинаем с таблицы Words
        .filter(
            Words.TextID == text_id,  # Фильтр по тексту
            Words.TokenID == token_id  # Фильтр по токену
        )
        .group_by(Words.Part_of_speech)
        .all()
    )

    return {
        'total_sentences': total_sentences,
        'token_data': result,
        'dependent_words': dependent_words
    }

# Обработка каждого TextID и TokenID
for text_id in [1, 2]:  # Обрабатываем оба текста
    for token_id in target_token_ids:
        data = get_token_data(text_id, token_id)

        # Если данных нет, выводим сообщение и переходим к следующему токену
        if not data:
            print(f"Для TextID={text_id} и TokenID={token_id} данные не найдены.")
            continue

        # Вывод основных метрик
        print(f"Текст: {data['token_data'].Text_Title}")
        print(f"Всего предложений: {data['total_sentences']}")
        print(f"Токен: {data['token_data'].Token_Text}")
        print(f"Вхождений токена: {data['token_data'].Token_Total_Count}")
        print(f"Зависимых слов: {sum(word.Total_Frequency for word in data['dependent_words'])}")
        print(f"Количество предложений, содержащих токен: {data['token_data'].Sentence_Count}")

        # Вывод зависимых слов по частям речи
        print("Зависимые слова по частям речи:")
        for word in data['dependent_words']:
            print(f"  Часть речи: {word.Part_of_speech}, Частота: {word.Total_Frequency}")
        print()  # Пустая строка для разделения