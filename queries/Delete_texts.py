from sqlalchemy.orm import sessionmaker
from project_db.model_3 import *

# Создаем сессию для работы с базой данных
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
db_session = scoped_session(sessionmaker(bind=engine))

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


try:
    # Шаг 1: Получаем все TokenID, связанные с текстами ID 41 и ID 42
    token_ids_to_delete = (
        session.query(TokenID.TokenID)
        .filter(TokenID.TextID.in_([41, 42]))
        .all()
    )
    token_ids_to_delete = [token_id[0] for token_id in token_ids_to_delete]

    # Шаг 2: Получаем все SentenceID, связанные с текстами ID 41 и ID 42
    sentence_ids_to_delete = (
        session.query(Sentences.SentenceID)
        .filter(Sentences.TextID.in_([41, 42]))
        .all()
    )
    sentence_ids_to_delete = [sentence_id[0] for sentence_id in sentence_ids_to_delete]

    # Шаг 3: Удаляем записи из таблицы Cross
    session.query(Cross).filter(
        (Cross.TextID.in_([41, 42])) |
        (Cross.TokenID.in_(token_ids_to_delete)) |
        (Cross.SentenceID.in_(sentence_ids_to_delete))
    ).delete(synchronize_session=False)

    # Шаг 4: Удаляем записи из таблицы Words
    session.query(Words).filter(Words.TextID.in_([41, 42])).delete(synchronize_session=False)

    # Шаг 5: Удаляем записи из таблицы Sentences
    session.query(Sentences).filter(Sentences.TextID.in_([41, 42])).delete(synchronize_session=False)

    # Шаг 6: Удаляем записи из таблицы TokenID
    session.query(TokenID).filter(TokenID.TextID.in_([41, 42])).delete(synchronize_session=False)

    # Шаг 7: Удаляем записи из таблицы DicTexts
    session.query(DicTexts).filter(DicTexts.TextID.in_([41, 42])).delete(synchronize_session=False)

    # Фиксируем изменения в базе данных
    session.commit()
    print("Записи успешно удалены.")

except Exception as e:
    # Откат изменений в случае ошибки
    session.rollback()
    print(f"Ошибка при удалении записей: {e}")

finally:
    # Закрываем сессию
    session.close()