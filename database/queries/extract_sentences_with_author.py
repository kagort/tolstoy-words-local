import csv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.model_3 import * # Импортируем модели

# Подключение к PostgreSQL
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
Session = sessionmaker(bind=engine)
session = Session()

# Функция для извлечения предложений и сохранения их в CSV
def extract_sentences_with_tokens(output_file):
    try:
        # Шаг 1: Найти все записи из TokenID, где Token_count > 0
        token_ids_with_count = [
            token_id for (token_id,) in session.query(TokenID.TokenID).filter(TokenID.Token_count > 0).all()
        ]

        # Шаг 2: Получить все SentenceID, связанные с этими TokenID через таблицу Cross
        sentence_ids = [
            cross.SentenceID
            for cross in session.query(Cross.SentenceID).filter(Cross.TokenID.in_(token_ids_with_count)).all()
        ]

        # Удаление дубликатов SentenceID
        unique_sentence_ids = set(sentence_ids)

        # Шаг 3: Извлечь предложения по SentenceID
        sentences = (
            session.query(Sentences.Sentence_text, Sentences.TextID)
            .filter(Sentences.SentenceID.in_(unique_sentence_ids))
            .all()
        )

        # Шаг 4: Получить информацию о текстах (автор и название) из таблицы DicTexts
        text_info = {
            text.TextID: {"Author": text.Text_Author, "Title": text.TextTitle}
            for text in session.query(DicTexts.TextID, DicTexts.Text_Author, DicTexts.TextTitle).all()
        }

        # Подсчет общего количества извлеченных предложений
        total_sentences = len(sentences)

        # Шаг 5: Сохранить результаты в CSV-файл
        with open(output_file, mode='w', encoding='utf-8', newline='') as csvfile:
            fieldnames = ['TextID', 'Author', 'Title', 'Sentence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for sentence in sentences:
                text_id = sentence.TextID
                author = text_info.get(text_id, {}).get("Author", "Unknown")
                title = text_info.get(text_id, {}).get("Title", "Unknown")
                writer.writerow({
                    'TextID': text_id,
                    'Author': author,
                    'Title': title,
                    'Sentence': sentence.Sentence_text
                })

        print(f"Предложения успешно сохранены в файл {output_file}")
        print(f"Общее количество извлеченных предложений: {total_sentences}")

    except Exception as e:
        print(f"Ошибка: {e}")

    finally:
        session.close()


# Вызов функции
if __name__ == '__main__':
    output_csv_file = '../../data/processed/sentences_with_tokens.csv'
    extract_sentences_with_tokens(output_csv_file)