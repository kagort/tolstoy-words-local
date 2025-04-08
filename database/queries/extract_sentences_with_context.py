import csv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.model_3 import Sentences, Cross, TokenID

# Подключение к PostgreSQL
engine = create_engine('postgresql://postgres:ouganda77@localhost:5432/tolstoy_words_csv')
Session = sessionmaker(bind=engine)
session = Session()

# Функция для извлечения предложений и сохранения их в CSV
def extract_sentences_with_context(output_file):
    try:
        # Шаг 1: Найти все записи из Cross, где Token_count > 0
        token_ids_with_count = [
            token_id for (token_id,) in session.query(TokenID.TokenID).filter(TokenID.Token_count > 0).all()
        ]

        # Шаг 2: Получить все SentenceID, связанные с этими TokenID
        sentence_ids = [
            cross.SentenceID
            for cross in session.query(Cross.SentenceID).filter(Cross.TokenID.in_(token_ids_with_count)).all()
        ]

        # Удаление дубликатов SentenceID
        unique_sentence_ids = set(sentence_ids)

        # Шаг 3: Извлечь все предложения из базы данных
        all_sentences = (
            session.query(Sentences.SentenceID, Sentences.Sentence_text, Sentences.TextID)
            .order_by(Sentences.SentenceID)  # Убедитесь, что предложения отсортированы по ID
            .all()
        )

        # Создаем словарь для быстрого доступа к индексам предложений по SentenceID
        sentence_index_map = {sentence.SentenceID: idx for idx, sentence in enumerate(all_sentences)}

        # Список для хранения результатов
        result_sentences = []

        print(f"Начинаем обработку {len(unique_sentence_ids)} уникальных предложений...")
        for i, sentence_id in enumerate(unique_sentence_ids, start=1):
            sentence_index = sentence_index_map.get(sentence_id)
            if sentence_index is not None:
                # Определяем диапазон индексов для контекста (-2 и +2 предложения)
                start_index = max(0, sentence_index - 2)
                end_index = min(len(all_sentences), sentence_index + 3)  # +3, потому что range не включает верхнюю границу

                # Извлекаем предложения из диапазона
                context_sentences = all_sentences[start_index:end_index]

                # Добавляем их в результат
                for sent in context_sentences:
                    result_sentences.append({
                        'TextID': sent.TextID,
                        'Sentence': sent.Sentence_text
                    })

            # Выводим прогресс каждые 100 итераций
            if i % 100 == 0:
                print(f"Обработано {i}/{len(unique_sentence_ids)} предложений...")

        # Удаляем дубликаты предложений (если они есть)
        unique_result_sentences = []
        seen_sentences = set()
        for sentence in result_sentences:
            sentence_tuple = (sentence['TextID'], sentence['Sentence'])
            if sentence_tuple not in seen_sentences:
                seen_sentences.add(sentence_tuple)
                unique_result_sentences.append(sentence)

        # Шаг 5: Сохраняем результаты в CSV-файл
        with open(output_file, mode='w', encoding='utf-8', newline='') as csvfile:
            fieldnames = ['TextID', 'Sentence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for sentence in unique_result_sentences:
                writer.writerow(sentence)

        print(f"Предложения успешно сохранены в файл {output_file}")
        print(f"Общее количество извлеченных предложений: {len(unique_result_sentences)}")

    except Exception as e:
        print(f"Ошибка: {e}")

    finally:
        session.close()


# Вызов функции
if __name__ == '__main__':
    output_csv_file = '../../data/processed/sentences_with_context.csv'
    extract_sentences_with_context(output_csv_file)