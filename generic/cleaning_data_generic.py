import re
import unicodedata


def split_text_into_sentences(text: str) -> list[str]:
    return re.compile(r'(?<=[.!?…])').split(text.strip())


def normalize_sentence(sentence):
    s = unicodedata.normalize('NFKC', sentence)
    s = re.sub(r'\s+', ' ', s)            # несколько пробелов заменяем на один
    s = s.replace('"', '').strip().lower()
    return s


def remove_and_check_duplicates(input_file, output_file):
    seen            = set()
    result          = []
    duplicate_count = 0
    total_sentences = 0
    processed = 0

    # Сначала прочитаем весь файл и соберём все предложения
    all_sentences = []

    print("[INFO] Чтение файла и разделение на предложения...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    all_sentences   = split_text_into_sentences(text)
    total_sentences = len(all_sentences)

    print(f"[INFO] Найдено предложений: {total_sentences}")

    # Удаление дубликатов
    print("[INFO] Начинаю обработку предложений...\n")
    for sentence in all_sentences:
        normalized = normalize_sentence(sentence)
        processed += 1

        if normalized in seen:
            duplicate_count += 1
        else:
            seen.add(normalized)
            result.append(normalized + '\n')

        # Прогресс в консоли
        print(f"\r[PROGRESS] Обработано {processed} из {total_sentences} предложений...", end='', flush=True)

    print(f"\n[INFO] Обработка завершена. Удалено дубликатов: {duplicate_count}")

    # Запись результата
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(result)

    print(f"[INFO] Очищенный файл сохранён как: {output_file}")

    # Проверка оставшихся дубликатов
    seen_again = set()
    duplicates_found = []

    print("[INFO] Проверяю очищенный файл на повторы...")

    with open(output_file, 'r', encoding='utf-8') as f:
        cleaned_text = f.read()
        cleaned_sentences = split_text_into_sentences(cleaned_text)

        for idx, sentence in enumerate(cleaned_sentences):
            normalized = normalize_sentence(sentence)
            if normalized in seen_again:
                duplicates_found.append((idx + 1, sentence))
            else:
                seen_again.add(normalized)

    if duplicates_found:
        print("[ERROR] В очищенном файле найдены повторяющиеся предложения:")
        for line_num, text in duplicates_found:
            print(f"Предложение {line_num}: {text[:50]}...")
    else:
        print("[SUCCESS] Все дубликаты успешно удалены. Повторений не найдено.")


# --- Использование ---
input_path = "pretrained_yandex_texts.txt"
output_path = "cleaned.txt"

remove_and_check_duplicates(input_path, output_path)