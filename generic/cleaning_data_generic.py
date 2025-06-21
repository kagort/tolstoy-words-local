def remove_duplicate_sentences(input_file, output_file):
    seen = set()
    result = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # Убираем лишние пробелы и переносы
            if line and line not in seen:
                seen.add(line)
                result.append(line + '\n')  # Возвращаем строку с переносом

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(result)

# Использование
input_path = r'C:\Users\User\PycharmProjects\tolstoy-words-local\generic\pretrained_yandex_texts.txt'     # Укажи свой путь к исходному файлу
output_path = r'C:\Users\User\PycharmProjects\tolstoy-words-local\generic\cleaned_file.txt' # Путь для файла без дубликатов

remove_duplicate_sentences(input_path, output_path)