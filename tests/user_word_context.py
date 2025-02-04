import pandas as pd
import nltk
import spacy
from collections import defaultdict
from tqdm import tqdm
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from pymorphy3 import MorphAnalyzer

# Загрузка моделей
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()

# Загрузка текста
with open("../voina_i_mir.txt", 'r', encoding='utf-8') as file:
    text = file.read()

# Токенизация текста на предложения
sentences = sent_tokenize(text, language='russian')

# Ввод слова для поиска
search_word = input("Введите слово для поиска: ").strip().lower()

# Генерация форм слова
parsed_word = morph.parse(search_word)[0]
forms = {form.word for form in parsed_word.lexeme}  # Все формы слова
lemma = parsed_word.normal_form  # Лемма слова
print(f"Формы слова '{search_word}': {forms}")

# Первичная фильтрация предложений по словоформам
filtered_sentences = []
for sent in tqdm(sentences, desc="Фильтрация по словоформам"):
    if any(form in sent.lower() for form in forms):
        filtered_sentences.append(sent)

# Вторичная фильтрация по лемме
final_sentences = []
for sent in tqdm(filtered_sentences, desc="Фильтрация по лемме"):
    doc = nlp(sent)
    if any(token.lemma_ == lemma for token in doc):
        final_sentences.append(sent)

# Индексация предложений без сортировки
sentence_data = {idx + 1: sent for idx, sent in enumerate(final_sentences)} # Уточнить про синтаксис

# Создание таблицы предложений
sentence_df = pd.DataFrame([{"Sentence ID": idx, "Sentence": sent} for idx, sent in sentence_data.items()])

# Словари для хранения окружения с частотностью и номерами предложений
pos_data = {
    "ADJ": defaultdict(list),       # Прилагательные
    "NOUN": defaultdict(list),      # Существительные с падежом
    "VERB_HEAD": defaultdict(list), # Глаголы, от которых слово зависит
    "VERB_CHILD": defaultdict(list), # Глаголы, зависящие от слова
    "ADV": defaultdict(list),       # Наречия
    "PRON": defaultdict(list),      # Местоимения
    "OTHER": defaultdict(list)      # Прочее
}

# Обработка предложений и поиск окружения
for idx, sentence in tqdm(sentence_data.items(), desc="Анализ окружения"):
    doc = nlp(sentence)  # Обработка предложения spaCy
    for token in doc:
        if token.lemma_ == lemma:  # Проверяем лемму вместо формы
            # Проверяем, от какого глагола зависит слово (head)
            if token.head.pos_ == "VERB" and token.head != token:
                pos_data["VERB_HEAD"][token.head.lemma_].append(idx)

            # Проверяем зависимых детей слова
            for child in token.children:
                if child.pos_ == "VERB":
                    pos_data["VERB_CHILD"][child.lemma_].append(idx)
                elif child.pos_ == "NOUN":
                    case = child.morph.get("Case")
                    case_label = f"{child.lemma_} ({case[0] if case else 'Неизвестный падеж'})"
                    pos_data["NOUN"][case_label].append(idx)
                elif child.pos_ == "ADJ":
                    pos_data["ADJ"][child.lemma_].append(idx)
                elif child.pos_ == "ADV":
                    pos_data["ADV"][child.lemma_].append(idx)
                elif child.pos_ == "PRON":
                    pos_data["PRON"][child.lemma_].append(idx)
                else:
                    pos_data["OTHER"][child.lemma_].append(idx)

# Преобразование данных в DataFrame
rows = []
for pos, words in pos_data.items():
    for word, sentence_numbers in words.items():
        rows.append({
            "Part of Speech": pos,
            "Word": word,
            "Frequency": len(sentence_numbers),
            "Sentence ID": ", ".join(map(str, sentence_numbers))
        })

# Создание DataFrame
context_df = pd.DataFrame(rows)

# Вывод таблиц
print("\nСгруппированное окружение слова в виде таблицы:")
print(context_df)
print("\nТаблица предложений с ID:")
print(sentence_df)

# Сохранение в файлы
context_df.to_csv("context_analysis.csv", index=False, encoding="utf-8")
sentence_df.to_csv("sentences_with_id.csv", index=False, encoding="utf-8")

print("Обработка завершена. Результаты сохранены в файлы 'context_analysis.csv' и 'sentences_with_id.csv'")