import string
import nltk
import spacy
from pymorphy3 import MorphAnalyzer

# Инициализация библиотек
nltk.download('punkt', quiet=True)
nlp = spacy.load("ru_core_news_sm")
morph = MorphAnalyzer()

# Удаление пунктуации
def remove_punctuation(word):
    return word.translate(str.maketrans('', '', string.punctuation))

# Очистка текста
def clean_text(text):
    return text.replace('\x00', '').strip()

