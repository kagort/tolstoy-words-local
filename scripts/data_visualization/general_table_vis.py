import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загрузка данных из CSV
df = pd.read_csv('comparison_table.csv')

# Убедимся, что данные загружены корректно
print(df.head())

# 1. Столбчатая диаграмма для сравнения total_token_count
plt.figure(figsize=(15, 8))
sns.barplot(data=df, x='Token_text', y='total_token_count_Natural', color='skyblue', label='Natural')
sns.barplot(data=df, x='Token_text', y='total_token_count_Generic', color='salmon', label='Generic')
plt.title('Сравнение total_token_count между Natural и Generic')
plt.xlabel('Токен')
plt.ylabel('Количество упоминаний')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Столбчатая диаграмма для сравнения avg_word_count
plt.figure(figsize=(15, 8))
sns.barplot(data=df, x='Token_text', y='avg_word_count_Natural', color='skyblue', label='Natural')
sns.barplot(data=df, x='Token_text', y='avg_word_count_Generic', color='salmon', label='Generic')
plt.title('Сравнение avg_word_count между Natural и Generic')
plt.xlabel('Токен')
plt.ylabel('Средняя длина предложения')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Столбчатая диаграмма для сравнения dependent_word_count
plt.figure(figsize=(15, 8))
sns.barplot(data=df, x='Token_text', y='dependent_word_count_Natural', color='skyblue', label='Natural')
sns.barplot(data=df, x='Token_text', y='dependent_word_count_Generic', color='salmon', label='Generic')
plt.title('Сравнение dependent_word_count между Natural и Generic')
plt.xlabel('Токен')
plt.ylabel('Количество уникальных зависимых слов')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Тепловая карта для сравнения POS-распределения
pos_columns = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB', 'VERB_HEAD']

# Создаем две тепловые карты
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Тепловая карта для Natural
sns.heatmap(df[pos_columns].add_suffix('_Natural'), annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[0])
axes[0].set_title('POS-распределение для Natural')

# Тепловая карта для Generic
sns.heatmap(df[pos_columns].add_suffix('_Generic'), annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[1])
axes[1].set_title('POS-распределение для Generic')

plt.tight_layout()
plt.show()