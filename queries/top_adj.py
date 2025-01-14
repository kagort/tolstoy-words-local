import pandas as pd
from project_db.db3_from_csv import engine

# SQL-запрос
query = """
SELECT 
    t."TextTitle" AS TextTitle,
    tok."Token_text" AS TokenText,
    w."Word_text" AS DependentWord,
    SUM(w."Frequency") AS TotalFrequency,
    STRING_AGG(s."Sentence_text", '; ') AS Sentences
FROM 
    public.word_sentence_association wsa
JOIN 
    public.words w ON wsa."WordID" = w."WordID"
JOIN 
    public.sentences s ON wsa."SentenceID" = s."SentenceID"
JOIN 
    public.dictexts t ON wsa."TextID" = t."TextID"
JOIN 
    public.tokenid tok ON tok."TokenID" = wsa."TextID"  -- Соединяем через TextID
WHERE 
    tok."TokenID" IN (1, 2)
    AND w."Part_of_speech" = 'ADJ'
GROUP BY 
    t."TextTitle", tok."Token_text", w."Word_text"
ORDER BY 
    tok."TokenID", TotalFrequency DESC
LIMIT 10;
"""

# Выполнение запроса
df = pd.read_sql(query, con=engine)

# Проверка результата
print(df)

# Сохранение результата в CSV
df.to_csv('top_adjectives_by_token.csv', index=False, encoding='utf-8')
print("Результаты сохранены в top_adjectives_by_token.csv")
