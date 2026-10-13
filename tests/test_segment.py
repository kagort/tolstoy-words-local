from app.nlp.normalize import extract_text, normalize_text, paragraphs
from app.nlp.segment import split_paragraph


def test_initials_are_not_sentence_breaks():
    sents = split_paragraph("Это написал Л. Н. Толстой. Он жил в Ясной Поляне.")
    assert len(sents) == 2
    assert sents[0].text.endswith("Толстой.")


def test_direct_speech_with_author_words_is_one_sentence():
    sents = split_paragraph("— Дрыхнешь, дьявол! — крикнул он. Никто не ответил.")
    assert [s.text for s in sents] == ["— Дрыхнешь, дьявол! — крикнул он.", "Никто не ответил."]
    assert sents[0].is_direct_speech


def test_token_offsets_point_into_sentence():
    s = split_paragraph("Запах сирени был сладким.")[0]
    for t in s.tokens:
        assert s.text[t.start : t.end] == t.text


def test_normalize_hyphenation_entities_and_paragraphs():
    raw = "Обо-\nняние  у  него\r\n\r\nG&#234;nes\x00 и  Лукка"
    text = normalize_text(raw)
    assert paragraphs(text) == ["Обоняние у него", "Gênes и Лукка"]


def test_extract_cp1251_txt():
    data = "Запах сирени".encode("cp1251")
    assert extract_text("a.txt", data) == "Запах сирени"
