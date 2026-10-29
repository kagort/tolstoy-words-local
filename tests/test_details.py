"""Мелкие доработки 0.14.2: «Создать категорию», «Непроверенные», значки вместо эмодзи, страница корпуса, «Без корпуса»."""

import io
import os
import re

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF⬅-⬇]")


@pytest.fixture(scope="module")
def client():
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from app.db import engine
    from app.main import app
    from app.services.jobs import run_pending

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")
    c = TestClient(app)
    c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    r = c.post("/corpora", data={"name": "Русская литература", "description": ""}, follow_redirects=False)
    c.corpus_id = int(r.headers["location"].rsplit("/", 1)[1])
    c.post("/texts/upload", files=[("files", ("a.txt", io.BytesIO("Духи пахли сиренью. Запах стоял.".encode()), "text/plain"))],
           data={"corpus_ids": str(c.corpus_id), "f_names": ["a.txt"], "f_titles": ["В корпусе"]})
    c.post("/texts/upload", files=[("files", ("b.txt", io.BytesIO("Сад и запах.".encode()), "text/plain"))],
           data={"f_names": ["b.txt"], "f_titles": ["Сам по себе"]})
    run_pending()
    c.post("/dictionaries", data={"name": "Запахи", "words": "духи, запах, сирень"})
    run_pending()
    from app.db import SessionLocal
    from app.models import Dictionary

    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Запахи").one()
        c.dict_id, c.entries = d.id, {e.surface: e.id for e in d.entries}
    return c


def test_texts_without_corpus_filter(client):
    page = client.get("/texts?corpus_id=none").text
    assert "Без корпуса (1)" in page and "Сам по себе" in page and "В корпусе" not in page
    page = client.get(f"/texts?corpus_id={client.corpus_id}").text
    assert "В корпусе" in page and "Сам по себе" not in page


def test_corpus_page_lists_only_its_texts(client):
    page = client.get(f"/corpora/{client.corpus_id}").text
    members = page.split("Добавить тексты в корпус")[0]
    assert "Тексты корпуса" in members and "В корпусе" in members and "Сам по себе" not in members
    assert "Сам по себе" in page.split("Добавить тексты в корпус")[1] and "без корпуса" in page
    # добавление из свёрнутого блока: отмеченный текст входит в корпус, прежние остаются
    from app.db import SessionLocal
    from app.models import Corpus, TextDoc

    with SessionLocal() as s:
        ids = {t.title: t.id for t in s.query(TextDoc)}
    client.post(f"/corpora/{client.corpus_id}", data={"name": "Русская литература", "text_ids": [str(ids["В корпусе"]), str(ids["Сам по себе"])]})
    with SessionLocal() as s:
        assert {t.title for t in s.get(Corpus, client.corpus_id).texts} == {"В корпусе", "Сам по себе"}


def test_unverified_filter(client):
    page = client.get(f"/dictionaries/{client.dict_id}?show=unverified").text
    rows = re.findall(r'<tr id="e(\d+)"', page)
    assert "непроверенные (" in page and str(client.entries["духи"]) in rows  # «духи» — неоднозначно
    assert str(client.entries["сирень"]) not in rows
    client.post(f"/dictionaries/{client.dict_id}/entries/{client.entries['духи']}", data={"option": "духи|NOUN", "match_mode": "lemma"})
    rows = re.findall(r'<tr id="e(\d+)"', client.get(f"/dictionaries/{client.dict_id}?show=unverified").text)
    assert str(client.entries["духи"]) not in rows  # после «Сохранить» — проверено


def test_create_category(client):
    from app.db import SessionLocal
    from app.models import Dictionary

    page = client.get(f"/dictionaries/{client.dict_id}").text
    assert "Создать категорию" in page and 'form="catform" name="entry_ids"' in page
    ids = [str(client.entries["духи"]), str(client.entries["сирень"])]
    r = client.post(f"/dictionaries/{client.dict_id}/assign-category", data={"category": " ЗАПАХИ  ЦВЕТОВ ", "entry_ids": ids}, follow_redirects=False)
    assert "ЗАПАХИ ЦВЕТОВ" in client.get(r.headers["location"]).text
    with SessionLocal() as s:
        cats = {e.surface: e.category for e in s.get(Dictionary, client.dict_id).entries}
        assert cats == {"духи": "ЗАПАХИ ЦВЕТОВ", "сирень": "ЗАПАХИ ЦВЕТОВ", "запах": None}
    client.post(f"/dictionaries/{client.dict_id}/assign-category", data={"category": "", "entry_ids": [ids[1]]})
    with SessionLocal() as s:
        assert {e.surface: e.category for e in s.get(Dictionary, client.dict_id).entries}["сирень"] is None
    r = client.post(f"/dictionaries/{client.dict_id}/assign-category", data={"category": "X"}, follow_redirects=False)
    assert "Отметьте слова" in client.get(r.headers["location"]).text


def test_no_emoji_in_pages(client):
    for url in ("/texts", "/corpora", f"/corpora/{client.corpus_id}", "/dictionaries", f"/dictionaries/{client.dict_id}",
                f"/analysis?dictionary_id={client.dict_id}", f"/analysis?tab=freq&dictionary_id={client.dict_id}"):
        for ui in ("v2", "classic"):
            client.cookies.set("qv_ui", ui)
            assert not EMOJI.search(client.get(url).text), (url, ui)
    client.cookies.set("qv_ui", "v2")


def test_categories_view_rename_delete(client):
    from app.db import SessionLocal
    from app.models import Dictionary

    ids = {k: str(v) for k, v in client.entries.items()}
    client.post(f"/dictionaries/{client.dict_id}/assign-category", data={"category": "", "entry_ids": list(ids.values())})
    client.post(f"/dictionaries/{client.dict_id}/assign-category", data={"category": "ЗАПАХ", "entry_ids": [ids["запах"]]})
    client.post(f"/dictionaries/{client.dict_id}/assign-category", data={"category": "ЦВЕТЫ", "entry_ids": [ids["сирень"]]})
    page = client.get(f"/dictionaries/{client.dict_id}?sort=cats").text
    assert page.count('class="cat-head"') == 3 and "Без категории" in page and "Удалить категорию" in page
    # переименование в существующее название — слияние
    client.post(f"/dictionaries/{client.dict_id}/category", data={"name": "ЦВЕТЫ", "new_name": "ЗАПАХ", "action": "rename"})
    with SessionLocal() as s:
        cats = {e.surface: e.category for e in s.get(Dictionary, client.dict_id).entries}
    assert cats["сирень"] == "ЗАПАХ" and cats["запах"] == "ЗАПАХ"
    r = client.post(f"/dictionaries/{client.dict_id}/category", data={"name": "ЗАПАХ", "action": "delete"}, follow_redirects=False)
    assert "удалена: 2 слов" in client.get(r.headers["location"]).text
    with SessionLocal() as s:
        d = s.get(Dictionary, client.dict_id)
        assert len(d.entries) == 3 and all(e.category is None for e in d.entries)  # слова остались, без категории
