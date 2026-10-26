"""Повторы в словаре (одна лемма — одна запись) и настройка «Без категорий»."""

import io
import os

import pytest


def test_clean_surface():
    from app.services.dictionaries import clean_surface

    assert clean_surface("благоуха́ния") == "благоухания"
    assert clean_surface(" духи​ ") == "духи"
    assert clean_surface("здравый  смысл") == "здравый смысл"
    assert clean_surface("ёлка") == "ёлка"  # ё не трогаем


needs_db = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = "Благоухание сада. Сад благоухал, и благоухание стояло до ночи. Тонкий аромат роз. Аромат и запах."


@pytest.fixture(scope="module")
def client():
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from app.db import engine
    from app.main import app

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")
    c = TestClient(app)
    c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    r = c.post("/corpora", data={"name": "Сад", "description": ""}, follow_redirects=False)
    cid = int(r.headers["location"].rsplit("/", 1)[1])
    c.post("/texts/upload", files=[("files", ("sad.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain"))],
           data={"corpus_ids": str(cid), "f_names": ["sad.txt"], "f_titles": ["Сад"]})
    from app.services.jobs import run_pending

    run_pending()
    c.corpus_id = cid
    return c


@needs_db
def test_duplicates_skipped_on_create_and_add(client):
    from app.db import SessionLocal
    from app.models import Dictionary

    csv_text = "слово;категория\nблагоухание;АРОМАТ\nблагоухания;АРОМАТ\nблагоуха́ние;АРОМАТ\nаромат;АРОМАТ\nаромат​;АРОМАТ\n"
    r = client.post("/dictionaries", data={"name": "Ароматы", "words": ""}, follow_redirects=False,
                    files={"file": ("a.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")})
    assert "Повторы по лемме не добавлены" in client.get(r.headers["location"]).text
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Ароматы").one()
        surfaces = sorted(e.surface for e in d.entries)
        assert surfaces == ["аромат", "благоухание"]
        assert "также: благоухания" in next(e.comment for e in d.entries if e.surface == "благоухание")
        dict_id = d.id
    client.post(f"/dictionaries/{dict_id}/add", data={"words": "ароматы, запах"})
    with SessionLocal() as s:
        assert sorted(e.surface for e in s.get(Dictionary, dict_id).entries) == ["аромат", "благоухание", "запах"]


@needs_db
def test_merge_existing_duplicates(client):
    from app.db import SessionLocal
    from app.models import Dictionary
    from app.services.dictionaries import create_dictionary, make_entry
    from app.services.jobs import run_pending

    with SessionLocal() as s:  # словарь из прежней версии: повторы уже лежат в базе
        d = create_dictionary(s, "Старый", [{"surface": "аромат", "category": None, "upos": None, "comment": None}])
        for w in ("ароматы", "аромат​"):
            d.entries.append(make_entry({"surface": w, "category": "АРОМАТ", "upos": None, "comment": None}))
        s.commit()
        dict_id = d.id
    run_pending()
    page = client.get(f"/dictionaries/{dict_id}").text
    assert "Повторы по лемме: 1" in page and "Объединить повторы" in page
    client.post(f"/dictionaries/{dict_id}/merge-duplicates")
    with SessionLocal() as s:
        d = s.get(Dictionary, dict_id)
        assert [e.surface for e in d.entries] == ["аромат"]
        assert d.entries[0].category == "АРОМАТ" and "также: ароматы" in d.entries[0].comment
    assert "Повторы по лемме" not in client.get(f"/dictionaries/{dict_id}").text


@needs_db
def test_no_categories_setting(client):
    from app.db import SessionLocal
    from app.models import Dictionary
    from app.services.selection import from_params

    with SessionLocal() as s:
        dict_id = s.query(Dictionary).filter_by(name="Ароматы").one().id
    base = f"dictionary_id={dict_id}&corpus_id={client.corpus_id}"
    assert "По категориям словаря" in client.get(f"/analysis?tab=freq&{base}").text
    client.post(f"/dictionaries/{dict_id}/settings", data={"name": "Ароматы", "no_categories": "1", "merge_propn": "1"})
    with SessionLocal() as s:
        assert s.get(Dictionary, dict_id).settings["no_categories"] is True
        assert from_params(s, {"dictionary_id": [str(dict_id)], "corpus_id": [str(client.corpus_id)], "unit": ["cat"]}).unit == ""
    page = client.get(f"/analysis?tab=freq&{base}&unit=cat").text
    assert "По категориям словаря" not in page and 'name="unit"' not in page
    assert "Словарь без категорий" in client.get(f"/dictionaries/{dict_id}").text
    assert "<th>Категория</th>" not in page
    csv = client.get(f"/export/dictionary/{dict_id}.csv").content.decode("utf-8-sig")
    assert csv.splitlines()[0] == "слово;часть речи;комментарий;лемма;режим"
    import pandas as pd

    freq = pd.read_excel(io.BytesIO(client.get(f"/export/frequency.xlsx?{base}").content), sheet_name=None)
    assert "Категория" not in freq["Частоты"].columns and "По категориям" not in freq
    for tab in ("overview", "map", "report", "conc"):
        assert client.get(f"/analysis?tab={tab}&{base}&unit=cat").status_code == 200, tab


def test_import_by_header_names():
    from app.services.dictionaries import parse_dictionary_text

    items = parse_dictionary_text("слово;часть речи;комментарий\nзапах;NOUN;главное\nаромат;;\n")
    assert items[0] == {"surface": "запах", "category": None, "upos": "NOUN", "comment": "главное"}
    items = parse_dictionary_text("запах;ЗАПАХИ;NOUN\nвонь;ЗАПАХИ;\n")  # без заголовка — по порядку колонок
    assert items[0]["category"] == "ЗАПАХИ" and items[0]["upos"] == "NOUN"
