"""«Имена собственные: не учитывать» охватывает и слова словаря внутри названий мест: «чёрный» в «Чёрное море»."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = "Корабль вышел в Чёрное море. Чёрный кот сидел у порога. Он видел чёрную тучу и чёрного кота."


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
    r = c.post("/corpora", data={"name": "Море", "description": ""}, follow_redirects=False)
    c.corpus_id = int(r.headers["location"].rsplit("/", 1)[1])
    c.post("/texts/upload", files=[("files", ("m.txt", io.BytesIO(TEXT.encode()), "text/plain"))],
           data={"corpus_ids": str(c.corpus_id), "f_names": ["m.txt"], "f_titles": ["Море"]})
    run_pending()
    c.post("/dictionaries", data={"name": "Цвета", "words": "чёрный"})
    run_pending()
    from app.db import SessionLocal
    from app.models import Dictionary

    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Цвета").one()
        c.dict_id, c.entry_id = d.id, d.entries[0].id
    return c


def _count(noprop: bool, dict_id: int, corpus_id: int) -> int:
    from app.db import SessionLocal
    from app.services import analytics as A
    from app.services.selection import from_params

    with SessionLocal() as s:
        sel = from_params(s, {"corpus_id": [str(corpus_id)], "noprop": ["1"] if noprop else []})
        try:
            return A.frequency(s, dict_id, sel.ids)["rows"][0]["count"]
        finally:
            A.set_noprop(False)


def test_place_names_excluded_only_with_noprop(client):
    from app.db import SessionLocal
    from app.models import Hit

    with SessionLocal() as s:
        flags = sorted((h.form, h.in_propn, h.in_name) for h in s.query(Hit).filter_by(dictionary_id=client.dict_id))
    assert ("Чёрное", True, False) in flags and ("Чёрный", False, False) in flags and ("чёрную", False, False) in flags
    assert _count(False, client.dict_id, client.corpus_id) == 4  # по умолчанию всё как раньше
    assert _count(True, client.dict_id, client.corpus_id) == 3  # «Чёрное море» не считается
    base = f"dictionary_id={client.dict_id}&corpus_id={client.corpus_id}"
    page = client.get(f"/analysis?tab=freq&{base}&noprop=1").text
    assert "1 вхождений слов словаря внутри имён и названий" in page
    assert "вхождений слов словаря внутри имён" not in client.get(f"/analysis?tab=freq&{base}").text


def test_manual_ok_counts_even_with_noprop(client):
    from app.db import SessionLocal
    from app.models import Hit

    with SessionLocal() as s:
        h = s.query(Hit).filter_by(dictionary_id=client.dict_id, form="Чёрное").one()
        h.is_valid = True
        s.commit()
    assert _count(True, client.dict_id, client.corpus_id) == 4


def test_character_with_some_place_mentions_not_in_places(client):
    """Персонаж, у которого модель приняла часть упоминаний за место, не попадает в «места × слова словаря»."""
    from app.db import SessionLocal
    from app.models import Entity, EntityAlias, SentenceRow
    from app.services.entities import entity_dictionary, entity_index

    with SessionLocal() as s:
        sents = s.query(SentenceRow).order_by(SentenceRow.idx).all()
        tid = sents[0].text_id
        for sent, etype, norm in ((sents[1], "PER", "Клим Самгин"), (sents[2], "PER", "Клим Самгин"), (sents[1], "LOC", "Самгин")):
            s.add(Entity(text_id=tid, sentence_id=sent.id, etype=etype, start_char=0, end_char=1, surface=norm, norm=norm))
        s.add(EntityAlias(norm="Самгин", canonical="Клим Самгин"))  # вариант объединён с персонажем
        s.commit()
        loc = entity_dictionary(s, [tid], client.dict_id, etype="LOC")
        per = entity_dictionary(s, [tid], client.dict_id, etype="PER")
        assert "Клим Самгин" not in loc["names"] and "Клим Самгин" in per["names"]
        assert "Клим Самгин" not in [i["name"] for i in entity_index(s, [tid], "LOC")["items"]]
