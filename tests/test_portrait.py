"""Портрет словосочетания: ключевые слова контекстов конкорданса."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = """Русский мужик пашет землю.
Старый русский мужик пашет землю на рассвете.
Русский мужик снова пашет землю.
Мужик пил чай в трактире.
Мужик сидел в трактире и пил чай.
Барин читал газету в кабинете.
Барин курил трубку в кабинете.

Русский мужик долго пашет землю. Потом он отдыхает у реки."""


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
    return c


@pytest.fixture(scope="module")
def corpus(client):
    from app.db import SessionLocal
    from app.models import Dictionary, TextDoc
    from app.services.jobs import run_pending

    files = [("files", ("muzhik.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain"))]
    client.post("/texts/upload", files=files, data={"author": "Автор", "f_names": ["muzhik.txt"], "f_titles": ["Мужики"]})
    run_pending()
    r = client.post("/dictionaries", data={"name": "Люди", "words": "мужик\nбарин"}, follow_redirects=False)
    assert r.status_code == 303
    run_pending()
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Люди").one()
        return {"dict_id": d.id, "text_ids": [t.id for t in s.query(TextDoc).all()],
                "entries": {e.surface: e.id for e in d.entries}}


def test_context_keywords(corpus):
    from app.db import SessionLocal
    from app.services import analytics as A
    from app.services.keywords import context_keywords, context_sentences

    ids, eid = corpus["text_ids"], corpus["entries"]["мужик"]
    with SessionLocal() as s:
        sids, nodes = A.concordance_sentence_ids(s, corpus["dict_id"], ids, entry_id=eid, related_lemma="русский")
        assert len(sids) == 4 and len(nodes) == 4
        kw = context_keywords(s, sids, ids, node_tokens=nodes, exclude={"русский"}, min_count=2)
        lemmas = [r["lemma"] for r in kw["positive"]]
        assert lemmas[:2] and set(lemmas[:2]) == {"пахать", "земля"}
        assert "мужик" not in lemmas and "русский" not in lemmas and "чай" not in lemmas
        assert kw["words_a"] > 0 and kw["words_b"] > 0
        # абзац и соседние предложения расширяют контекст
        assert len(context_sentences(s, sids, "win")) > len(sids)
        para = context_sentences(s, sids, "para")
        assert len(para) > len(sids)


def test_context_keywords_page(client, corpus):
    q = f"/analysis?tab=conc&dictionary_id={corpus['dict_id']}&entry_id={corpus['entries']['мужик']}&related=русский"
    r = client.get(q)
    assert r.status_code == 200 and "Ключевые слова этих контекстов" in r.text
    r = client.get(q + "&ctxkw=para")
    assert r.status_code == 200 and 'id="ctx-kw"' in r.text and "земля" in r.text
