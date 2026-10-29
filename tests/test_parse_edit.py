"""Ручная правка разбора на странице предложения: пересчёт, откат, защита от переразбора."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = "Русский мужик пашет землю. Барин курил трубку в кабинете."


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
    from app.models import Dictionary, SentenceRow, TextDoc
    from app.services.jobs import run_pending

    files = [("files", ("m.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain"))]
    client.post("/texts/upload", files=files, data={"author": "Автор", "f_names": ["m.txt"], "f_titles": ["Мужик"]})
    run_pending()
    client.post("/dictionaries", data={"name": "Люди", "words": "мужик\nземля"}, follow_redirects=False)
    run_pending()
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Люди").one()
        t = s.query(TextDoc).one()
        sid = s.query(SentenceRow).filter_by(text_id=t.id, idx=1).one().id
        return {"dict_id": d.id, "text_id": t.id, "sid": sid}


def _tok(s, sid, form):
    from app.models import TokenRow

    return s.query(TokenRow).filter_by(sentence_id=sid, form=form).one()


def _hit(s, sid, form):
    from app.models import Hit

    return s.query(Hit).filter_by(sentence_id=sid, form=form).one()


def test_validate_cycle_and_root(corpus):
    from app.db import SessionLocal
    from app.services.parse_edit import save_edit

    sid = corpus["sid"]
    with SessionLocal() as s:
        verb, noun = _tok(s, sid, "пашет"), _tok(s, sid, "мужик")
        assert noun.head == verb.idx
        assert "цикл" in save_edit(s, sid, verb.idx, "VERB", noun.idx, "acl")["error"]
        assert "root" in save_edit(s, sid, noun.idx, "NOUN", 0, "nsubj")["error"]
        assert "часть речи" in save_edit(s, sid, noun.idx, "XYZ", verb.idx, "nsubj")["error"]


def test_edit_rematches_and_reverts(client, corpus):
    from app.db import SessionLocal
    from app.models import TokenEdit, TokenRelation

    sid = corpus["sid"]
    with SessionLocal() as s:
        verb, noun, land = _tok(s, sid, "пашет"), _tok(s, sid, "мужик"), _tok(s, sid, "землю")
        h = _hit(s, sid, "мужик")
        assert h.role == "subject"
        h.is_valid = True  # ручная отметка должна пережить пересчёт
        s.commit()
        v_idx, n_idx, l_idx = verb.idx, noun.idx, land.idx
    # «землю» — подлежащее, «мужик» — дополнение (нарочно неверно — проверяем пересчёт)
    r = client.post(f"/sentences/{sid}/edit", data={"idx": n_idx, "upos": "NOUN", "head": v_idx, "deprel": "obj"}, follow_redirects=False)
    assert r.status_code == 303
    client.post(f"/sentences/{sid}/edit", data={"idx": l_idx, "upos": "NOUN", "head": v_idx, "deprel": "nsubj"})
    with SessionLocal() as s:
        h = _hit(s, sid, "мужик")
        assert h.role == "object" and h.is_valid is True
        assert _hit(s, sid, "землю").role == "subject"
        assert s.query(TokenEdit).count() == 2
        rels = {(r.rel_type, r.related_lemma) for r in s.query(TokenRelation).filter_by(hit_id=h.id)}
        assert ("governor", "пахать") in rels
    page = client.get(f"/sentences/{sid}?token={n_idx}")
    assert "исправлено вручную" in page.text and "вернуть весь разбор парсера" in page.text
    client.post(f"/sentences/{sid}/edit/revert", data={"idx": str(l_idx)})
    with SessionLocal() as s:
        assert _hit(s, sid, "землю").role == "object" and s.query(TokenEdit).count() == 1


def test_edit_survives_reparse(corpus):
    from app.db import SessionLocal
    from app.models import PARSE_FAST, SentenceRow, TokenEdit
    from app.services.deepparse import upgrade_sentences
    from app.services.ingest import run_ingest

    sid = corpus["sid"]
    with SessionLocal() as s:
        s.get(SentenceRow, sid).parse_level = PARSE_FAST
        s.commit()
    assert upgrade_sentences([sid]) == 1  # точный разбор заново — правка накладывается поверх
    with SessionLocal() as s:
        assert _tok(s, sid, "мужик").deprel == "obj"
    run_ingest(None, corpus["text_id"])  # повторный разбор текста: новые id предложений
    with SessionLocal() as s:
        new = s.query(SentenceRow).filter_by(text_id=corpus["text_id"], idx=1).one().id
        assert new != sid and _tok(s, new, "мужик").deprel == "obj"
        e = s.query(TokenEdit).one()
        assert e.orig_deprel == "nsubj"
