"""Ручное исключение неправильных связей из анализа (страница предложения → «✕ убрать»)."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = "Пчёлы жужжали в саду и летали над цветами. Пчёлы гудели."


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
    r = c.post("/corpora", data={"name": "Пасека", "description": ""}, follow_redirects=False)
    c.corpus_id = int(r.headers["location"].rsplit("/", 1)[1])
    c.post("/texts/upload", files=[("files", ("p.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain"))],
           data={"corpus_ids": str(c.corpus_id), "f_names": ["p.txt"], "f_titles": ["Пасека"]})
    run_pending()
    c.post("/dictionaries", data={"name": "Пчёлы", "words": "пчела"})
    run_pending()
    return c


def _rels(lemma):
    from app.db import SessionLocal
    from app.models import Hit, TokenRelation

    with SessionLocal() as s:
        q = s.query(TokenRelation, Hit).join(Hit, Hit.id == TokenRelation.hit_id).filter(TokenRelation.related_lemma == lemma)
        return [(h.sentence_id, h.token_idx, r.related_idx, r.rel_type) for r, h in q]


def test_unlink_excludes_from_analysis_and_survives_rematch(client):
    from app.db import SessionLocal
    from app.models import Dictionary, RelationOverride
    from app.services import analytics as A
    from app.services.jobs import run_pending

    before = _rels("летать")
    assert before and before[0][3] == "predicate"  # «пчёлы» — подлежащее при однородном «летали»
    sid, focus, other, _ = before[0]
    page = client.get(f"/sentences/{sid}?token={focus}").text
    assert "✕ убрать" in page and f'action="/sentences/{sid}/unlink"' in page

    r = client.post(f"/sentences/{sid}/unlink", data={"focus": focus, "other": other}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith(f"/sentences/{sid}?token={focus}")
    assert _rels("летать") == []
    assert _rels("жужжать")  # другие связи не тронуты
    page = client.get(r.headers["location"]).text
    assert "rel-off" in page and "вернуть" in page and "Связь убрана" in page

    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Пчёлы").one()
        entry_id, dict_id = d.entries[0].id, d.id
        preds = [x["lemma"] for x in A.word_profile(s, entry_id, [_text_id(s)])["relations"]["predicate"]["items"]]
        assert "летать" not in preds and "жужжать" in preds
    client.post(f"/dictionaries/{dict_id}/rematch")  # пересчёт вхождений: отметка сохраняется
    run_pending()
    assert _rels("летать") == []

    client.post(f"/sentences/{sid}/unlink", data={"focus": focus, "other": other, "undo": "1"})
    assert _rels("летать")
    with SessionLocal() as s:
        assert s.query(RelationOverride).count() == 0
    assert client.post(f"/sentences/{sid}/unlink", data={"focus": focus, "other": 999}).status_code == 400


def _text_id(s):
    from app.models import TextDoc

    return s.query(TextDoc).one().id
