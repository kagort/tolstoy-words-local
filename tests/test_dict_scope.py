"""Словарь для одного корпуса: область при создании, пересчёт одного корпуса, число посчитанных текстов."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")


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
    c.corpora = {}
    for name, texts in (("Малый", ["Запах сирени стоял в саду.", "Аромат роз и запах травы."]),
                        ("Большой", ["Запах дыма. Сад горел.", "В саду пахло мёдом, запах был густой.", "Сад и запах."])):
        r = c.post("/corpora", data={"name": name, "description": ""}, follow_redirects=False)
        cid = int(r.headers["location"].rsplit("/", 1)[1])
        files = [("files", (f"{name}{i}.txt", io.BytesIO(t.encode()), "text/plain")) for i, t in enumerate(texts)]
        c.post("/texts/upload", files=files, data={"corpus_ids": str(cid), "f_names": [f[1][0] for f in files],
                                                   "f_titles": [f[1][0] for f in files]})
        c.corpora[name] = cid
    run_pending()
    return c


def _hits_by_corpus(dict_id):
    from sqlalchemy import func, select

    from app.db import SessionLocal
    from app.models import Hit, corpus_texts

    with SessionLocal() as s:
        return dict(s.execute(select(corpus_texts.c.corpus_id, func.count()).join(Hit, Hit.text_id == corpus_texts.c.text_id)
                              .where(Hit.dictionary_id == dict_id).group_by(corpus_texts.c.corpus_id)).all())


def test_scope_at_creation_and_partial_rematch(client):
    from app.db import SessionLocal
    from app.models import Dictionary
    from app.services.dictionaries import matched_texts
    from app.services.jobs import run_pending

    small, big = client.corpora["Малый"], client.corpora["Большой"]
    page = client.get("/dictionaries").text
    assert 'name="scope_corpus_ids"' in page  # область можно выбрать сразу при создании
    client.post("/dictionaries", data={"name": "Запахи", "words": "запах, сад", "scope_corpus_ids": [str(small)]})
    run_pending()
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Запахи").one()
        dict_id = d.id
        assert d.settings["corpus_ids"] == [small]
        assert matched_texts(s, d) == {"n": 2, "total": 2, "exact": True}
    assert set(_hits_by_corpus(dict_id)) == {small}  # большой корпус не сопоставлялся
    page = client.get("/dictionaries").text
    assert "корпуса: Малый" in page
    detail = client.get(f"/dictionaries/{dict_id}").text
    assert "Посчитан на <strong>2</strong> текстах из 2" in detail and "Пересчитать корпус" in detail

    # корпус вне области — пересчёт не запускается, подсказка
    r = client.post(f"/dictionaries/{dict_id}/rematch", data={"corpus_id": str(big)}, follow_redirects=False)
    assert "добавьте корпус" in client.get(r.headers["location"]).text

    # область — все тексты, но пересчитываем только большой корпус: малый не трогается
    with SessionLocal() as s:
        d = s.get(Dictionary, dict_id)
        d.settings = {**d.settings, "corpus_ids": []}
        s.commit()
        assert matched_texts(s, d)["n"] == 2 and matched_texts(s, d)["total"] == 5
    before_small = _hits_by_corpus(dict_id)[small]
    client.post(f"/dictionaries/{dict_id}/rematch", data={"corpus_id": str(big)})
    run_pending()
    hits = _hits_by_corpus(dict_id)
    assert hits[small] == before_small and hits[big] > 0
    with SessionLocal() as s:
        assert matched_texts(s, s.get(Dictionary, dict_id)) == {"n": 5, "total": 5, "exact": True}
    # полный пересчёт — все тексты области
    client.post(f"/dictionaries/{dict_id}/rematch")
    run_pending()
    with SessionLocal() as s:
        assert matched_texts(s, s.get(Dictionary, dict_id))["n"] == 5
