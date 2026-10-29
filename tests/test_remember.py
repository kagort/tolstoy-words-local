"""Запоминание выбора на страницах анализа и дашборда (cookie qv_sel): словарь, корпус, стоп-слова, флажки."""

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
    c.cids = []
    for name in ("Первый", "Второй"):
        r = c.post("/corpora", data={"name": name, "description": ""}, follow_redirects=False)
        c.cids.append(int(r.headers["location"].rsplit("/", 1)[1]))
    c.post("/texts/upload", files=[("files", ("a.txt", io.BytesIO("Запах сада стоял долго.".encode()), "text/plain"))],
           data={"corpus_ids": str(c.cids[1]), "f_names": ["a.txt"], "f_titles": ["Сад"]})
    run_pending()
    for name in ("Первый словарь", "Второй словарь"):
        c.post("/dictionaries", data={"name": name, "words": "запах, сад"})
    run_pending()
    from app.db import SessionLocal
    from app.models import Dictionary

    with SessionLocal() as s:
        c.dids = [d.id for d in s.query(Dictionary).order_by(Dictionary.id)]
    return c


def test_selection_is_remembered_and_reset(client):
    d2, c2 = client.dids[1], client.cids[1]
    r = client.get(f"/analysis?tab=freq&dictionary_id={d2}&corpus_id={c2}&stop=ru&noprop=1&texts=999")
    assert r.status_code == 200 and "qv_sel" in r.headers.get("set-cookie", "")
    r = client.get("/analysis?tab=overview", follow_redirects=False)  # «Анализ» из меню — подставляется сохранённое
    loc = r.headers["location"]
    assert r.status_code == 303 and f"dictionary_id={d2}" in loc and f"corpus_id={c2}" in loc and "stop=ru" in loc
    assert "noprop=1" in loc and "tab=overview" in loc and "texts=" not in loc  # отобранные тексты не запоминаются
    page = client.get(loc).text
    assert "сбросить запомненный выбор" in page
    # явный выбор в адресе важнее запомненного
    assert client.get(f"/analysis?dictionary_id={client.dids[0]}", follow_redirects=False).status_code == 200
    # дашборд — тоже
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303 and f"dictionary_id={client.dids[0]}" in r.headers["location"]
    # сброс
    r = client.get("/analysis?reset=1&tab=freq", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/analysis?tab=freq"
    client.cookies.delete("qv_sel")
    assert client.get("/analysis", follow_redirects=False).status_code == 200


def test_deleted_dictionary_not_recalled(client):
    from urllib.parse import urlencode

    client.cookies.set("qv_sel", urlencode([("dictionary_id", "99999"), ("corpus_id", str(client.cids[0]))]))
    r = client.get("/analysis", follow_redirects=False)
    assert r.status_code == 303 and "99999" not in r.headers["location"] and f"corpus_id={client.cids[0]}" in r.headers["location"]
    client.cookies.delete("qv_sel")
