"""«Найти слово»: любое слово без своего словаря — служебный словарь «Быстрый поиск», только по выборке."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = "Русский мужик пашет землю. Старый мужик пашет землю на рассвете. Барин курил трубку в кабинете."


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
    from app.services.jobs import run_pending

    c.post("/texts/upload", files=[("files", ("m.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain"))],
           data={"f_names": ["m.txt"], "f_titles": ["Мужики"]})
    c.post("/dictionaries", data={"name": "Люди", "words": "мужик\nбарин"})
    run_pending()
    return c


def test_find_word(client):
    from app.db import SessionLocal
    from app.models import Dictionary, Hit, Job
    from app.services.jobs import run_pending
    from app.services.quicksearch import is_quick

    with SessionLocal() as s:
        people = s.query(Dictionary).filter_by(name="Люди").one().id
    page = client.get(f"/analysis?tab=profile&dictionary_id={people}")
    assert 'action="/analysis/find"' in page.text
    r = client.post("/analysis/find", data={"word": "земля", "back": f"tab=profile&dictionary_id={people}"}, follow_redirects=False)
    assert r.status_code == 303 and "tab=profile" in r.headers["location"] and f"prev_dict={people}" in r.headers["location"]
    with SessionLocal() as s:
        qd = next(d for d in s.query(Dictionary) if is_quick(d))
        entry = qd.entries[0]
        assert entry.lemma == "земля" and s.query(Job).filter_by(status="queued").count() == 1
    # пока идёт поиск — плашка «ищем…»
    wait = client.get(r.headers["location"])
    assert "Быстрый поиск" in wait.text and "Ищем в текстах выборки" in wait.text and "вернуться к словарю «Люди»" in wait.text
    run_pending()
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(entry_id=entry.id).count() == 2
        qd = s.get(Dictionary, qd.id)
        assert qd.settings["quick_done"][str(entry.id)]
    done = client.get(r.headers["location"])
    assert "Ищем в текстах выборки" not in done.text
    assert f"dictionary_id%3D{qd.id}" not in client.cookies.get("qv_sel", "") and f"dictionary_id={qd.id}" not in client.cookies.get("qv_sel", "")
    # повторный поиск того же слова в той же выборке — без новой задачи; словосочетание — тоже работает
    client.post("/analysis/find", data={"word": "Земля", "back": f"dictionary_id={people}"})
    with SessionLocal() as s:
        assert s.query(Job).filter_by(status="queued").count() == 0
    client.post("/analysis/find", data={"word": "пашет землю", "back": f"dictionary_id={people}"})
    run_pending()
    with SessionLocal() as s:
        phrase = next(e for e in s.get(Dictionary, qd.id).entries if e.phrase)
        assert s.query(Hit).filter_by(entry_id=phrase.id).count() == 2
    # ошибки ввода
    r = client.post("/analysis/find", data={"word": "  ", "back": ""}, follow_redirects=False)
    assert r.status_code == 303
    # новый текст: обычный словарь сопоставляется, служебный — нет
    client.post("/texts/upload", files=[("files", ("n.txt", io.BytesIO("Мужик видел землю.".encode("utf-8")), "text/plain"))],
                data={"f_names": ["n.txt"], "f_titles": ["Новый"]})
    run_pending()
    with SessionLocal() as s:
        jobs = [j.params for j in s.query(Job).filter_by(kind="match")]
        assert not any(p.get("dictionary_id") == qd.id and not p.get("entry_ids") for p in jobs)
