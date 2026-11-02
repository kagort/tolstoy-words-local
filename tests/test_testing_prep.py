"""Перед открытым тестированием: отзывы, «кто загрузил» и предупреждение при удалении, статус резервных копий."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def app_client():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    from app.db import engine

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture(autouse=True)
def tester_password(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tester_password", "testers-pass")


def _login(name, password="testers-pass"):
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    c.post("/login", data={"username": name, "password": password, "next": "/"})
    return c


def test_tester_login(app_client):
    assert "Тестовый режим" in _login("x", "wrong").get("/login").text
    assert _login("Анна", "wrong").get("/texts", follow_redirects=False).status_code == 303  # неверный пароль
    assert _login("admin", "testers-pass").get("/texts", follow_redirects=False).status_code == 303  # не администратор
    assert _login("admin", "test-password").get("/texts").status_code == 200
    assert "Анна" in _login("  Анна   ").get("/texts").text


def test_feedback_flow(app_client):
    from app.db import SessionLocal
    from app.models import Feedback

    c = _login("Анна")
    page = c.get("/texts")
    assert 'id="fb-dialog"' in page.text and "data-feedback" in page.text
    r = c.post("/feedback", data={"text": "Не нашлось слово «запах»", "kind": "error", "page": "/analysis?tab=conc"},
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/analysis?tab=conc"
    r = c.post("/feedback", data={"text": "x", "page": "https://evil.example/"}, follow_redirects=False)
    assert r.headers["location"] == "/"  # на чужой сайт не уводим
    assert c.post("/feedback", data={"text": "  ", "page": "/"}, follow_redirects=False).status_code == 303
    with SessionLocal() as s:
        fbs = s.query(Feedback).order_by(Feedback.id).all()
        assert len(fbs) == 2 and fbs[0].author == "Анна" and fbs[0].kind == "error" and fbs[0].status == "new"
        fid = fbs[0].id
    lst = c.get("/feedback")
    assert "Не нашлось слово" in lst.text and "новых: 2" in c.get("/texts").text
    c.post(f"/feedback/{fid}", data={"status": "done", "reply": "исправлено в 0.18.3"})
    with SessionLocal() as s:
        f = s.get(Feedback, fid)
        assert f.status == "done" and f.reply == "исправлено в 0.18.3"
    assert "исправлено в 0.18.3" in c.get("/feedback?status=done").text
    x = c.get("/export/feedback.xlsx")
    assert x.status_code == 200 and x.content[:2] == b"PK"


def test_created_by_and_delete_warning(app_client):
    from app.db import SessionLocal
    from app.models import Corpus, Dictionary, TextDoc

    anna = _login("Анна")
    anna.post("/texts/upload", files=[("files", ("a.txt", io.BytesIO("Мама мыла раму.".encode("utf-8")), "text/plain"))],
              data={"f_names": ["a.txt"], "f_titles": ["Рама"]})
    anna.post("/corpora", data={"name": "Корпус Анны", "description": ""})
    anna.post("/dictionaries", data={"name": "Словарь Анны", "words": "рама"})
    with SessionLocal() as s:
        t = s.query(TextDoc).filter_by(title="Рама").one()
        c = s.query(Corpus).filter_by(name="Корпус Анны").one()
        d = s.query(Dictionary).filter_by(name="Словарь Анны").one()
        assert t.created_by == c.created_by == d.created_by == "Анна"
    # Анна у себя — без предупреждения; Борис видит, что данные чужие
    assert "загрузил(а) другой человек" not in anna.get(f"/texts/{t.id}").text
    boris = _login("Борис")
    page = boris.get(f"/texts/{t.id}").text
    assert "Этот текст загрузил(а) другой человек: Анна" in page and "Загрузил(а): Анна" in page
    assert "data-other" in boris.get("/texts").text
    assert "Этот корпус создал(а) другой человек: Анна" in boris.get(f"/corpora/{c.id}").text
    assert "Этот словарь создал(а) другой человек: Анна" in boris.get(f"/dictionaries/{d.id}").text


def test_backup_status(tmp_path, monkeypatch):
    import os as _os
    import time

    from app.config import settings
    from app.web.pages import backup_status

    monkeypatch.setattr(settings, "backup_dir", str(tmp_path / "нет"))
    assert backup_status() is None
    monkeypatch.setattr(settings, "backup_dir", str(tmp_path))
    assert backup_status() == {"n": 0}
    old, new = tmp_path / "tolstoy-2026-10-01_0300.dump", tmp_path / "tolstoy-2026-11-02_0300.dump"
    old.write_bytes(b"x" * 10)
    new.write_bytes(b"x" * 2_000_000)
    _os.utime(old, (time.time() - 5 * 86400,) * 2)
    st = backup_status()
    assert st["n"] == 2 and st["name"] == new.name and st["size_mb"] == 1.9 and not st["stale"]
    _os.utime(new, (time.time() - 3 * 86400,) * 2)
    assert backup_status()["stale"]
