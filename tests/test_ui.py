"""Новый интерфейс (v2) и откат к классическому: переключатель у пользователя и настройка по умолчанию."""

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

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")
    c = TestClient(app)
    c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    return c


def test_v2_by_default_and_switch_to_classic(client):
    page = client.get("/texts").text
    assert 'class="v2"' in page and "/static/v2.css" in page and "side-nav" in page and "bottom-nav" in page
    assert "Классический интерфейс" in page
    r = client.get("/ui/classic?next=/texts", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/texts" and "qv_ui=classic" in r.headers["set-cookie"]
    client.cookies.set("qv_ui", "classic")
    page = client.get("/texts").text
    assert "/static/v2.css" not in page and 'header class="top"' in page and "новый интерфейс" in page
    client.cookies.set("qv_ui", "v2")
    assert "/static/v2.css" in client.get("/texts").text
    # чужой адрес в next не принимается
    assert client.get("/ui/v2?next=//evil.example", follow_redirects=False).headers["location"] == "/"


def test_default_from_settings(client, monkeypatch):
    from app.config import settings

    client.cookies.delete("qv_ui")
    monkeypatch.setattr(settings, "ui_default", "classic")
    assert "/static/v2.css" not in client.get("/texts").text
    monkeypatch.setattr(settings, "ui_default", "v2")
    assert "/static/v2.css" in client.get("/texts").text


def test_fonts_are_local(client):
    css = client.get("/static/fonts.css").text
    assert "fonts.googleapis" not in css and "fonts.gstatic" not in css and "/static/fonts/" in css
    for path in {line.split("url(")[1].split(")")[0] for line in css.splitlines() if "url(" in line}:
        assert client.get(path).status_code == 200
