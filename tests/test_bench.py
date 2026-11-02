"""Замер скорости разбора: команда работает и проверяет, что разбор не меняется от настроек."""

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

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")
    c = TestClient(app)
    c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    return c


def test_bench_parse(client):
    from app.db import SessionLocal
    from app.nlp.parsers import available_cpus
    from app.services.bench import bench
    from app.services.jobs import run_pending

    text = "Русский мужик пашет землю. Барин курил трубку в кабинете. " * 10
    client.post("/texts/upload", files=[("files", ("b.txt", io.BytesIO(text.encode("utf-8")), "text/plain"))],
                data={"f_names": ["b.txt"], "f_titles": ["Замер"]})
    run_pending()
    lines = []
    with SessionLocal() as s:
        rows = bench(s, n=50, batches=[4, 32], backends=["natasha"], echo=lines.append)
    assert len(rows) == 2 and all(r["same_pct"] == 100.0 and r["words_per_sec"] for r in rows)
    assert "Доступно ядер" in lines[0] and available_cpus() >= 1
