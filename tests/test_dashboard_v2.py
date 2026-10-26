"""Дашборд v2 «корпус × словарь» (services/dashboard_v2.py): разделы с выводом одной фразой, автономный HTML."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXTS = {
    "a.txt": ("Утро", "Иван Петров", "1850", ("Радость и любовь наполняли дом. Сад цвёл, и радость была в каждом. "
              "Любовь к саду не проходила. Иван смотрел на сад с радостью. Радость и сад, любовь и сад.")),
    "b.txt": ("Полдень", "Пётр Сидоров", "1870", ("Тоска и печаль стояли над городом. Печаль не уходила, тоска росла. "
              "Сад опустел, и печаль в саду. Пётр говорил о тоске и печали. Радость ушла, осталась тоска.")),
    "c.txt": ("Вечер", "Пётр Сидоров", "1890", ("Печаль и радость сменяли друг друга. Сад шумел, и радость вернулась. "
              "Любовь и печаль, тоска и сад. Пётр нашёл радость в саду. Печаль стихла, любовь осталась.")),
}


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
    r = c.post("/corpora", data={"name": "Времена", "description": ""}, follow_redirects=False)
    c.corpus_id = int(r.headers["location"].rsplit("/", 1)[1])
    names = list(TEXTS)
    c.post("/texts/upload", files=[("files", (n, io.BytesIO(TEXTS[n][3].encode("utf-8")), "text/plain")) for n in names],
           data={"corpus_ids": str(c.corpus_id), "f_names": names, "f_titles": [TEXTS[n][0] for n in names],
                 "f_authors": [TEXTS[n][1] for n in names], "f_years": [TEXTS[n][2] for n in names]})
    run_pending()
    csv_text = "слово;категория\nрадость;СВЕТЛОЕ\nлюбовь;СВЕТЛОЕ\nпечаль;ТЁМНОЕ\nтоска;ТЁМНОЕ\nсад;МЕСТО\nзакат;МЕСТО\n"
    c.post("/dictionaries", data={"name": "Настроения", "words": ""},
           files={"file": ("n.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")})
    run_pending()
    from app.db import SessionLocal
    from app.models import Dictionary

    with SessionLocal() as s:
        c.dict_id = s.query(Dictionary).filter_by(name="Настроения").one().id
    return c


def test_build_v2_sections(client):
    from app.db import SessionLocal
    from app.services.dashboard_v2 import build_v2
    from app.services.selection import from_params

    with SessionLocal() as s:
        sel = from_params(s, {"corpus_id": [str(client.corpus_id)]})
        v = build_v2(s, sel, client.dict_id)["v2"]
    assert v["has_cats"] and [c["name"] for c in v["categories"]][:1] in (["СВЕТЛОЕ"], ["ТЁМНОЕ"], ["МЕСТО"])
    cov = v["coverage"]
    assert cov["found"] == 5 and cov["entries"] == 6 and cov["dead"] == ["закат"] and "Найдено 5 из 6" in cov["say"]
    assert v["core"] and len(v["core"]["points"]) == 5
    assert v["group_kind"] == "author" and v["profile"]["form"] == "radar" and len(v["profile"]["groups"]) == 2
    assert v["distinct"]["pair"] and {v["distinct"]["pair"]["a"], v["distinct"]["pair"]["b"]} == set(v["distinct"]["groups"])
    assert v["time"] and len(v["time"]["periods"]) >= 3 and v["time"]["periods"][0].startswith("1850")
    assert v["syntax"] and len(v["syntax"]["rows"][0]["pct"]) == 6
    assert v["quality"]["items"] and v["quotes"] and all(q["kw"] for q in v["quotes"])
    for key in ("coverage", "core", "profile", "distinct", "time", "syntax", "quality"):
        assert v[key]["say"], key


def test_page_v2_and_classic(client):
    q = f"corpus_id={client.corpus_id}&dictionary_id={client.dict_id}"
    page = client.get(f"/dashboard?{q}").text
    assert "Корпус × словарь" in page and "Ядро и периферия" in page and "Во времени" in page and "dv2-say" in page
    assert "/static/dash_v2.js" in page and "Цитаты" in page
    client.cookies.set("qv_ui", "classic")
    page = client.get(f"/dashboard?{q}").text
    assert "Корпус × словарь" not in page and "Частые слова словаря" in page
    client.cookies.set("qv_ui", "v2")


def test_standalone_v2(client):
    html = client.get(f"/export/dashboard.html?corpus_id={client.corpus_id}&dictionary_id={client.dict_id}&cdn=1").text
    assert "Корпус × словарь" in html and "function drawDashV2" in html and "--cat-1" in html
    assert "data:font/woff2;base64," in html and "/static/fonts/" not in html  # шрифты внутри файла
    assert "/static/" not in html.split("<body", 1)[1]
