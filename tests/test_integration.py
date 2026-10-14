"""Сквозной сценарий через веб-интерфейс: загрузка → разбор → словарь → анализ → экспорт."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = """Сладкий, тяжёлый запах сирени, который шёл из сада, наполнял комнату и кружил голову.
— Чем это пахнет? — спросил Л. Н. Толстой.
Пахло дымом и гарью. Запах был сладким. Он почувствовал и вдохнул запах.
Не запах, а вонь стояла в комнате. Почувствовав аромат, он остановился."""


@pytest.fixture(scope="module")
def client():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    from app.db import engine

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")

    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    r = c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"}, follow_redirects=False)
    assert r.status_code == 303
    return c


def run_jobs():
    from app.services.jobs import run_pending

    run_pending()


def test_requires_login():
    from fastapi.testclient import TestClient

    from app.main import app

    r = TestClient(app).get("/texts", follow_redirects=False)
    assert r.status_code == 303 and "/login" in r.headers["location"]


def test_full_flow(client):
    r = client.post("/corpora", data={"name": "Тест", "description": ""}, follow_redirects=False)
    assert r.status_code == 303
    corpus_id = int(r.headers["location"].rsplit("/", 1)[1])

    # несколько файлов сразу: у каждого свои сведения (форма) или из таблицы метаданных
    files = [
        ("files", ("siren.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain")),
        ("files", ("dym.txt", io.BytesIO("Пахло дымом. Запах гари стоял над рекой.".encode("utf-8")), "text/plain")),
        ("files", ("gen.txt", io.BytesIO("Аромат кофе наполнял кухню.".encode("utf-8")), "text/plain")),
    ]
    meta_csv = "файл;название;автор;год;жанр;тип\ngen;Текст модели;Qwen;2025;эссе;нейросеть\n".encode("utf-8")
    files.append(("meta_file", ("meta.csv", io.BytesIO(meta_csv), "text/csv")))
    data = {"author": "Общий Автор", "years": "1900", "genre": "рассказ", "source_type": "natural",
            "corpus_ids": str(corpus_id),
            "f_names": ["siren.txt", "dym.txt", "gen.txt"], "f_titles": ["Сирень", "Дым", ""],
            "f_authors": ["Автор Сирени", "", ""], "f_years": ["1880", "", ""], "f_genres": ["", "очерк", ""],
            "f_types": ["", "", ""], "f_generators": ["", "", ""]}
    r = client.post("/texts/upload", files=files, data=data)
    assert r.status_code == 200
    run_jobs()
    assert "готов" in client.get("/texts").text
    from app.db import SessionLocal as _S
    from app.models import TextDoc as _T

    with _S() as s:
        by_title = {t.title: t for t in s.query(_T).all()}
        assert by_title["Сирень"].author == "Автор Сирени" and by_title["Сирень"].year_from == 1880
        assert by_title["Сирень"].genre == "рассказ"  # общее поле
        assert by_title["Дым"].author == "Общий Автор" and by_title["Дым"].genre == "очерк"
        assert by_title["Текст модели"].author == "Qwen" and by_title["Текст модели"].source_type == "generated"

    r = client.post("/dictionaries", data={"name": "Запахи", "words": "запах, аромат, вонь, пахнуть, дым"},
                    follow_redirects=False)
    assert r.status_code == 303
    dict_id = int(r.headers["location"].rsplit("/", 1)[1])
    run_jobs()

    page = client.get(f"/dictionaries/{dict_id}")
    assert page.status_code == 200 and "запах" in page.text

    base = f"dictionary_id={dict_id}&corpus_id={corpus_id}"
    for tab in ("overview", "freq", "conc", "profile", "compare", "structure"):
        r = client.get(f"/analysis?tab={tab}&{base}")
        assert r.status_code == 200, tab

    from app.db import SessionLocal
    from app.models import DictionaryEntry, Hit
    from app.services import analytics as A

    with SessionLocal() as s:
        entry = s.query(DictionaryEntry).filter_by(dictionary_id=dict_id, surface="запах").one()
        ids = A.corpus_text_ids(s, corpus_id)
        freq = {row["word"]: row["count"] for row in A.frequency(s, dict_id, ids)["rows"]}
        assert freq["запах"] >= 4
        prof = A.word_profile(s, entry.id, ids)
        assert prof["total"] == freq["запах"]
        hit_id = s.query(Hit.id).filter_by(entry_id=entry.id).first()[0]
        sent_id = s.query(Hit.sentence_id).filter_by(entry_id=entry.id).first()[0]

    r = client.get(f"/analysis?tab=profile&{base}&entry_id={entry.id}")
    assert "logDice" in r.text

    # ручная отметка ложного вхождения
    r = client.post(f"/api/hits/{hit_id}/valid", json={"valid": False})
    assert r.json()["is_valid"] is False
    with SessionLocal() as s:
        assert A.frequency(s, dict_id, ids)["rows"][0]["count"] >= 1

    page = client.get(f"/sentences/{sent_id}")
    assert page.status_code == 200 and f"Предложение ID {sent_id}" in page.text and "(NOUN)" in page.text
    ctx = client.get(f"/api/sentences/{sent_id}/context?n=2").json()
    assert any(x["current"] and x["id"] == sent_id for x in ctx["sentences"]) and len(ctx["sentences"]) >= 2
    r = client.get(f"/go/sentence?id={sent_id}", follow_redirects=False)
    assert r.headers["location"] == f"/sentences/{sent_id}"
    conc = client.get(f"/analysis?tab=conc&{base}").text
    assert f'data-ctx="{sent_id}"' in conc or "data-ctx=" in conc
    assert "(nsubj" in conc or "(obj" in conc or "(obl" in conc

    for dim in ("author", "text", "genre", "source_type", "period"):
        r = client.get(f"/analysis?tab=overview&{base}&dim={dim}&step=10&metric=ipm&words={entry.id}")
        assert r.status_code == 200 and "chart-ov" in r.text, dim
    with SessionLocal() as s:
        ov = A.overview(s, dict_id, ids, [entry.id], "author", 10)
        now = {row["word"]: row["count"] for row in A.frequency(s, dict_id, ids)["rows"]}
        assert sum(ov["words"][0]["count"]) == now["запах"]  # одно вхождение уже отмечено ложным
        assert A.filter_texts(s, ids, year_from=1870, year_to=1890) and not A.filter_texts(s, ids, year_from=3000)
    for url in (f"/export/overview.xlsx?{base}&dim=period&step=25", f"/export/frequency.xlsx?{base}", f"/export/frequency.csv?{base}", f"/export/concordance.xlsx?{base}",
                f"/export/profile.json?{base}&entry_id={entry.id}", f"/export/compare.xlsx?{base}&corpus_b=",
                f"/export/structure.csv?{base}", f"/export/hits.conllu?{base}", f"/export/dictionary/{dict_id}.csv"):
        r = client.get(url)
        assert r.status_code == 200, url
        assert len(r.content) > 50, url
    assert "# text =" in client.get(f"/export/hits.conllu?{base}").text

    # пересчёт словаря сохраняет ручную отметку
    client.post(f"/dictionaries/{dict_id}/rematch")
    run_jobs()
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(entry_id=entry.id, is_valid=False).count() == 1
