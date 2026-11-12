"""Корпус архивом ZIP (очищенные тексты, метаданные, леммы, CoNLL-U) и сравнение текстов из разных корпусов."""

import io
import os
import zipfile

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def setup():
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from app.db import SessionLocal, engine
    from app.main import app
    from app.models import Corpus, TextDoc
    from app.services.jobs import run_pending

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")
    c = TestClient(app)
    c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    ids = {}
    for cname, fname, title, body in [("Проза", "a.txt", "Первый", "Мама  мыла раму.\n\nПапа читал газету."),
                                      ("Поэзия", "b.txt", "Второй", "Буря мглою небо кроет, вихри снежные крутя.")]:
        r = c.post("/corpora", data={"name": cname, "description": ""}, follow_redirects=False)
        cid = int(r.headers["location"].rsplit("/", 1)[1])
        ids[cname] = cid
        c.post("/texts/upload", files=[("files", (fname, io.BytesIO(body.encode("utf-8")), "text/plain"))],
               data={"f_names": [fname], "f_titles": [title], "f_authors": ["Иван Петрович Сидоров"], "corpus_ids": str(cid)})
    c.post("/dictionaries", data={"name": "Слова", "words": "мама\nбуря"})  # «Анализ» открывается при наличии словаря
    run_pending()
    with SessionLocal() as s:
        texts = {t.title: t.id for t in s.query(TextDoc)}
        assert {cn: len(s.get(Corpus, i).texts) for cn, i in ids.items()} == {"Проза": 1, "Поэзия": 1}
    return c, ids, texts


def test_corpus_zip(setup):
    c, ids, texts = setup
    page = c.get(f"/corpora/{ids['Проза']}")
    assert 'action="/export/corpus.zip"' in page.text
    r = c.get(f"/export/corpus.zip?corpus_id={ids['Проза']}&parts=lemmas&parts=conllu")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    stem = [n for n in names if n.startswith("texts/")][0][len("texts/"):-4]
    assert stem.startswith(f"{texts['Первый']:04d}_") and "Сидоров" in stem
    assert {f"texts/{stem}.txt", f"lemmas/{stem}.txt", f"conllu/{stem}.conllu", "metadata.csv", "README.txt"} <= set(names)
    assert not any("Второй" in n for n in names)  # только тексты корпуса
    clean = z.read(f"texts/{stem}.txt").decode("utf-8")
    assert "Мама мыла раму." in clean and "\n\n" in clean  # двойной пробел убран, абзацы сохранены
    assert z.read(f"lemmas/{stem}.txt").decode("utf-8").splitlines()[0].split() == ["мама", "мыть", "рама"]
    assert "# newdoc id" in z.read(f"conllu/{stem}.conllu").decode("utf-8")
    meta = z.read("metadata.csv").decode("utf-8-sig")
    assert "Первый" in meta and "Иван Петрович Сидоров" in meta
    only = zipfile.ZipFile(io.BytesIO(c.get(f"/export/corpus.zip?corpus_id={ids['Поэзия']}").content)).namelist()
    assert not any(n.startswith(("lemmas/", "conllu/")) for n in only)
    assert c.get("/export/corpus.zip?corpus_id=99999").status_code == 404


def test_textcmp_across_corpora(setup):
    c, ids, texts = setup
    base = f"/analysis?tab=textcmp&corpus_id={ids['Проза']}"
    page = c.get(base)
    assert "Другие тексты базы" in page.text and "Поэзия" in page.text  # текст другого корпуса предлагается
    r = c.get(base + f"&cmp={texts['Первый']}&cmp={texts['Второй']}")
    assert r.status_code == 200 and "Лепестковая диаграмма" in r.text and "Второй" in r.text
    x = c.get(f"/export/textcmp.xlsx?corpus_id={ids['Проза']}&cmp={texts['Первый']}&cmp={texts['Второй']}")
    assert x.status_code == 200 and x.content[:2] == b"PK"


def test_punctuation(setup):
    from app.db import SessionLocal
    from app.services.punctuation import analyze, classify, examples

    c, ids, texts = setup
    assert [classify(f) for f in (",", "…", "...", "?!", "!..", "–", "«", "*")] == \
        ["comma", "ellipsis", "ellipsis", "qe", "exclam", "dash", "quote", None]
    with SessionLocal() as s:
        pu = analyze(s, list(texts.values()), "text")
        first = next(g for g in pu["groups"] if g["full"].endswith("«Первый»"))
        assert first["counts"]["period"] == 2 and first["endings"]["period"] == 100.0
        second = next(g for g in pu["groups"] if g["full"].endswith("«Второй»"))
        assert second["counts"]["comma"] == 1 and second["per1000"]["comma"] > 0
        assert pu["total"]["words"] == first["words"] + second["words"]
        ex = examples(s, list(texts.values()), "comma")
        assert ex["total"] == 1 and any(k == "kw" for _, k in ex["items"][0]["parts"])
    page = c.get(f"/analysis?tab=punct&dictionary_id=1&pdim=text&pmark=comma")
    assert page.status_code == 200 and "Профиль пунктуации" in page.text and "Буря мглою" in page.text
    x = c.get("/export/punct.xlsx?pdim=author")
    assert x.status_code == 200 and x.content[:2] == b"PK"
