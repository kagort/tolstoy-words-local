"""Двухступенчатая обработка: предварительный разбор всего текста, точный — только для кандидатов."""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = """Запах сирени стоял в саду. Пахло дымом и гарью.
Он почувствовал сладкий запах и остановился. Вечер был тихий.
Дети бегали по двору. Никто не заметил, как пришла ночь. Запахи исчезли к утру."""


@pytest.fixture(scope="module", autouse=True)
def two_stage_db():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    from app.config import settings
    from app.db import engine

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")
    old = settings.processing_mode, settings.fast_parser
    # в тестах обе ступени — Natasha (Stanza не установлена), но это разные экземпляры парсера
    settings.processing_mode, settings.fast_parser = "two_stage", "natasha-fast"
    assert settings.two_stage
    yield
    settings.processing_mode, settings.fast_parser = old


def test_two_stage_flow():
    from app.db import SessionLocal
    from app.models import PARSE_PRECISE, Hit, Job, SentenceRow, TextDoc, TokenRow
    from app.services.deepparse import parse_levels
    from app.services.dictionaries import create_dictionary
    from app.services.ingest import create_text
    from app.services.jobs import enqueue, run_pending

    with SessionLocal() as s:
        doc = create_text(s, filename="t.txt", data=TEXT.encode("utf-8"), title="Двухступенчатый")
        s.commit()
        tid = doc.id
    run_pending()
    with SessionLocal() as s:
        doc = s.get(TextDoc, tid)
        assert doc.status == "ready" and "двухступенчатый" in doc.pipeline_version
        lv = parse_levels(s, tid)
        assert lv["fast"] == lv["total"] > 0 and lv["precise"] == 0  # словарей ещё нет — всё предварительно
        # у предварительного разбора есть дерево: портрет и структура работают сразу
        assert s.query(SentenceRow).filter(SentenceRow.text_id == tid, SentenceRow.tree_depth > 0).count() == lv["total"]

        d = create_dictionary(s, "Запахи", [{"surface": "запах"}], None, {})
        s.commit()
        did = d.id
    run_pending()
    with SessionLocal() as s:
        lv = parse_levels(s, tid)
        precise_ids = set(s.scalars(s.query(SentenceRow.id).filter_by(text_id=tid, parse_level=PARSE_PRECISE).statement))
        # точно разобраны только предложения со словом «запах» (в любой форме): 3 из 7
        assert lv["precise"] == 3 and lv["fast"] == lv["total"] - 3
        hits = s.query(Hit).filter_by(dictionary_id=did).all()
        assert len(hits) == 3 and {h.sentence_id for h in hits} <= precise_ids
        token_ids = {h.token_id for h in hits}

        # точный разбор всего текста по кнопке: id токенов сохраняются, вхождения не теряются
        enqueue(s, "deepparse", {"text_id": tid}, "Точный разбор")
        s.commit()
    run_pending()
    with SessionLocal() as s:
        lv = parse_levels(s, tid)
        assert lv["precise"] == lv["total"] and lv["fast"] == 0
        hits = s.query(Hit).filter_by(dictionary_id=did).all()
        assert len(hits) == 3 and {h.token_id for h in hits} == token_ids
        assert all(s.get(TokenRow, t) is not None for t in token_ids)
        assert s.query(Job).filter(Job.status == "error").count() == 0


def test_precise_parse_of_single_sentence():
    from fastapi.testclient import TestClient

    from app.db import SessionLocal
    from app.main import app
    from app.models import PARSE_FAST, PARSE_PRECISE, SentenceRow
    from app.services.ingest import create_text
    from app.services.jobs import run_pending

    with SessionLocal() as s:
        doc = create_text(s, filename="u.txt", data="Тишина. Птицы молчали. Ветер стих.".encode("utf-8"), title="Тишина")
        s.commit()
        tid = doc.id
    run_pending()
    with SessionLocal() as s:
        sid = s.query(SentenceRow.id).filter_by(text_id=tid, parse_level=PARSE_FAST).order_by(SentenceRow.idx).first()[0]
    c = TestClient(app)
    c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    page = c.get(f"/sentences/{sid}")
    assert page.status_code == 200 and "предварительный разбор" in page.text
    assert "предварительный" in c.get(f"/texts/{tid}").text
    r = c.post(f"/sentences/{sid}/precise", follow_redirects=False)
    assert r.status_code == 303
    run_pending()
    with SessionLocal() as s:
        assert s.get(SentenceRow, sid).parse_level == PARSE_PRECISE
        # остальные предложения текста не тронуты
        assert s.query(SentenceRow).filter_by(text_id=tid, parse_level=PARSE_FAST).count() == 2
    assert c.post(f"/texts/{tid}/precise", follow_redirects=False).status_code == 303
    run_pending()
    with SessionLocal() as s:
        assert s.query(SentenceRow).filter_by(text_id=tid, parse_level=PARSE_FAST).count() == 0
    assert c.get(f"/texts/{tid}/stats").status_code == 200
