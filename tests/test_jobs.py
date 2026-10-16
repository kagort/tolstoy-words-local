"""Очередь задач: отмена, удаление, возврат брошенных задач, защита от бесконечных падений."""

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")


@pytest.fixture(scope="module", autouse=True)
def fresh_db():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    from app.db import engine

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")


def _text(s, body="Запах сирени стоял в саду. Пахло дымом."):
    from app.services.ingest import create_text

    doc = create_text(s, filename="t.txt", data=body.encode("utf-8"), title="Т")
    s.commit()
    return doc


def test_cancel_queued_ingest_marks_text():
    from app.db import SessionLocal
    from app.models import Job, TextDoc
    from app.services.jobs import cancel_job

    with SessionLocal() as s:
        doc = _text(s)
        job = s.query(Job).filter_by(kind="ingest").order_by(Job.id.desc()).first()
        cancel_job(s, job)
        s.commit()
        assert job.status == "cancelled"
    with SessionLocal() as s:
        doc = s.get(TextDoc, doc.id)
        assert doc.status == "error" and "отмен" in doc.error


def test_cancel_running_job_stops_on_next_step():
    from app.db import SessionLocal
    from app.models import Job, SentenceRow, TextDoc
    from app.services.jobs import claim_next, run_job

    with SessionLocal() as s:
        doc = _text(s, " ".join(f"Предложение номер {i} о запахе." for i in range(300)))
        tid = doc.id
        job = claim_next(s)
        job.cancel_requested = True  # как будто нажали «Отменить», пока задача выполняется
        s.commit()
        jid = job.id
    from app.config import settings

    old = settings.parse_batch_size
    settings.parse_batch_size = 20
    try:
        run_job(jid)
    finally:
        settings.parse_batch_size = old
    with SessionLocal() as s:
        assert s.get(Job, jid).status == "cancelled"
        assert s.get(TextDoc, tid).status == "error"
        assert s.query(SentenceRow).filter_by(text_id=tid).count() == 0  # частичный разбор убран


def test_requeue_stale_and_attempt_limit():
    from app.config import settings
    from app.db import SessionLocal
    from app.models import Job
    from app.services.jobs import enqueue, requeue_stale

    old = datetime.now(timezone.utc) - timedelta(hours=1)
    with SessionLocal() as s:
        fresh = enqueue(s, "match", {"dictionary_id": 999}, "живая")
        dead = enqueue(s, "match", {"dictionary_id": 999}, "брошенная")
        hopeless = enqueue(s, "match", {"dictionary_id": 999}, "падает всегда")
        for j, hb, att in ((fresh, datetime.now(timezone.utc), 1), (dead, old, 1), (hopeless, old, settings.job_max_attempts)):
            j.status, j.heartbeat_at, j.attempts = "running", hb, att
        s.commit()
        ids = fresh.id, dead.id, hopeless.id
    assert requeue_stale() == 2
    with SessionLocal() as s:
        f, d, h = (s.get(Job, i) for i in ids)
        assert f.status == "running"  # чужая живая задача не тронута (несколько обработчиков)
        assert d.status == "queued"
        assert h.status == "error" and "прерывался" in h.error


def test_delete_and_clear_jobs():
    from app.db import SessionLocal
    from app.models import Job
    from app.services.jobs import delete_jobs, enqueue

    with SessionLocal() as s:
        j1 = enqueue(s, "match", {"dictionary_id": 999}, "готовая")
        j1.status = "done"
        j2 = enqueue(s, "match", {"dictionary_id": 999}, "идёт")
        j2.status = "running"
        s.commit()
        assert delete_jobs(s, [j1, j2]) == 1
        s.commit()
        assert s.get(Job, j2.id) is not None


def test_manual_marks_survive_interrupted_match():
    """Отметки «ложное» сохраняются в параметрах задачи и не теряются, если пересчёт прервался."""
    from app.db import SessionLocal
    from app.models import Hit, Job
    from app.services.dictionaries import create_dictionary
    from app.services.jobs import JobCancelled, enqueue, run_pending
    from app.services.matching import run_match

    with SessionLocal() as s:
        s.query(Job).delete()
        s.commit()
        _text(s, "Запах сирени. Запах дыма. Запах гари.")
    run_pending()
    with SessionLocal() as s:
        d = create_dictionary(s, "Запахи-тест", [{"surface": "запах"}], None, {})
        s.commit()
        did = d.id
    run_pending()
    with SessionLocal() as s:
        hit = s.query(Hit).filter_by(dictionary_id=did).first()
        hit.is_valid = False
        token_id = hit.token_id
        job = enqueue(s, "match", {"dictionary_id": did}, "пересчёт")
        s.commit()
        jid = job.id

    import app.services.matching as M

    real = M._match_entries

    def crash(*a, **kw):
        raise JobCancelled()

    M._match_entries = crash
    try:
        with pytest.raises(JobCancelled):
            run_match(jid, did)
    finally:
        M._match_entries = real
    with SessionLocal() as s:
        params = s.get(Job, jid).params
        assert [x for x in params["manual"] if x[1] == token_id and x[2] is False]
        assert s.query(Hit).filter_by(dictionary_id=did).count() == 0  # пересчёт оборвался после удаления
    run_match(jid, did, manual=params["manual"])  # повторный запуск той же задачи
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(dictionary_id=did, token_id=token_id).one().is_valid is False
