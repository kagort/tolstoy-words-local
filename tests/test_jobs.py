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


def test_delete_texts_with_all_links():
    """Удаление текста убирает предложения, токены, вхождения, связи, имена, членство в корпусах и задачи текста."""
    from fastapi.testclient import TestClient

    from app.db import SessionLocal
    from app.main import app
    from app.models import Corpus, Entity, Hit, Job, SentenceRow, TextDoc, TokenRelation, TokenRow
    from app.services.dictionaries import create_dictionary
    from app.services.jobs import run_pending

    with SessionLocal() as s:
        s.query(Job).delete()
        c = Corpus(name="Для удаления")
        s.add(c)
        s.flush()
        doc = _text(s, "Пьер Безухов почувствовал запах сирени в Москве. Сладкий запах стоял в саду.")
        doc.corpora = [c]
        keep = _text(s, "Запах дыма. Наташа Ростова молчала.")
        s.commit()
        tid, keep_id, cid = doc.id, keep.id, c.id
    run_pending()
    with SessionLocal() as s:
        create_dictionary(s, "Запахи для удаления", [{"surface": "запах"}], None, {})
        s.commit()
    run_pending()
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(text_id=tid).count() > 0 and s.query(Entity).filter_by(text_id=tid).count() > 0
        assert s.query(TokenRelation).filter_by(text_id=tid).count() > 0
    c = TestClient(app)
    c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    assert 'name="text_ids"' in c.get("/texts").text
    r = c.post("/texts/delete", data={"text_ids": [str(tid)]}, follow_redirects=False)
    assert r.status_code == 303
    with SessionLocal() as s:
        assert s.get(TextDoc, tid).status == "deleting"
    run_pending()
    with SessionLocal() as s:
        assert s.get(TextDoc, tid) is None
        for model in (SentenceRow, TokenRow, Hit, TokenRelation, Entity):
            assert s.query(model).filter_by(text_id=tid).count() == 0, model
        assert s.get(Corpus, cid) is not None and not s.get(Corpus, cid).texts
        assert s.get(TextDoc, keep_id) is not None and s.query(Hit).filter_by(text_id=keep_id).count() > 0  # другой текст не тронут
        assert s.query(Job).filter(Job.status == "error").count() == 0


def test_delete_cancels_text_jobs_and_corpus_with_texts():
    from app.db import SessionLocal
    from app.models import Corpus, Job, TextDoc
    from app.services.jobs import run_pending
    from app.services.textdelete import request_delete

    with SessionLocal() as s:
        s.query(Job).delete()
        doc = _text(s, "Тишина.")  # разбор ещё в очереди
        s.commit()
        tid = doc.id
        job = request_delete(s, [tid])
        s.commit()
        ingest = s.query(Job).filter_by(kind="ingest").one()
        assert ingest.status == "cancelled" and job.kind == "delete_text"
        assert request_delete(s, [tid]) is None  # повторное удаление не ставит вторую задачу
    run_pending()
    with SessionLocal() as s:
        assert s.get(TextDoc, tid) is None
        c = Corpus(name="Корпус с текстами")
        s.add(c)
        s.flush()
        d1 = _text(s, "Первый текст.")
        d1.corpora = [c]
        s.commit()
        cid, d1_id = c.id, d1.id
    from fastapi.testclient import TestClient

    from app.main import app

    cl = TestClient(app)
    cl.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    assert cl.post(f"/corpora/{cid}/delete", data={"with_texts": "1"}, follow_redirects=False).status_code == 303
    run_pending()
    with SessionLocal() as s:
        assert s.get(Corpus, cid) is None and s.get(TextDoc, d1_id) is None


def test_ai_check_calibrated_flow():
    """С образцами обоих типов модель калибруется и отличает «машинный» шаблон от живого текста."""
    import random

    from app.db import SessionLocal
    from app.services import aicheck as C
    from app.services.ingest import create_text
    from app.services.jobs import run_pending

    rnd = random.Random(1)
    human_bits = ["— Ты куда? — спросил он.", "Ну и пусть!", "Я долго молчал, глядя в окно, где медленно падал снег и кто-то шёл по двору.",
                  "Она засмеялась.", "А что потом было, я уже не помню, да и вспоминать не хочу, честное слово.", "Тишина.",
                  "Мы вышли на крыльцо, и ветер сразу ударил в лицо?", "— Пойдём, — сказала мать."]
    ai_bits = ["Важно отметить, что аромат играет ключевую роль в создании уникальной атмосферы гармонии.",
               "Кроме того, насыщенный запах трав создаёт удивительное ощущение, наполняя пространство гармонией.",
               "Таким образом, богатый аромат цветов является неотъемлемой частью незабываемой атмосферы."]
    with SessionLocal() as s:
        for k in range(2):
            h = "\n".join(" ".join(rnd.choice(human_bits) for _ in range(rnd.randint(1, 6))) for _ in range(260))
            a = "\n".join(" ".join(rnd.choice(ai_bits) for _ in range(3)) for _ in range(160))
            create_text(s, filename=f"h{k}.txt", data=h.encode("utf-8"), title=f"Люди {k}", source_type="natural")
            create_text(s, filename=f"a{k}.txt", data=a.encode("utf-8"), title=f"Модель {k}", source_type="generated")
        s.commit()
    run_pending()
    with SessionLocal() as s:
        C._CALIB.clear()
        m = C.calibrate(s)
        assert m["ok"] and m["n_ai"] >= 10 and m["n_human"] >= 10 and m["cv_accuracy"] is not None
        ai = C.report(s, C.parse_raw(" ".join(ai_bits) * 40))
        hu = C.report(s, C.parse_raw("\n".join(" ".join(rnd.choice(human_bits) for _ in range(4)) for _ in range(60))))
        assert ai["enough"] and ai["prob"] > hu["prob"] and ai["level"] in ("много", "заметно")
        assert any(r["key"] == "markers" and r["position"] > 0.5 for r in ai["rows"])


def test_stanza_models_job(tmp_path):
    """Без моделей Stanza обработчик ставит задачу загрузки; упавшие из-за моделей задачи потом возвращаются в очередь."""
    import os

    from app.config import settings
    from app.db import SessionLocal
    from app.models import Job
    from app.services import models as M
    from app.services.jobs import claim_next, enqueue

    old = settings.nlp_backend, settings.stanza_dir
    settings.nlp_backend, settings.stanza_dir = "stanza", str(tmp_path)
    try:
        with SessionLocal() as s:
            s.query(Job).delete()
            s.commit()
            other = enqueue(s, "match", {"dictionary_id": 999}, "раньше в очереди")
            s.commit()
        assert not M.stanza_ready() and M.needs_models()
        jid = M.ensure_download_job()
        assert jid and M.ensure_download_job() == jid  # вторая задача не ставится
        with SessionLocal() as s:
            assert claim_next(s).id == jid  # загрузка моделей — вне очереди
            from app.nlp.parsers import get_parser
            try:
                get_parser("stanza")
                raise AssertionError("должно быть исключение")
            except RuntimeError as exc:
                assert M.MISSING_MARK in str(exc)
            job = s.get(Job, other.id)
            job.status, job.error = "error", f"{M.MISSING_MARK}: дождитесь…"
            s.commit()
        os.makedirs(tmp_path / "ru" / "depparse")
        (tmp_path / "resources.json").write_text("{}")
        assert M.stanza_ready() and not M.needs_models() and M.ensure_download_job() is None
        assert M.requeue_waiting() == 1
        with SessionLocal() as s:
            assert s.query(Job).filter_by(kind="match", status="queued").count() == 1
    finally:
        settings.nlp_backend, settings.stanza_dir = old
