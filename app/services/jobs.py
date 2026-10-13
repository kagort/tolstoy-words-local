"""Очередь фоновых задач в таблице jobs (без Redis/Celery)."""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Job

log = logging.getLogger(__name__)


def enqueue(session: Session, kind: str, params: dict, title: str = "") -> Job:
    job = Job(kind=kind, params=params, title=title, status="queued")
    session.add(job)
    session.flush()
    return job


def set_progress(job_id: int, progress: float, message: str | None = None) -> None:
    """Пишет прогресс отдельной транзакцией, чтобы его было видно в интерфейсе сразу."""
    with SessionLocal() as s:
        s.execute(update(Job).where(Job.id == job_id).values(progress=min(max(progress, 0.0), 1.0), message=message))
        s.commit()


def claim_next(session: Session) -> Job | None:
    row = session.execute(
        text(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY id "
            "FOR UPDATE SKIP LOCKED LIMIT 1"
        )
    ).first()
    if not row:
        return None
    job = session.get(Job, row[0])
    job.status = "running"
    job.started_at = datetime.now(timezone.utc)
    job.progress = 0.0
    session.commit()
    return job


def run_job(job_id: int) -> None:
    from app.services.ingest import run_ingest
    from app.services.matching import run_match

    handlers = {"ingest": run_ingest, "match": run_match}
    with SessionLocal() as s:
        job = s.get(Job, job_id)
        kind, params = job.kind, dict(job.params or {})
    try:
        handlers[kind](job_id, **params)
        status, err = "done", None
    except Exception as exc:  # noqa: BLE001 — задача не должна ронять обработчик
        log.exception("Задача %s упала", job_id)
        status, err = "error", f"{exc}\n\n{traceback.format_exc()}"
    with SessionLocal() as s:
        s.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status=status, error=err, finished_at=datetime.now(timezone.utc), progress=1.0 if status == "done" else Job.progress)
        )
        s.commit()


def run_pending(limit: int | None = None) -> int:
    """Выполняет задачи из очереди (используется обработчиком и тестами)."""
    done = 0
    while limit is None or done < limit:
        with SessionLocal() as s:
            job = claim_next(s)
            job_id = job.id if job else None
        if job_id is None:
            break
        run_job(job_id)
        done += 1
    return done


def requeue_stale() -> int:
    """После перезапуска обработчика возвращает «зависшие» задачи в очередь."""
    with SessionLocal() as s:
        res = s.execute(update(Job).where(Job.status == "running").values(status="queued"))
        s.commit()
        return res.rowcount or 0


def worker_loop(poll_seconds: float = 2.0) -> None:
    n = requeue_stale()
    if n:
        log.info("Возвращено в очередь задач: %s", n)
    log.info("Обработчик задач запущен")
    while True:
        try:
            processed = run_pending(limit=1)
        except Exception:  # noqa: BLE001
            log.exception("Ошибка обработчика")
            processed = 0
        if not processed:
            time.sleep(poll_seconds)


def active_jobs(session: Session, limit: int = 20) -> list[Job]:
    return list(session.scalars(select(Job).order_by(Job.id.desc()).limit(limit)))
