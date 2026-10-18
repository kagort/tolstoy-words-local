"""Очередь фоновых задач в таблице jobs (без Redis/Celery).

Надёжность:
- каждый шаг задачи обновляет «пульс» (heartbeat_at); задачу без пульса дольше JOB_STALE_MINUTES
  любой обработчик возвращает в очередь — так переживаются падения и перезапуски, в том числе
  при нескольких обработчиках (`--scale worker=N`): чужие живые задачи не трогаются;
- после JOB_MAX_ATTEMPTS падений задача помечается ошибкой (например, не хватает памяти);
- отмена: задача в очереди отменяется сразу, выполняющаяся — на ближайшем шаге (JobCancelled).
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Job

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
FINISHED = ("done", "error", "cancelled")


class JobCancelled(Exception):
    """Пользователь отменил задачу — обработчик прекращает её на ближайшем шаге."""


def now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue(session: Session, kind: str, params: dict, title: str = "") -> Job:
    job = Job(kind=kind, params=params, title=title, status="queued")
    session.add(job)
    session.flush()
    return job


def set_progress(job_id: int, progress: float, message: str | None = None) -> None:
    """Пишет прогресс и пульс отдельной транзакцией (видно в интерфейсе сразу); проверяет запрос отмены."""
    with SessionLocal() as s:
        cancel = s.execute(
            update(Job).where(Job.id == job_id)
            .values(progress=min(max(progress, 0.0), 1.0), message=message, heartbeat_at=now())
            .returning(Job.cancel_requested)
        ).scalar()
        s.commit()
    if cancel:
        raise JobCancelled()


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
    job.started_at = job.heartbeat_at = now()
    job.progress = 0.0
    job.attempts = (job.attempts or 0) + 1
    session.commit()
    return job


def _on_cancel(kind: str, params: dict) -> None:
    """Приводит данные в порядок после отмены или удаления задачи."""
    if kind == "ingest" and params.get("text_id"):
        from app.services.ingest import mark_interrupted

        mark_interrupted(params["text_id"], "Разбор отменён")


def run_job(job_id: int) -> None:
    from app.services.deepparse import run_deepparse
    from app.services.ingest import run_ingest
    from app.services.matching import run_match

    from app.services.entities import run_ner

    handlers = {"ingest": run_ingest, "match": run_match, "deepparse": run_deepparse, "ner": run_ner}
    with SessionLocal() as s:
        job = s.get(Job, job_id)
        kind, params = job.kind, dict(job.params or {})
    try:
        handlers[kind](job_id, **params)
        status, err, msg = "done", None, None
    except JobCancelled:
        log.info("Задача %s отменена", job_id)
        _on_cancel(kind, params)
        status, err, msg = "cancelled", None, "отменена пользователем"
    except Exception as exc:  # noqa: BLE001 — задача не должна ронять обработчик
        log.exception("Задача %s упала", job_id)
        status, err, msg = "error", f"{exc}\n\n{traceback.format_exc()}", None
    values = dict(status=status, error=err, finished_at=now(), progress=1.0 if status == "done" else Job.progress)
    if msg:
        values["message"] = msg
    with SessionLocal() as s:
        s.execute(update(Job).where(Job.id == job_id).values(**values))
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


def requeue_stale(stale_minutes: float | None = None) -> int:
    """Возвращает в очередь задачи, чей обработчик перестал подавать «пульс» (упал или перезапущен).

    Задачи, которые падали уже JOB_MAX_ATTEMPTS раз, помечаются ошибкой, чтобы не крутиться вечно.
    """
    minutes = settings.job_stale_minutes if stale_minutes is None else stale_minutes
    border = now() - timedelta(minutes=minutes)
    n = 0
    with SessionLocal() as s:
        stale = list(s.scalars(
            select(Job).where(Job.status == "running", (Job.heartbeat_at.is_(None)) | (Job.heartbeat_at < border))
            .with_for_update(skip_locked=True)
        ))
        for job in stale:
            if job.cancel_requested:
                job.status, job.finished_at, job.message = "cancelled", now(), "отменена пользователем"
                _on_cancel(job.kind, dict(job.params or {}))
            elif (job.attempts or 0) >= settings.job_max_attempts:
                job.status, job.finished_at = "error", now()
                job.error = (f"Обработчик прерывался {job.attempts} раз на этой задаче (возможно, не хватает памяти "
                             f"или процесс перезапускали). Задача остановлена; нажмите «Повторить», когда причина устранена.")
                if job.kind == "ingest" and (job.params or {}).get("text_id"):
                    from app.services.ingest import mark_interrupted

                    mark_interrupted(job.params["text_id"], "Разбор прерван: обработчик падал несколько раз")
            else:
                job.status, job.message = "queued", "возвращена в очередь после перезапуска обработчика"
            n += 1
        s.commit()
    return n


def cancel_job(session: Session, job: Job) -> str:
    """Отмена из интерфейса. Возвращает текст для пользователя."""
    if job.status == "queued":
        job.status, job.finished_at, job.message = "cancelled", now(), "отменена до запуска"
        session.flush()
        _on_cancel(job.kind, dict(job.params or {}))
        return "Задача отменена"
    if job.status == "running":
        job.cancel_requested = True
        return "Задача будет остановлена на ближайшем шаге (обычно в течение минуты)"
    return "Задача уже завершена"


def delete_jobs(session: Session, jobs: list[Job]) -> int:
    """Удаляет завершённые (и ещё не начатые) задачи. Выполняющиеся не удаляются — их сначала отменяют."""
    n = 0
    for job in jobs:
        if job.status == "running":
            continue
        if job.status == "queued":
            _on_cancel(job.kind, dict(job.params or {}))
        session.delete(job)
        n += 1
    return n


def worker_loop(poll_seconds: float = 2.0) -> None:
    try:  # при первом запуске новой версии миграции применяет контейнер web — ошибка здесь не должна ронять обработчик
        n = requeue_stale()
        if n:
            log.info("Возвращено в очередь задач: %s", n)
    except Exception:  # noqa: BLE001
        log.exception("Не удалось проверить брошенные задачи при запуске")
    log.info("Обработчик задач запущен")
    last_check = time.monotonic()
    while True:
        try:
            processed = run_pending(limit=1)
        except Exception:  # noqa: BLE001
            log.exception("Ошибка обработчика")
            processed = 0
        if time.monotonic() - last_check > 60:  # раз в минуту подбираем брошенные задачи других обработчиков
            try:
                requeue_stale()
            except Exception:  # noqa: BLE001
                log.exception("Не удалось проверить брошенные задачи")
            last_check = time.monotonic()
        if not processed:
            time.sleep(poll_seconds)


def active_jobs(session: Session, limit: int = 20) -> list[Job]:
    return list(session.scalars(select(Job).order_by(Job.id.desc()).limit(limit)))
