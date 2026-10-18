"""Удаление текстов со всеми связями: предложения, токены, вхождения словарей и их связи, имена собственные,
статистический портрет, членство в корпусах, задачи этого текста в очереди.

Удаление большого текста (сотни тысяч токенов) идёт фоновой задачей «delete_text», по шагам и в порядке
«от зависимых к главным» — так база не проверяет каскадные ссылки построчно. Пока задача идёт, текст
помечен «удаляется» и уже не участвует в анализе.
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Entity, Hit, Job, SentenceRow, TextDoc, TokenRelation, TokenRow, corpus_texts
from app.services.jobs import cancel_job, enqueue, set_progress

log = logging.getLogger(__name__)

TEXT_JOBS = ("ingest", "ner", "deepparse")


def _job_texts(job: Job) -> set[int]:
    p = job.params or {}
    out = set(p.get("text_ids") or [])
    if p.get("text_id"):
        out.add(p["text_id"])
    return out


def request_delete(session: Session, text_ids: list[int]) -> Job | None:
    """Помечает тексты «удаляется», отменяет их задачи и ставит удаление в очередь. Возвращает задачу."""
    docs = list(session.scalars(select(TextDoc).where(TextDoc.id.in_(text_ids or [-1]), TextDoc.status != "deleting")))
    if not docs:
        return None
    ids = {d.id for d in docs}
    for job in session.scalars(select(Job).where(Job.status.in_(["queued", "running"]), Job.kind != "delete_text")):
        affected = _job_texts(job)
        if not affected or not affected & ids:
            continue
        if job.kind in TEXT_JOBS or affected <= ids:
            cancel_job(session, job)  # в очереди — сразу, выполняющаяся — на ближайшем шаге
        else:  # пересчёт словаря по нескольким текстам — убираем из него удаляемые
            job.params = {**job.params, "text_ids": sorted(affected - ids)}
    for d in docs:
        d.status = "deleting"
    titles = ", ".join(f"«{d.title}»" for d in docs)
    job = enqueue(session, "delete_text", {"text_ids": sorted(ids)}, title=f"Удаление текстов: {titles}"[:480])
    return job


def delete_text_now(text_id: int, job_id: int | None = None, progress: tuple[float, float] = (0.0, 1.0)) -> None:
    lo, hi = progress
    steps = [
        ("связи вхождений", lambda s: s.execute(delete(TokenRelation).where(TokenRelation.text_id == text_id))),
        ("вхождения словарей", lambda s: s.execute(delete(Hit).where(Hit.text_id == text_id))),
        ("имена собственные", lambda s: s.execute(delete(Entity).where(Entity.text_id == text_id))),
        ("токены", lambda s: s.execute(delete(TokenRow).where(TokenRow.text_id == text_id))),
        ("предложения", lambda s: s.execute(delete(SentenceRow).where(SentenceRow.text_id == text_id))),
        ("текст", lambda s: (s.execute(delete(corpus_texts).where(corpus_texts.c.text_id == text_id)),
                             s.execute(delete(TextDoc).where(TextDoc.id == text_id)))),
    ]
    for k, (label, step) in enumerate(steps):
        with SessionLocal() as s:
            step(s)
            s.commit()
        if job_id:
            set_progress(job_id, lo + (hi - lo) * (k + 1) / len(steps), f"Текст {text_id}: удалено — {label}")


def run_delete(job_id: int | None, text_ids: list[int]) -> None:
    """Задача «delete_text». Повторный запуск после сбоя безопасен: уже удалённое просто пропускается."""
    n = len(text_ids)
    for i, tid in enumerate(text_ids):
        delete_text_now(tid, job_id, (i / n, (i + 1) / n))
    log.info("Удалено текстов: %s", n)
