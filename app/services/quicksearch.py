"""«Найти слово» в «Анализе»: любое слово или словосочетание без своего словаря.

Слово заводится в служебном словаре «Быстрый поиск» (settings.quick = True) и сопоставляется только с текстами
текущей выборки; какие тексты уже просмотрены для слова — settings.quick_done = {entry_id: [text_id, …]} (повторный
поиск в той же выборке не запускает задачу). Новые тексты со служебным словарём автоматически не сопоставляются —
иначе с каждым найденным словом загрузка текстов становилась бы дольше.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Dictionary, Job
from app.services.dictionaries import clean_surface, entry_key, make_entry
from app.services.jobs import enqueue

QUICK_NAME = "Быстрый поиск"
QUICK_DESCRIPTION = "Служебный словарь: слова из поля «Найти слово» в «Анализе». Его можно очищать — слова добавятся снова при поиске."
MAX_WORDS = 5  # «словосочетание» длиннее — вероятно, вставили фразу по ошибке


def is_quick(d: Dictionary | None) -> bool:
    return bool(d is not None and (d.settings or {}).get("quick"))


def quick_dictionary(session: Session, create: bool = True) -> Dictionary | None:
    d = next((x for x in session.scalars(select(Dictionary)) if is_quick(x)), None)
    if d is None and create:
        name = QUICK_NAME
        if session.scalar(select(Dictionary.id).where(Dictionary.name == name)):  # имя занято обычным словарём
            name = f"{QUICK_NAME} (служебный)"
        d = Dictionary(name=name, description=QUICK_DESCRIPTION, settings={"quick": True, "quick_done": {}})
        session.add(d)
        session.flush()
    return d


def find(session: Session, word: str, text_ids: list[int]) -> dict:
    """Завести слово (или найти уже заведённое) и поставить сопоставление с ещё не просмотренными текстами выборки.

    Возвращает {"error": …} или {"dictionary": d, "entry": e, "job": Job | None, "new": bool}.
    """
    surface = clean_surface(word or "")
    if not surface:
        return {"error": "введите слово"}
    if len(surface.split()) > MAX_WORDS:
        return {"error": f"слишком длинно: до {MAX_WORDS} слов (словосочетание)"}
    if not text_ids:
        return {"error": "в выборке нет текстов"}
    d = quick_dictionary(session)
    probe = make_entry({"surface": surface})
    entry = next((e for e in d.entries if entry_key(e) == entry_key(probe)), None)
    new = entry is None
    if new:
        d.entries.append(probe)
        session.flush()
        entry = probe
    done = set((d.settings or {}).get("quick_done", {}).get(str(entry.id), []))
    missing = sorted(set(text_ids) - done)
    job = None
    if missing:
        job = _pending_job(session, d.id, entry.id, missing) or enqueue(
            session, "match", {"dictionary_id": d.id, "entry_ids": [entry.id], "text_ids": missing},
            title=f"Поиск слова «{entry.surface}» ({len(missing)} текст.)")
    return {"dictionary": d, "entry": entry, "job": job, "new": new}


def _pending_job(session: Session, dict_id: int, entry_id: int, text_ids: list[int]) -> Job | None:
    """Такой же поиск уже в очереди или идёт (повторное нажатие «Найти») — новую задачу не ставим."""
    for j in session.scalars(select(Job).where(Job.kind == "match", Job.status.in_(["queued", "running"]))):
        p = j.params or {}
        if p.get("dictionary_id") == dict_id and p.get("entry_ids") == [entry_id] and set(text_ids) <= set(p.get("text_ids") or []):
            return j
    return None


def pending(session: Session, d: Dictionary | None, entry_id: int | None = None) -> bool:
    """Идёт ли поиск по служебному словарю (для плашки «ищем…» и автообновления страницы)."""
    if not is_quick(d):
        return False
    for j in session.scalars(select(Job).where(Job.kind == "match", Job.status.in_(["queued", "running"]))):
        p = j.params or {}
        if p.get("dictionary_id") == d.id and (entry_id is None or entry_id in (p.get("entry_ids") or [])):
            return True
    return False


def mark_done(session: Session, dictionary_id: int, entry_ids: list[int] | None, text_ids: list[int] | None) -> None:
    """После сопоставления (matching.run_match): запомнить просмотренные тексты для слов служебного словаря."""
    d = session.get(Dictionary, dictionary_id)
    if not is_quick(d) or not entry_ids or not text_ids:
        return
    st = dict(d.settings or {})
    done = dict(st.get("quick_done") or {})
    for eid in entry_ids:
        done[str(eid)] = sorted(set(done.get(str(eid), [])) | set(text_ids))
    d.settings = {**st, "quick_done": done}


def forget_text(session: Session, text_id: int) -> None:
    """Текст разобран заново (вхождения удалены) — для быстрого поиска он снова «не просмотрен»."""
    d = quick_dictionary(session, create=False)
    if d is None:
        return
    st = dict(d.settings or {})
    done = {k: [t for t in v if t != text_id] for k, v in (st.get("quick_done") or {}).items()}
    d.settings = {**st, "quick_done": done}
