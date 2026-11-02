"""Замер скорости разбора на машине пользователя: python -m app.cli bench-parse.

Берёт предложения уже загруженного текста (те же токены, что при разборе), разбирает их быстрым и точным парсером
с разными настройками (потоки PyTorch, предложений за вызов) и печатает слов в секунду. Каждый вариант сравнивается
с первым: доля слов с тем же разбором (лемма, часть речи, граммемы, главное слово, связь) — ускорение не должно
менять результат.
"""

from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import SentenceRow, TextDoc, TokenRow


def sample(session: Session, text_id: int | None = None, n: int = 400) -> tuple[str, list[list[str]]]:
    """Первые n предложений текста (по умолчанию — самого длинного готового) как списки словоформ."""
    if text_id is None:
        text_id = session.scalar(select(TextDoc.id).where(TextDoc.status == "ready").order_by(TextDoc.n_words.desc()).limit(1))
    doc = session.get(TextDoc, text_id) if text_id else None
    if doc is None:
        return "", []
    sids = list(session.scalars(select(SentenceRow.id).where(SentenceRow.text_id == doc.id).order_by(SentenceRow.idx).limit(n)))
    forms: dict[int, list[str]] = {sid: [] for sid in sids}
    for sid, form in session.execute(select(TokenRow.sentence_id, TokenRow.form).where(TokenRow.sentence_id.in_(sids or [-1]))
                                     .order_by(TokenRow.sentence_id, TokenRow.idx)):
        forms[sid].append(form)
    return doc.title, [f for f in forms.values() if f]


def _key(words) -> list[tuple]:
    return [(w.lemma, w.upos, tuple(sorted(w.feats.items())), w.head, w.deprel) for s in words for w in s]


def run(parser, sentences: list[list[str]], batch: int) -> tuple[float, list]:
    t0 = time.perf_counter()
    out = []
    for k in range(0, len(sentences), batch):
        out.extend(parser.parse(sentences[k: k + batch]))
    return time.perf_counter() - t0, out


def bench(session: Session, text_id: int | None = None, n: int = 400, threads: list[int] | None = None,
          batches: list[int] | None = None, backends: list[str] | None = None, echo=print) -> list[dict]:
    """Таблица замеров: [{backend, threads, batch, seconds, words_per_sec, same_pct}]."""
    from app.config import settings
    from app.nlp.parsers import available_cpus, get_parser

    title, sents = sample(session, text_id, n)
    if not sents:
        echo("Нет готовых текстов для замера.")
        return []
    n_words = sum(len(s) for s in sents)
    echo(f"Текст «{title}»: {len(sents)} предложений, {n_words} токенов. Доступно ядер: {available_cpus()}.")
    rows = []
    for backend in backends or [settings.fast_parser, settings.nlp_backend]:
        try:
            parser = get_parser(backend)
        except Exception as exc:  # noqa: BLE001 — модель не скачана или не установлена
            echo(f"{backend}: недоступен ({exc})")
            continue
        torch_ok = backend.startswith("stanza")
        ref = None
        for th in (threads or [0]) if torch_ok else [0]:
            if torch_ok:
                from app.nlp.parsers import set_torch_threads

                th = set_torch_threads(th or None)
            for b in batches or [settings.parse_batch_size, settings.precise_batch_size]:
                run(parser, sents[:8], b)  # прогрев
                dt, out = run(parser, sents, b)
                key = _key(out)
                if ref is None:
                    ref = key
                same = 100.0 * sum(1 for a, c in zip(ref, key) if a == c) / max(len(ref), 1)
                row = {"backend": parser.version, "threads": th if torch_ok else None, "batch": b, "seconds": round(dt, 2),
                       "words_per_sec": round(n_words / dt) if dt else None, "same_pct": round(same, 2)}
                rows.append(row)
                echo(f"{row['backend']:<22} потоков {str(row['threads'] or '—'):>3}  предложений за вызов {b:>4}  "
                     f"{row['seconds']:>7} с  {row['words_per_sec']:>6} слов/с  совпадает с первым вариантом: {row['same_pct']} %")
    return rows
