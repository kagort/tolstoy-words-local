"""Словари: импорт, верификация разборов, правка."""

from __future__ import annotations

import csv
import io
import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Dictionary, DictionaryEntry, TokenRow
from app.nlp.morph import parse_options
from app.nlp.normalize import decode_bytes
from app.nlp.tree import norm
from app.services.jobs import enqueue

DEFAULT_SETTINGS = {"include_participles": False, "include_gerunds": False, "merge_propn": True}

HEADER_WORDS = {"слово", "word", "лемма", "lemma", "токен", "token"}


def parse_dictionary_file(filename: str, data: bytes) -> list[dict]:
    """CSV/TSV/TXT/XLSX → [{surface, category, upos, comment}]."""
    name = filename.lower()
    rows: list[list[str]] = []
    if name.endswith(".xlsx"):
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(data), read_only=True)
        for r in wb.active.iter_rows(values_only=True):
            rows.append(["" if v is None else str(v) for v in r])
    else:
        content = decode_bytes(data)
        return parse_dictionary_text(content)
    return _rows_to_entries(rows)


def parse_dictionary_text(content: str) -> list[dict]:
    lines = [ln for ln in content.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    sample = "\n".join(lines[:20])
    if any(sep in sample for sep in (";", "\t")) or (lines[0].count(",") >= 1 and len(lines) > 1 and all(ln.count(",") <= 3 for ln in lines[:20])):
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";\t,")
            rows = list(csv.reader(lines, dialect))
            if rows and max(len(r) for r in rows) > 1:
                return _rows_to_entries(rows)
        except csv.Error:
            pass
    # простой список: по слову в строке или через запятую
    words: list[str] = []
    for ln in lines:
        words += [w for w in re.split(r"[,;]", ln)]
    return _rows_to_entries([[w] for w in words])


def _rows_to_entries(rows: list[list[str]]) -> list[dict]:
    if rows and rows[0] and rows[0][0].strip().lower() in HEADER_WORDS:
        rows = rows[1:]
    out, seen = [], set()
    for r in rows:
        cells = [c.strip() for c in r] + ["", "", "", ""]
        surface = cells[0]
        if not surface or surface.lower() in seen:
            continue
        seen.add(surface.lower())
        out.append({"surface": surface, "category": cells[1] or None, "upos": cells[2].upper() or None, "comment": cells[3] or None})
    return out


def make_entry(item: dict) -> DictionaryEntry:
    surface = item["surface"]
    opts = parse_options(surface.split()[0] if surface else surface)
    wanted_upos = item.get("upos")
    best = None
    if wanted_upos:
        best = next((o for o in opts if o.upos == wanted_upos), None)
    best = best or (opts[0] if opts else None)
    lemma = best.lemma if best else surface.lower()
    upos = best.upos if best else wanted_upos
    lemmas = {o.lemma for o in opts if o.score >= 0.05}
    pos_set = {o.upos for o in opts if o.score >= 0.05}
    return DictionaryEntry(
        surface=surface,
        lemma=lemma,
        lemma_norm=norm(lemma),
        upos=upos,
        match_mode="lemma",
        category=item.get("category"),
        comment=item.get("comment"),
        verified=bool(wanted_upos),
        ambiguous=len(lemmas) > 1 or len(pos_set) > 1,
        options=[{"lemma": o.lemma, "upos": o.upos, "tag": o.tag, "score": round(o.score, 3)} for o in opts[:6]],
    )


def create_dictionary(session: Session, name: str, items: list[dict], description: str | None = None,
                      settings: dict | None = None, enqueue_match: bool = True) -> Dictionary:
    d = Dictionary(name=name.strip(), description=description, settings={**DEFAULT_SETTINGS, **(settings or {})})
    for it in items:
        d.entries.append(make_entry(it))
    session.add(d)
    session.flush()
    if enqueue_match:
        enqueue(session, "match", {"dictionary_id": d.id}, title=f"Сопоставление словаря «{d.name}»")
    return d


def add_entries(session: Session, d: Dictionary, items: list[dict]) -> list[DictionaryEntry]:
    existing = {e.surface.lower() for e in d.entries}
    new = []
    for it in items:
        if it["surface"].lower() in existing:
            continue
        e = make_entry(it)
        d.entries.append(e)
        new.append(e)
    session.flush()
    if new:
        enqueue(session, "match", {"dictionary_id": d.id, "entry_ids": [e.id for e in new]},
                title=f"Сопоставление новых слов словаря «{d.name}»")
    return new


def update_entry(session: Session, entry: DictionaryEntry, *, lemma: str | None = None, upos: str | None = None,
                 match_mode: str | None = None, category: str | None = None) -> None:
    changed = False
    if lemma is not None and lemma.strip() and lemma.strip() != entry.lemma:
        entry.lemma = lemma.strip()
        entry.lemma_norm = norm(entry.lemma)
        changed = True
    if upos is not None and (upos or None) != entry.upos:
        entry.upos = upos or None
        changed = True
    if match_mode in ("lemma", "forms") and match_mode != entry.match_mode:
        entry.match_mode = match_mode
        changed = True
    if category is not None:
        entry.category = category or None
    entry.verified = True
    session.flush()
    if changed:
        enqueue(session, "match", {"dictionary_id": entry.dictionary_id, "entry_ids": [entry.id]},
                title=f"Пересчёт слова «{entry.surface}»")


def option_counts(session: Session, entry: DictionaryEntry) -> dict[str, int]:
    """Сколько токенов в базе у каждого варианта разбора (для экрана верификации)."""
    out = {}
    for o in entry.options or []:
        q = select(func.count()).select_from(TokenRow).where(TokenRow.lemma_norm == norm(o["lemma"]))
        if o.get("upos") and o["upos"] != "X":
            q = q.where(TokenRow.upos == o["upos"])
        out[f"{o['lemma']}|{o['upos']}"] = session.scalar(q) or 0
    return out
