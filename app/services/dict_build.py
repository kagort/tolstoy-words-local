"""Словари, собранные из данных корпуса: из указателя имён (персонаж = категория, варианты имени = записи) и из мест
географической карты (категория — место, вид объекта или страна).

Записи-имена: однословные — часть речи PROPN (при сопоставлении берутся имена собственные и существительные
с заглавной буквы: «Вера», но не «вера»), многословные — словосочетания. В настройках словаря count_names: слова
такого словаря не исключаются как «слово внутри имени» — они сами имена.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Dictionary
from app.services.dictionaries import create_dictionary, make_entry

NAME_MODES = {"person": "персонаж (варианты имени — в одной категории)", "etype": "класс (персоны, места, организации)",
              "none": "без категорий"}
PLACE_MODES = {"place": "место (варианты названия — в одной категории)", "kind": "вид объекта (город, река, страна…)",
               "country": "страна", "none": "без категорий"}


def _items(groups: list[tuple[str | None, list[str], str | None]]) -> list[dict]:
    """[(категория, [варианты], комментарий)] → записи словаря без повторов (первое вхождение важнее)."""
    out, seen = [], set()
    for cat, variants, comment in groups:
        for v in variants:
            v = " ".join(v.split())
            if not v or not any(ch.isalpha() for ch in v) or v.lower() in seen:
                continue
            seen.add(v.lower())
            out.append({"surface": v, "category": cat, "upos": "PROPN" if " " not in v else None, "comment": comment})
    return out


def name_groups(items: list[dict], mode: str = "person", etype_labels: dict | None = None) -> list[tuple]:
    """Элементы указателя имён (entity_index) → группы для словаря."""
    groups = []
    for it in items:
        variants = [it["name"], *it.get("all_variants", it.get("variants", []))]
        if mode == "person":
            cat = it["name"]
        elif mode == "etype":
            cat = (etype_labels or {}).get(it["etype"], it["etype"])
        else:
            cat = None
        groups.append((cat, variants, f"из указателя имён: {it['name']}" if len(variants) > 1 else None))
    return groups


def place_groups(points: list[dict], mode: str = "place") -> list[tuple]:
    """Места карты (place_map: points и unresolved) → группы. В режиме «место» названия одной точки (исторические
    синонимы: «Петроград» и «Санкт-Петербург») — одна категория."""
    from app.services.geo import KIND_LABELS, country_names

    names = country_names()
    by_point: dict[tuple, str] = {}
    groups = []
    for p in sorted(points, key=lambda x: -x["count"]):
        variants = [p["name"], *p.get("all_variants", p.get("variants", []))]
        if mode == "place":
            key = (round(p["lat"], 2), round(p["lon"], 2)) if p.get("lat") is not None else ("name", p["name"])
            cat = by_point.setdefault(key, p["name"])
        elif mode == "kind":
            cat = KIND_LABELS.get(p.get("kind") or "", "не найдено на карте" if p.get("lat") is None else "место")
        elif mode == "country":
            cat = names.get(p.get("cc") or "", "вне стран и без координат")
        else:
            cat = None
        groups.append((cat, variants, None))
    return groups


def build_dictionary(session: Session, name: str, groups: list[tuple], description: str) -> Dictionary:
    name = name.strip()
    if not name:
        raise ValueError("Впишите название словаря")
    if session.scalar(select(Dictionary).where(Dictionary.name == name)):
        raise ValueError(f"Словарь «{name}» уже есть — выберите другое название")
    items = _items(groups)
    if not items:
        raise ValueError("Нет ни одного имени для словаря")
    d = create_dictionary(session, name, [], description, {"count_names": True}, enqueue_match=False)
    for it in items:
        e = make_entry(it)
        if it["upos"] == "PROPN":  # разбор pymorphy может предложить NOUN — для имени оставляем PROPN
            e.upos, e.verified = "PROPN", True
        d.entries.append(e)
    session.flush()
    from app.services.jobs import enqueue

    enqueue(session, "match", {"dictionary_id": d.id}, title=f"Сопоставление словаря «{d.name}»")
    return d
