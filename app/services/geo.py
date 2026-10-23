"""Географическая карта: места из вкладки «Имена» (тип LOC) → координаты.

Откуда координаты (по порядку):
1. ручная правка пользователя (таблица places: source = manual — точка, none — «не на карте»);
2. дополнения app/data/places_extra_ru.tsv: реки, горы, моря, регионы, исторические и привычные названия;
3. справочник app/data/gazetteer_ru.tsv.gz (GeoNames, города больше 1000 жителей и страны с русскими названиями).

Если в справочнике несколько мест с таким названием (Париж во Франции и в Техасе), выбирается место в стране, которая
чаще встречается среди однозначно найденных мест этой выборки, иначе — самое крупное. Другие варианты можно выбрать
вручную. Интернет не нужен.
"""

from __future__ import annotations

import gzip
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Place

DATA = Path(__file__).resolve().parents[1] / "data"
SMALL_TOWN = 20_000  # меньше — помечаем «проверить»
KIND_LABELS = {"city": "город", "country": "страна", "region": "регион", "river": "река", "sea": "море, озеро",
               "lake": "озеро", "mountain": "горы", "feature": "объект"}


@dataclass
class Cand:
    lat: float
    lon: float
    pop: int
    cc: str
    kind: str
    note: str = ""


def key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower().replace("ё", "е"))


@lru_cache(maxsize=1)
def gazetteer() -> dict[str, list[Cand]]:
    out: dict[str, list[Cand]] = {}
    path = DATA / "gazetteer_ru.tsv.gz"
    if not path.exists():
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            n, lat, lon, pop, cc, kind = line.rstrip("\n").split("\t")
            out.setdefault(n, []).append(Cand(float(lat), float(lon), int(pop), cc, kind))
    for cands in out.values():
        cands.sort(key=lambda c: -c.pop)
    return out


@lru_cache(maxsize=1)
def country_names() -> dict[str, str]:
    """Код страны → короткое русское название (для подписей: «Париж — Франция»)."""
    names: dict[str, str] = {}
    for n, cands in gazetteer().items():
        for c in cands:
            if c.kind == "country" and (c.cc not in names or len(n) < len(names[c.cc])):
                names[c.cc] = n
    return {cc: n[:1].upper() + n[1:] for cc, n in names.items()}


@lru_cache(maxsize=1)
def extras() -> dict[str, Cand]:
    path = DATA / "places_extra_ru.tsv"
    raw: dict[str, list[str]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            raw[key(parts[0])] = parts[1:]
    out: dict[str, Cand] = {}
    for n, parts in raw.items():
        if parts and parts[0].startswith("="):
            target = key(parts[0][1:])
            if target in raw and not raw[target][0].startswith("="):
                t = raw[target]
                out[n] = Cand(float(t[0]), float(t[1]), 0, "", t[2] if len(t) > 2 else "region", f"= {parts[0][1:]}")
            elif gazetteer().get(target):
                c = gazetteer()[target][0]
                out[n] = Cand(c.lat, c.lon, c.pop, c.cc, c.kind, f"= {parts[0][1:]}")
        elif len(parts) >= 2:
            out[n] = Cand(float(parts[0]), float(parts[1]), 0, "", parts[2] if len(parts) > 2 else "region",
                          parts[3] if len(parts) > 3 else "")
    return out


def candidates(name: str) -> list[Cand]:
    k = key(name)
    out = []
    if k in extras():
        out.append(extras()[k])
    out += gazetteer().get(k, [])
    return out


def describe(c: Cand, name: str = "") -> str:
    parts = [KIND_LABELS.get(c.kind, c.kind)]
    if c.cc and c.kind != "country":
        parts.append(country_names().get(c.cc, c.cc))
    if c.pop and c.kind == "city":
        parts.append(f"{c.pop:,} жит.".replace(",", " "))
    if c.note:
        parts.append(c.note)
    return ", ".join(parts)


def resolve_all(session: Session, names: list[str]) -> dict[str, dict]:
    """{имя: {lat, lon, source, label, options}} — source: manual | auto | none (не на карте) | missing (не найдено)."""
    manual = {p.norm: p for p in session.scalars(select(Place).where(Place.norm.in_([key(n) for n in names] or ["-"])))}
    cands = {n: candidates(n) for n in names}
    # страны однозначно найденных мест — подсказка для неоднозначных («Париж» рядом с «Лионом» — во Франции)
    context = Counter(c[0].cc for c in cands.values() if len(c) == 1 and c[0].cc)
    out = {}
    for n in names:
        opts = cands[n]
        m = manual.get(key(n))
        options = [{"index": i, "label": describe(c, n), "lat": c.lat, "lon": c.lon} for i, c in enumerate(opts[:6])]
        if m is not None and m.source == "none":
            out[n] = {"source": "none", "lat": None, "lon": None, "label": "не на карте", "options": options}
            continue
        if m is not None and m.lat is not None:
            out[n] = {"source": "manual", "lat": m.lat, "lon": m.lon, "label": m.label or "поставлено вручную", "options": options}
            continue
        if not opts:
            out[n] = {"source": "missing", "lat": None, "lon": None, "label": "нет в справочнике", "options": []}
            continue
        best = opts[0]
        if best.kind == "city" and len(opts) > 1 and context:
            best = max(opts, key=lambda c: (context.get(c.cc, 0) > 0, c.kind == "country", c.pop))
        # совпадение с маленьким населённым пунктом часто случайно: выдуманное имение («Лысые Горы»), полк, фамилия
        doubtful = best.kind == "city" and best.pop < SMALL_TOWN
        out[n] = {"source": "auto", "lat": best.lat, "lon": best.lon, "label": describe(best, n), "options": options,
                  "ambiguous": len(opts) > 1, "doubtful": doubtful}
    return out


def place_map(session: Session, text_ids: list[int], top: int = 300) -> dict:
    """Точки карты для выборки: места (LOC) с координатами, частотой и порядком первого упоминания."""
    from app.services.entities import entity_index

    idx = entity_index(session, text_ids, etype="LOC", top=top)
    items = idx["items"]
    res = resolve_all(session, [it["name"] for it in items])
    points, unresolved = [], []
    for it in items:
        r = res[it["name"]]
        row = {**it, **r}
        (points if r["lat"] is not None else unresolved).append(row)
    route = sorted(points, key=lambda p: p["first"]) if len(text_ids) == 1 else []
    return {"points": points, "unresolved": unresolved, "route": [{"name": p["name"], "lat": p["lat"], "lon": p["lon"]} for p in route],
            "total": idx["total"], "n_mentions": sum(p["count"] for p in points), "single_text": len(text_ids) == 1,
            "to_check": sum(1 for p in points if p.get("ambiguous") or p.get("doubtful")),
            "missing": idx["missing"]}


def set_place(session: Session, name: str, action: str, coords: str = "", option: int | None = None) -> str:
    """Ручная правка: action = coords («55.75, 37.62»), option (вариант из справочника), none (не на карте), reset."""
    k = key(name)
    p = session.get(Place, k)
    if action == "reset":
        if p is not None:
            session.delete(p)
        return "Место будет найдено в справочнике автоматически"
    if p is None:
        p = Place(norm=k)
        session.add(p)
    if action == "none":
        p.source, p.lat, p.lon, p.label = "none", None, None, "не на карте"
        return f"«{name}» не показывается на карте"
    if action == "option" and option is not None:
        opts = candidates(name)
        if not 0 <= option < len(opts):
            raise ValueError("Нет такого варианта")
        c = opts[option]
        p.source, p.lat, p.lon, p.label, p.country = "manual", c.lat, c.lon, describe(c, name), c.cc or None
        return f"«{name}»: {p.label}"
    m = re.match(r"^\s*(-?\d+(?:[.,]\d+)?)\s*[,; ]\s*(-?\d+(?:[.,]\d+)?)\s*$", coords or "")
    if not m:
        raise ValueError("Координаты — два числа: широта и долгота, например «55.75, 37.62»")
    lat, lon = float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Широта — от −90 до 90, долгота — от −180 до 180")
    p.source, p.lat, p.lon, p.label = "manual", lat, lon, "поставлено вручную"
    return f"«{name}»: {lat}, {lon}"
