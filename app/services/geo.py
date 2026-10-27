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
               "lake": "озеро", "mountain": "горы", "feature": "объект", "fictional": "вымышленное"}
# виды, которые пользователь выбирает в правке места (озеро и море — один вид)
EDIT_KINDS = {"city": "город", "country": "страна", "region": "регион", "river": "река", "mountain": "гора, горы",
              "sea": "море, озеро", "fictional": "вымышленное (не на карте)"}
SAME_KIND = {"lake": "sea", "feature": "region"}


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


# привычные названия вместо официальных («Российская Федерация» → «Россия»)
COUNTRY_SHORT = {"RU": "Россия", "US": "США", "GB": "Великобритания", "KP": "КНДР", "KR": "Южная Корея",
                 "CZ": "Чехия", "MD": "Молдавия", "BY": "Белоруссия", "VA": "Ватикан"}


@lru_cache(maxsize=1)
def countries() -> dict[str, tuple[str, str]]:
    """Код страны → (код ISO-3, русское название) из app/data/countries_ru.tsv."""
    out = {}
    path = DATA / "countries_ru.tsv"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            cc, iso3, name = line.split("\t")
            out[cc] = (iso3, COUNTRY_SHORT.get(cc, name))
    return out


def country_names() -> dict[str, str]:
    """Код страны → короткое русское название (для подписей: «Париж — Франция»)."""
    return {cc: name for cc, (_, name) in countries().items()}


@lru_cache(maxsize=1)
def _city_grid() -> dict[tuple[int, int], list[tuple[float, float, str]]]:
    """Города от 15 000 жителей по клеткам 1°×1° — для определения страны по координатам."""
    grid: dict[tuple[int, int], list] = {}
    seen = set()
    for cands in gazetteer().values():
        for c in cands:
            if c.kind == "city" and c.pop >= 15000 and (c.lat, c.lon) not in seen:
                seen.add((c.lat, c.lon))
                grid.setdefault((int(c.lat // 1), int(c.lon // 1)), []).append((c.lat, c.lon, c.cc))
    return grid


def nearest_country(lat: float, lon: float, max_deg: int = 3) -> str | None:
    """Страна ближайшего крупного города (не дальше ~300 км) — для мест с координатами, поставленными вручную."""
    grid = _city_grid()
    ci, cj = int(lat // 1), int(lon // 1)
    best, best_d = None, None
    for r in range(max_deg + 1):
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                if max(abs(i - ci), abs(j - cj)) != r:
                    continue
                for la, lo, cc in grid.get((i, j), ()):
                    d = (la - lat) ** 2 + ((lo - lon) * 0.6) ** 2
                    if best_d is None or d < best_d:
                        best, best_d = cc, d
        if best is not None:
            return best
    return None


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
            out[n] = Cand(float(parts[0]), float(parts[1]), 0, (parts[4] if len(parts) > 4 else "").strip().upper(),
                          parts[2] if len(parts) > 2 else "region", parts[3] if len(parts) > 3 else "")
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
    cands = {}
    for n in names:
        m = manual.get(key(n))
        opts = candidates(n)
        if m is not None and m.kind and m.kind != "fictional":  # пользователь указал вид: «Монблан» — гора, а не город
            opts = [c for c in opts if SAME_KIND.get(c.kind, c.kind) == m.kind]
        cands[n] = opts
    # страны однозначно найденных мест — подсказка для неоднозначных («Париж» рядом с «Лионом» — во Франции)
    context = Counter(c[0].cc for c in cands.values() if len(c) == 1 and c[0].cc)
    out = {}
    for n in names:
        opts = cands[n]
        m = manual.get(key(n))
        options = [{"index": i, "label": describe(c, n), "lat": c.lat, "lon": c.lon} for i, c in enumerate(opts[:6])]
        kind = m.kind if m is not None else None
        if m is not None and (m.source == "none" or kind == "fictional"):
            out[n] = {"source": "none", "lat": None, "lon": None, "options": options, "kind": kind,
                      "label": "вымышленное место" if kind == "fictional" else "не на карте"}
            continue
        if m is not None and m.lat is not None:
            cc = m.country or (nearest_country(m.lat, m.lon) if (kind or "city") == "city" else None)
            out[n] = {"source": "manual", "lat": m.lat, "lon": m.lon, "label": m.label or "поставлено вручную", "options": options,
                      "kind": kind or "city", "cc": cc}
            continue
        if not opts:
            label = f"нет в справочнике как «{KIND_LABELS.get(kind, kind)}» — введите координаты" if kind else "нет в справочнике"
            out[n] = {"source": "missing", "lat": None, "lon": None, "label": label, "options": [], "kind": kind}
            continue
        best = opts[0]
        if best.kind == "city" and len(opts) > 1 and context:
            best = max(opts, key=lambda c: (context.get(c.cc, 0) > 0, c.kind == "country", c.pop))
        # совпадение с маленьким населённым пунктом часто случайно: выдуманное имение («Лысые Горы»), полк, фамилия
        doubtful = best.kind == "city" and best.pop < SMALL_TOWN
        out[n] = {"source": "auto", "lat": best.lat, "lon": best.lon, "label": describe(best, n), "options": options,
                  "ambiguous": len(opts) > 1, "doubtful": doubtful, "kind": kind or best.kind, "cc": best.cc or None}
    for n in names:  # вид, заданный вручную (для формы правки)
        m = manual.get(key(n))
        out[n]["kind_manual"] = m.kind if m is not None else None
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
    countries_rows = country_stats(points)
    return {"points": points, "unresolved": unresolved, "route": [{"name": p["name"], "lat": p["lat"], "lon": p["lon"]} for p in route],
            "total": idx["total"], "n_mentions": sum(p["count"] for p in points), "single_text": len(text_ids) == 1,
            "to_check": sum(1 for p in points if p.get("ambiguous") or p.get("doubtful")),
            "countries": countries_rows, "outside": next((c for c in countries_rows if c["cc"] is None), None),
            "missing": idx["missing"]}


def country_stats(points: list[dict]) -> list[dict]:
    """Страны выборки: упоминания самой страны + всех её мест (города, регионы с известной страной).

    Реки, моря, горы и регионы без страны — строкой «вне стран» (cc = None). Границы современные.
    """
    agg: dict[str | None, dict] = {}
    names = country_names()
    iso = countries()
    for p in points:
        cc = p.get("cc") if p.get("kind") in ("city", "country", "region", None) else None
        a = agg.setdefault(cc, {"cc": cc, "iso3": iso.get(cc, ("", ""))[0] if cc else "",
                                "name": names.get(cc, cc) if cc else "вне стран (реки, моря, горы, регионы)",
                                "total": 0, "self": 0, "places": []})
        a["total"] += p["count"]
        if p.get("kind") == "country":
            a["self"] += p["count"]
        else:
            a["places"].append({"name": p["name"], "count": p["count"]})
    rows = sorted(agg.values(), key=lambda a: (a["cc"] is None, -a["total"]))
    for a in rows:
        a["places"].sort(key=lambda x: -x["count"])
    return rows


def set_place(session: Session, name: str, action: str, coords: str = "", option: int | None = None, kind: str = "") -> str:
    """Ручная правка: action = coords («55.75, 37.62»), option (вариант из справочника), kind (вид объекта),
    none (не на карте), reset."""
    k = key(name)
    p = session.get(Place, k)
    if action == "reset":
        if p is not None:
            session.delete(p)
        return "Место будет найдено в справочнике автоматически"
    if p is None:
        p = Place(norm=k, source="auto")
        session.add(p)
    if action == "kind":
        if kind and kind not in EDIT_KINDS:
            raise ValueError("Неизвестный вид объекта")
        p.kind = kind or None
        if p.source == "none" and kind != "fictional":
            p.source = "auto"
        return f"«{name}»: вид — {EDIT_KINDS.get(kind, 'как в справочнике')}"
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
