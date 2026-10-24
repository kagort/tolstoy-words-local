"""Сборка справочника мест app/data/gazetteer_ru.tsv.gz для географической карты.

Источник — данные GeoNames (CC BY 4.0, https://www.geonames.org) из пакета geonamescache (города больше 1000
жителей с альтернативными названиями) и русские названия стран из pycountry. Запускается разработчиком один раз;
в образ попадает готовый файл, интернет при работе не нужен.

    pip install geonamescache==3.0.2 pycountry==26.2.16
    python scripts/build_gazetteer.py

Строка файла: название (строчными, ё → е) \t широта \t долгота \t население \t код страны \t вид (city | country).
Для каждого названия хранится не больше MAX_PER_NAME самых крупных мест — из них при совпадении выбирается одно
(см. app/services/geo.py). Реки, горы, моря, регионы и исторические названия — в app/data/places_extra_ru.tsv (вручную).
"""

from __future__ import annotations

import gettext
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "app" / "data" / "gazetteer_ru.tsv.gz"
COUNTRIES = OUT.parent / "countries_ru.tsv"
MAX_PER_NAME = 5
RU_NAME = re.compile(r"^[а-яё][а-яё\- ]*[а-яё]$")  # только русский алфавит: украинские, казахские и др. варианты отбрасываются


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower().replace("ё", "е"))


def main() -> None:
    import geonamescache
    import pycountry

    data_dir = Path(geonamescache.__file__).parent / "data"
    cities = json.loads((data_dir / "cities1000.json").read_text(encoding="utf-8"))
    by_name: dict[str, list[tuple]] = defaultdict(list)
    country_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c in cities.values():
        lat, lon, pop, cc = round(c["latitude"], 4), round(c["longitude"], 4), int(c["population"] or 0), c["countrycode"]
        country_points[cc].append((lat, lon))
        names = {norm(a) for a in c.get("alternatenames", []) if RU_NAME.match(norm(a))}
        for n in names:
            by_name[n].append((lat, lon, pop, cc, "city"))

    # страны: русские названия ISO 3166 + сокращённые формы («Республика Корея» → «Корея, Республика» не нужна)
    tr = gettext.translation("iso3166-1", localedir=str(Path(pycountry.__file__).parent / "locales"), languages=["ru"])
    for country in pycountry.countries:
        pts = country_points.get(country.alpha_2)
        if not pts:
            continue
        lat = round(statistics.median(p[0] for p in pts), 3)
        lon = round(statistics.median(p[1] for p in pts), 3)
        ru = tr.gettext(country.name)
        for n in {norm(ru), norm(re.sub(r"\s*\(.*?\)", "", ru))}:
            if RU_NAME.match(n):
                by_name[n].insert(0, (lat, lon, 10**9, country.alpha_2, "country"))

    rows = []
    for n, cands in by_name.items():
        uniq = {}
        for cand in sorted(cands, key=lambda x: -x[2]):
            uniq.setdefault((cand[3], round(cand[0], 1), round(cand[1], 1)), cand)
        for lat, lon, pop, cc, kind in list(uniq.values())[:MAX_PER_NAME]:
            rows.append(f"{n}\t{lat}\t{lon}\t{pop}\t{cc}\t{kind}")
    rows.sort()
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("# GeoNames (CC BY 4.0, geonames.org) via geonamescache; страны — pycountry. Сборка: scripts/build_gazetteer.py\n")
        f.write("\n".join(rows) + "\n")
    print(f"{OUT}: {len(rows)} строк, {len(by_name)} названий, {OUT.stat().st_size // 1024} КБ")
    # коды стран: двухбуквенный → трёхбуквенный (раскраска стран на карте Plotly) и короткое русское название
    with open(COUNTRIES, "w", encoding="utf-8") as f:
        f.write("# код\tкод ISO-3\tназвание (pycountry, ru)\n")
        for country in sorted(pycountry.countries, key=lambda c: c.alpha_2):
            ru = re.sub(r"\s*\(.*?\)", "", tr.gettext(getattr(country, "common_name", None) or country.name))
            f.write(f"{country.alpha_2}\t{country.alpha_3}\t{ru}\n")
    print(f"{COUNTRIES}: {len(pycountry.countries)} стран")


if __name__ == "__main__":
    main()
