"""Волна 4¾: слова внутри имён, персонажи, время задач, категории/слова, география (вид, страны), словарь для корпусов, дашборд."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

CHERRY = """Любовь Андреевна вернулась из Парижа. Варя встретила её на станции.
Любовь Андреевна плакала от радости. Любовь сильнее смерти, говорила Варя.
Раневская смотрела на сад. Её любовь к саду не проходила. Лопахин купил имение."""


@pytest.fixture(scope="module")
def client():
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from app.db import engine
    from app.main import app

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")
    c = TestClient(app)
    c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"})
    return c


@pytest.fixture(scope="module")
def data(client):
    from app.db import SessionLocal
    from app.models import Corpus, Dictionary, TextDoc
    from app.services.jobs import run_pending

    r = client.post("/corpora", data={"name": "Чехов", "description": ""}, follow_redirects=False)
    cid = int(r.headers["location"].rsplit("/", 1)[1])
    client.post("/corpora", data={"name": "Прочее", "description": ""})
    files = [("files", ("sad.txt", io.BytesIO(CHERRY.encode("utf-8")), "text/plain"))]
    client.post("/texts/upload", files=files, data={"author": "Антон Павлович Чехов", "corpus_ids": str(cid),
                                                    "f_names": ["sad.txt"], "f_titles": ["Вишнёвый сад"]})
    run_pending()
    csv_text = "слово;категория\nлюбовь;ЧУВСТВА\nрадость;ЧУВСТВА\nсад;МЕСТО\nсмерть;ЧУВСТВА\n"
    client.post("/dictionaries", data={"name": "Чувства", "words": ""},
                files={"file": ("f.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")})
    run_pending()
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Чувства").one()
        return {"dict_id": d.id, "corpus_id": cid, "text_id": s.query(TextDoc).one().id,
                "entries": {e.surface: e.id for e in d.entries},
                "other_corpus": s.query(Corpus).filter_by(name="Прочее").one().id}


def _freq(dict_id, text_ids):
    from app.db import SessionLocal
    from app.services import analytics as A

    with SessionLocal() as s:
        return {r["word"]: r["count"] for r in A.frequency(s, dict_id, text_ids)["rows"]}


def test_words_inside_names_not_counted(client, data):
    from app.db import SessionLocal
    from app.models import Hit

    e = data["entries"]
    with SessionLocal() as s:
        in_name = s.query(Hit).filter_by(entry_id=e["любовь"], in_name=True).count()
        total = s.query(Hit).filter_by(entry_id=e["любовь"]).count()
    assert total == 4 and in_name == 2  # «Любовь Андреевна» ×2 — имя; «Любовь сильнее», «Любовь к саду» — слово
    assert _freq(data["dict_id"], [data["text_id"]])["любовь"] == 2
    page = client.get(f"/dictionaries/{data['dict_id']}").text
    assert "2 в именах" in page
    base = f"dictionary_id={data['dict_id']}&corpus_id={data['corpus_id']}"
    r = client.get(f"/analysis?tab=conc&{base}&entry_id={e['любовь']}&names=1")
    assert "Найдено: 2" in r.text and "в имени" in r.text
    # слово-исключение: считать и внутри имён
    client.post(f"/dictionaries/{data['dict_id']}/entries/{e['любовь']}/names")
    assert _freq(data["dict_id"], [data["text_id"]])["любовь"] == 4
    client.post(f"/dictionaries/{data['dict_id']}/entries/{e['любовь']}/names")
    assert _freq(data["dict_id"], [data["text_id"]])["любовь"] == 2
    # настройка словаря «считать внутри имён» — без пересчёта
    form = {"name": "Чувства", "description": "", "merge_propn": "1", "count_names": "1"}
    client.post(f"/dictionaries/{data['dict_id']}/settings", data=form)
    assert _freq(data["dict_id"], [data["text_id"]])["любовь"] == 4
    form.pop("count_names")
    client.post(f"/dictionaries/{data['dict_id']}/settings", data=form)
    assert _freq(data["dict_id"], [data["text_id"]])["любовь"] == 2
    # ручная отметка ✓ возвращает конкретное вхождение
    with SessionLocal() as s:
        hid = s.query(Hit).filter_by(entry_id=e["любовь"], in_name=True).first().id
    client.post(f"/api/hits/{hid}/valid", json={"valid": True})
    assert _freq(data["dict_id"], [data["text_id"]])["любовь"] == 3
    client.post(f"/api/hits/{hid}/valid", json={"valid": None})
    # «это не имя» на вкладке «Имена» возвращает вхождения в частоты
    client.post("/entities/override", data={"norm": "Любовь Андреевна", "hidden": "1", "back": "/texts"})
    assert _freq(data["dict_id"], [data["text_id"]])["любовь"] == 4
    client.post("/entities/override", data={"norm": "Любовь Андреевна", "back": "/texts"})
    assert _freq(data["dict_id"], [data["text_id"]])["любовь"] == 2


def test_job_eta():
    import time

    from app.models import Job
    from app.services.jobs import _next_eta, eta_seconds, human_duration

    t = time.time()
    e = _next_eta(None, 0.0, t - 100)
    e = _next_eta(e, 0.10, t - 50)   # 0,1 за 50 с
    e = _next_eta(e, 0.20, t)        # ещё 0,1 за 50 с → 0,002 в секунду
    assert abs(e["rate"] - 0.002) < 1e-6
    j = Job(status="running", progress=0.2, eta=e)
    assert 390 < eta_seconds(j) <= 400 and human_duration(eta_seconds(j)) == "7 мин"
    assert eta_seconds(Job(status="running", progress=0.02, eta=e)) is None  # рано
    assert human_duration(30) == "меньше минуты" and human_duration(3 * 3600 + 600) == "3 ч 10 мин"


def test_jobs_page_eta(client, data):
    r = client.get("/api/jobs")
    assert r.status_code == 200 and "queue" in r.json()
    assert client.get("/jobs").status_code == 200


def test_unit_categories_and_rest_groups(client, data):
    from app.db import SessionLocal
    from app.services import analytics as A
    from app.services.selection import from_params
    from app.services.textmap import text_map

    tid = data["text_id"]
    with SessionLocal() as s:
        ov = A.overview(s, data["dict_id"], [tid], [], "text", by="cat")
        cats = {w["word"]: w["count"][0] for w in ov["words"]}
        assert cats["ЧУВСТВА"] == 4 and cats["МЕСТО"] >= 1 and ov["by"] == "cat"  # любовь 2 + радость 1 + смерть 1
        only = A.overview(s, data["dict_id"], [tid], [data["entries"]["сад"]], "text", by="cat")
        assert [w["word"] for w in only["words"]] == ["МЕСТО"]
        heat = A.frequency_by_text(s, data["dict_id"], [tid], by="cat")
        assert heat["by"] == "cat" and "ЧУВСТВА" in heat["words"]
        tm = text_map(s, tid, data["dict_id"], None, 10, by="cat")
        assert {r["label"] for r in tm["rows"]} >= {"ЧУВСТВА", "МЕСТО"}
        sel = from_params(s, {"corpus_id": [str(data["corpus_id"])], "unit": ["cat"]})
        assert sel.unit == "cat" and "unit=cat" in sel.query() and "по категориям" in sel.describe()
    # сворачивание групп: из двух текстов показывается один (с бо́льшим числом вхождений) + «остальные (1)»
    from app.services.jobs import run_pending

    files = [("files", ("other.txt", io.BytesIO("Радость и любовь. Сад цвёл.".encode("utf-8")), "text/plain"))]
    client.post("/texts/upload", files=files, data={"corpus_ids": str(data["other_corpus"]), "f_names": ["other.txt"],
                                                    "f_titles": ["Другое"]})
    run_pending()
    with SessionLocal() as s:
        texts = A.corpus_text_ids(s, None)
        assert len(texts) == 2
        ov = A.overview(s, data["dict_id"], texts, [data["entries"]["любовь"]], "text", groups_top=1)
        assert [g["label"] for g in ov["groups"]][-1] == "остальные (1)" and ov["hidden_groups"] == 1
        assert ov["groups"][0]["key"] == tid and ov["words"][0]["count"] == [2, 1]
    base = f"dictionary_id={data['dict_id']}&corpus_id={data['corpus_id']}&unit=cat"
    for tab in ("overview", "freq", "map", "report"):
        r = client.get(f"/analysis?tab={tab}&{base}")
        assert r.status_code == 200, tab
    r = client.get(f"/analysis?tab=overview&dictionary_id={data['dict_id']}&dim=text&gtop=1&words={data['entries']['любовь']}")
    assert "Показать все группы" in r.text
    assert client.get(f"/export/overview.xlsx?{base}").status_code == 200


def test_merge_characters(client, data):
    from app.db import SessionLocal
    from app.services.entities import entity_index, merge_suggestions

    tid = data["text_id"]
    with SessionLocal() as s:
        names = {it["name"] for it in entity_index(s, [tid], "PER")["items"]}
    assert {"Любовь Андреевна", "Раневская"} <= names
    base = f"/analysis?tab=entities&dictionary_id={data['dict_id']}&corpus_id={data['corpus_id']}"
    r = client.post("/entities/merge", data={"names": ["Любовь Андреевна", "Раневская"], "canonical": "Раневская Любовь Андреевна",
                                             "scope": str(tid), "back": base})
    assert "Объединено в «Раневская Любовь Андреевна»" in r.text
    with SessionLocal() as s:
        items = {it["name"]: it for it in entity_index(s, [tid], "PER")["items"]}
    assert "Раневская Любовь Андреевна" in items and "Раневская" not in items
    assert items["Раневская Любовь Андреевна"]["count"] == 3
    assert set(items["Раневская Любовь Андреевна"]["variants"]) == {"Любовь Андреевна", "Раневская"}
    # слияние только в этом тексте: в другом тексте «Раневская» осталась бы собой (проверяем через resolve)
    from app.services.entities import overrides, resolve

    with SessionLocal() as s:
        ovr = overrides(s)
    assert resolve("Раневская", "PER", ovr, tid)[0] == "Раневская Любовь Андреевна"
    assert resolve("Раневская", "PER", ovr, tid + 999)[0] == "Раневская"
    # отделить вариант
    client.post("/entities/split", data={"variant": "Раневская", "text_ids": [str(tid)], "back": base})
    with SessionLocal() as s:
        items = {it["name"]: it for it in entity_index(s, [tid], "PER")["items"]}
    assert "Раневская" in items and items["Раневская Любовь Андреевна"]["count"] == 2
    # подсказки: общее слово
    hints = merge_suggestions([{"name": "Андрей", "etype": "PER", "count": 5}, {"name": "князь Андрей", "etype": "PER", "count": 3},
                               {"name": "Пьер", "etype": "PER", "count": 9}])
    assert hints == [["Андрей", "князь Андрей"]]
    assert "Объединить</button>" in client.get(base).text


def test_geo_kind_and_countries(client, data):
    from app.db import SessionLocal
    from app.services import geo as G

    assert G.nearest_country(55.7, 37.6) == "RU" and G.nearest_country(48.85, 2.35) == "FR"
    assert G.country_names()["RU"] == "Россия" and G.countries()["RU"][0] == "RUS"
    assert G.candidates("Монблан")[0].kind == "mountain"
    tid = data["text_id"]
    base = f"/analysis?tab=geo&dictionary_id={data['dict_id']}&corpus_id={data['corpus_id']}"
    with SessionLocal() as s:
        m = G.place_map(s, [tid])
        paris = {p["name"]: p for p in m["points"]}["Париж"]
        assert paris["kind"] == "city" and paris["cc"] == "FR"
        fr = next(c for c in m["countries"] if c["cc"] == "FR")
        assert fr["total"] == 1 and fr["iso3"] == "FRA" and fr["places"][0]["name"] == "Париж"
    # вид «гора»: у Парижа нет варианта-горы → без координат; «вымышленное» → не на карте; сброс
    client.post("/places", data={"name": "Париж", "action": "kind", "kind": "mountain", "back": base})
    with SessionLocal() as s:
        m = G.place_map(s, [tid])
        p = {x["name"]: x for x in m["unresolved"]}["Париж"]
        assert "как «горы»" in p["label"] and p["kind_manual"] == "mountain"
    client.post("/places", data={"name": "Париж", "action": "kind", "kind": "fictional", "back": base})
    with SessionLocal() as s:
        assert {x["name"]: x for x in G.place_map(s, [tid])["unresolved"]}["Париж"]["label"] == "вымышленное место"
    client.post("/places", data={"name": "Париж", "action": "reset", "back": base})
    r = client.get(base + "&geo_mode=countries")
    assert r.status_code == 200 and "choropleth" in r.text and "Франция" in r.text
    assert client.get(f"/export/geo.xlsx?dictionary_id={data['dict_id']}&corpus_id={data['corpus_id']}").status_code == 200


def test_dictionary_scope(client, data):
    from app.db import SessionLocal
    from app.models import Hit, TextDoc
    from app.services.jobs import run_pending

    dict_id, cid, other = data["dict_id"], data["corpus_id"], data["other_corpus"]
    with SessionLocal() as s:
        other_tid = s.query(TextDoc).filter_by(title="Другое").one().id
        assert s.query(Hit).filter_by(dictionary_id=dict_id, text_id=other_tid).count() > 0
    form = {"name": "Чувства", "description": "", "merge_propn": "1", "scope_corpus_ids": [str(cid)]}
    client.post(f"/dictionaries/{dict_id}/settings", data=form)
    run_pending()
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(dictionary_id=dict_id, text_id=other_tid).count() == 0  # вне области
        assert s.query(Hit).filter_by(dictionary_id=dict_id, text_id=data["text_id"]).count() > 0
    page = client.get(f"/analysis?tab=freq&dictionary_id={dict_id}&corpus_id={other}").text
    assert "не сопоставлены" in page and "Сопоставить с этим корпусом" in page
    client.post(f"/dictionaries/{dict_id}/scope/add", data={"corpus_id": str(other), "back": "/texts"})
    run_pending()
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(dictionary_id=dict_id, text_id=other_tid).count() > 0
    # текст убрали из корпуса области — вхождения словаря для него удаляются
    client.post(f"/corpora/{other}", data={"name": "Прочее", "description": "", "text_ids": []})
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(dictionary_id=dict_id, text_id=other_tid).count() == 0
    # вернуть область «все тексты»
    form.pop("scope_corpus_ids")
    client.post(f"/dictionaries/{dict_id}/settings", data=form)
    run_pending()
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(dictionary_id=dict_id, text_id=other_tid).count() > 0


def test_dashboard(client, data):
    client.cookies.set("qv_ui", "classic")  # прежний дашборд; дашборд v2 — tests/test_dashboard_v2.py
    r = client.get(f"/dashboard?corpus_id={data['corpus_id']}&dictionary_id={data['dict_id']}")
    assert r.status_code == 200 and "Частые слова словаря" in r.text and "Персонажи" in r.text
    r = client.get(f"/dashboard?texts={data['text_id']}")
    assert r.status_code == 200 and "Вишнёвый сад" in r.text and "Где в тексте" in r.text
    r = client.get(f"/export/dashboard.html?corpus_id={data['corpus_id']}&dictionary_id={data['dict_id']}")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    html = r.text
    assert "PlotlyGeoAssets" in html and "plotly.js v" in html and len(html) > 3_000_000  # всё внутри
    assert "/static/" not in html.split("<body>", 1)[1]  # никаких внешних файлов приложения
    small = client.get(f"/export/dashboard.html?corpus_id={data['corpus_id']}&cdn=1").text
    assert len(small) < 600_000 and "cdn.jsdelivr.net/npm/plotly.js-dist-min@" in small
    client.cookies.set("qv_ui", "v2")
    r = client.get(f"/dashboard?texts={data['text_id']}")
    assert "Корпус × словарь" in r.text and "Карта текста" in r.text and "/static/dash_v2.js" in r.text
