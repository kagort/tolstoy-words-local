"""Волна 5: указатель имён (поиск, шум, групповые правки), словари из имён и мест, однокоренные слова, категории."""

import io
import os

import pytest

TEXT = """Пьер вошёл в гостиную. Наташа улыбнулась Пьеру, и Пьер покраснел.
Впрочем, Наташа уже смотрела на Андрея. Андрей Болконский стоял у окна.
Впрочем, вечер был тёплым. Пьер уехал в Москву, а Наташа осталась в Петербурге.
Москва встретила Пьера холодом. Страх и страшная тоска не отпускали Пьера.
Милосердие и милосердный поступок спасли его. Он поступил милосердно и немилосердно сразу."""


# ------------------------------------------------------------------ модульные (без базы)


def test_match_mask():
    from app.services.entities import match_mask

    assert match_mask("Пь*", ["Пьер"]) and not match_mask("Пь*", ["Наташа"])
    assert match_mask("*ович", ["Иван Петрович"]) and match_mask("=пьер", ["Пьер"]) and not match_mask("=Пье", ["Пьер"])
    assert match_mask("ёж", ["Ежов"]) and match_mask("люба", ["Раневская", "Люба"])  # подстрока, ё = е, варианты


def test_noise_score():
    from app.services.entities import NOISE_MIN, noise_score

    # первое слово предложения, со строчной буквы встречается часто, наречие
    s, why = noise_score("Впрочем", {"n": 3, "init": 3, "propn": 0, "lower": 40}, 1)
    assert s >= NOISE_MIN and any("в начале предложения" in w for w in why)
    # настоящее имя: в середине предложения с заглавной
    assert noise_score("Наташа", {"n": 30, "init": 4, "propn": 30, "lower": 0}, 1)[0] < NOISE_MIN
    # «Вера» — и имя, и слово; в середине предложения с заглавной ≥ 2 раз — не шум
    assert noise_score("Вера", {"n": 12, "init": 2, "propn": 10, "lower": 30}, 1)[0] < NOISE_MIN
    # одной латиницы мало: «Pierre» во французской реплике
    assert noise_score("Pierre", {"n": 1, "init": 0, "propn": 1, "lower": 0}, 1)[0] < NOISE_MIN


def test_stems():
    from app.services.wordfamily import stems

    assert "милосерд" in stems("милосердие") and "милосерд" in stems("милосердный")
    assert {"страх", "страш"} <= stems("страх")
    assert stems("закон и порядок") == set()


# ------------------------------------------------------------------ с базой

needs_db = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")


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
    from app.models import Dictionary, TextDoc
    from app.services.jobs import run_pending

    r = client.post("/corpora", data={"name": "Толстой", "description": ""}, follow_redirects=False)
    cid = int(r.headers["location"].rsplit("/", 1)[1])
    files = [("files", ("wp.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain"))]
    client.post("/texts/upload", files=files, data={"author": "Лев Толстой", "corpus_ids": str(cid),
                                                    "f_names": ["wp.txt"], "f_titles": ["Война и мир"]})
    run_pending()
    client.post("/dictionaries", data={"name": "Чувства", "words": "страх, милосердие, тоска, вечер, окно, гостиная"})
    run_pending()
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Чувства").one()
        return {"dict_id": d.id, "corpus_id": cid, "text_id": s.query(TextDoc).one().id,
                "base": f"dictionary_id={d.id}&corpus_id={cid}"}


def _names(client, data, extra=""):
    r = client.get(f"/analysis/entities-index?tab=entities&{data['base']}{extra}")
    assert r.status_code == 200
    import json
    import re

    return json.loads(re.search(r'<script type="application/json" id="ent-all">(.*?)</script>', r.text, re.DOTALL).group(1))


@needs_db
def test_index_search_flags_and_fragment(client, data):
    names = _names(client, data)
    assert "Пьер" in names and "Наташа" in names
    assert _names(client, data, "&q=Пь*") == ["Пьер"]
    assert "Андрей Болконский" not in _names(client, data, "&f=one")
    page = client.get(f"/analysis?tab=entities&{data['base']}").text
    assert 'id="ent-index"' in page and "Найдено:" in page and "Словарь из отмеченных" in page
    noise = client.get(f"/analysis?tab=entities&{data['base']}&f=noise")
    assert noise.status_code == 200 and "Почему" in noise.text


@needs_db
def test_hide_restore_and_etype(client, data):
    from app.db import SessionLocal
    from app.services.entities import entity_index

    target = "Москва" if "Москва" in _names(client, data) else _names(client, data)[-1]
    r = client.post("/api/entities/hide", json={"names": [target], "hidden": True})
    assert r.json()["n"] == 1
    assert target not in _names(client, data) and target in _names(client, data, "&hidden=1")
    client.post("/api/entities/hide", json={"names": [target], "hidden": False})
    assert target in _names(client, data)
    assert client.post("/api/entities/hide", json={"names": []}).status_code == 400
    # перенос в класс группой («отметить все найденные» — names_all)
    back = f"/analysis?tab=entities&{data['base']}"
    client.post("/entities/etype", data={"names_all": "Пьер\nНаташа", "etype": "ORG", "back": back})
    with SessionLocal() as s:
        items = {x["name"]: x["etype"] for x in entity_index(s, [data["text_id"]], top=None)["items"]}
    assert items["Пьер"] == "ORG" and items["Наташа"] == "ORG"
    client.post("/entities/etype", data={"names": ["Пьер", "Наташа"], "etype": "", "back": back})
    with SessionLocal() as s:
        items = {x["name"]: x["etype"] for x in entity_index(s, [data["text_id"]], top=None)["items"]}
    assert items["Пьер"] == "PER"


@needs_db
def test_dictionary_from_names_and_places(client, data):
    from app.db import SessionLocal
    from app.models import Dictionary, Hit
    from app.services import analytics as A
    from app.services.jobs import run_pending

    back = f"/analysis?tab=entities&{data['base']}"
    r = client.post("/entities/to-dictionary", data={"names": ["Пьер", "Наташа"], "dict_name": "Герои", "mode": "person", "back": back},
                    follow_redirects=False)
    assert r.status_code == 303 and "/dictionaries/" in r.headers["location"]
    assert "создан из указателя имён" in client.get(r.headers["location"]).text
    run_pending()
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Герои").one()
        assert d.settings["count_names"] and {e.category for e in d.entries} == {"Пьер", "Наташа"}
        assert all(e.upos == "PROPN" for e in d.entries if " " not in e.surface)
        cf = {r["category"]: r["count"] for r in A.category_frequency(s, d.id, [data["text_id"]])["rows"]}
        assert cf["Пьер"] >= 5 and cf["Наташа"] >= 3  # «Пьеру», «Пьера» — формы имени
        assert s.query(Hit).filter_by(dictionary_id=d.id).count() >= 8
    # повтор названия — ошибка, словарь не создаётся
    r = client.post("/entities/to-dictionary", data={"names": ["Пьер"], "dict_name": "Герои", "back": back}, follow_redirects=False)
    assert r.headers["location"] == back and "уже есть" in client.get(back).text
    # места: категории по виду объекта
    r = client.post("/places/to-dictionary", data={"dict_name": "Города", "mode": "kind", "back": f"/analysis?tab=geo&{data['base']}"},
                    follow_redirects=False)
    assert r.status_code == 303
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Города").one()
        assert {"Москва", "Петербург"} <= {e.surface for e in d.entries} and "город" in {e.category for e in d.entries}


@needs_db
def test_cognates_and_categories(client, data):
    from app.db import SessionLocal
    from app.models import Dictionary

    page = client.get(f"/dictionaries/{data['dict_id']}/cognates").text
    assert "милосердный" in page and "страшный" in page
    r = client.post(f"/dictionaries/{data['dict_id']}/add-suggested",
                    data={"rows": ["милосердный\tДОБРО\tот: милосердие"]}, follow_redirects=False)
    assert r.status_code == 303
    with SessionLocal() as s:
        d = s.get(Dictionary, data["dict_id"])
        e = next(e for e in d.entries if e.surface == "милосердный")
        assert e.category == "ДОБРО" and e.comment == "от: милосердие"
    page = client.get(f"/dictionaries/{data['dict_id']}/categories?k=2").text
    assert "Применить" in page and 'name="name_1"' in page
    with SessionLocal() as s:
        d = s.get(Dictionary, data["dict_id"])
        empty = [e.id for e in d.entries if not e.category]
    form = {"name_1": "ПЕРВАЯ", "name_2": "ВТОРАЯ", **{f"entry_{i}": "1" for i in empty}}
    client.post(f"/dictionaries/{data['dict_id']}/categories", data=form)
    with SessionLocal() as s:
        d = s.get(Dictionary, data["dict_id"])
        cats = {e.surface: e.category for e in d.entries}
    assert cats["милосердный"] == "ДОБРО"  # назначенная раньше категория не меняется
    assert all(cats[e] == "ПЕРВАЯ" for e in cats if e != "милосердный")
    # создание словаря с «сразу предложить однокоренные» ведёт на страницу однокоренных
    r = client.post("/dictionaries", data={"name": "Ещё", "words": "страх", "suggest": "1"}, follow_redirects=False)
    assert r.headers["location"].endswith("/cognates")

