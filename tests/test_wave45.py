"""Волна 4½: словосочетания (поиск и записи словаря), частоты по категориям, имена собственные, справка, карта."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = """Мы вернём закон и порядок в наши города. Это революция здравого смысла.
Порядок, основанный на праве, защищает права человека. Здравый смысл подсказывал другое.
Закон суров, но это закон. В Москве и Петербурге говорили о порядке. Андрей уехал в Москву."""


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
def corpus(client):
    from app.db import SessionLocal
    from app.models import Corpus, Dictionary, TextDoc
    from app.services.jobs import run_pending

    r = client.post("/corpora", data={"name": "Речи", "description": ""}, follow_redirects=False)
    corpus_id = int(r.headers["location"].rsplit("/", 1)[1])
    files = [("files", ("speech.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain")),
             ("files", ("other.txt", io.BytesIO("Закон и порядок. Москва спала.".encode("utf-8")), "text/plain"))]
    client.post("/texts/upload", files=files, data={"author": "Дональд Трамп", "corpus_ids": str(corpus_id),
                                                    "f_names": ["speech.txt", "other.txt"], "f_titles": ["Речь", "Другое"],
                                                    "f_authors": ["", "Энтони Блинкен"]})
    run_pending()
    csv_text = ("слово;категория\nзакон;ЗАКОН И ПОРЯДОК\nпорядок;ЗАКОН И ПОРЯДОК\nздравый смысл;ЗДРАВЫЙ СМЫСЛ\n"
                "закон и порядок;ЗАКОН И ПОРЯДОК\nправа человека;ЦЕННОСТИ\nсмысл;ЗДРАВЫЙ СМЫСЛ\n")
    r = client.post("/dictionaries", data={"name": "Политика", "words": ""},
                    files={"file": ("pol.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}, follow_redirects=False)
    assert r.status_code == 303
    run_pending()
    with SessionLocal() as s:
        d = s.query(Dictionary).filter_by(name="Политика").one()
        texts = {t.title: t.id for t in s.query(TextDoc).all()}
        cid = s.query(Corpus).filter_by(name="Речи").one().id
        return {"dict_id": d.id, "corpus_id": cid, "texts": texts, "entries": {e.surface: e for e in d.entries}}


def test_phrase_entries(corpus):
    from app.db import SessionLocal
    from app.models import Hit

    e = corpus["entries"]
    assert e["здравый смысл"].phrase and e["здравый смысл"].lemma == "смысл"
    assert e["закон и порядок"].phrase and e["закон и порядок"].lemma == "закон"
    assert e["права человека"].lemma == "право" and e["закон"].phrase is None
    with SessionLocal() as s:
        def n(word):
            return s.query(Hit).filter_by(entry_id=e[word].id).count()
        assert n("здравый смысл") == 2  # «здравого смысла», «Здравый смысл»
        assert n("закон и порядок") == 2  # в обоих текстах; «Закон суров» — нет
        assert n("права человека") == 1
        assert n("закон") == 4 and n("смысл") == 2  # одиночные слова считаются как раньше


def test_phrase_search_in_concordance(client, corpus):
    from app.db import SessionLocal
    from app.services import analytics as A

    ids = list(corpus["texts"].values())
    with SessionLocal() as s:
        c = A.concordance(s, corpus["dict_id"], ids, entry_id=corpus["entries"]["порядок"].id, phrase="закон * порядок")
        assert c["total"] == 2 and c["phrase_sentences"] == 2
        assert all("закон и порядок" in " ".join(i["phrase"]).lower() for i in c["items"])
        c = A.concordance(s, corpus["dict_id"], ids, phrase="порядок основанный на право")
        assert c["phrase_sentences"] == 1 and "Порядок, основанный на праве" in c["items"][0]["phrase"][0]
        # без выбранного слова — только вхождения внутри словосочетания (закон, «закон и порядок», порядок), а не «города»
        c = A.concordance(s, corpus["dict_id"], ids, phrase="закон и порядок")
        assert c["total"] > 0 and {i["kw"].lower() for i in c["items"]} <= {"закон", "порядок"}
        assert A.concordance(s, corpus["dict_id"], ids, phrase="порядок закон")["total"] == 0
        assert A.concordance(s, corpus["dict_id"], ids, phrase="порядок закон", phrase_any_order=True)["total"] > 0
        # запись-словосочетание показывает найденный отрывок
        c = A.concordance(s, corpus["dict_id"], ids, entry_id=corpus["entries"]["здравый смысл"].id)
        assert {x.lower() for i in c["items"] for x in i["phrase"]} == {"здравого смысла", "здравый смысл"}
    base = f"dictionary_id={corpus['dict_id']}&corpus_id={corpus['corpus_id']}"
    r = client.get(f"/analysis?tab=conc&{base}&phrase=закон+*+порядок")
    assert r.status_code == 200 and "предложений со словосочетанием: 2" in r.text
    r = client.get(f"/export/concordance.csv?{base}&phrase=здравый+смысл")
    assert r.status_code == 200 and "Словосочетание" in r.content.decode("utf-8-sig")


def test_category_frequency(client, corpus):
    import io as _io

    import pandas as pd

    from app.db import SessionLocal
    from app.services import analytics as A

    ids = list(corpus["texts"].values())
    with SessionLocal() as s:
        cf = A.category_frequency(s, corpus["dict_id"], ids, dim="author")
        rows = {r["category"]: r for r in cf["rows"]}
        assert cf["has_categories"] and rows["ЗАКОН И ПОРЯДОК"]["entries"] == 3
        # закон (4) + порядок (4) + «закон и порядок» (2), но слово «закон» в «законе и порядке» — одно вхождение категории
        assert rows["ЗАКОН И ПОРЯДОК"]["count"] == 8
        assert abs(sum(r["share"] for r in cf["rows"]) - 100) < 0.5
        assert [g["label"] for g in cf["groups"]] == ["Д. Трамп", "Э. Блинкен"]
        only = A.category_frequency(s, corpus["dict_id"], ids, entry_ids=[corpus["entries"]["закон"].id])
        assert [r["count"] for r in only["rows"]] == [4]
    base = f"dictionary_id={corpus['dict_id']}&corpus_id={corpus['corpus_id']}"
    r = client.get(f"/analysis?tab=freq&{base}&cat_dim=text")
    assert r.status_code == 200 and "По категориям словаря" in r.text
    x = pd.read_excel(_io.BytesIO(client.get(f"/export/frequency.xlsx?{base}").content), sheet_name=None)
    assert "По категориям" in x and "ipm: Дональд Трамп" in x["По категориям"].columns


def test_word_dropdowns_sorted_by_frequency(client, corpus):
    base = f"dictionary_id={corpus['dict_id']}&corpus_id={corpus['corpus_id']}"
    for tab in ("profile", "conc", "colloc", "semantic", "structure", "compare"):
        page = client.get(f"/analysis?tab={tab}&{base}").text
        assert page.index(">закон (4)</option>") < page.index(">права человека (1)</option>"), tab


def test_no_proper_names(client, corpus):
    from app.db import SessionLocal
    from app.services.entities import name_lemmas
    from app.services.keywords import tfidf_by_text
    from app.services.selection import from_params

    ids = list(corpus["texts"].values())
    with SessionLocal() as s:
        names = name_lemmas(s, ids)
        assert {"москва", "андрей"} <= names and "закон" not in names and "порядок" not in names
        sel = from_params(s, {"corpus_id": [str(corpus["corpus_id"])], "noprop": ["1"]})
        assert sel.noprop and "москва" in sel.stopset and "noprop=1" in sel.query() and "без имён собственных" in sel.describe()
        lemmas = {w["lemma"] for t in tfidf_by_text(s, ids, stop=sel.stopset)["texts"] for w in t["words"]}
        assert "москва" not in lemmas
        assert "москва" in {w["lemma"] for t in tfidf_by_text(s, ids)["texts"] for w in t["words"]}
    base = f"dictionary_id={corpus['dict_id']}&corpus_id={corpus['corpus_id']}&noprop=1"
    for tab in ("keywords", "colloc", "topics", "semantic", "stylo"):
        r = client.get(f"/analysis?tab={tab}&{base}")
        assert r.status_code == 200 and 'name="noprop" value="1" checked' in r.text, tab


def test_help_and_terms(client, corpus):
    from app.services import help as H

    terms = H.glossary()
    assert {"ipm", "logdice", "g2", "tfidf", "phrase", "noprop", "cosine_delta"} <= set(terms)
    assert all(t["short"] and not t["short"].startswith("#") for t in terms.values())
    r = client.get("/help")
    assert r.status_code == 200 and "Глоссарий" in r.text and "Руководство пользователя" in r.text
    r = client.get("/help?q=коллокаций")
    assert r.status_code == 200 and "logDice" in r.text
    r = client.get("/help/glossary")
    assert r.status_code == 200 and 'id="logdice"' in r.text and "{#" not in r.text
    assert client.get("/help/term/ipm.json").json()["title"] == "ipm"
    assert client.get("/help/term/nope.json").status_code == 404
    assert client.get("/help/files/political_discourse_ru.csv").status_code == 200
    assert client.get("/help/files/..%2F..%2F.env").status_code == 404
    page = client.get("/help/political-discourse").text
    assert 'href="/help/files/political_discourse_ru.csv"' in page
    # значки «?» у показателей на страницах анализа
    base = f"dictionary_id={corpus['dict_id']}&corpus_id={corpus['corpus_id']}"
    assert 'data-term="ipm"' in client.get(f"/analysis?tab=freq&{base}").text
    assert 'data-term="logdice"' in client.get(f"/analysis?tab=colloc&{base}").text


def test_geo_map(client, corpus):
    from app.db import SessionLocal
    from app.services import geo as G

    # справочник: неоднозначность, исторические названия, дополнения важнее городов
    assert G.candidates("Париж")[0].cc == "FR"
    assert abs(G.candidates("Тифлис")[0].lat - 41.69) < 0.1
    assert G.candidates("Дон")[0].kind == "river" and G.candidates("Россия")[0].cc == "RU"
    ids = list(corpus["texts"].values())
    with SessionLocal() as s:
        m = G.place_map(s, ids)
        names = {p["name"]: p for p in m["points"]}
        assert "Москва" in names and names["Москва"]["source"] == "auto" and abs(names["Москва"]["lat"] - 55.75) < 0.1
        assert not names["Москва"]["doubtful"]
        assert "Андрей" not in names  # персоны на карту не попадают
        s.commit()
    base = f"dictionary_id={corpus['dict_id']}&corpus_id={corpus['corpus_id']}"
    r = client.get(f"/analysis?tab=geo&{base}")
    assert r.status_code == 200 and "chart-geo" in r.text and "Москва" in r.text
    # ручная правка: координаты, ошибка ввода, «не на карте», сброс
    back = f"/analysis?tab=geo&{base}"
    client.post("/places", data={"name": "Москва", "action": "coords", "coords": "55,1; 37,2", "back": back})
    with SessionLocal() as s:
        p = {x["name"]: x for x in G.place_map(s, ids)["points"]}["Москва"]
        assert p["source"] == "manual" and p["lat"] == 55.1 and p["lon"] == 37.2
    r = client.post("/places", data={"name": "Москва", "action": "coords", "coords": "север", "back": back})
    assert "широта и долгота" in r.text
    client.post("/places", data={"name": "Москва", "action": "none", "back": back})
    with SessionLocal() as s:
        m = G.place_map(s, ids)
        assert "Москва" not in {x["name"] for x in m["points"]} and "Москва" in {x["name"] for x in m["unresolved"]}
    client.post("/places", data={"name": "Москва", "action": "reset", "back": back})
    with SessionLocal() as s:
        assert {x["name"]: x for x in G.place_map(s, ids)["points"]}["Москва"]["source"] == "auto"
    assert client.get(f"/export/geo.csv?{base}").status_code == 200
    assert client.get("/static/topojson/world_110m.json").status_code == 200
