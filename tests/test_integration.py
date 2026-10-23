"""Сквозной сценарий через веб-интерфейс: загрузка → разбор → словарь → анализ → экспорт."""

import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

TEXT = """Сладкий, тяжёлый запах сирени, который шёл из сада, наполнял комнату и кружил голову.
— Чем это пахнет? — спросил Л. Н. Толстой.
Пахло дымом и гарью. Запах был сладким. Он почувствовал и вдохнул запах.
Не запах, а вонь стояла в комнате. Почувствовав аромат, он остановился."""


@pytest.fixture(scope="module")
def client():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    from app.db import engine

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(Config("alembic.ini"), "head")

    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    r = c.post("/login", data={"username": "admin", "password": "test-password", "next": "/"}, follow_redirects=False)
    assert r.status_code == 303
    return c


def page_text(client):
    return client.get("/analysis").text


def run_jobs():
    from app.services.jobs import run_pending

    run_pending()


def test_requires_login():
    from fastapi.testclient import TestClient

    from app.main import app

    r = TestClient(app).get("/texts", follow_redirects=False)
    assert r.status_code == 303 and "/login" in r.headers["location"]


def test_full_flow(client):
    r = client.post("/corpora", data={"name": "Тест", "description": ""}, follow_redirects=False)
    assert r.status_code == 303
    corpus_id = int(r.headers["location"].rsplit("/", 1)[1])

    # несколько файлов сразу: у каждого свои сведения (форма) или из таблицы метаданных
    files = [
        ("files", ("siren.txt", io.BytesIO(TEXT.encode("utf-8")), "text/plain")),
        ("files", ("dym.txt", io.BytesIO("Пахло дымом. Запах гари стоял над рекой.".encode("utf-8")), "text/plain")),
        ("files", ("gen.txt", io.BytesIO("Аромат кофе наполнял кухню.".encode("utf-8")), "text/plain")),
    ]
    meta_csv = "файл;название;автор;год;жанр;тип\ngen;Текст модели;Qwen;2025;эссе;нейросеть\n".encode("utf-8")
    files.append(("meta_file", ("meta.csv", io.BytesIO(meta_csv), "text/csv")))
    data = {"author": "Общий Автор", "years": "1900", "genre": "рассказ", "source_type": "natural",
            "corpus_ids": str(corpus_id),
            "f_names": ["siren.txt", "dym.txt", "gen.txt"], "f_titles": ["Сирень", "Дым", ""],
            "f_authors": ["Автор Сирени", "", ""], "f_years": ["1880", "", ""], "f_genres": ["", "очерк", ""],
            "f_types": ["", "", ""], "f_generators": ["", "", ""]}
    r = client.post("/texts/upload", files=files, data=data)
    assert r.status_code == 200
    run_jobs()
    assert "готов" in client.get("/texts").text
    from app.db import SessionLocal as _S
    from app.models import TextDoc as _T

    with _S() as s:
        by_title = {t.title: t for t in s.query(_T).all()}
        assert by_title["Сирень"].author == "Автор Сирени" and by_title["Сирень"].year_from == 1880
        assert by_title["Сирень"].genre == "рассказ"  # общее поле
        assert by_title["Дым"].author == "Общий Автор" and by_title["Дым"].genre == "очерк"
        assert by_title["Текст модели"].author == "Qwen" and by_title["Текст модели"].source_type == "generated"

    r = client.post("/dictionaries", data={"name": "Запахи", "words": "запах, аромат, вонь, пахнуть, дым"},
                    follow_redirects=False)
    assert r.status_code == 303
    dict_id = int(r.headers["location"].rsplit("/", 1)[1])
    run_jobs()

    page = client.get(f"/dictionaries/{dict_id}")
    assert page.status_code == 200 and "запах" in page.text

    base = f"dictionary_id={dict_id}&corpus_id={corpus_id}"
    for tab in ("overview", "freq", "conc", "profile", "compare", "structure"):
        r = client.get(f"/analysis?tab={tab}&{base}")
        assert r.status_code == 200, tab

    from app.db import SessionLocal
    from app.models import DictionaryEntry, Hit
    from app.services import analytics as A

    with SessionLocal() as s:
        entry = s.query(DictionaryEntry).filter_by(dictionary_id=dict_id, surface="запах").one()
        ids = A.corpus_text_ids(s, corpus_id)
        freq = {row["word"]: row["count"] for row in A.frequency(s, dict_id, ids)["rows"]}
        assert freq["запах"] >= 4
        prof = A.word_profile(s, entry.id, ids)
        assert prof["total"] == freq["запах"]
        hit_id = s.query(Hit.id).filter_by(entry_id=entry.id).first()[0]
        sent_id = s.query(Hit.sentence_id).filter_by(entry_id=entry.id).first()[0]

    r = client.get(f"/analysis?tab=profile&{base}&entry_id={entry.id}")
    assert "logDice" in r.text

    # ручная отметка ложного вхождения
    r = client.post(f"/api/hits/{hit_id}/valid", json={"valid": False})
    assert r.json()["is_valid"] is False
    with SessionLocal() as s:
        assert A.frequency(s, dict_id, ids)["rows"][0]["count"] >= 1

    page = client.get(f"/sentences/{sent_id}")
    assert page.status_code == 200 and f"Предложение ID {sent_id}" in page.text and "(NOUN)" in page.text
    ctx = client.get(f"/api/sentences/{sent_id}/context?n=2").json()
    assert any(x["current"] and x["id"] == sent_id for x in ctx["sentences"]) and len(ctx["sentences"]) >= 2
    r = client.get(f"/go/sentence?id={sent_id}", follow_redirects=False)
    assert r.headers["location"] == f"/sentences/{sent_id}"
    conc = client.get(f"/analysis?tab=conc&{base}").text
    assert f'data-ctx="{sent_id}"' in conc or "data-ctx=" in conc
    assert "(nsubj" in conc or "(obj" in conc or "(obl" in conc

    for dim in ("author", "text", "genre", "source_type", "period"):
        r = client.get(f"/analysis?tab=overview&{base}&dim={dim}&step=10&metric=ipm&words={entry.id}")
        assert r.status_code == 200 and "chart-ov" in r.text, dim
    with SessionLocal() as s:
        ov = A.overview(s, dict_id, ids, [entry.id], "author", 10)
        now = {row["word"]: row["count"] for row in A.frequency(s, dict_id, ids)["rows"]}
        assert sum(ov["words"][0]["count"]) == now["запах"]  # одно вхождение уже отмечено ложным
        assert A.filter_texts(s, ids, year_from=1870, year_to=1890) and not A.filter_texts(s, ids, year_from=3000)
    # статистический портрет текста
    with SessionLocal() as s:
        tid = s.query(Hit.text_id).filter_by(entry_id=entry.id).first()[0]
    r = client.get(f"/texts/{tid}/stats")
    assert r.status_code == 200 and "MATTR" in r.text and "ch-zipf" in r.text
    from app.services.textstats import get_stats, mattr, yule_k

    with SessionLocal() as s:
        st = get_stats(s, tid)
        assert st["size"]["words"] > 0 and st["lexical"]["n_lemmas"] > 0 and st["morphology"]["pos"]
    assert mattr(["а", "б", "а", "б"], window=2) == 1.0 and yule_k([1, 1, 1]) < yule_k([3])
    assert client.get(f"/export/text/{tid}/stats.xlsx").status_code == 200
    for url in (f"/export/overview.xlsx?{base}&dim=period&step=25", f"/export/frequency.xlsx?{base}", f"/export/frequency.csv?{base}", f"/export/concordance.xlsx?{base}",
                f"/export/profile.json?{base}&entry_id={entry.id}", f"/export/compare.xlsx?{base}&corpus_b=",
                f"/export/structure.csv?{base}", f"/export/hits.conllu?{base}", f"/export/dictionary/{dict_id}.csv"):
        r = client.get(url)
        assert r.status_code == 200, url
        assert len(r.content) > 50, url
    assert "# text =" in client.get(f"/export/hits.conllu?{base}").text

    # пересчёт словаря сохраняет ручную отметку
    client.post(f"/dictionaries/{dict_id}/rematch")
    run_jobs()
    with SessionLocal() as s:
        assert s.query(Hit).filter_by(entry_id=entry.id, is_valid=False).count() == 1


def test_selection_report_stopwords(client):
    """Панель «Выборка», выбор слов, отчёт, стоп-слова, выгрузка предложений, страница задач."""
    from app.db import SessionLocal
    from app.models import Corpus, Dictionary, DictionaryEntry, TextDoc
    from app.services import analytics as A
    from app.services.selection import from_params

    with SessionLocal() as s:
        corpus = s.query(Corpus).filter_by(name="Тест").one()
        d = s.query(Dictionary).filter_by(name="Запахи").one()
        e_zapah = s.query(DictionaryEntry).filter_by(dictionary_id=d.id, surface="запах").one()
        e_dym = s.query(DictionaryEntry).filter_by(dictionary_id=d.id, surface="дым").one()
        texts = {t.title: t.id for t in s.query(TextDoc).all()}
        all_ids = A.corpus_text_ids(s, corpus.id)
        # отмеченные тексты, фильтры и чужие отметки (текст не из корпуса не обнуляет выборку)
        sel = from_params(s, {"corpus_id": [str(corpus.id)], "texts": [str(texts["Сирень"])]})
        assert sel.ids == [texts["Сирень"]]
        sel = from_params(s, {"corpus_id": [str(corpus.id)], "texts": ["999999"]})
        assert sel.ids == all_ids
        sel = from_params(s, {"corpus_id": [str(corpus.id)], "stypes": ["generated"]})
        assert sel.ids == [texts["Текст модели"]]
        sel = from_params(s, {"corpus_id": [str(corpus.id)], "authors": ["Автор Сирени"], "stop": ["ru"]})
        assert sel.ids == [texts["Сирень"]] and "и" in sel.stopset

    base = f"dictionary_id={d.id}&corpus_id={corpus.id}"
    words = f"&words={e_zapah.id}&words={e_dym.id}"
    # все вкладки с выбранными словами и отмеченным текстом
    for tab in ("overview", "report", "freq", "conc", "profile", "compare", "structure"):
        r = client.get(f"/analysis?tab={tab}&{base}{words}&texts={texts['Сирень']}&texts={texts['Дым']}&stop=ru")
        assert r.status_code == 200, tab
    # ссылки «Скачать» сохраняют все повторяющиеся параметры выборки (раньше оставался последний автор/текст/слово)
    r = client.get(f"/analysis?tab=overview&{base}{words}&texts={texts['Сирень']}&texts={texts['Дым']}"
                   "&authors=Автор+Сирени&authors=Другой")
    link = r.text.split('href="/export/overview.xlsx', 1)[1].split('"', 1)[0].replace("&amp;", "&")
    for p in (f"words={e_zapah.id}", f"words={e_dym.id}", f"texts={texts['Сирень']}", f"texts={texts['Дым']}",
              "authors=%D0%94%D1%80%D1%83%D0%B3%D0%BE%D0%B9", "authors=%D0%90%D0%B2%D1%82%D0%BE%D1%80"):
        assert p in link, p
    r = client.get(f"/analysis?tab=report&{base}{words}&report_b=")
    assert r.status_code == 200 and "Сводный отчёт" in r.text and "rep-freq" in r.text and "Сравнение с корпусом" in r.text
    # отчёт без выбранных слов берёт самые частые
    assert "10 самых частых" in client.get(f"/analysis?tab=report&{base}").text
    # частоты: при выбранных словах — только они, с долей; all_words — весь словарь
    r = client.get(f"/analysis?tab=freq&{base}&words={e_dym.id}")
    assert "Показаны <strong>1 выбранных слов" in r.text and "показать весь словарь" in r.text
    with SessionLocal() as s:
        f = A.only_words(A.frequency(s, d.id, all_ids), [e_dym.id, e_zapah.id])
        assert {x["entry_id"] for x in f["rows"]} == {e_dym.id, e_zapah.id}
        assert abs(sum(x["share"] for x in f["rows"]) - 100) < 0.2
    assert "Показан весь словарь" in client.get(f"/analysis?tab=freq&{base}&words={e_dym.id}&all_words=1").text
    assert client.get(f"/export/frequency.xlsx?{base}&words={e_dym.id}").status_code == 200
    # корпус A выбирается в форме вкладки сравнения
    r = client.get(f"/analysis?tab=compare&{base}&corpus_b=")
    assert 'name="corpus_id" form="sel-form"' in r.text and "disabled" not in r.text.split("Корпус A")[1][:200]
    # конкорданс ограничен выбранными словами и текстами
    with SessionLocal() as s:
        conc = A.concordance(s, d.id, [texts["Дым"]], entry_ids=[e_dym.id])
        assert conc["total"] >= 1 and all(i["word"] == "дым" for i in conc["items"])
    # выгрузки по выборке: отчёт, предложения отмеченных слов, CoNLL-U
    for url in (f"/export/report.xlsx?{base}{words}", f"/export/report.json?{base}{words}&report_b=",
                f"/export/concordance.xlsx?{base}{words}", f"/export/concordance.csv?{base}&entry_id={e_zapah.id}",
                f"/export/profile.xlsx?{base}&entry_id={e_zapah.id}&stop=ru"):
        r = client.get(url)
        assert r.status_code == 200 and len(r.content) > 50, url
    csv_text = client.get(f"/export/concordance.csv?{base}&words={e_dym.id}").content.decode("utf-8-sig")
    assert "дым" in csv_text.lower() and "запах;" not in csv_text.split("\n", 1)[1][:30]
    conllu = client.get(f"/export/hits.conllu?{base}&words={e_dym.id}").text
    assert "# text =" in conllu

    # стоп-слова: встроенный список, свой список, фильтр в портрете и профиле
    # скрипты и стили подключаются с меткой версии — браузер не возьмёт старый app.js из кэша
    assert "/static/app.js?v=" in page_text(client) and "function checkTop" in client.get("/static/app.js").text
    page = client.get("/stopwords")
    assert page.status_code == 200 and "Встроенный" in page.text
    assert "быть" in client.get("/stopwords/builtin.txt").text
    r = client.post("/stopwords", data={"name": "Мои", "words": "Сладкий, тяжёлый\n# комментарий\nкомнату", "description": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    from app.models import StopList

    with SessionLocal() as s:
        sl = s.query(StopList).filter_by(name="Мои").one()
        assert "сладкий" in sl.words and "тяжелый" in sl.words and "и" not in sl.words  # флажок «включить встроенный» не отмечен
        sl_id = sl.id
        prof_all = A.word_profile(s, e_zapah.id, all_ids)
        prof_stop = A.word_profile(s, e_zapah.id, all_ids, stop=frozenset({"сладкий"}))
        lemmas = lambda p: {x["lemma"] for b in p["relations"].values() for x in b["items"]}
        assert "сладкий" in lemmas(prof_all) and "сладкий" not in lemmas(prof_stop)
        tid = texts["Сирень"]
    r = client.post(f"/stopwords/{sl_id}", data={"name": "Мои", "words": "сладкий", "description": "x"}, follow_redirects=False)
    assert r.status_code == 303
    stats_page = client.get(f"/texts/{tid}/stats?stop={sl_id}")
    assert stats_page.status_code == 200 and "без стоп-слов" in stats_page.text
    from app.services.stopwords import resolve
    from app.services.textstats import frequency_lists, get_stats

    with SessionLocal() as s:
        st = get_stats(s, tid)
        assert any(x["lemma"] == "и" for x in st["frequency"]["top_all"])
        assert not any(x["lemma"] == "и" for x in frequency_lists(st, resolve(s, "ru"))["top_all"])
    assert client.get(f"/export/text/{tid}/stats.xlsx?stop=ru").status_code == 200
    assert client.post(f"/stopwords/{sl_id}/delete", follow_redirects=False).status_code == 303
    client.post("/stopwords", data={"name": "Встроенный+", "words": "князь", "with_builtin": "1"})
    with SessionLocal() as s:
        assert {"и", "князь"} <= set(s.query(StopList).filter_by(name="Встроенный+").one().words)

    # волна 2: коллокации (окно и синтаксис), ключевые слова, карта текста, фильтр «рядом» в конкордансе
    from app.services.collocations import collocations, orbit_layout
    from app.services.keywords import keywords, reference_ids, tfidf_by_text
    from app.services.textmap import passage, text_map

    with SessionLocal() as s:
        win = collocations(s, e_zapah.id, all_ids, mode="window", window=3, min_count=1, pos="all")
        lem = {r["lemma"]: r for r in win["rows"]}
        assert win["fx"] >= 4 and "сладкий" in lem and lem["сладкий"]["logdice"] is not None
        syn = collocations(s, e_zapah.id, all_ids, mode="syntax", min_count=1, pos="all")
        # синтаксические коллокаты берутся из связей (вхождение со «сладкий» выше отмечено как ложное и не учитывается)
        assert syn["rows"] and all(r["rel"] and r["rel_label"] for r in syn["rows"]) and syn["span"] == 1
        assert "сладкий" not in {r["lemma"] for r in collocations(s, e_zapah.id, all_ids, min_count=1, pos="all",
                                                                    stop=frozenset({"сладкий"}))["rows"]}
        orb = orbit_layout(win["rows"], "logdice")
        assert orb[0]["ring"] == 1 and all("x" in r for r in orb)
        near = A.concordance(s, d.id, all_ids, entry_id=e_zapah.id, near="сладкий", window=3)
        assert near["total"] == lem["сладкий"]["count"]
        ref, _ = reference_ids(s, [texts["Сирень"]], "")
        kw = keywords(s, [texts["Сирень"]], ref, min_count=1)
        assert kw["words_a"] > 0 and kw["positive"] and kw["positive"][0]["ll"] >= kw["positive"][-1]["ll"]
        tf = tfidf_by_text(s, all_ids)
        assert len(tf["texts"]) == len(all_ids)
        tm = text_map(s, texts["Сирень"], d.id, [e_zapah.id], 10)
        assert tm["rows"][0]["total"] == sum(tm["rows"][0]["count"]) >= 1 and len(tm["seg_words"]) == 10
        pas = passage(s, texts["Сирень"], 1, 3, d.id)
        assert pas and any(x["marks"] for x in pas)
    for tab, extra in (("colloc", f"&entry_id={e_zapah.id}"), ("colloc", f"&entry_id={e_zapah.id}&mode=syntax&min_count=1"),
                       ("keywords", ""), ("keywords", f"&ref={corpus.id}"), ("map", f"&text_id={texts['Сирень']}&segments=20")):
        r = client.get(f"/analysis?tab={tab}&{base}{extra}&stop=ru")
        assert r.status_code == 200, (tab, extra)
    assert "chart-orbit" in client.get(f"/analysis?tab=colloc&{base}&entry_id={e_zapah.id}&min_count=1&pos=all").text
    assert "chart-map" in client.get(f"/analysis?tab=map&{base}&text_id={texts['Сирень']}").text
    assert client.get(f"/api/texts/{texts['Сирень']}/passage?from=1&to=5&dictionary_id={d.id}").json()["sentences"]
    for url in (f"/export/collocations.xlsx?{base}&entry_id={e_zapah.id}&min_count=1", f"/export/collocations.csv?{base}&entry_id={e_zapah.id}&mode=syntax&min_count=1",
                f"/export/keywords.xlsx?{base}", f"/export/keywords.csv?{base}&ref={corpus.id}",
                f"/export/textmap.xlsx?{base}&text_id={texts['Сирень']}", f"/export/concordance.csv?{base}&entry_id={e_zapah.id}&near=сладкий&window=3"):
        r = client.get(url)
        assert r.status_code == 200 and len(r.content) > 50, url

    # волна 3: имена собственные (извлекаются после разбора автоматически), правки, семантика, темы
    from app.services import entities as E
    from app.services import semantic as S

    with SessionLocal() as s:
        ix = E.entity_index(s, all_ids)
        assert not ix["missing"] and ix["n_mentions"] > 0
        names = {x["name"]: x for x in ix["items"]}
        assert any("Толстой" in n for n in names), names.keys()
        tol = next(n for n in names if "Толстой" in n)
        assert E.mentions(s, all_ids, tol)["sentences"]
        E.set_override(s, tol, "PER", "Лев Толстой")
        s.commit()
        assert "Лев Толстой" in {x["name"] for x in E.entity_index(s, all_ids)["items"]}
        E.set_override(s, "Лев Толстой", None, None, hidden=True)
        s.commit()
        assert "Лев Толстой" not in {x["name"] for x in E.entity_index(s, all_ids)["items"]}
        sim = S.similar_words(s, ["запах"], all_ids, only_in_corpus=False)
        assert sim["items"] and all(w["word"] != "запах" for w in sim["items"])
        assert S.dictionary_map(s, d.id)["points"]
        assert S.topics(s, all_ids, k=3)["n_chunks"] >= 1
    r = client.post("/entities/override", data={"norm": "Лев Толстой", "etype": "", "canonical": ""}, follow_redirects=False)
    assert r.status_code == 303
    for tab, extra in (("entities", ""), ("entities", "&etype=PER&unit=paragraph&cross=PER&nolatin=1"),
                       ("semantic", f"&entry_id={e_zapah.id}&k=2"), ("semantic", "&seed=запах, аромат&any=1&groups=3"), ("topics", "&k=3")):
        r = client.get(f"/analysis?tab={tab}&{base}{extra}")
        assert r.status_code == 200, (tab, extra)
    assert client.get(f"/api/entities/mentions?{base}&name={tol}").status_code == 200
    assert client.get(f"/api/entities/cooccurrence?{base}&name={tol}&entry={e_zapah.id}").status_code == 200
    r = client.post(f"/dictionaries/{d.id}/add-words", data={"words": ["аромат", "благоухание"]}, follow_redirects=False)
    assert r.status_code == 303
    with SessionLocal() as s:
        assert s.query(DictionaryEntry).filter_by(dictionary_id=d.id, surface="благоухание").count() == 1
    for url in (f"/export/entities.xlsx?{base}", f"/export/entities.csv?{base}&etype=PER"):
        r = client.get(url)
        assert r.status_code == 200 and len(r.content) > 50, url
    assert client.post("/entities/extract", data={"text_ids": [str(texts["Дым"])]}, follow_redirects=False).status_code == 303
    run_jobs()

    # волна 4: авторство, сравнение текстов, проверка на признаки машинного текста
    from app.services import aicheck as C
    from app.services.stylometry import analyze
    from app.services.textcompare import compare

    with SessionLocal() as s:
        st = analyze(s, all_ids, mfw=50, unit="text", target=texts["Сирень"])
        assert st["matrix"] and len(st["units"]) == len(all_ids) and st["ranking"]["rows"]
        assert "Нужно хотя бы 3" in analyze(s, all_ids[:2], unit="text").get("error", "")
        tc = compare(s, all_ids[:3])
        assert len(tc["texts"]) == 3 and tc["rows"] and tc["delta"] and len(tc["jaccard"]) == 3
        rep = C.report(s, C.parse_raw("Короткий текст про запах сирени."))
        assert not rep["enough"] and rep["rows"]
        assert rep["model"]["ok"] is False  # в тестовой базе мало образцов — калибровки нет, вердикта нет
    for tab, extra in (("stylo", "&mfw=50"), ("stylo", f"&unit=chunk&chunk=2000&target={texts['Сирень']}"), ("textcmp", ""),
                       ("textcmp", f"&cmp={texts['Сирень']}&cmp={texts['Дым']}")):
        r = client.get(f"/analysis?tab={tab}&{base}{extra}")
        assert r.status_code == 200, (tab, extra)
    r = client.post("/check", data={"text": "Важно отметить, что запах играет ключевую роль. " * 80})
    assert r.status_code == 200 and "Отчёт" in r.text and "не откалибрована" in r.text
    assert client.get(f"/check?text_id={texts['Сирень']}").status_code == 200
    assert client.post("/check", files={"file": ("a.txt", "Запах сирени.".encode("utf-8"), "text/plain")}).status_code == 200
    for url in (f"/export/stylometry.xlsx?{base}&mfw=50", f"/export/textcmp.xlsx?{base}"):
        r = client.get(url)
        assert r.status_code == 200 and len(r.content) > 50, url

    # страница задач: удаление выполненных
    assert client.get("/jobs").status_code == 200
    assert client.post("/jobs/clear", data={"what": "finished"}, follow_redirects=False).status_code == 303
