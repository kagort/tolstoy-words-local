"""Пакетная загрузка: много файлов или ZIP + таблица метаданных, где имя файла — ID текста."""

import io
import os
import zipfile

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="нужен TEST_DATABASE_URL")

# формат таблицы пользователя: запятые, кавычки, прочерки, диапазоны лет
META = """файл,название,автор,год,жанр,тип,модель
1,Война и мир,Лев Николаевич Толстой,1863,роман,литература,-
2,Анна Каренина,Лев Николаевич Толстой,1873,роман,литература,-
3,"Мы (роман, сборник)",Евгений Иванович Замятин,1920,роман/сборник,литература,-
4,На дне,Максим Горький,1901-1902,-,литература,-
5,Двенадцать стульев,"Илья Ильф, Евгений Петров",1928,роман,литература,-
"""


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


def test_metadata_parsing():
    from app.services.metadata import file_keys, lookup, parse_meta_table, split_corpora, unmatched_rows

    meta = parse_meta_table("meta.csv", META.encode("utf-8"))
    assert lookup(meta, "1.txt")["title"] == "Война и мир"
    assert lookup(meta, "001.txt")["title"] == "Война и мир"  # ведущие нули
    assert lookup(meta, "3.docx")["title"] == "Мы (роман, сборник)"
    assert lookup(meta, "4.txt")["genre"] == "" and lookup(meta, "4.txt")["generator"] == ""  # прочерк — нет данных
    assert lookup(meta, "5.txt")["author"] == "Илья Ильф, Евгений Петров"
    assert lookup(meta, "99.txt") == {}
    used = {id(lookup(meta, "1.txt")), id(lookup(meta, "2.txt"))}
    assert unmatched_rows(meta, used) == ["3", "4", "5"]
    assert split_corpora("Проза; Толстой | -") == ["Проза", "Толстой"]
    assert "abc" not in file_keys("abc.txt")[2:]  # нечисловые имена не превращаются в «число»
    from app.services.batchupload import author_variants, surname

    assert surname("Иван Сергеевич Шмелев") == surname("Шмелев И.") == "шмелев"
    assert surname("Илья Ильф, Евгений Петров") is None
    assert author_variants({"Иван Сергеевич Шмелев", "Шмелев И.", "Максим Горький"}) == [("Иван Сергеевич Шмелев", "Шмелев И.")]
    assert author_variants({"Лев Николаевич Толстой", "Алексей Николаевич Толстой"}) == []  # разные люди


def test_batch_upload_files_and_meta(client):
    from app.db import SessionLocal
    from app.models import Corpus, TextDoc

    files = [("files", (f"{i}.txt", io.BytesIO(f"Текст номер {i}. Запах сирени стоял в саду.".encode("utf-8")), "text/plain"))
             for i in (1, 2, 4, 7)]
    files.append(("meta_file", ("table-metadata.csv", io.BytesIO(META.encode("utf-8")), "text/csv")))
    # браузер присылает автозаполненное имя файла в ручной таблице — оно не должно перебить название из таблицы
    data = {"f_names": ["1.txt", "2.txt", "4.txt", "7.txt"], "f_titles": ["1", "", "", ""], "f_authors": ["", "", "", ""],
            "f_years": ["", "", "", ""], "f_genres": ["", "", "", ""], "f_types": ["", "", "", ""], "f_generators": ["", "", "", ""],
            "new_corpus": "Русская проза", "by_author": "1", "skip_existing": "1", "source_type": "natural"}
    r = client.post("/texts/upload", files=files, data=data, follow_redirects=False)
    assert r.status_code == 303
    page = client.get("/texts").text
    assert "Нет строки в таблице метаданных (1): 7.txt" in page
    assert "Строки таблицы без файла (2): 3, 5" in page
    with SessionLocal() as s:
        by_file = {t.source_filename: t for t in s.query(TextDoc).all()}
        wp = by_file["1.txt"]
        assert (wp.title, wp.author, wp.year_from, wp.genre, wp.source_type) == ("Война и мир", "Лев Николаевич Толстой", 1863, "роман", "natural")
        assert wp.meta.get("file_id") == "1" and wp.generator is None
        nd = by_file["4.txt"]
        assert (nd.year_from, nd.year_to, nd.genre) == (1901, 1902, None)
        assert by_file["7.txt"].title == "7"  # без строки — имя файла
        names = {c.name: {t.source_filename for t in c.texts} for c in s.query(Corpus).all()}
        assert names["Русская проза"] == {"1.txt", "2.txt", "4.txt", "7.txt"}
        assert names["Лев Николаевич Толстой"] == {"1.txt", "2.txt"} and names["Максим Горький"] == {"4.txt"}
    # повторная загрузка тех же файлов пропускается
    files2 = [("files", ("1.txt", io.BytesIO("Повтор.".encode("utf-8")), "text/plain"))]
    client.post("/texts/upload", files=files2, data={"skip_existing": "1"}, follow_redirects=False)
    assert "Пропущены уже загруженные файлы (1): 1.txt" in client.get("/texts").text
    with SessionLocal() as s:
        assert s.query(TextDoc).filter_by(source_filename="1.txt").count() == 1


def test_batch_upload_zip_with_meta_inside(client):
    from app.db import SessionLocal
    from app.models import Corpus, TextDoc

    meta = 'файл,название,автор,год,корпус\n10,Тихий Дон,Михаил Александрович Шолохов,1925,"Эпопеи; XX век"\n11,Конармия,Исаак Бабель,1923,XX век\n'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("корпус/10.txt", "Дон. Запах полыни.")
        zf.writestr("корпус/11.txt", "Конармия. Пахло дымом.")
        zf.writestr("корпус/table-metadata.csv", meta)
        zf.writestr("__MACOSX/._10.txt", "мусор")
    r = client.post("/texts/upload", files=[("files", ("pack.zip", io.BytesIO(buf.getvalue()), "application/zip"))],
                    data={"skip_existing": "1"}, follow_redirects=False)
    assert r.status_code == 303
    with SessionLocal() as s:
        td = s.query(TextDoc).filter_by(source_filename="10.txt").one()
        assert td.title == "Тихий Дон" and td.year_from == 1925
        corp = {c.name: {t.source_filename for t in c.texts} for c in s.query(Corpus).all()}
        assert corp["XX век"] == {"10.txt", "11.txt"} and corp["Эпопеи"] == {"10.txt"}
        assert s.query(TextDoc).filter(TextDoc.source_filename.like("%._10%")).count() == 0
