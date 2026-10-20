"""Командная строка: python -m app.cli <команда>

Команды:
  import-dir ПАПКА [--meta CSV] [--corpus ИМЯ] [--type natural|generated]
        массовая загрузка .txt/.docx/.fb2 из папки; метаданные — из CSV
        (колонки: файл/TextID, название, автор, год, жанр)
  import-dictionary ФАЙЛ --name ИМЯ    загрузить словарь
  seed-demo                            демо: 3 текста из data/raw + словарь запахов
  run-jobs                             выполнить задачи из очереди и выйти
  check-nlp ["предложение"]            проверить, что парсер работает (печатает разбор и связи)
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

from sqlalchemy import select

from app.db import session_scope
from app.models import Corpus
from app.nlp.normalize import SUPPORTED_EXTENSIONS, decode_bytes
from app.services.dictionaries import create_dictionary, parse_dictionary_file
from app.services.ingest import create_text
from app.services.jobs import run_pending

ROOT = Path(__file__).resolve().parent.parent

DEMO_DICTIONARY = (
    "духи, одеколон, аромат, букет, вонь, запах, перегар, смрад, парфюм, душок, благовоние, благоухание, "
    "зловоние, запашок, фимиам, миазм, амбре, амбра, пригарь, тухлятина, испарение, дуновение, ладан, "
    "скверна, дым, навоз, веять, вонять, благоухать, попахивать, разить, смердеть, чадить, чад, пахнуть, "
    "душистый, вонючий, пахучий, пот, нюхать, обонять, обоняние, ароматный, зловонный, гарь"
)


def _year(value: str | None) -> tuple[int | None, int | None]:
    years = [int(y) for y in re.findall(r"\d{4}", value or "")]
    if not years:
        return None, None
    return years[0], years[-1]


def _get_corpus(session, name: str | None) -> list[int]:
    if not name:
        return []
    c = session.scalar(select(Corpus).where(Corpus.name == name))
    if c is None:
        c = Corpus(name=name)
        session.add(c)
        session.flush()
    return [c.id]


def load_meta(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    text = decode_bytes(path.read_bytes())
    dialect = csv.Sniffer().sniff(text[:2000], delimiters=",;\t")
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    meta = {}
    for r in rows:
        r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
        key = r.get("файл") or r.get("file") or r.get("textid") or r.get("id")
        if key:
            meta[key] = r
    return meta


def import_dir(folder: Path, meta_csv: Path | None, corpus: str | None, source_type: str) -> int:
    meta = load_meta(meta_csv)
    n = 0
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    with session_scope() as s:
        corpus_ids = _get_corpus(s, corpus)
        for p in files:
            m = meta.get(p.name) or meta.get(p.stem) or {}
            y1, y2 = _year(m.get("год") or m.get("year") or m.get("text_year_creation"))
            create_text(
                s, filename=p.name, data=p.read_bytes(),
                title=m.get("название") or m.get("title") or m.get("texttitle") or p.stem,
                author=m.get("автор") or m.get("author") or m.get("text_author"),
                year_from=y1, year_to=y2,
                genre=m.get("жанр") or m.get("genre") or m.get("text_genre"),
                source_type=source_type, corpus_ids=corpus_ids,
            )
            n += 1
    return n


def seed_demo() -> None:
    raw = ROOT / "data" / "raw"
    demo = [("38.txt", "Собачье сердце", "Михаил Афанасьевич Булгаков", 1925, "повесть"),
            ("66.txt", "Левша", "Николай Семенович Лесков", 1881, "повесть"),
            ("70.txt", "Рассказы (сборник)", "Михаил Михайлович Зощенко", 1927, "рассказ")]
    with session_scope() as s:
        corpus_ids = _get_corpus(s, "Демо: литература")
        for fn, title, author, year, genre in demo:
            p = raw / fn
            if p.exists():
                create_text(s, filename=fn, data=p.read_bytes(), title=title, author=author, year_from=year,
                            genre=genre, corpus_ids=corpus_ids)
        gen = ROOT / "data" / "generated" / "cleaned-raw-qwen_text.txt"
        if gen.exists():
            gen_ids = _get_corpus(s, "Демо: тексты нейросетей")
            create_text(s, filename=gen.name, data=gen.read_bytes(), title="Тексты Qwen (пример)",
                        source_type="generated", generator="Qwen", corpus_ids=gen_ids)
        items = [{"surface": w.strip(), "category": None, "upos": None, "comment": None} for w in DEMO_DICTIONARY.split(",")]
        create_dictionary(s, "Запахи (демо)", items)


def check_nlp(sentence: str) -> None:
    from app.nlp.parsers import get_parser
    from app.nlp.pipeline import process_text
    from app.nlp.syntax import REL_LABELS, SentenceAnalysis

    from app.config import settings

    parser = get_parser()
    print(f"Парсер: {parser.version}")
    if settings.two_stage:
        print(f"Двухступенчатая обработка: весь текст — {get_parser(settings.fast_parser).version}, "
              f"предложения со словами словарей — {parser.version}")
    print()
    for batch in process_text(sentence, parser):
        for ps in batch:
            print(ps.sentence.to_conllu())
            an = SentenceAnalysis(ps.sentence)
            for t in ps.sentence.tokens:
                if t.upos in ("NOUN", "PROPN"):
                    print(f"«{t.form}» — {an.role(t.idx).label}")
                    for r in an.relations(t.idx):
                        print(f"    {REL_LABELS.get(r.rel_type, r.rel_type)}: {r.related_lemma} ({r.scope}{', ' + r.rule if r.rule else ''})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("import-dir")
    p1.add_argument("folder", type=Path)
    p1.add_argument("--meta", type=Path)
    p1.add_argument("--corpus")
    p1.add_argument("--type", default="natural", choices=["natural", "generated", "other"])
    p2 = sub.add_parser("import-dictionary")
    p2.add_argument("file", type=Path)
    p2.add_argument("--name", required=True)
    sub.add_parser("seed-demo")
    sub.add_parser("run-jobs")
    p3 = sub.add_parser("check-nlp")
    p3.add_argument("sentence", nargs="?", default="Сладкий запах сирени, который шёл из сада, наполнял комнату и кружил голову.")
    a = ap.parse_args()
    if a.cmd == "import-dir":
        n = import_dir(a.folder, a.meta, a.corpus, a.type)
        print(f"Поставлено в очередь текстов: {n}. Обработчик (worker) разберёт их в фоне.")
    elif a.cmd == "import-dictionary":
        with session_scope() as s:
            d = create_dictionary(s, a.name, parse_dictionary_file(a.file.name, a.file.read_bytes()))
            print(f"Словарь «{d.name}»: {len(d.entries)} слов")
    elif a.cmd == "seed-demo":
        seed_demo()
        print("Демо-данные поставлены в очередь.")
    elif a.cmd == "check-nlp":
        check_nlp(a.sentence)
    elif a.cmd == "run-jobs":
        print(f"Выполнено задач: {run_pending()}")


if __name__ == "__main__":
    main()
