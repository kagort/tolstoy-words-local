"""База знаний: статьи в Markdown (app/help/*.md и избранные файлы docs/), глоссарий метрик, поиск.

Глоссарий (`app/help/glossary.md`) — единый источник подсказок: статья «### Название {#id}», первый абзац —
краткое определение для всплывающей подсказки у значка «?» (`term('ipm')` в шаблонах), остальное — подробности.
Поиск — по леммам (pymorphy3), поэтому «коллокаций» находит статью «Коллокация».
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELP_DIR = ROOT / "app" / "help"
DOCS_DIR = ROOT / "docs"
# статьи из docs/ (копируются в образ Docker, см. .dockerignore); остальные документы — для разработчиков
DOC_ARTICLES = {"guide": "USER_GUIDE.md", "political-discourse": "POLITICAL_DISCOURSE.md"}
FILES_DIR = DOCS_DIR / "dictionaries"  # файлы, на которые ссылаются статьи (словари, таблицы метаданных)
ORDER = ["guide", "glossary", "political-discourse"]


@dataclass
class Section:
    article: str
    anchor: str
    title: str
    text: str
    lemmas: dict[str, int] = field(default_factory=dict)
    title_lemmas: set[str] = field(default_factory=set)


@dataclass
class Article:
    slug: str
    title: str
    html: str
    toc: list[tuple[int, str, str]]  # (уровень, id, заголовок)
    sections: list[Section]


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_]+", "-", s)[:60] or "section"


def _md():
    from markdown_it import MarkdownIt

    return MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table").enable("strikethrough")


def _rewrite_link(href: str) -> str:
    """Ссылки между статьями и на файлы: «dictionaries/x.csv» → /help/files/x.csv, «USER_GUIDE.md#…» → /help/guide#…"""
    if re.match(r"^(https?:|mailto:|/|#)", href):
        return href
    path, _, frag = href.partition("#")
    name = Path(path).name
    for slug, fname in DOC_ARTICLES.items():
        if name == fname:
            return f"/help/{slug}" + (f"#{frag}" if frag else "")
    if name.endswith(".md") and (HELP_DIR / name).exists():
        return f"/help/{name[:-3]}" + (f"#{frag}" if frag else "")
    if (FILES_DIR / name).exists():
        return f"/help/files/{name}"
    return href


def render(slug: str, source: str) -> Article:
    md = _md()
    tokens = md.parse(source)
    toc, used = [], set()
    title = slug
    for k, tok in enumerate(tokens):
        if tok.type == "heading_open":
            inline = tokens[k + 1]
            text = inline.content
            m = re.search(r"\s*\{#([\w-]+)\}\s*$", text)
            if m:
                anchor = m.group(1)
                text = text[: m.start()]
                inline.content = text
                for ch in reversed(inline.children or []):
                    if ch.type == "text":
                        ch.content = re.sub(r"\s*\{#[\w-]+\}\s*$", "", ch.content)
                        break
            else:
                anchor = slugify(text)
            base, n = anchor, 2
            while anchor in used:
                anchor, n = f"{base}-{n}", n + 1
            used.add(anchor)
            tok.attrSet("id", anchor)
            level = int(tok.tag[1])
            if level == 1 and title == slug:
                title = text
            else:
                toc.append((level, anchor, text))
        elif tok.type == "inline":
            for ch in tok.children or []:
                if ch.type == "link_open" and ch.attrGet("href"):
                    ch.attrSet("href", _rewrite_link(ch.attrGet("href")))
    body = md.renderer.render(tokens, md.options, {})
    return Article(slug, title, body, toc, _sections(slug, tokens))


def _sections(slug: str, tokens) -> list[Section]:
    out: list[Section] = []
    cur = Section(slug, "", "", "")
    parts: list[str] = []

    def flush():
        cur.text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if cur.text or cur.title:
            out.append(cur)

    k = 0
    while k < len(tokens):
        tok = tokens[k]
        if tok.type == "heading_open":
            flush()
            cur = Section(slug, tok.attrGet("id") or "", tokens[k + 1].content, "")
            parts = []
            k += 2
            continue
        if tok.type == "inline":
            # для поиска и отрывков — текст без разметки: **жирный**, `код`, [ссылка](адрес)
            text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", tok.content)
            parts.append(re.sub(r"[*_`]+", "", text))
        k += 1
    flush()
    return out


def _lemmas(text: str) -> list[str]:
    from app.nlp.parsers import _morph

    morph = _morph()
    return [morph.parse(w)[0].normal_form for w in re.findall(r"[A-Za-zА-Яа-яЁё0-9²³]+", text.lower().replace("ё", "е"))]


@lru_cache(maxsize=1)
def articles() -> dict[str, Article]:
    sources: dict[str, str] = {}
    for p in sorted(HELP_DIR.glob("*.md")):
        sources[p.stem] = p.read_text(encoding="utf-8")
    for slug, fname in DOC_ARTICLES.items():
        p = DOCS_DIR / fname
        if p.exists():
            sources[slug] = p.read_text(encoding="utf-8")
    out = {}
    for slug in sorted(sources, key=lambda s: (ORDER.index(s) if s in ORDER else 99, s)):
        art = render(slug, sources[slug])
        for sec in art.sections:
            counts: dict[str, int] = {}
            for lem in _lemmas(sec.title + " " + sec.text):
                counts[lem] = counts.get(lem, 0) + 1
            sec.lemmas = counts
            sec.title_lemmas = set(_lemmas(sec.title))
        out[slug] = art
    return out


@lru_cache(maxsize=1)
def glossary() -> dict[str, dict]:
    """{id: {"title", "short", "url"}} из статей глоссария (заголовки с {#id})."""
    src = (HELP_DIR / "glossary.md").read_text(encoding="utf-8") if (HELP_DIR / "glossary.md").exists() else ""
    terms = {}
    for m in re.finditer(r"^###\s+(.+?)\s*\{#([\w-]+)\}\s*\n(.*?)(?=\n\s*\n|\Z)", src, re.MULTILINE | re.DOTALL):
        title, tid, para = m.group(1), m.group(2), m.group(3)
        terms[tid] = {"title": title, "short": re.sub(r"\s+", " ", para).strip(), "url": f"/help/glossary#{tid}"}
    return terms


def search(query: str, limit: int = 20) -> list[dict]:
    q = [lem for lem in _lemmas(query) if len(lem) > 1]
    if not q:
        return []
    results = []
    for art in articles().values():
        for sec in art.sections:
            hit = [lem for lem in q if lem in sec.lemmas]
            if not hit:
                continue
            score = len(set(hit)) * 10 + sum(sec.lemmas[lem] for lem in hit) + 15 * len(set(hit) & sec.title_lemmas)
            results.append({"article": art.slug, "article_title": art.title, "anchor": sec.anchor,
                            "title": sec.title or art.title, "snippet": _snippet(sec.text, query), "score": score})
    results.sort(key=lambda r: -r["score"])
    return results[:limit]


def _snippet(text: str, query: str, width: int = 220) -> str:
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    low = text.lower()
    pos = min((low.find(w[: max(3, len(w) - 2)]) for w in words if low.find(w[: max(3, len(w) - 2)]) >= 0), default=0)
    start = max(0, pos - width // 3)
    s = text[start : start + width]
    return ("…" if start else "") + s + ("…" if start + width < len(text) else "")


def term_html(tid: str) -> str:
    """Значок «?» с подсказкой из глоссария (для шаблонов: {{ term('ipm') }})."""
    t = glossary().get(tid)
    if not t:
        return ""
    return (f'<a class="term-q" href="{t["url"]}" data-term="{html.escape(tid)}" tabindex="0" '
            f'aria-label="{html.escape(t["title"])}: что это">?</a>')
