"""Общие вещи для веб-слоя: шаблоны, вход, сообщения."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app import __version__
from app.config import settings
from app.nlp.morph import POS_LABELS, pos_label
from app.nlp.syntax import REL_LABELS, ROLE_LABELS, SENT_TYPE_LABELS, rel_label_ud, role_label_ud

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

SOURCE_TYPES = {"natural": "литература / естественный", "generated": "сгенерирован нейросетью", "other": "другое"}
STATUS_LABELS = {
    "uploaded": "загружен", "queued": "в очереди", "processing": "разбирается", "ready": "готов", "error": "ошибка",
    "running": "выполняется", "done": "готово", "cancelled": "отменена", "deleting": "удаляется",
}
SCOPE_LABELS = {"direct": "прямая связь", "shared": "общий член ряда", "uncertain": "спорно", "private": "частный"}


def _qs(request: Request, **changes) -> str:
    # повторяющиеся параметры (authors=…&authors=…, texts, words) сохраняются все — dict() оставил бы последний
    params = [(k, v) for k, v in request.query_params.multi_items() if k not in changes]
    params += [(k, v) for k, v in changes.items() if v is not None and v != ""]
    return "?" + urlencode(params)


def _asset_version() -> str:
    """Метка версии для /static/app.js и app.css: меняется при любом изменении файлов,
    поэтому браузер после обновления программы не берёт старые скрипты из кэша."""
    import hashlib

    h = hashlib.md5(__version__.encode())
    for name in ("app.js", "app.css", "v2.css", "fonts.css", "charts_v2.js", "dash_v2.js"):
        f = TEMPLATES_DIR.parent / "static" / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


# Линейные значки для кнопок и ссылок (вместо эмодзи): тот же стиль, что у значков бокового меню (base_v2.html).
ICONS = {
    "tree": "M12 3v5M5 14v-2a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v2M12 10v4M4 14h2v4H4zM11 14h2v4h-2zM18 14h2v4h-2zM10 3h4v3h-4z",
    "trash": "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
    "chart": "M4 20V10M10 20V4M16 20v-7M22 20H2",
    "report": "M9 3h6v3H9zM7 4.5H5V21h14V4.5h-2M8 11h8M8 15h6",
    "book": "M5 4h11a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3zM5 17a3 3 0 0 1 3-3h11",
    "folder": "M3 6h6l2 2h10v11H3zM3 10h18",
    "download": "M12 4v11M7 10l5 5 5-5M5 20h14",
    "print": "M7 9V3h10v6M7 17H4v-7h16v7h-3M7 14h10v7H7z",
    "sprout": "M12 21v-9M12 12c0-4 3-6.5 7.5-6.5 0 4-3 6.5-7.5 6.5zM12 14.5c0-3-2.2-5.5-6.5-5.5 0 3 2.2 5.5 6.5 5.5z",
    "tag": "M3 12V4h8l10 10-8 8zM7.5 8.5h.01",
    "filter": "M3 5h18l-7 8v6l-4 2v-8z",
}


def icon(name: str, title: str = "") -> Markup:
    """SVG-значок 1em в цвет текста; title — подсказка (иначе значок скрыт от экранного чтения)."""
    d = ICONS.get(name, "")
    a11y = f'role="img" aria-label="{title}"' if title else 'aria-hidden="true"'
    return Markup(f'<svg class="ic" viewBox="0 0 24 24" {a11y}><path d="{d}" fill="none" stroke="currentColor" '
                  'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>')


UI_MODES = ("v2", "classic")


def ui_mode(request: Request | None) -> str:
    """Какой интерфейс показывать: выбор пользователя (cookie qv_ui), иначе настройка UI_DEFAULT."""
    choice = request.cookies.get("qv_ui") if request is not None else None
    if choice in UI_MODES:
        return choice
    return settings.ui_default if settings.ui_default in UI_MODES else "v2"


templates.env.globals.update(
    asset_version=_asset_version(),
    ui_mode=ui_mode,
    icon=icon,
    site_title=settings.site_title,
    app_version=__version__,
    ROLE_LABELS=ROLE_LABELS,
    REL_LABELS=REL_LABELS,
    POS_LABELS=POS_LABELS,
    SOURCE_TYPES=SOURCE_TYPES,
    STATUS_LABELS=STATUS_LABELS,
    SCOPE_LABELS=SCOPE_LABELS,
    SENT_TYPE_LABELS=SENT_TYPE_LABELS,
    qs=_qs,
    pos_label=pos_label,
    role_label=role_label_ud,
    rel_label=rel_label_ud,
    term=lambda tid: Markup(_term_html(tid)),
)


def _term_html(tid: str) -> str:
    from app.services.help import term_html

    return term_html(tid)


def flash(request: Request, message: str, kind: str = "info") -> None:
    request.session.setdefault("flash", []).append({"message": message, "kind": kind})


def pop_flash(request: Request) -> list[dict]:
    return request.session.pop("flash", [])


def render(request: Request, name: str, **ctx):
    ctx.setdefault("user", request.session.get("user"))
    ctx["flashes"] = pop_flash(request)
    ctx["request"] = request
    return templates.TemplateResponse(request, name, ctx)


def current_user(request: Request) -> str | None:
    return request.session.get("user")
