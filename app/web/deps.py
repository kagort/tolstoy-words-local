"""Общие вещи для веб-слоя: шаблоны, вход, сообщения."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import Request
from fastapi.templating import Jinja2Templates

from app import __version__
from app.config import settings
from app.nlp.morph import POS_LABELS, pos_label
from app.nlp.syntax import REL_LABELS, ROLE_LABELS, SENT_TYPE_LABELS, rel_label_ud, role_label_ud

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

SOURCE_TYPES = {"natural": "литература / естественный", "generated": "сгенерирован нейросетью", "other": "другое"}
STATUS_LABELS = {
    "uploaded": "загружен", "queued": "в очереди", "processing": "разбирается", "ready": "готов", "error": "ошибка",
    "running": "выполняется", "done": "готово",
}
SCOPE_LABELS = {"direct": "прямая связь", "shared": "общий член ряда", "uncertain": "спорно", "private": "частный"}


def _qs(request: Request, **changes) -> str:
    params = dict(request.query_params)
    for k, v in changes.items():
        if v is None or v == "":
            params.pop(k, None)
        else:
            params[k] = v
    return "?" + urlencode(params)


templates.env.globals.update(
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
)


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
