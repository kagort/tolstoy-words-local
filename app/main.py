"""Веб-приложение: uvicorn app.main:app"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app import __version__
from app.config import settings
from app.web import export, pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
PUBLIC_PATHS = ("/login", "/static/", "/static-vendor/", "/healthz", "/favicon.ico")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith(PUBLIC_PATHS) and not request.session.get("user"):
            if path.startswith("/api/") or path.startswith("/export/"):
                return JSONResponse({"error": "Требуется вход"}, status_code=401)
            return RedirectResponse(f"/login?next={request.url.path}", status_code=303)
        return await call_next(request)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.site_title, version=__version__, docs_url=None, redoc_url=None)
    secret = settings.secret_key
    if secret in ("", "change-me"):
        secret = secrets.token_urlsafe(32)
        log.warning("SECRET_KEY не задан: сгенерирован временный ключ (после перезапуска придётся войти заново)")
    # порядок важен: сначала добавляем Auth, затем Session — Session будет внешним слоем
    app.add_middleware(AuthMiddleware)
    app.add_middleware(SessionMiddleware, secret_key=secret, same_site="lax", https_only=os.getenv("HTTPS_ONLY") == "1",
                       max_age=14 * 24 * 3600)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/static-vendor/plotly.min.js", include_in_schema=False)
    def plotly_js():
        import plotly

        path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        return FileResponse(path, media_type="application/javascript", headers={"Cache-Control": "public, max-age=604800"})

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok", "version": __version__}

    app.include_router(pages.router)
    app.include_router(export.router)
    return app


app = create_app()
