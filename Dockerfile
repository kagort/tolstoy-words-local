FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-stanza.txt ./
RUN pip install -r requirements.txt

# NLP_BACKEND=stanza (по умолчанию) — ставим PyTorch для CPU, Stanza и скачиваем русскую модель в образ.
# NLP_BACKEND=natasha — лёгкий образ без PyTorch.
ARG NLP_BACKEND=stanza
ENV NLP_BACKEND=${NLP_BACKEND} \
    STANZA_DIR=/models/stanza
RUN if [ "$NLP_BACKEND" = "stanza" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install -r requirements-stanza.txt && \
        python -c "import stanza; stanza.download('ru', model_dir='/models/stanza', processors='tokenize,pos,lemma,depparse')"; \
    fi

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd --create-home --uid 1000 tw && chown -R tw /app
USER tw

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["web"]
