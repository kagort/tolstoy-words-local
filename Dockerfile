FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-stanza.txt ./
RUN pip install -r requirements.txt

# NLP_BACKEND=stanza (по умолчанию) — ставим PyTorch для CPU и Stanza. Сами модели (~600 МБ) в образ НЕ скачиваются:
# их при первом запуске скачивает обработчик в том Docker «stanza_models» (с повторами и докачкой) — см. app/services/models.py.
# Кэш huggingface_hub (HF_HOME) тоже в томе — недокачанные файлы переживают перезапуск контейнера.
# NLP_BACKEND=natasha — лёгкий образ без PyTorch.
ARG NLP_BACKEND=stanza
ENV NLP_BACKEND=${NLP_BACKEND} \
    STANZA_DIR=/models/stanza \
    HF_HOME=/models/stanza/.hf \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    HF_HUB_ETAG_TIMEOUT=30 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10
RUN if [ "$NLP_BACKEND" = "stanza" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install -r requirements-stanza.txt; \
    fi

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY docs ./docs
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd --create-home --uid 1000 tw && mkdir -p /models/stanza && chown -R tw /app /models
USER tw

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["web"]
