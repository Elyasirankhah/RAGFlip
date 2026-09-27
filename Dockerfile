FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ci/cd/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt

COPY app.py start_server.py ./
COPY rag ./rag
COPY utils ./utils
COPY ui ./ui
COPY ci/cd/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PORT=8000
EXPOSE 8000

# API key is injected at runtime. Do not bake secrets into the image.
ENTRYPOINT ["/entrypoint.sh"]
