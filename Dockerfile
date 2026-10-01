# Hugging Face Space (Docker SDK) running the API in server.py on CPU.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-space.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements-space.txt

COPY --chown=user . .
USER user

# Bake the weights into the image so cold starts skip the download.
RUN python -m scripts.download_models

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
