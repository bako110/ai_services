FROM python:3.11-slim

# ffmpeg requis par app/pipelines/video_pipeline.py (extraction frames/audio).
# build-essential/cmake : filet de securite pour llama-cpp-python si jamais
# aucune wheel precompilee ne correspond a cette plateforme malgre l'index
# dedie dans requirements.txt — sans ca, pip install echouerait purement et
# simplement (pas de gcc/cmake sur une image slim).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
