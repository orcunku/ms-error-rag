FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from fastembed import TextEmbedding; import json, numpy as np; chunks = json.load(open('data/chunks.json')); m = TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); np.save('data/embeddings.npy', np.array(list(m.embed([c['text'] for c in chunks]))))"
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

CMD ["python", "app.py"]