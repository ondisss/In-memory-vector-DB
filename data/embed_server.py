#!/usr/bin/env python3
import sys
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)
print("READY", flush=True)


while True:
    line = sys.stdin.readline()
    if not line:  
        break
    text = line.rstrip("\n")
    if not text:
        continue
    emb = model.encode(text, normalize_embeddings=True)
    print(" ".join(f"{x:.8f}" for x in emb), flush=True)
