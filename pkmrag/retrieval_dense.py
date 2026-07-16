"""Dense multilingual retrieval using sentence-transformers.

Default model: BAAI/bge-m3 -- strong cross-lingual alignment for en/ja/zh,
so a Chinese query retrieves English documents and vice versa.
Runs on your machine / CARC (needs HuggingFace access + optionally a GPU).

Usage:
    r = DenseRetriever("data/processed/corpus.jsonl")   # builds/loads index
    r.search("皮卡丘的特性是什么", k=5)

A fine-tuned checkpoint from finetune/train_embedder.py can be passed as
`model_name` to compare against the base model.
"""

import json
from pathlib import Path

import numpy as np


class DenseRetriever:
    def __init__(self, corpus_path: str | Path,
                 model_name: str = "BAAI/bge-m3",
                 index_path: str | Path | None = None,
                 device: str | None = None,
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.docs = [json.loads(line)
                     for line in open(corpus_path, encoding="utf-8")]
        self.model = SentenceTransformer(model_name, device=device)

        index_path = Path(index_path) if index_path else \
            Path(corpus_path).with_suffix(".npy")
        if index_path.exists():
            self.emb = np.load(index_path)
            assert len(self.emb) == len(self.docs), \
                "Index/corpus mismatch -- delete the .npy and re-run."
        else:
            print(f"Encoding {len(self.docs)} docs with {model_name} ...")
            self.emb = self.model.encode(
                [d["text"] for d in self.docs],
                batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=True)
            np.save(index_path, self.emb)
            print(f"Saved index -> {index_path}")

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q                      # cosine (normalized)
        order = np.argsort(-scores)[:k]
        return [{**self.docs[i], "score": round(float(scores[i]), 4)}
                for i in order]
