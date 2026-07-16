"""FastAPI backend for the multilingual Pokemon RAG system,
with the Gradio chat UI mounted at / as a temporary frontend.

Run:
    python server.py                          # sparse BM25 + extractive
    python server.py --retriever dense        # BGE-M3
    python server.py --backend ollama         # answers via local Ollama
    # or directly:  uvicorn server:app  (configure via PKMRAG_* env vars)

Then:
    UI:       http://localhost:8000
    API:      http://localhost:8000/api/ask   (POST)
    Docs:     http://localhost:8000/docs      (auto-generated OpenAPI)

Example:
    curl -X POST localhost:8000/api/ask \
         -H 'Content-Type: application/json' \
         -d '{"question": "皮卡丘的属性和特性是什么？", "k": 4}'
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pkmrag.rag import RAGPipeline
from pkmrag.retrieval_sparse import SparseRetriever, detect_lang

CORPUS = Path(__file__).parent / "data" / "processed" / "corpus.jsonl"

EXAMPLES = [
    "What is Pikachu called in Japanese and Chinese?",
    "皮卡丘的属性和特性是什么？",
    "フシギダネの図鑑説明を教えて",
    "Which Pokemon is the Flame Pokemon?",
    "妙蛙种子的英文名字是什么？",
]

# --------------------------------------------------------------------------
# configuration (env vars so `uvicorn server:app` works too)
# --------------------------------------------------------------------------

CONFIG = {
    "retriever": os.environ.get("PKMRAG_RETRIEVER", "sparse"),
    "backend": os.environ.get("PKMRAG_BACKEND", "extractive"),
    "embedder": os.environ.get("PKMRAG_EMBEDDER", "BAAI/bge-m3"),
    "ollama_model": os.environ.get("PKMRAG_OLLAMA_MODEL", "qwen2.5:7b"),
    "k": int(os.environ.get("PKMRAG_K", "4")),
}

# pipelines share one retriever; keyed by generation backend so a request
# can override the default backend cheaply (e.g. try ollama vs extractive)
_STATE: dict = {"retriever": None, "pipelines": {}}


def _get_pipeline(backend: str | None = None) -> RAGPipeline:
    backend = backend or CONFIG["backend"]
    if backend not in ("extractive", "ollama", "transformers"):
        raise HTTPException(422, f"unknown backend: {backend!r}")
    # loading a HF model on a per-request basis would be too slow
    if backend == "transformers" and CONFIG["backend"] != "transformers":
        raise HTTPException(
            422, "transformers backend must be enabled at startup "
                 "(--backend transformers)")
    pipes = _STATE["pipelines"]
    if backend not in pipes:
        pipes[backend] = RAGPipeline(
            _STATE["retriever"], backend=backend,
            ollama_model=CONFIG["ollama_model"])
    return pipes[backend]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CORPUS.exists():
        raise RuntimeError(
            f"corpus not found at {CORPUS} -- run "
            "`python scripts/download_data.py && python -m pkmrag.build_corpus`")
    if CONFIG["retriever"] == "dense":
        from pkmrag.retrieval_dense import DenseRetriever
        _STATE["retriever"] = DenseRetriever(
            CORPUS, model_name=CONFIG["embedder"])
    else:
        _STATE["retriever"] = SparseRetriever(CORPUS)
    _get_pipeline()  # warm the default pipeline
    yield
    _STATE["pipelines"].clear()
    _STATE["retriever"] = None


app = FastAPI(title="pkm-rag API",
              description="Multilingual Pokemon RAG (EN / 日本語 / 中文)",
              version="0.1.0", lifespan=lifespan)

# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=[EXAMPLES[0]])
    k: int = Field(default=None, ge=1, le=20,
                   description="number of documents to retrieve")
    backend: str | None = Field(
        default=None,
        description="override generation backend for this request: "
                    "'extractive' | 'ollama'")


class RetrievedDoc(BaseModel):
    id: str
    name: str
    lang: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    query_lang: str
    retriever: str
    backend: str
    docs: list[RetrievedDoc]


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    r = _STATE["retriever"]
    return {"status": "ok" if r is not None else "starting",
            "corpus_docs": len(getattr(r, "docs", []) or []) if r else 0}


@app.get("/api/config")
def config() -> dict:
    return {**CONFIG, "examples": EXAMPLES}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    pipe = _get_pipeline(req.backend)
    result = pipe.answer(req.question, k=req.k or CONFIG["k"])
    return AskResponse(
        answer=result["answer"],
        query_lang=detect_lang(req.question),
        retriever=CONFIG["retriever"],
        backend=req.backend or CONFIG["backend"],
        docs=[RetrievedDoc(**{k: d[k] for k in
                              ("id", "name", "lang", "score", "text")})
              for d in result["docs"]],
    )


# --------------------------------------------------------------------------
# temporary Gradio frontend, mounted at /
# --------------------------------------------------------------------------


def _chat(message: str, history) -> str:
    result = _get_pipeline().answer(message, k=CONFIG["k"])
    sources = "\n".join(
        f"- `{d['id']}` {d['name']} ({d['lang']}, score {d['score']})"
        for d in result["docs"])
    return f"{result['answer']}\n\n---\n**Sources**\n{sources}"


demo = gr.ChatInterface(
    fn=_chat,
    title="Multilingual Pokemon RAG (EN / 日本語 / 中文)",
    description=(f"Retriever: **{CONFIG['retriever']}** | "
                 f"Generator: **{CONFIG['backend']}** — fully local. "
                 "API docs at [/docs](/docs)."),
    examples=EXAMPLES,
)

app = gr.mount_gradio_app(app, demo, path="")


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--retriever", choices=["sparse", "dense"],
                   default=CONFIG["retriever"])
    p.add_argument("--embedder", default=CONFIG["embedder"])
    p.add_argument("--backend",
                   choices=["extractive", "ollama", "transformers"],
                   default=CONFIG["backend"])
    p.add_argument("--ollama-model", default=CONFIG["ollama_model"])
    p.add_argument("--k", type=int, default=CONFIG["k"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    CONFIG.update(retriever=args.retriever, embedder=args.embedder,
                  backend=args.backend, ollama_model=args.ollama_model,
                  k=args.k)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
