"""Gradio chat UI for the multilingual Pokemon RAG system.

Run locally:
    python app.py                          # sparse BM25 + extractive
    python app.py --retriever dense        # BGE-M3 (downloads model once)
    python app.py --backend ollama         # answers via local Ollama LLM
    python app.py --retriever dense --backend ollama   # full setup

Then open http://localhost:7860
"""

import argparse
from pathlib import Path

import gradio as gr

from pkmrag.rag import RAGPipeline
from pkmrag.retrieval_sparse import SparseRetriever

CORPUS = Path(__file__).parent / "data" / "processed" / "corpus.jsonl"

EXAMPLES = [
    "What is Pikachu called in Japanese and Chinese?",
    "皮卡丘的属性和特性是什么？",
    "フシギダネの図鑑説明を教えて",
    "Which Pokemon is the Flame Pokemon?",
    "妙蛙种子的英文名字是什么？",
]


def build_pipeline(args) -> RAGPipeline:
    if args.retriever == "dense":
        from pkmrag.retrieval_dense import DenseRetriever
        retriever = DenseRetriever(CORPUS, model_name=args.embedder)
    else:
        retriever = SparseRetriever(CORPUS)
    return RAGPipeline(retriever, backend=args.backend,
                       ollama_model=args.ollama_model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--retriever", choices=["sparse", "dense"],
                   default="sparse")
    p.add_argument("--embedder", default="BAAI/bge-m3",
                   help="dense embedder (or path to fine-tuned checkpoint)")
    p.add_argument("--backend",
                   choices=["extractive", "ollama", "transformers"],
                   default="extractive")
    p.add_argument("--ollama-model", default="qwen2.5:7b")
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()

    pipe = build_pipeline(args)

    def chat(message, history):
        result = pipe.answer(message, k=args.k)
        sources = "\n".join(
            f"- `{d['id']}` {d['name']} ({d['lang']}, score {d['score']})"
            for d in result["docs"])
        return f"{result['answer']}\n\n---\n**Sources**\n{sources}"

    demo = gr.ChatInterface(
        fn=chat,
        title="Multilingual Pokemon RAG (EN / 日本語 / 中文)",
        description=(f"Retriever: **{args.retriever}** | "
                     f"Generator: **{args.backend}** — fully local."),
        examples=EXAMPLES,
    )
    demo.launch()


if __name__ == "__main__":
    main()
