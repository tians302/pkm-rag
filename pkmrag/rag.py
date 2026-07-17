"""RAG pipeline: retrieve -> (optionally) generate with a local LLM.

Generation backends (all local, no cloud APIs):
  * "ollama"       -- calls a local Ollama server (default model qwen2.5:7b,
                      which is natively strong in en/ja/zh)
  * "transformers" -- loads a HF chat model in-process
                      (default Qwen/Qwen2.5-1.5B-Instruct for CPU-friendliness)
  * "extractive"   -- no LLM: returns the top passages verbatim
                      (used in environments with no model access)
"""

import json
import urllib.request

from .retrieval_sparse import detect_lang

SYSTEM = ("You are a helpful Pokedex assistant. Answer using ONLY the "
          "provided context documents. Answer in the same language as the "
          "question. If the context is insufficient, say so.")

PROMPT = ("Context documents:\n{context}\n\n"
          "Question: {question}\n\n"
          "Answer (same language as the question, cite species names):")


def _format_context(docs: list[dict]) -> str:
    return "\n\n".join(f"[{i+1}] ({d['lang']}) {d['text']}"
                       for i, d in enumerate(docs))


# --------------------------------------------------------------------------
# generation backends
# --------------------------------------------------------------------------

def generate_ollama(prompt: str, model: str = "qwen2.5:7b",
                    host: str = "http://localhost:11434") -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["message"]["content"]


class TransformersGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto",
            device_map=device or "auto")

    def __call__(self, prompt: str, max_new_tokens: int = 512) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        return self.tok.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True)


EXTRACTIVE_HEADER = {
    "en": "No local LLM configured -- top retrieved passages:",
    "ja": "ローカルLLM未設定のため、検索上位の文書を表示します:",
    "zh": "未配置本地LLM，以下为检索到的最相关文档:",
}


def generate_extractive(question: str, docs: list[dict]) -> str:
    head = EXTRACTIVE_HEADER.get(detect_lang(question),
                                 EXTRACTIVE_HEADER["en"])
    body = "\n\n".join(f"[{i+1}] {d['name']} ({d['lang']}, "
                       f"score {d['score']}):\n{d['text']}"
                       for i, d in enumerate(docs))
    return f"{head}\n\n{body}"


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, retriever, backend: str = "extractive",
                 ollama_model: str = "qwen2.5:7b",
                 hf_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 rewriter=None):
        self.retriever = retriever
        self.backend = backend
        self.ollama_model = ollama_model
        self.rewriter = rewriter    # optional cross-lingual query rewriter
        self._hf = TransformersGenerator(hf_model) \
            if backend == "transformers" else None

    def _retrieve(self, question: str, k: int) -> list[dict]:
        docs = self.retriever.search(question, k=k)
        if self.rewriter is None:
            return docs
        rq = self.rewriter.rewrite(question)
        if not rq:
            return docs
        from pkmrag.query_rewrite import fuse
        return fuse(docs, self.retriever.search(rq, k=k), k)

    def answer(self, question: str, k: int = 4) -> dict:
        docs = self._retrieve(question, k)
        if not docs:
            return {"answer": "No relevant documents found.", "docs": []}
        prompt = PROMPT.format(context=_format_context(docs),
                               question=question)
        if self.backend == "ollama":
            try:
                ans = generate_ollama(prompt, model=self.ollama_model)
            except Exception as e:                     # server not running
                ans = (f"[Ollama unavailable: {e}]\n\n"
                       + generate_extractive(question, docs))
        elif self.backend == "transformers":
            ans = self._hf(prompt)
        else:
            ans = generate_extractive(question, docs)
        return {"answer": ans, "docs": docs}
