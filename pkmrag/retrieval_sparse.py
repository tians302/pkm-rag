"""Sparse BM25 retrieval with CJK-aware tokenization.

No model downloads needed -- this is the fallback / demo retriever.
Latin script is tokenized into lowercase words; Japanese/Chinese spans are
tokenized into character unigrams + bigrams (a standard trick that makes
BM25 work reasonably well for CJK without a segmenter).
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CJK = (r"\u3040-\u30ff"      # hiragana + katakana
       r"\u4e00-\u9fff"      # CJK unified ideographs
       r"\u3400-\u4dbf")     # extension A
TOKEN_RE = re.compile(rf"([{CJK}]+)|([a-z0-9]+)")


def detect_lang(text: str) -> str:
    """Cheap script-based language guess: en / ja / zh."""
    if re.search(r"[\u3040-\u30ff]", text):        # kana => Japanese
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):        # han only => Chinese
        return "zh"
    return "en"


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for cjk, latin in TOKEN_RE.findall(text.lower()):
        if latin:
            tokens.append(latin)
        else:
            tokens.extend(cjk)                                   # unigrams
            tokens.extend(cjk[i:i + 2] for i in range(len(cjk) - 1))  # bigrams
    return tokens


class SparseRetriever:
    NAME_WEIGHT = 4    # repeat name tokens: title field > body field
    ALIAS_BOOST = 2.0  # exact name match in query (any language) doubles
    # the score of that species' docs. Without this, "Pikachu" retrieves
    # Mimikyu, whose Pokedex entries mention Pikachu more often than
    # Pikachu's own entries do (BM25 TF saturation makes the name-field
    # weighting alone insufficient).

    def __init__(self, corpus_path: str | Path):
        corpus_path = Path(corpus_path)
        self.docs = [json.loads(line)
                     for line in open(corpus_path, encoding="utf-8")]
        self.bm25 = BM25Okapi(
            [tokenize(d["name"]) * self.NAME_WEIGHT + tokenize(d["text"])
             for d in self.docs])

        # alias -> species_ids, from the translation linkage table
        self.aliases: dict[str, set[int]] = {}
        link = corpus_path.parent / "names_link.json"
        if link.exists():
            for sid, nm in json.load(open(link, encoding="utf-8")).items():
                for name in nm.values():
                    self.aliases.setdefault(name.lower(), set()).add(int(sid))

    def _matched_species(self, query: str) -> set[int]:
        q = query.lower()
        return {sid for alias, sids in self.aliases.items()
                if alias in q for sid in sids}

    def search(self, query: str, k: int = 5,
               lang_boost: float = 1.15) -> list[dict]:
        """Top-k docs; boosts for exact name matches + query language."""
        qlang = detect_lang(query)
        named = self._matched_species(query)
        scores = self.bm25.get_scores(tokenize(query))
        boosted = []
        for doc, s in zip(self.docs, scores):
            if doc["species_id"] in named:
                s *= self.ALIAS_BOOST
            dl = doc["lang"]
            match = (dl == qlang or
                     (qlang == "zh" and dl.startswith("zh")))
            boosted.append(s * lang_boost if match else s)
        order = sorted(range(len(boosted)),
                       key=lambda i: boosted[i], reverse=True)[:k]
        return [{**self.docs[i], "score": round(boosted[i], 3)}
                for i in order if boosted[i] > 0]
