"""Dictionary-based cross-lingual query rewriting (CLIR).

Some corpus domains only exist in English (competitive sets). A Chinese or
Japanese battle question can't match them lexically, and even a fine-tuned
multilingual embedder ranks same-language species docs above them. But we
own an exact entity dictionary for free: names_link.json plus the alias
fields on move/ability/type docs map every CJK entity name to English,
keyed by database id -- no machine translation involved.

rewrite() swaps entities for their English names, maps a small glossary of
battle vocabulary, and drops the grammar words retrieval never needed.
The result is meant as a SECOND retrieval channel fused with the original
query (see fuse()), not a replacement: same-language docs are still the
best answer for same-language dex questions.
"""

import json
import re
from pathlib import Path

_CJK_RE = re.compile(r"[぀-ヿ一-鿿㐀-䶿]")

# battle / dex vocabulary that isn't an entity name
GLOSSARY = {
    "配招": "moveset", "招式": "move", "技能": "move", "努力值": "EVs",
    "性格": "nature", "道具": "item", "特性": "ability",
    "对战": "competitive", "對戰": "competitive", "弱点": "weakness",
    "弱點": "weakness", "属性": "type", "屬性": "type", "进化": "evolution",
    "進化": "evolution", "种族值": "base stats", "種族值": "base stats",
    "太晶": "Tera", "分级": "tier", "分級": "tier",
    "育成論": "competitive set", "持ち物": "item", "わざ": "move",
    "とくせい": "ability", "型": "set", "調整": "EV spread",
    "たいせん": "competitive", "対戦": "competitive", "しんか": "evolution",
}

# CJK entity aliases that double as everyday words -- never treat as entities
BLOCKLIST = {"一般"}   # "usually" / zh-hant name of the Normal type


class QueryRewriter:
    def __init__(self, corpus_path: str | Path):
        corpus_path = Path(corpus_path)
        self.entities: dict[str, str] = {}
        link = corpus_path.parent / "names_link.json"
        if link.exists():
            for nm in json.load(open(link, encoding="utf-8")).values():
                en = nm.get("en")
                if en:
                    for lang, n in nm.items():
                        if lang != "en" and n not in BLOCKLIST:
                            self.entities[n] = en
        for line in open(corpus_path, encoding="utf-8"):
            d = json.loads(line)
            if d.get("aliases") and d["lang"] == "en":
                for a in d["aliases"]:
                    if _CJK_RE.search(a) and a not in BLOCKLIST:
                        self.entities[a] = d["name"]
        # longest match first, so 十万伏特 wins over 伏特
        self._order = sorted(self.entities, key=len, reverse=True)
        self._gloss = sorted(GLOSSARY, key=len, reverse=True)

    def rewrite(self, query: str) -> str:
        """English keyword query, or "" when rewriting adds nothing."""
        if not _CJK_RE.search(query):
            return ""
        out, hit_entity = [], False
        for k in self._order:
            if k in query:
                out.append(self.entities[k])
                query = query.replace(k, " ")
                hit_entity = True
        for k in self._gloss:
            if k in query:
                out.append(GLOSSARY[k])
                query = query.replace(k, " ")
        # keep any latin words the user already mixed in
        out += re.findall(r"[A-Za-z0-9]+", query)
        return " ".join(out) if hit_entity else ""


def fuse(primary: list[dict], secondary: list[dict], k: int) -> list[dict]:
    """Reciprocal-rank fusion of two result lists (dedup by doc id)."""
    C = 60.0
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in (primary, secondary):
        for rank, d in enumerate(results, 1):
            scores[d["id"]] = scores.get(d["id"], 0.0) + 1.0 / (C + rank)
            docs.setdefault(d["id"], d)
    order = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**docs[i], "score": round(scores[i], 4)} for i in order]
