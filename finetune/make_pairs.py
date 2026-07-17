"""Phase 2, step 1: build cross-lingual (query, positive) training pairs.

Pair types (all linked via species_id, so labels are free):
  1. name-bridge:  "皮卡丘"  <->  the English document for that species
  2. doc-bridge:   Japanese document <-> English document (same species)
  3. query-style:  templated question in lang A <-> document in lang B

Output: data/processed/train_pairs.jsonl  {"query": ..., "positive": ...}
Held-out species go to eval_species.json for the retrieval eval.
"""

import json
import random
from pathlib import Path

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
random.seed(42)

QUERY_TEMPLATES = {
    "en": ["Tell me about {n}", "What type is {n}?", "{n} Pokedex entry",
           "What are {n}'s abilities?"],
    "ja": ["{n}について教えて", "{n}のタイプは？", "{n}の図鑑説明",
           "{n}の特性は？"],
    "zh-hans": ["介绍一下{n}", "{n}是什么属性？", "{n}的图鉴说明",
                "{n}的特性是什么？"],
    "zh-hant": ["介紹一下{n}", "{n}是什麼屬性？", "{n}的圖鑑說明",
                "{n}的特性是什麼？"],
}


def main(eval_frac: float = 0.1) -> None:
    docs = {}                            # (sid, lang) -> text
    for line in open(PROC / "corpus.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d.get("kind", "species") != "species":
            continue                     # type-chart docs are not pair material
        docs[(d["species_id"], d["lang"])] = d["text"]
    names = {int(k): v for k, v in
             json.load(open(PROC / "names_link.json",
                            encoding="utf-8")).items()}

    sids = sorted({sid for sid, _ in docs})
    random.shuffle(sids)
    n_eval = max(1, int(len(sids) * eval_frac))
    eval_sids, train_sids = set(sids[:n_eval]), sids[n_eval:]
    json.dump(sorted(eval_sids), open(PROC / "eval_species.json", "w"))

    langs = ["en", "ja", "zh-hans", "zh-hant"]
    n = 0
    with open(PROC / "train_pairs.jsonl", "w", encoding="utf-8") as out:
        def emit(q, p):
            nonlocal n
            out.write(json.dumps({"query": q, "positive": p},
                                 ensure_ascii=False) + "\n")
            n += 1

        for sid in train_sids:
            nm = names.get(sid, {})
            for la in langs:
                if (sid, la) not in docs:
                    continue
                # 1. name in lang A -> doc in lang B
                for lb in langs:
                    if lb != la and la in nm and (sid, lb) in docs:
                        emit(nm[la], docs[(sid, lb)])
                # 2. doc bridge (one random cross-lingual doc pair)
                others = [lb for lb in langs
                          if lb != la and (sid, lb) in docs]
                if others:
                    emit(docs[(sid, la)],
                         docs[(sid, random.choice(others))])
                # 3. templated query in lang A -> doc in random lang B
                if la in nm and others:
                    tmpl = random.choice(QUERY_TEMPLATES[la])
                    emit(tmpl.format(n=nm[la]),
                         docs[(sid, random.choice(others))])
    print(f"{n} pairs -> {PROC / 'train_pairs.jsonl'}")
    print(f"{len(eval_sids)} held-out species -> "
          f"{PROC / 'eval_species.json'}")


if __name__ == "__main__":
    main()
