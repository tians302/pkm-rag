"""Cross-lingual retrieval evaluation: recall@k and MRR.

Task: given a Pokemon's name in language A, retrieve any document of the
correct species in language B (A != B), over held-out species from
make_pairs.py (falls back to all species if no split file exists).

Works with both retrievers, so you get three comparable numbers:
    python eval/eval_crosslingual.py --retriever sparse
    python eval/eval_crosslingual.py --retriever dense                 # base
    python eval/eval_crosslingual.py --retriever dense --model runs/pkm-embedder
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))

LANGS = ["en", "ja", "zh-hans", "zh-hant"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--retriever", choices=["sparse", "dense"],
                   default="sparse")
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    if args.retriever == "dense":
        from pkmrag.retrieval_dense import DenseRetriever
        index = PROC / f"index_{Path(args.model).name.replace('/', '_')}.npy"
        retriever = DenseRetriever(PROC / "corpus.jsonl",
                                   model_name=args.model, index_path=index)
    else:
        from pkmrag.retrieval_sparse import SparseRetriever
        retriever = SparseRetriever(PROC / "corpus.jsonl")

    names = {int(k): v for k, v in
             json.load(open(PROC / "names_link.json",
                            encoding="utf-8")).items()}
    split = PROC / "eval_species.json"
    sids = json.load(open(split)) if split.exists() else sorted(names)
    doc_langs = defaultdict(set)
    for d in retriever.docs:
        doc_langs[d["species_id"]].add(d["lang"])

    hits = defaultdict(int)
    mrr = defaultdict(float)
    total = defaultdict(int)
    for sid in sids:
        for la in LANGS:
            if la not in names.get(sid, {}):
                continue
            targets = doc_langs[sid] - {la}
            if not targets:
                continue
            total[la] += 1
            results = retriever.search(names[sid][la], k=args.k) \
                if args.retriever == "dense" \
                else retriever.search(names[sid][la], k=args.k,
                                      lang_boost=1.0)  # no boost: pure x-ling
            for rank, r in enumerate(results, 1):
                if r["species_id"] == sid and r["lang"] != la:
                    hits[la] += 1
                    mrr[la] += 1 / rank
                    break

    print(f"\nCross-lingual name->doc retrieval "
          f"({args.retriever}, k={args.k}, {len(sids)} species)")
    print(f"{'query lang':<10} {'recall@k':>9} {'MRR':>7} {'n':>6}")
    for la in LANGS:
        if total[la]:
            print(f"{la:<10} {hits[la] / total[la]:>9.3f} "
                  f"{mrr[la] / total[la]:>7.3f} {total[la]:>6}")
    n = sum(total.values())
    print(f"{'ALL':<10} {sum(hits.values()) / n:>9.3f} "
          f"{sum(mrr.values()) / n:>7.3f} {n:>6}")


if __name__ == "__main__":
    main()
