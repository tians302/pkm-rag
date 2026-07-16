"""Phase 2, step 2: contrastive fine-tuning of the multilingual embedder.

Uses MultipleNegativesRankingLoss (in-batch negatives) on the cross-lingual
pairs from make_pairs.py. In-batch negatives are other Pokemon, which is
exactly the discrimination we want the model to learn.

Base model default is paraphrase-multilingual-MiniLM-L12-v2 (fast, fits on
a laptop GPU or even CPU); swap in BAAI/bge-m3 with --model for the strong
version (needs a real GPU, e.g. one CARC node).

Run:
    python finetune/make_pairs.py
    python finetune/train_embedder.py --epochs 2 --batch-size 64
    python eval/eval_crosslingual.py --model runs/pkm-embedder   # compare
"""

import argparse
import json
from pathlib import Path

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default="sentence-transformers/"
                           "paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--out", default="runs/pkm-embedder")
    args = p.parse_args()

    from sentence_transformers import (InputExample, SentenceTransformer,
                                       losses)
    from torch.utils.data import DataLoader

    examples = [InputExample(texts=[r["query"], r["positive"]])
                for r in map(json.loads,
                             open(PROC / "train_pairs.jsonl",
                                  encoding="utf-8"))]
    print(f"{len(examples)} training pairs")

    model = SentenceTransformer(args.model)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size,
                        drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    steps = len(loader) * args.epochs
    model.fit(train_objectives=[(loader, loss)],
              epochs=args.epochs,
              warmup_steps=int(steps * args.warmup_frac),
              optimizer_params={"lr": args.lr},
              output_path=args.out,
              show_progress_bar=True)
    print(f"Saved fine-tuned model -> {args.out}")


if __name__ == "__main__":
    main()
