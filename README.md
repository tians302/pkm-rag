# pkm-rag — Multilingual Pokemon RAG (EN / 日本語 / 中文)

A fully local retrieval-augmented question-answering system over the Pokedex,
with English, Japanese, Simplified Chinese, and Traditional Chinese linked at
the database level. No cloud APIs anywhere in the pipeline.

Ask in any of the three languages, get answers grounded in retrieved
Pokedex documents:

```
Q: 皮卡丘的属性和特性是什么？        -> 皮卡丘 doc (zh-hans)
Q: フシギダネの図鑑説明を教えて      -> フシギダネ doc (ja)
Q: What is Pikachu called in Japanese and Chinese?  -> Pikachu doc (en)
```

## Data

All data comes from the [PokeAPI database](https://github.com/PokeAPI/pokeapi)
(CSV tables, veekun-derived — the same data behind pokeapi.co):

- **1,025 species** with names + genus in `en`, `ja`, `ja-hrkt`, `zh-hans`,
  `zh-hant`, linked by species ID → `data/processed/names_link.json`
- **Pokedex flavor text**: ~14.5k EN / ~5.8k JA / ~2.9k each zh-Hans & zh-Hant
  entries (official Chinese text exists from Gen 7 onward)
- Localized **type and ability** names
- The **type-effectiveness matrix** (`type_efficacy.csv`)
- **~900 moves** with localized names, stats, and descriptions
  (`moves.csv`, `move_names.csv`, `move_flavor_text.csv`), plus level-up
  learnsets (`pokemon_moves.csv`)

The corpus builder emits **7,768 documents**:

- **4,100 species docs** (one per species × language): a localized fact card
  (names in all languages, genus, types, abilities, height/weight, base
  stats, a computed weakness/resistance line that accounts for dual typing,
  and the level-up learnset from that Pokemon's newest game version) plus
  deduplicated Pokedex entries.
- **3,596 move docs** (one per move × language): type, damage class,
  power/accuracy/PP, names in all languages, localized effect description.
- **72 type-chart docs** (18 types × 4 languages) phrased the way questions
  are asked ("the most effective attacks against Fire-type Pokemon are…") —
  small local LLMs answer reliably from that phrasing but flip the
  attack/defense direction when given chart-style wording ("Fire is weak
  to…").

## Architecture

```
CSVs ──build_corpus──> corpus.jsonl + names_link.json
                            │
              ┌─────────────┴──────────────┐
        SparseRetriever               DenseRetriever
        (BM25, CJK bigrams,           (BGE-M3, cross-lingual
         name-alias boost)             semantic retrieval)
              └─────────────┬──────────────┘
                       RAGPipeline
              ┌─────────────┼──────────────┐
          extractive      Ollama       transformers
          (no LLM)     (qwen2.5:7b)  (Qwen2.5-1.5B-Instruct)
                            │
                       Gradio chat UI
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/download_data.py
python -m pkmrag.build_corpus

# minimal demo (BM25 + extractive answers, zero model downloads)
python app.py

# full local setup (BGE-M3 retrieval + Qwen via Ollama)
ollama pull qwen2.5:7b
python app.py --retriever dense --backend ollama
```

Open http://localhost:7860.

## Web app (FastAPI + Gradio)

`server.py` exposes the pipeline as a REST API with the Gradio chat mounted
on top — same flags as `app.py`:

```bash
python server.py                    # UI at http://localhost:8000
```

| route | what |
|-------|------|
| `/` | Gradio chat UI |
| `POST /api/ask` | `{"question": "...", "k": 4, "backend": "extractive"\|"ollama"}` → answer + retrieved docs + detected query language |
| `GET /api/health`, `GET /api/config` | status / current settings |
| `/docs` | interactive OpenAPI docs |

```bash
curl -X POST localhost:8000/api/ask -H 'Content-Type: application/json' \
     -d '{"question": "皮卡丘的属性和特性是什么？"}'
```

## Retrieval details

**Sparse (BM25)** — works with zero model downloads:
- CJK-aware tokenization: latin words + character uni/bigrams for JA/ZH
- Name-field weighting (title > body)
- **Name-alias boost** via the translation table: an exact species name in
  the query (any of the 5 language variants) doubles that species' scores.
  This fixed a fun failure mode: without it, "Pikachu" retrieves *Mimikyu*,
  whose Pokedex entries mention Pikachu more often than Pikachu's own do.
  The same boost applies to type names, for the same reason: every type-chart
  doc shares the question's vocabulary ("attack", "effective"), so without it
  BM25 TF saturation ranks the wrong type's chart above the one named in the
  query.
- Language boost: docs matching the query's script rank slightly higher

**Dense (BGE-M3)** — `pkmrag/retrieval_dense.py`. Multilingual embeddings
give true cross-lingual semantic retrieval (a Chinese query matches English
docs by meaning, and handles non-entity questions like "which Pokemon is the
Flame Pokemon?" that keyword BM25 gets wrong). Embeddings are cached to
`.npy` next to the corpus.

## Evaluation

Cross-lingual name→document retrieval on 102 held-out species
(query = name in language A, hit = correct species in language B ≠ A):

| query lang | recall@5 | MRR | n |
|-----------|---------|------|-----|
| en | 1.000 | 0.495 | 102 |
| ja | 1.000 | 0.497 | 102 |
| zh-hans | 1.000 | 0.529 | 102 |
| zh-hant | 1.000 | 0.691 | 102 |

(Sparse retriever. MRR < 1 mostly because the same-language doc — excluded
by the cross-lingual criterion — often ranks first, which is desirable in
actual use.) Run `python eval/eval_crosslingual.py --retriever dense` to get
comparable numbers for the dense model, before and after fine-tuning.

## Phase 2: fine-tuning the embedder

Contrastive fine-tuning on cross-lingual pairs mined from the linkage table
(18,460 pairs: name↔doc bridges, doc↔doc bridges, templated queries), with
in-batch negatives (= other Pokemon):

```bash
python finetune/make_pairs.py
python finetune/train_embedder.py --epochs 2 --batch-size 64   # MiniLM default
python eval/eval_crosslingual.py --retriever dense --model runs/pkm-embedder
```

Default base model is `paraphrase-multilingual-MiniLM-L12-v2` (laptop-
friendly); pass `--model BAAI/bge-m3` on a GPU node for the strong version.
Held-out species from `make_pairs.py` keep the eval honest — the model never
sees them during training.

## Repo structure

```
pkm-rag/
├── app.py                      # Gradio chat UI
├── scripts/download_data.py    # fetch PokeAPI CSVs
├── pkmrag/
│   ├── build_corpus.py         # link translations, emit documents
│   ├── retrieval_sparse.py     # BM25 + CJK tokenizer + alias boost
│   ├── retrieval_dense.py      # BGE-M3 semantic retrieval
│   └── rag.py                  # pipeline + local LLM backends
├── finetune/
│   ├── make_pairs.py           # cross-lingual training pairs
│   └── train_embedder.py       # contrastive fine-tuning
└── eval/eval_crosslingual.py   # recall@k / MRR, sparse vs dense vs tuned
```

## Known limitations

- BM25 fails on semantic/genus queries ("the Flame Pokemon") — use the dense
  retriever for those.
- Chinese Pokedex flavor text only covers Gen 7+ games; fact cards cover all
  1,025 species in all languages regardless.
- Learnsets cover level-up moves from each Pokemon's newest version group
  only — no TM/HM, egg, or tutor moves yet (`pokemon_moves.csv` has them
  all if needed).

## License

Code is released under the [MIT License](LICENSE).

The Pokédex data itself is **not** covered by that license: it is sourced from
the [PokeAPI database](https://github.com/PokeAPI/pokeapi) and the underlying
names, flavor text, and species data remain the property of Nintendo / Game
Freak / The Pokémon Company. Nothing here is affiliated with or endorsed by
them. No corpus data is committed to this repo — `scripts/download_data.py`
fetches it at build time.
