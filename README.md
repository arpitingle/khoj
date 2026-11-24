# Khoj

Khoj is a local semantic-search prototype for the Bharat government documents collection. It uses the precomputed `documents-embedded.parquet` vectors, stores document metadata and text in SQLite, and runs cosine search over a memory-mapped float32 matrix.

## Requirements

- Python 3.10 or newer
- About 7 GB of free disk space for the 2.04 GB source Parquet, roughly 4.5 GB local index, and temporary files. The current build measured a 1.02 GB vector matrix and 3.82 GB SQLite metadata store.
- A Hugging Face account with access to `google/embeddinggemma-300m`; accept the model terms on its model page before loading the query encoder.

## Build and run locally

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf auth login
python build_index.py
python -m uvicorn server:app --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001>. The first build downloads the embedded Parquet into the Hugging Face cache and creates `.index/vectors.f32`, `.index/metadata.sqlite`, and a manifest. The next startup reuses the index. `.index/` is ignored by Git.

The embedded Parquet has 330,688 rows: 55,050 marked `good` and 275,638 marked `review` in the dataset manifest. These are heuristic extraction/source/subject labels, not manual fact-checks. The builder preserves the status. Search includes all rows by default; users can opt out of review rows to search only `good` records. The API embeds each query with `SentenceTransformer.encode_query()` and computes cosine scores in chunks over eligible vectors; it does not load a separate ANN index. This keeps the first implementation simple, with a roughly 1 GB vector matrix and no hosted vector database.

## API

- `GET /api/health` — index/model readiness and `good`/`review` row counts
- `GET /api/facets` — available categories and states
- `POST /api/search` — JSON body with `query`, optional `category`, `state`, `include_review` (default `true`), and a `limit` from 1 to 50

Example:

```sh
curl -X POST http://127.0.0.1:8001/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who can apply for crop insurance?","limit":5}'
```

## Current limits

- This is a local prototype; it is not deployed and has no multi-user queue or rate limiting.
- Query embedding requires the matching EmbeddingGemma model. Model access terms and model download are separate from access to the dataset.
- Search is exhaustive cosine scoring over a memory-mapped matrix. It should be benchmarked locally before choosing a free host; the query model and memory/disk needs may exceed small free-tier limits.
- The embedding table is self-contained for results: it has `document_id`, `title`, `text`, and `url` beside each 768-dimensional vector. Source links still point to original publishers.
