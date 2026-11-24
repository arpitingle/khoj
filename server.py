#!/usr/bin/env python3
"""Local API and static frontend for semantic search over the embedded dataset."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
INDEX_DIR = Path(__import__("os").environ.get("INDGOV_INDEX_DIR", ROOT / ".index")).expanduser().resolve()
MODEL_ID = "google/embeddinggemma-300m"
logger = logging.getLogger("indgov")


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    limit: int = Field(default=12, ge=1, le=50)
    category: str | None = None
    state: str | None = None
    include_review: bool = True


class VectorIndex:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No index found at {index_dir}; run build_index.py first")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.rows = int(self.manifest["rows"])
        self.dimensions = int(self.manifest["dimensions"])
        self.vectors = np.memmap(
            index_dir / self.manifest["vector_file"],
            dtype=np.float32,
            mode="r",
            shape=(self.rows, self.dimensions),
        )
        self.database_path = index_dir / self.manifest["metadata_file"]
        with closing(self.connect()) as connection:
            counts = dict(connection.execute("SELECT status, COUNT(*) FROM documents GROUP BY status"))
        self.good_rows = int(counts.get("good", 0))
        self.review_rows = int(counts.get("review", 0))

    def connect(self) -> sqlite3.Connection:
        uri = f"file:{self.database_path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def facets(self) -> dict[str, list[str]]:
        with closing(self.connect()) as connection:
            categories = [row[0] for row in connection.execute(
                "SELECT DISTINCT category FROM documents WHERE category IS NOT NULL AND category != '' ORDER BY category LIMIT 100"
            )]
            states = [row[0] for row in connection.execute(
                "SELECT DISTINCT state FROM documents WHERE state IS NOT NULL AND state != '' ORDER BY state LIMIT 100"
            )]
        return {"categories": categories, "states": states}

    def search(self, query_vector: np.ndarray, request: SearchRequest) -> list[dict[str, Any]]:
        if query_vector.shape != (self.dimensions,):
            raise ValueError(f"Expected a {self.dimensions}-dimension query vector, got {query_vector.shape}")

        clauses: list[str] = []
        params: list[str] = []
        if request.category:
            clauses.append("category = ?")
            params.append(request.category)
        if request.state:
            clauses.append("state = ?")
            params.append(request.state)
        if not request.include_review:
            clauses.append("status = ?")
            params.append("good")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self.connect()) as connection:
            if clauses:
                eligible = np.fromiter(
                    (row[0] for row in connection.execute(f"SELECT row_index FROM documents{where}", params)),
                    dtype=np.int64,
                )
            else:
                eligible = None

        if eligible is not None and eligible.size == 0:
            return []

        best_indices = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        chunk_size = 16384
        if eligible is None:
            chunks = ((start, np.arange(start, min(start + chunk_size, self.rows), dtype=np.int64))
                      for start in range(0, self.rows, chunk_size))
        else:
            chunks = ((start, eligible[start:start + chunk_size]) for start in range(0, eligible.size, chunk_size))

        for _, row_indices in chunks:
            vectors = np.asarray(self.vectors[row_indices], dtype=np.float32)
            scores = vectors @ query_vector
            local_count = min(request.limit, scores.size)
            if local_count == 0:
                continue
            local_positions = np.argpartition(scores, -local_count)[-local_count:]
            candidate_indices = row_indices[local_positions]
            candidate_scores = scores[local_positions]
            combined_indices = np.concatenate((best_indices, candidate_indices))
            combined_scores = np.concatenate((best_scores, candidate_scores))
            keep = min(request.limit, combined_scores.size)
            selected = np.argpartition(combined_scores, -keep)[-keep:]
            best_indices = combined_indices[selected]
            best_scores = combined_scores[selected]

        ordering = np.argsort(best_scores)[::-1]
        best_indices = best_indices[ordering]
        best_scores = best_scores[ordering]
        records: dict[int, tuple[Any, ...]] = {}
        with closing(self.connect()) as connection:
            for start in range(0, best_indices.size, 500):
                ids = [int(value) for value in best_indices[start:start + 500]]
                placeholders = ",".join("?" for _ in ids)
                rows = connection.execute(
                    f"SELECT row_index, document_id, title, text, url, host, kind, trust, quality, status, state, category, fetched_at, topics FROM documents WHERE row_index IN ({placeholders})",
                    ids,
                )
                records.update({int(row[0]): row for row in rows})

        results = []
        for row_index, score in zip(best_indices, best_scores, strict=True):
            row = records.get(int(row_index))
            if row is None:
                continue
            results.append({
                "document_id": row[1], "title": row[2] or "Untitled government document",
                "text": row[3], "url": row[4], "host": row[5], "kind": row[6],
                "trust": row[7], "quality": row[8], "status": row[9], "state": row[10], "category": row[11],
                "fetched_at": row[12], "topics": row[13], "score": round(float(score), 4),
            })
        return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = None
    app.state.model = None
    if (INDEX_DIR / "manifest.json").exists():
        try:
            app.state.index = VectorIndex(INDEX_DIR)
            logger.info("Loaded %s vectors from %s", app.state.index.rows, INDEX_DIR)
            from sentence_transformers import SentenceTransformer
            app.state.model = SentenceTransformer(MODEL_ID, device="cpu")
            logger.info("Loaded query encoder %s on CPU", MODEL_ID)
        except Exception:
            logger.exception("Could not load index or query encoder")
    yield


app = FastAPI(title="Khoj semantic search", version="0.1.0", lifespan=lifespan)

cors_origins = [
    origin.strip()
    for origin in __import__("os").environ.get("INDGOV_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(ROOT / "styles.css", media_type="text/css")


@app.get("/semantic.css")
def semantic_styles() -> FileResponse:
    return FileResponse(ROOT / "semantic.css", media_type="text/css")


@app.get("/app.js")
def script() -> FileResponse:
    return FileResponse(ROOT / "app.js", media_type="application/javascript")


@app.get("/api/health")
def health() -> dict[str, Any]:
    index = app.state.index
    return {
        "index_ready": index is not None,
        "model_ready": app.state.model is not None,
        "rows": index.rows if index else 0,
        "good_rows": index.good_rows if index else 0,
        "review_rows": index.review_rows if index else 0,
        "dimensions": index.dimensions if index else 0,
        "model": MODEL_ID,
    }


@app.get("/api/facets")
def facets() -> dict[str, list[str]]:
    index = app.state.index
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not built. Run python build_index.py first.")
    return index.facets()


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    index = app.state.index
    model = app.state.model
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not built. Run python build_index.py first.")
    if model is None:
        raise HTTPException(status_code=503, detail="Query encoder is unavailable. Check model access and installation.")
    try:
        vector = np.asarray(model.encode_query(request.query, normalize_embeddings=True), dtype=np.float32)
        results = index.search(vector, request)
    except Exception as exc:
        logger.exception("Semantic search failed")
        raise HTTPException(status_code=500, detail="Semantic search failed; see server logs.") from exc
    return {"query": request.query, "results": results, "count": len(results), "metric": "cosine"}
