#!/usr/bin/env python3
"""Download the embedded dataset once and build a local cosine-search index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "ankitjh4/bharat-government-documents"
FILENAME = "documents-embedded.parquet"
MODEL_ID = "google/embeddinggemma-300m"
DIMENSION = 768
COLUMNS = [
    "document_id", "title", "text", "url", "host", "kind", "trust",
    "quality", "status", "state", "category", "fetched_at", "topics", "embedding",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path(".index"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    index_dir = args.index_dir.expanduser().resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_path = index_dir / "vectors.f32"
    database_path = index_dir / "metadata.sqlite"
    manifest_path = index_dir / "manifest.json"
    for path in (vector_path, database_path, manifest_path):
        if path.exists():
            path.unlink()

    print(f"Checking Hub checksum for {REPO_ID}/{FILENAME}…", flush=True)
    try:
        file_info = HfApi().get_paths_info(repo_id=REPO_ID, repo_type="dataset", paths=[FILENAME])[0]
        expected_sha = getattr(getattr(file_info, "lfs", None), "sha256", None)
        if not expected_sha:
            raise RuntimeError("The Hub did not provide a SHA-256 checksum for the embedded Parquet")
    except Exception as exc:
        print(f"Could not verify the public Parquet file: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    parquet_path = ""
    for attempt in range(2):
        print(f"Fetching Parquet into the Hugging Face cache (attempt {attempt + 1}/2)…", flush=True)
        try:
            parquet_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=FILENAME,
                force_download=attempt > 0,
            )
        except Exception as exc:
            print(f"Could not download the public Parquet file: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        print("Verifying downloaded file checksum…", flush=True)
        digest = hashlib.sha256()
        with open(parquet_path, "rb") as downloaded:
            for chunk in iter(lambda: downloaded.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() == expected_sha:
            break
        print(f"Checksum mismatch: expected {expected_sha}, got {digest.hexdigest()}", file=sys.stderr)
        if attempt == 1:
            raise SystemExit(1)
    else:
        raise SystemExit(1)

    parquet = pq.ParquetFile(parquet_path)
    schema_names = set(parquet.schema_arrow.names)
    missing = set(COLUMNS) - schema_names
    if missing:
        print(f"The embedded file is missing expected columns: {', '.join(sorted(missing))}", file=sys.stderr)
        raise SystemExit(1)
    vector_type = parquet.schema_arrow.field("embedding").type
    if not hasattr(vector_type, "list_size") or vector_type.list_size != DIMENSION:
        print(f"Expected a fixed-size {DIMENSION}-dimension embedding, got {vector_type}", file=sys.stderr)
        raise SystemExit(1)

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("""
        CREATE TABLE documents (
            row_index INTEGER PRIMARY KEY,
            document_id TEXT,
            title TEXT,
            text TEXT NOT NULL,
            url TEXT,
            host TEXT,
            kind TEXT,
            trust TEXT,
            quality TEXT,
            status TEXT,
            state TEXT,
            category TEXT,
            fetched_at TEXT,
            topics TEXT
        )
    """)
    connection.execute("CREATE INDEX idx_documents_category ON documents(category)")
    connection.execute("CREATE INDEX idx_documents_state ON documents(state)")
    connection.execute("CREATE INDEX idx_documents_status ON documents(status)")

    row_count = 0
    columns = [name for name in COLUMNS if name != "embedding"] + ["embedding"]
    metadata_columns = [name for name in columns if name != "embedding"]
    try:
        with vector_path.open("wb") as vector_file:
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
                table = batch.select(metadata_columns).to_pydict()
                vectors = np.asarray(batch.column(batch.schema.get_field_index("embedding")).values).reshape(-1, DIMENSION)
                vectors = vectors.astype(np.float32, copy=True)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors /= np.maximum(norms, 1e-12)
                vectors.tofile(vector_file)

                length = batch.num_rows
                metadata_records = zip(
                    range(row_count, row_count + length),
                    table["document_id"], table["title"], (value or "" for value in table["text"]), table["url"],
                    table["host"], table["kind"], table["trust"], table["quality"], table["status"],
                    table["state"], table["category"], table["fetched_at"], table["topics"],
                    strict=True,
                )
                connection.executemany(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", metadata_records
                )
                connection.commit()
                row_count += length
                print(f"Indexed {row_count:,}/{parquet.metadata.num_rows:,} rows", flush=True)
    except Exception:
        connection.close()
        raise
    finally:
        if connection:
            connection.close()

    manifest = {
        "dataset": REPO_ID,
        "file": FILENAME,
        "model": MODEL_ID,
        "rows": row_count,
        "dimensions": DIMENSION,
        "dtype": "float32",
        "metric": "cosine",
        "vector_file": vector_path.name,
        "metadata_file": database_path.name,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=index_dir, delete=False) as temporary:
        json.dump(manifest, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, manifest_path)
    print(f"\nIndex ready in {index_dir}")
    print(f"Rows: {row_count:,}; vectors: {vector_path.stat().st_size / 1e9:.2f} GB")
    print(f"Metadata: {database_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
