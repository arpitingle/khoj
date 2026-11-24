#!/usr/bin/env python3
"""Inspect the Bharat government document embedding Parquet without bulk download.

By default this reads Parquet footers and repository metadata from the Hugging
Face Hub. Optional row sampling reads projected columns from the Parquet files.
Install dependencies with:

    python -m pip install -r requirements-spike.txt

Pass --smoke-query to optionally load EmbeddingGemma and rank a small sample.
The model may require accepting its Hugging Face access terms first.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Any, Iterable

DATASET_REPO = "ankitjh4/bharat-government-documents"
EMBEDDING_REPO = "anudit/bharat-government-documents-embeddinggemma-300m"
MODEL_ID = "google/embeddinggemma-300m"
ID_NAMES = ("doc_id", "document_id", "id", "source_id", "record_id", "url", "source_url")
TEXT_NAMES = ("text", "cleaned_text", "normalized_text", "document_text", "content", "body", "full_text")


def require_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet
        from huggingface_hub import HfApi, HfFileSystem
    except ImportError as exc:
        print(
            "Missing spike dependencies. Run: python -m pip install -r requirements-spike.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return np, pa, (HfApi, HfFileSystem)


def file_paths(api: Any, repo: str) -> list[Any]:
    return list(api.list_repo_tree(repo_id=repo, repo_type="dataset", recursive=True, expand=True))


def parquet_paths(files: Iterable[Any]) -> list[Any]:
    return [item for item in files if item.path.lower().endswith(".parquet")]


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1000 or unit == "TB":
            return f"{amount:.2f} {unit}"
        amount /= 1000
    return f"{amount:.2f} TB"


def open_parquet(fs: Any, repo: str, path: str, pa: Any) -> Any:
    handle = fs.open(f"datasets/{repo}/{path}", "rb")
    try:
        return pa.parquet.ParquetFile(handle), handle
    except Exception:
        handle.close()
        raise


def read_sample(fs: Any, repo: str, path: str, columns: list[str], limit: int, pa: Any) -> list[dict[str, Any]]:
    parquet, handle = open_parquet(fs, repo, path, pa)
    rows: list[dict[str, Any]] = []
    try:
        for batch in parquet.iter_batches(batch_size=min(max(limit, 1), 512), columns=columns):
            rows.extend(batch.to_pylist())
            if len(rows) >= limit:
                return rows[:limit]
        return rows
    finally:
        handle.close()


def is_float_vector(field: Any, pa: Any) -> bool:
    data_type = field.type
    if pa.types.is_fixed_size_list(data_type) or pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return pa.types.is_floating(data_type.value_type)
    return False


def find_vector_fields(schema: Any, pa: Any) -> list[str]:
    fields = [field for field in schema if is_float_vector(field, pa)]
    fields.sort(key=lambda field: (not re.search(r"embed|vector|emb", field.name, re.I), field.name))
    return [field.name for field in fields]


def find_text_fields(schema: Any, pa: Any) -> list[str]:
    fields = [field for field in schema if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)]
    fields.sort(key=lambda field: (field.name.lower() not in TEXT_NAMES, field.name))
    return [field.name for field in fields]


def find_id_fields(schema: Any) -> list[str]:
    names = {field.name for field in schema}
    preferred = [name for name in ID_NAMES if name in names]
    other = [name for name in names if re.search(r"(^|_)(doc|document|record|source)?_?id$", name, re.I)]
    return preferred + sorted(set(other) - set(preferred))


def scalar_key(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    return str(value).strip() or None


def inspect_file(fs: Any, repo: str, file: Any, pa: Any, sample_rows: int) -> tuple[Any, list[dict[str, Any]]]:
    parquet, handle = open_parquet(fs, repo, file.path, pa)
    try:
        schema = parquet.schema_arrow
        print(f"\n  {file.path}")
        print(f"    stored size: {human_bytes(getattr(file, 'size', None))}")
        print(f"    rows: {parquet.metadata.num_rows:,}; row groups: {parquet.metadata.num_row_groups}")
        print(f"    columns: {schema}")
        vectors = find_vector_fields(schema, pa)
        texts = find_text_fields(schema, pa)
        ids = find_id_fields(schema)
        if vectors:
            print(f"    vector candidates: {', '.join(vectors)}")
        if texts:
            print(f"    text candidates: {', '.join(texts[:12])}")
        if ids:
            print(f"    ID candidates: {', '.join(ids)}")
        columns = list(dict.fromkeys(vectors[:1] + texts[:1] + ids[:4]))
        rows: list[dict[str, Any]] = []
        if columns and sample_rows > 0:
            for batch in parquet.iter_batches(batch_size=min(sample_rows, 512), columns=columns):
                rows.extend(batch.to_pylist())
                if len(rows) >= sample_rows:
                    break
        return {"schema": schema, "vectors": vectors, "texts": texts, "ids": ids, "rows": parquet.metadata.num_rows}, rows[:sample_rows]
    finally:
        handle.close()


def vector_stats(rows: list[dict[str, Any]], vector_field: str, np: Any) -> dict[str, Any]:
    vectors = [row.get(vector_field) for row in rows if row.get(vector_field) is not None]
    if not vectors:
        return {"count": 0}
    lengths = [len(vector) for vector in vectors]
    consistent = len(set(lengths)) == 1
    dimensions = lengths[0] if consistent else None
    norms: list[float] = []
    if consistent:
        sample = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(sample, axis=1).tolist()
    return {
        "count": len(vectors),
        "dimensions": dimensions,
        "consistent_dimensions": consistent,
        "norm_min": min(norms) if norms else None,
        "norm_max": max(norms) if norms else None,
    }


def run_smoke_query(rows: list[dict[str, Any]], vector_field: str, text_field: str, query: str, np: Any) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print("\nSmoke query needs sentence-transformers. Install it with: python -m pip install sentence-transformers", file=sys.stderr)
        raise SystemExit(2) from exc

    usable = [row for row in rows if row.get(vector_field) is not None and row.get(text_field)]
    if not usable:
        print("\nSmoke query skipped: sample rows do not contain both vectors and text.")
        return

    print(f"\nLoading {MODEL_ID} for a {len(usable):,}-row smoke query…")
    model = SentenceTransformer(MODEL_ID)
    query_vector = np.asarray(model.encode_query(query, normalize_embeddings=True), dtype=np.float32)
    matrix = np.asarray([row[vector_field] for row in usable], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    scores = normalized @ query_vector
    top = np.argsort(scores)[::-1][:5]
    print(f"Top 5 sample matches for: {query}")
    for rank, index in enumerate(top, start=1):
        row = usable[int(index)]
        snippet = re.sub(r"\s+", " ", str(row[text_field]))[:240]
        print(f"  {rank}. cosine={scores[index]:.4f} | {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rows", type=int, default=0, help="Optional rows to read per candidate file; 0 inspects metadata only (default: 0)")
    parser.add_argument("--smoke-query", help="Optional query to rank within the sampled embedded rows")
    parser.add_argument("--max-source-files", type=int, default=12, help="Maximum main-dataset Parquet files used for ID overlap sampling")
    args = parser.parse_args()
    if args.sample_rows < 0:
        parser.error("--sample-rows cannot be negative")

    np, pa, hf_imports = require_dependencies()
    HfApi, HfFileSystem = hf_imports
    api = HfApi()
    fs = HfFileSystem()

    try:
        main_files = parquet_paths(file_paths(api, DATASET_REPO))
        embedding_files = parquet_paths(file_paths(api, EMBEDDING_REPO))
    except Exception as exc:
        print(f"Could not list Hugging Face dataset files: {exc}", file=sys.stderr)
        print("Check network access, repository visibility, and any required Hugging Face authentication.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Main dataset: {DATASET_REPO} ({len(main_files)} Parquet files)")
    for item in main_files:
        print(f"  {item.path} — {human_bytes(getattr(item, 'size', None))}")
    print(f"\nEmbedding dataset: {EMBEDDING_REPO} ({len(embedding_files)} Parquet files)")
    for item in embedding_files:
        print(f"  {item.path} — {human_bytes(getattr(item, 'size', None))}")

    if not embedding_files:
        print("\nNo Parquet files found in the embedding repository.", file=sys.stderr)
        raise SystemExit(1)

    candidates = [item for item in embedding_files if re.search(r"embed|vector", item.path, re.I)] or embedding_files
    embed_info: list[tuple[Any, Any, list[dict[str, Any]]]] = []
    for file in candidates:
        try:
            info, rows = inspect_file(fs, EMBEDDING_REPO, file, pa, args.sample_rows)
            embed_info.append((file, info, rows))
        except Exception as exc:
            print(f"\n  Could not inspect {file.path}: {exc}", file=sys.stderr)

    if not embed_info:
        print("\nCould not inspect any candidate embedding Parquet files.", file=sys.stderr)
        raise SystemExit(1)

    id_hits: dict[str, set[str]] = defaultdict(set)
    for _, info, rows in embed_info:
        for name in info["ids"]:
            id_hits[name].update(key for row in rows if (key := scalar_key(row.get(name))))

    embedded_has_display_fields = any({"text", "url"}.issubset(set(info["schema"].names)) for _, info, _ in embed_info)
    if embedded_has_display_fields:
        print("\nSource-record join")
        print("  Not required for result display: an embedded file contains both text and URL fields.")
        print("  The embedding table also has a document_id field; matching it to the main repo can be checked separately if needed.")
    else:
        print("\nSource-record ID overlap (sampled)")
    if not embedded_has_display_fields and not id_hits:
        print("  No recognizable ID column found in the embedding sample.")
    elif not embedded_has_display_fields:
        overlap_counts: dict[str, int] = defaultdict(int)
        source_candidates = [item for item in main_files if not re.search(r"embed|vector", item.path, re.I)][: args.max_source_files]
        for file in source_candidates:
            try:
                parquet, handle = open_parquet(fs, DATASET_REPO, file.path, pa)
                try:
                    source_ids = find_id_fields(parquet.schema_arrow)
                    shared = sorted(set(source_ids) & set(id_hits))
                    if not shared:
                        continue
                    sampled = read_sample(fs, DATASET_REPO, file.path, shared, args.sample_rows, pa)
                    for name in shared:
                        main_keys = {key for row in sampled if (key := scalar_key(row.get(name)))}
                        overlap_counts[name] += len(main_keys & id_hits[name])
                finally:
                    handle.close()
            except Exception as exc:
                print(f"  Could not sample {file.path}: {exc}", file=sys.stderr)
        if not overlap_counts:
            print("  No same-named ID fields were found in the sampled main-dataset Parquet files.")
        else:
            for name, count in sorted(overlap_counts.items(), key=lambda pair: pair[1], reverse=True):
                embedded_count = len(id_hits[name])
                ratio = count / embedded_count if embedded_count else 0.0
                print(f"  {name}: {count:,}/{embedded_count:,} sampled embedding IDs matched ({ratio:.1%})")

    print("\nVector sample summary")
    for file, info, rows in embed_info:
        if not info["vectors"]:
            continue
        vector_type = info["schema"].field(info["vectors"][0]).type
        scalar_type = vector_type.value_type
        dimensions = getattr(vector_type, "list_size", None)
        stats = vector_stats(rows, info["vectors"][0], np)
        if dimensions:
            stats["schema_dimensions"] = dimensions
        bytes_per_value = 8 if pa.types.is_float64(scalar_type) else 4 if pa.types.is_float32(scalar_type) else None
        estimate = None
        if dimensions and bytes_per_value:
            estimate = info["rows"] * dimensions * bytes_per_value
        print(f"  {file.path}: {json.dumps(stats, sort_keys=True)}")
        if estimate is not None:
            print(f"    vector payload estimate (uncompressed): {human_bytes(estimate)}")
        if args.smoke_query and rows:
            if info["texts"]:
                run_smoke_query(rows, info["vectors"][0], info["texts"][0], args.smoke_query, np)
            else:
                print(f"  Semantic smoke query skipped for {file.path}: no text column in the sampled rows.")
        elif args.smoke_query:
            print(f"  Semantic smoke query skipped for {file.path}: rerun with --sample-rows to read a small sample.")

    print("\nNotes")
    print("  • Sampled ID overlap is a clue; verify that matched IDs identify the same document, not merely similarly named fields.")
    print("  • A smoke query ranks only the sampled rows; it does not measure full-corpus recall or production latency.")
    print("  • If the embedding repository requires access approval, accept it on Hugging Face and authenticate with `hf auth login`.")


if __name__ == "__main__":
    main()
