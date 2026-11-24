# Semantic search feasibility spike

This is a small, read-only audit of the two public Hugging Face datasets. By default it reads repository metadata and Parquet footers only; it does not download the full corpus or write to Hugging Face. Optional row sampling is available, but a remote Parquet row group can transfer more data than the requested row count.

## Run the schema and join check

Use Python 3.10+ on a machine with network access:

```sh
python -m pip install -r requirements-spike.txt
python scripts/inspect_embeddings.py --sample-rows 1000
```

The report lists file sizes and row counts, Parquet schemas, likely vector/text/ID columns, vector dimensions, and estimated uncompressed vector payload size. If the embedded table does not contain display text and a URL, it also samples ID overlap with the main dataset. The overlap is only a first signal: check that a matching key actually points to the same document.

## Optional retrieval smoke test

Install `sentence-transformers`, accept any Hugging Face access terms required by `google/embeddinggemma-300m`, sign in if needed, and run:

```sh
python -m pip install sentence-transformers
hf auth login
python scripts/inspect_embeddings.py --sample-rows 2000 --smoke-query "Who can apply for crop insurance?"
```

This ranks only the sampled embedding rows. It demonstrates that the query encoder and stored vectors are compatible; it is not a full-corpus relevance or latency benchmark. It needs a local model download and can take a while on CPU. Remote row sampling can transfer a full Parquet row group, so use a local copy if transfer size matters.

## Decision gate

Proceed to a real vector index only if the vector dimension is consistent, the row IDs join reliably to source text and URLs (or those fields already live in the embedding file), and several varied smoke queries return plausible documents. If IDs do not join, first find the correct mapping or re-embed a small, traceable subset.
