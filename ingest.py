"""Ingest a support-ticket CSV (from S3 or a local file) into Postgres + pgvector.

Usage:
    python ingest.py                                   # S3 bucket/key from .env
    python ingest.py --local-csv sample_data/sample_support.csv
    python ingest.py --limit 200                       # quick test on the first 200 rows

Safe to re-run: rows are upserted on (file_name, row_index), so no duplicates.
"""
import argparse
import csv
import io
import time

from src import db
from src.config import aws_kwargs, cfg, cfg_int
from src.providers import embed_dim, embed_provider, embed_texts


def _csv_list(name, default=""):
    return [c.strip() for c in cfg(name, default).split(",") if c.strip()]


# Column handling (override in .env to match your CSV)
COMPANY_COLUMN = cfg("COMPANY_COLUMN", "company_id")
EXCLUDE_COLUMNS = set(_csv_list("EXCLUDE_COLUMNS", "internal_notes"))  # never embedded or returned
TEXT_COLUMNS = _csv_list("TEXT_COLUMNS", "")  # empty = all non-excluded columns


def load_csv_text(local_csv=None):
    if local_csv:
        print(f"1. Reading local file {local_csv} ...")
        with open(local_csv, encoding="utf-8-sig") as f:
            return local_csv, f.read()
    import boto3

    bucket, key = cfg("S3_BUCKET_NAME"), cfg("S3_OBJECT_KEY")
    if not bucket or not key:
        raise SystemExit("Set S3_BUCKET_NAME and S3_OBJECT_KEY (or use --local-csv).")
    print(f"1. Downloading s3://{bucket}/{key} ...")
    s3 = boto3.client("s3", region_name=cfg("AWS_REGION", "us-east-1"), **aws_kwargs())
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8-sig")
    return key, body


def row_to_text(row):
    cols = TEXT_COLUMNS or [c for c in row if c not in EXCLUDE_COLUMNS]
    parts = []
    for c in cols:
        v = (row.get(c) or "").strip()
        if v:
            parts.append(f"{c}: {v}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"Embedding provider: {embed_provider()} (dim {embed_dim()})")
    db.init_schema()

    file_name, text = load_csv_text(args.local_csv)
    reader = csv.DictReader(io.StringIO(text))
    batch_size = cfg_int("INGEST_BATCH_SIZE", 64)

    print("2. Embedding and upserting ...")
    t0, total, batch = time.time(), 0, []

    def flush():
        nonlocal total, batch
        if not batch:
            return
        vecs = embed_texts([b[3] for b in batch], kind="document")
        db.upsert_documents([(*b, v) for b, v in zip(batch, vecs)])
        total += len(batch)
        batch = []
        print(f"   -> {total} rows ({total / (time.time() - t0):.1f} rows/s)")

    for idx, row in enumerate(reader):
        if args.limit and idx >= args.limit:
            break
        content = row_to_text(row)
        if not content:
            continue
        company = (row.get(COMPANY_COLUMN) or "").strip() or None
        metadata = {k: v for k, v in row.items() if k not in EXCLUDE_COLUMNS}
        batch.append((file_name, idx, company, content, metadata))
        if len(batch) >= batch_size:
            flush()
    flush()

    print("3. Building vector index ...")
    db.ensure_vector_index()
    print(f"Done: {total} rows ingested in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
