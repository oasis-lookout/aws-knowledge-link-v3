"""Create the pgvector extension and tables (sized for your embedding model).

Usage:
    python setup_db.py            # create if missing
    python setup_db.py --reset    # DROP and recreate csv_documents (needed after changing embedding model)
"""
import argparse

from src import db
from src.providers import embed_dim, embed_provider

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop csv_documents first")
    args = ap.parse_args()
    db.init_schema(reset=args.reset)
    print(f"OK: schema ready (embedding provider={embed_provider()}, vector({embed_dim()})).")
