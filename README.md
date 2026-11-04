# AWS Knowledge Link - RAG Support Assistant

Ask questions about customer-support tickets stored as a CSV in Amazon S3.
Rows are embedded, stored in PostgreSQL + **pgvector**, and answered through a **Streamlit** chat UI
with a similarity guardrail, per-company filtering and conversation memory.

```
S3 CSV --(ingest.py)--> embeddings --> Postgres + pgvector <--(search)-- Streamlit app.py --> LLM
                                       csv_documents / chat_history
```

**Zero-cost default stack** (works locally *and* online):

| Part | Default | Alternatives (set in `.env`) |
|---|---|---|
| Embeddings | `fastembed` (BGE-small, runs inside the app, no key) | `gemini`, `bedrock` (Titan V2), `ollama` |
| LLM | `groq` (free tier, Llama 3.3) | `gemini`, `bedrock` (Nova/Claude), `ollama` |
| Database | Postgres + pgvector (Docker locally, Neon/Supabase/RDS online) | - |
| Hosting | Streamlit Community Cloud (free) or Docker on EC2 | - |

Switching providers is only a config change, so you can build for free and switch to Bedrock for the final AWS-native evaluation.

## What's improved vs. the first version
- Provider interface (no hard dependency on a local Ollama) so it can run online
- `company_id` filter (tenant isolation), `internal_notes` excluded from embeddings and answers
- Idempotent batch ingestion (upsert; HNSW index built after load)
- Similarity threshold guardrail ("I can't answer") + conversation memory
- Connection pooling, per-session IDs, optional password gate
- Latency and top-similarity logged to `chat_history` (useful for dissertation evaluation)

## 1. Run locally in ~10 minutes

**Prereqs:** Python 3.10-3.12, Docker, a free [Groq API key](https://console.groq.com) (or Gemini key).

```bash
# 1) Postgres with pgvector
docker run -d --name pgvector -e POSTGRES_PASSWORD=change-me -p 5432:5432 pgvector/pgvector:pg16

# 2) Python env
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3) Config
cp .env.example .env        # then edit: DB_PASSWORD=change-me, GROQ_API_KEY=...

# 4) Create tables + ingest the 12-row sample (first run downloads the ~130 MB embedding model)
python setup_db.py
python ingest.py --local-csv sample_data/sample_support.csv

# 5) Launch
streamlit run app.py        # http://localhost:8501
```

Try (select company `ACME-PAY`): *"Why was user U-501's account blocked?"*, *"How do I reset my password?"*,
and an out-of-scope one like *"What's the weather?"* (should refuse via the guardrail).

## 2. Use your real S3 data

1. Put your CSV in S3 (e.g. bucket `aws-knowledge-link`, key `support_conversations_10k.csv`).
2. In `.env` set `S3_BUCKET_NAME`, `S3_OBJECT_KEY`, `AWS_REGION`. For credentials use `aws configure` locally
   (boto3 picks it up) or set `AWS_ACCESS_KEY_ID/SECRET` in `.env`. The identity needs `s3:GetObject` on the object.
3. Map your columns: `COMPANY_COLUMN` (default `company_id`), `EXCLUDE_COLUMNS` (default `internal_notes`),
   optionally `TEXT_COLUMNS` to embed only some columns.
4. `python ingest.py` (use `--limit 200` first as a test). Re-running is safe.

## 3. Put it online

### Option A - Streamlit Community Cloud + Neon/Supabase (free, easiest)
1. Push this repo to GitHub (section 5).
2. Create a free Postgres at [Neon](https://neon.tech) or [Supabase](https://supabase.com) (pgvector is available on both).
   Use its host/user/password in `.env` with `DB_SSLMODE=require`. For Supabase use the *session pooler* connection details.
3. Run `python setup_db.py` and `python ingest.py` **from your laptop** against that DB.
4. On [share.streamlit.io](https://share.streamlit.io): New app -> pick the repo, main file `app.py`.
   Under *Advanced settings -> Secrets* paste the contents of `.streamlit/secrets.toml.example` with real values.
5. Set `APP_PASSWORD` so the public URL is not open to everyone.

Free hosts have limited memory; `fastembed` BGE-small is small enough. If the app runs out of memory, use `EMBED_PROVIDER=gemini` (re-ingest required).

### Option B - Inside AWS (matches your architecture; keeps RDS private)
1. Launch a small EC2 instance (or App Runner/ECS) in the **same VPC as RDS**. Attach an IAM role with `s3:GetObject`
   (and `bedrock:InvokeModel` if you use Bedrock) - no access keys needed.
2. RDS security group: allow 5432 **only from the app's security group**; set Public access = No.
3. On the instance:
   ```bash
   git clone <your-repo-url> && cd aws-knowledge-link
   cp .env.example .env   # DB_HOST=<rds endpoint>, DB_SSLMODE=require, keys for Groq/Gemini
   docker build -t knowledge-link .
   docker run -d --env-file .env -p 8501:8501 --restart unless-stopped knowledge-link
   ```
   (Ingest once from the same instance: `docker run --rm --env-file .env knowledge-link python ingest.py`.)
4. Open port 8501 in the app's security group (restrict to your IP), or put an ALB with HTTPS in front.

## 4. Switch to AWS Bedrock (final evaluation)
```
EMBED_PROVIDER=bedrock        # Titan Text Embeddings V2, 1024 dims
LLM_PROVIDER=bedrock          # BEDROCK_LLM_MODEL, e.g. us.amazon.nova-lite-v1:0 or a Claude Haiku profile
BEDROCK_REGION=us-east-1
```
Then `python setup_db.py --reset && python ingest.py`. Enable model access in the Bedrock console first.
Bedrock is pay-per-use (not free), but embedding ~10k short rows costs very little; check current pricing.
Model IDs vary by region/account - copy the exact ID or inference-profile ID from the Bedrock console.

## 5. Push to GitHub
```bash
cd aws-knowledge-link
git init
git status                  # confirm .env is NOT listed
git add .
git commit -m "Initial commit: provider-agnostic RAG on S3 + pgvector"
git branch -M main
# create an empty repo on github.com, then:
git remote add origin https://github.com/<you>/aws-knowledge-link.git
git push -u origin main
```
Never commit `.env` or a real `secrets.toml` (both are in `.gitignore`). If a key was ever committed, rotate it.

## Configuration reference
See `.env.example`. Most useful: `MIN_SIMILARITY` (tune per embedding model; also a UI slider), `TOP_K`,
`HISTORY_TURNS`, `ALLOW_ALL_COMPANIES`, `EXCLUDE_COLUMNS`, `INGEST_BATCH_SIZE`.

## Troubleshooting
- **`extension "vector" is not available`** - use the `pgvector/pgvector` image locally; on RDS use PostgreSQL 15+ (pgvector is supported); on Neon/Supabase run it as the owner.
- **`csv_documents.embedding is vector(X) but ... uses Y`** - you changed embedding model: `python setup_db.py --reset`, then re-ingest.
- **Always "couldn't find relevant records"** - lower `MIN_SIMILARITY` (try 0.3) or check the retrieved-records expander for actual scores.
- **Groq/Gemini 429** - free-tier rate limit; the client retries with backoff. Wait a minute or switch provider.
- **Cannot connect to RDS** - security group, public access, and `DB_SSLMODE=require`.
- **Gemini embeddings during bulk ingest hit rate limits** - use `fastembed` for ingestion (free, no limits).

## Known limitations (good "future work" for the dissertation)
- Tenant filter is chosen in the UI; production needs real authentication that fixes `company_id` server-side.
- Count/aggregate questions ("how many times was user X flagged?") are better answered by SQL over metadata than by vector search; a text-to-SQL or metadata-query route would be a natural next step.
- No reranker or hybrid (keyword + vector) search yet.
- Add an evaluation set (recall@k, answer accuracy) and compare providers using the logged latency/similarity.

## Saved conversations and conversation memory
- Every Q&A turn is saved automatically to the `chat_history` table (session ID, company, question, answer,
  sources, similarity, latency).
- **Open an earlier conversation** from the sidebar to reload it and continue the same session.
- **Past conversations as context:** each grounded answer is also embedded, so later questions can retrieve
  similar earlier Q&A (toggle in the sidebar). Support records stay the source of truth; refusals are never
  reused. Past answers are only searched within the selected company.
- The tables migrate automatically on app start. Old turns saved before this feature have no embedding, so they
  appear in the conversation list but are not used as context.
- Privacy: conversations may contain personal data. Treat the database accordingly (restrict access, set a retention policy).
