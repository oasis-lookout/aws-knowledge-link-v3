"""Pluggable embedding + LLM providers.

EMBED_PROVIDER: fastembed (local, free, default) | gemini | bedrock | ollama
LLM_PROVIDER:   groq (free tier, default) | gemini | bedrock | ollama

Swap providers by changing env vars only. Remember: if you change the
embedding provider/model you must re-ingest (vector dimensions must match).
"""
import functools
import json
import time

import requests

from .config import aws_kwargs, cfg, cfg_int

DEFAULT_DIMS = {"fastembed": 384, "gemini": 768, "bedrock": 1024, "ollama": 768}
EMBED_PROVIDERS = tuple(DEFAULT_DIMS)
LLM_PROVIDERS = ("groq", "gemini", "bedrock", "ollama")


def embed_provider():
    p = cfg("EMBED_PROVIDER", "fastembed").lower()
    if p not in EMBED_PROVIDERS:
        raise ValueError(f"EMBED_PROVIDER must be one of {EMBED_PROVIDERS}, got '{p}'")
    return p


def llm_provider():
    p = cfg("LLM_PROVIDER", "groq").lower()
    if p not in LLM_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {LLM_PROVIDERS}, got '{p}'")
    return p


def embed_dim():
    return cfg_int("EMBED_DIM", DEFAULT_DIMS[embed_provider()])


def _require(name):
    v = cfg(name)
    if not v:
        raise RuntimeError(f"Missing required setting: {name}")
    return v


def _post(url, headers=None, payload=None, timeout=120, retries=4):
    delay = 2
    for attempt in range(retries + 1):
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(delay)
            delay *= 2
            continue
        if not r.ok:
            raise RuntimeError(f"{r.status_code} from {url.split('?')[0]}: {r.text[:500]}")
        return r.json()


@functools.lru_cache(maxsize=1)
def _bedrock():
    import boto3

    region = cfg("BEDROCK_REGION") or cfg("AWS_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region, **aws_kwargs())


# --------------------------------------------------------------- embeddings
@functools.lru_cache(maxsize=1)
def _fastembed_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=cfg("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))


def _embed_fastembed(texts, kind):
    m = _fastembed_model()
    if kind == "query" and hasattr(m, "query_embed"):
        it = m.query_embed(texts)
    elif hasattr(m, "passage_embed"):
        it = m.passage_embed(texts)
    else:
        it = m.embed(texts)
    return [v.tolist() for v in it]


def _embed_gemini(texts, kind):
    model = cfg("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    dim = embed_dim()
    task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
    headers = {"x-goog-api-key": _require("GEMINI_API_KEY"), "Content-Type": "application/json"}
    out = []
    for i in range(0, len(texts), 100):
        reqs = [
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": t}]},
                "taskType": task,
                "outputDimensionality": dim,
            }
            for t in texts[i : i + 100]
        ]
        data = _post(url, headers, {"requests": reqs})
        out.extend(e["values"] for e in data["embeddings"])
    return out


def _embed_bedrock(texts, kind):
    model = cfg("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    dim = embed_dim()
    out = []
    for t in texts:
        body = json.dumps({"inputText": t, "dimensions": dim, "normalize": True})
        resp = _bedrock().invoke_model(modelId=model, body=body)
        out.append(json.loads(resp["body"].read())["embedding"])
    return out


def _embed_ollama(texts, kind):
    base = cfg("OLLAMA_URL", "http://localhost:11434")
    model = cfg("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    out = []
    for i in range(0, len(texts), 64):
        data = _post(f"{base}/api/embed", payload={"model": model, "input": texts[i : i + 64]})
        out.extend(data["embeddings"])
    return out


_EMBEDDERS = {
    "fastembed": _embed_fastembed,
    "gemini": _embed_gemini,
    "bedrock": _embed_bedrock,
    "ollama": _embed_ollama,
}


def embed_texts(texts, kind="document"):
    """kind: 'document' (ingestion) or 'query' (search)."""
    vecs = _EMBEDDERS[embed_provider()](list(texts), kind)
    if vecs and len(vecs[0]) != embed_dim():
        raise RuntimeError(
            f"Embedding model returned {len(vecs[0])} dims but EMBED_DIM={embed_dim()}. "
            "Fix EMBED_DIM (or the model) and re-run: python setup_db.py --reset"
        )
    return vecs


def embed_query(text):
    return embed_texts([text], "query")[0]


# --------------------------------------------------------------- generation
def _gen_groq(system, messages):
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {_require('GROQ_API_KEY')}"},
        payload={
            "model": cfg("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": 0.2,
        },
    )
    return data["choices"][0]["message"]["content"]


def _gen_gemini(system, messages):
    model = cfg("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    data = _post(
        url,
        headers={"x-goog-api-key": _require("GEMINI_API_KEY"), "Content-Type": "application/json"},
        payload={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2},
        },
    )
    return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])


def _gen_bedrock(system, messages):
    resp = _bedrock().converse(
        modelId=cfg("BEDROCK_LLM_MODEL", "us.amazon.nova-lite-v1:0"),
        system=[{"text": system}],
        messages=[{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
        inferenceConfig={"temperature": 0.2, "maxTokens": 800},
    )
    return resp["output"]["message"]["content"][0]["text"]


def _gen_ollama(system, messages):
    base = cfg("OLLAMA_URL", "http://localhost:11434")
    data = _post(
        f"{base}/api/chat",
        payload={
            "model": cfg("OLLAMA_LLM_MODEL", "llama3.2"),
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
        },
        timeout=300,
    )
    return data["message"]["content"]


_GENERATORS = {"groq": _gen_groq, "gemini": _gen_gemini, "bedrock": _gen_bedrock, "ollama": _gen_ollama}


def generate(system, messages):
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...]"""
    return _GENERATORS[llm_provider()](system, messages)
