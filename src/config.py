"""Central configuration.

Values are read from (1) environment variables / .env, then (2) Streamlit
secrets (used on Streamlit Community Cloud), then (3) the default given.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def cfg(name, default=None):
    v = os.getenv(name)
    if v not in (None, ""):
        return v
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def cfg_int(name, default):
    return int(cfg(name, default))


def cfg_float(name, default):
    return float(cfg(name, default))


def cfg_bool(name, default=False):
    return str(cfg(name, default)).strip().lower() in ("1", "true", "yes", "on")


def aws_kwargs():
    """Explicit AWS keys if provided; otherwise boto3's default chain
    (IAM role on EC2/ECS/App Runner, ~/.aws/credentials, etc.)."""
    key, secret = cfg("AWS_ACCESS_KEY_ID"), cfg("AWS_SECRET_ACCESS_KEY")
    if key and secret:
        kw = {"aws_access_key_id": key, "aws_secret_access_key": secret}
        token = cfg("AWS_SESSION_TOKEN")
        if token:
            kw["aws_session_token"] = token
        return kw
    return {}
