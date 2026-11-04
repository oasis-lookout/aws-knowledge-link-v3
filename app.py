import hmac
import time
import uuid

import streamlit as st

from src import db
from src.config import cfg, cfg_bool, cfg_float, cfg_int
from src.providers import embed_dim, embed_provider, embed_query, embed_texts, generate, llm_provider

SYSTEM_PROMPT = (
    "You are a customer-support knowledge assistant. Answer ONLY from the material given in the "
    "context. 'Support records' are the source of truth. 'Earlier conversations' are previous "
    "answers from this assistant: use them only as supporting context, and prefer the support "
    "records if they disagree. If the context does not contain the answer, say you do not have "
    "enough information - do not guess. Cite ticket IDs or user IDs when they appear. "
    "Be concise and factual."
)
HISTORY_TURNS = cfg_int("HISTORY_TURNS", 3)  # previous turns of THIS chat sent to the LLM
NO_ANSWER = "I couldn't find relevant records for that question, so I can't answer it reliably."

st.set_page_config(page_title="AWS Knowledge Link Assistant", page_icon="💬", layout="wide")

# ----- optional password gate (set APP_PASSWORD) -----
app_pw = cfg("APP_PASSWORD")
if app_pw and not st.session_state.get("auth"):
    st.title("🔒 Knowledge Link")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered, app_pw):
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()


@st.cache_resource
def _init_db():
    db.init_schema()  # creates / migrates tables once per server process
    return True


st.title("💬 AWS Knowledge Link - Support Data Q&A")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----- sidebar -----
with st.sidebar:
    st.header("Settings")
    st.caption(f"Embeddings: `{embed_provider()}` ({embed_dim()}d) · LLM: `{llm_provider()}`")

    try:
        _init_db()
        companies = db.list_companies()
    except Exception as e:
        st.error(f"Database error: {e}")
        st.stop()

    allow_all = cfg_bool("ALLOW_ALL_COMPANIES", False)
    company = None
    if companies:
        options = (["(all companies)"] if allow_all else []) + companies
        choice = st.selectbox("Company (tenant filter)", options)
        company = None if choice == "(all companies)" else choice
    else:
        st.info("No company_id values found - searching all rows.")

    top_k = st.slider("Records to retrieve (top-k)", 1, 15, cfg_int("TOP_K", 5))
    min_sim = st.slider("Min similarity (guardrail)", 0.0, 1.0, cfg_float("MIN_SIMILARITY", 0.45), 0.01)

    st.divider()
    st.subheader("Conversation memory")
    use_past = st.toggle(
        "Use past conversations as context", value=cfg_bool("USE_PAST_CONVERSATIONS", True),
        help="Also searches earlier saved Q&A (from other sessions) for relevant context.",
    )
    past_k = st.slider("Past turns to retrieve", 1, 8, cfg_int("PAST_TURNS", 3), disabled=not use_past)

    st.divider()
    st.subheader("Saved conversations")
    st.caption(f"Current session: `{st.session_state.session_id}` (auto-saved)")
    try:
        sessions = db.list_sessions(company)
    except Exception:
        sessions = []
    if sessions:
        labels = {
            sid: f"{started:%d %b %H:%M} · {n} msg · {(first or '')[:40]}"
            for sid, started, n, first in sessions
        }
        picked = st.selectbox("Open an earlier conversation", list(labels), format_func=labels.get, index=None)
        if st.button("Open conversation") and picked:
            msgs = []
            for q, r, _t in db.load_history(picked):
                msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
            st.session_state.messages = msgs
            st.session_state.session_id = picked  # continue the same session
            st.rerun()
    else:
        st.caption("No saved conversations yet.")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()

# ----- chat -----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a question about your support data"):
    history = st.session_state.messages[-2 * HISTORY_TURNS :] if HISTORY_TURNS else []
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and thinking..."):
                t0 = time.time()
                qvec = embed_query(prompt)
                matches = db.search(qvec, top_k, company)
                top_sim = matches[0][2] if matches else 0.0
                relevant = [m for m in matches if m[2] >= min_sim]

                past_rel = []
                if use_past:
                    past = db.search_past(qvec, past_k, company, st.session_state.session_id)
                    past_rel = [p for p in past if p[3] >= min_sim]

                if not relevant and not past_rel:
                    answer = NO_ANSWER
                else:
                    parts = []
                    if relevant:
                        parts.append("Support records:\n" + "\n\n---\n\n".join(m[0] for m in relevant))
                    if past_rel:
                        parts.append(
                            "Earlier conversations:\n"
                            + "\n\n".join(f"Q: {q}\nA: {a}" for q, a, _t, _s in past_rel)
                        )
                    msgs = history + [
                        {"role": "user", "content": "Context:\n\n" + "\n\n=====\n\n".join(parts) + f"\n\nQuestion: {prompt}"}
                    ]
                    answer = generate(SYSTEM_PROMPT, msgs)
                latency_ms = int((time.time() - t0) * 1000)

            st.markdown(answer)
            st.caption(
                f"top similarity {top_sim:.2f} · {len(relevant)}/{len(matches)} records used · "
                f"{len(past_rel)} past turns used · {latency_ms} ms"
            )
            with st.expander("Retrieved records"):
                for content, _meta, sim in matches:
                    used = "✅" if sim >= min_sim else "⛔ below threshold"
                    st.markdown(f"**similarity {sim:.2f}** {used}")
                    st.text(content)
            if past_rel:
                with st.expander("Past conversations used"):
                    for q, a, t, sim in past_rel:
                        st.markdown(f"**similarity {sim:.2f}** · {t:%Y-%m-%d %H:%M}")
                        st.text(f"Q: {q}\nA: {a}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

            # save the turn; embed it only if it was a real, grounded answer so it can serve as future context
            grounded = bool(relevant or past_rel)
            turn_vec = None
            if grounded:
                try:
                    turn_vec = embed_texts([f"Q: {prompt}\nA: {answer}"], "document")[0]
                except Exception:
                    turn_vec = None
            db.save_chat(
                st.session_state.session_id, company, prompt, answer,
                [m[1] for m in relevant], top_sim, latency_ms, turn_vec, grounded,
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.session_state.messages.pop()  # drop the unanswered question
