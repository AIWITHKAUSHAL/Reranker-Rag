"""Interactive live retrieval and reranking workbench."""

import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import EMBEDDING_MODEL, GENERATION_MODEL, Chunk, Pipeline, chunk_text, evaluate

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="RankLab · Reranking RAG", page_icon="↗", layout="wide")
st.markdown(
    """
<style>
.stApp {background: #0c1220;}
.block-container {padding-top: 2.5rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -.035em;}
[data-testid="stSidebar"] {background: #131c2e;}
[data-testid="stMetric"] {background: #152238; padding: 18px; border-radius: 12px;}
.hero {border: 1px solid #2d4261; border-radius: 18px; padding: 30px;
background: linear-gradient(115deg, #172944, #101b2d); margin-bottom: 24px;}
.eyebrow {color: #66dec6; font-size: 12px; letter-spacing: .17em; font-weight: 700;}
.hero h1 {color: #f4f7fc; margin: 10px 0; font-size: 42px;}
.hero p {color: #b1bed2; max-width: 760px;}
</style>
<div class="hero"><div class="eyebrow">RANKLAB / LIVE RAG WORKBENCH</div>
<h1>Better context. Better answers.</h1>
<p>Follow every passage from semantic retrieval to relevance scoring.
Compare grounded answers with and without reranking, using the same context budget.</p></div>
""",
    unsafe_allow_html=True,
)


def load_sample() -> list[Chunk]:
    """Load the fictional handbook with stable evaluation IDs."""
    return [Chunk(**item) for item in json.loads((ROOT / "data/corpus.json").read_text())]


def show_comparison(result: dict) -> None:
    """Display answers, ranking evidence, and exact context passages."""
    st.subheader(result["question"])
    left, right = st.columns(2)
    for column, mode, title in (
        (left, "baseline", "Without reranking"),
        (right, "reranked", "With reranking"),
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.caption("Context: " + " · ".join(result[f"{mode}_ids"]))
                st.markdown(result["answers"][mode])
    st.markdown(
        "#### Initial Retrieval Rank · Initial Retrieval Score ↓ Reranker Score ↓ Final Rank"
    )
    st.caption(
        "Retrieval: cosine similarity (−1 to 1). Reranker: LLM relevance score (0–100), "
        "not a probability. Scores use different scales."
    )
    rows = [
        {
            "Chunk": hit["chunk"]["id"],
            "Source": hit["chunk"]["source"],
            "Initial Retrieval Rank": hit["initial_rank"],
            "Initial Retrieval Score": round(hit["initial_score"], 4),
            "Reranker Score": hit["reranker_score"],
            "Final Rank": hit["final_rank"],
            "Rank improvement": hit["initial_rank"] - hit["final_rank"],
            "Sent to LLM": hit["chunk"]["id"] in result["reranked_ids"],
        }
        for hit in result["ranking"]
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    timing = result["timings"]
    st.caption(
        f"Retrieval {timing['retrieval']:.2f}s · Reranking {timing['rerank']:.2f}s · "
        f"Baseline generation {timing['generation']['baseline']:.2f}s · "
        f"Reranked generation {timing['generation']['reranked']:.2f}s"
    )
    with st.expander("Inspect all candidate passages and citations"):
        for hit in result["ranking"]:
            st.markdown(f"**[{hit['chunk']['id']}] {hit['chunk']['source']}**")
            st.write(hit["chunk"]["text"])


with st.sidebar:
    st.markdown("## ↗ RankLab")
    st.caption("RETRIEVE → RERANK → GENERATE")
    api_key = st.text_input("EURI API key", type="password", value=os.getenv("EURI_API_KEY", ""))
    st.caption("Your key stays in this server session and is never included in reports.")
    source = st.radio("Knowledge source", ["Sample handbook", "Upload documents"])
    uploads = (
        st.file_uploader("UTF-8 text or Markdown", type=["txt", "md"], accept_multiple_files=True)
        if source == "Upload documents"
        else []
    )
    candidate_count = st.slider("Candidate chunks", 10, 20, 15)
    top_k = st.slider("Chunks sent to each answer", 3, 5, 4)
    st.divider()
    st.caption("EMBEDDING MODEL")
    st.code(EMBEDDING_MODEL, language=None)
    st.caption("GENERATION + RELEVANCE JUDGE")
    st.code(GENERATION_MODEL, language=None)
    st.caption("Endpoint: EURI · Live API calls only")

chunks = load_sample() if source == "Sample handbook" else []
try:
    for i, upload in enumerate(uploads):
        chunks.extend(chunk_text(upload.getvalue().decode("utf-8"), upload.name, f"U{i + 1}-"))
except UnicodeDecodeError:
    st.error("Please upload UTF-8 encoded text or Markdown files.")
    st.stop()
if len(chunks) > 500:
    st.error("Please limit this classroom demo to 500 chunks per index.")
    st.stop()
fingerprint = hashlib.sha256(
    json.dumps([vars(chunk) for chunk in chunks], sort_keys=True).encode() + api_key.encode()
).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:
    for key in ("pipeline", "comparison", "evaluation", "fingerprint", "result_settings"):
        st.session_state.pop(key, None)
settings = (candidate_count, top_k)
if st.session_state.get("result_settings", settings) != settings:
    st.session_state.pop("comparison", None)
    st.session_state.pop("evaluation", None)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Knowledge chunks", len(chunks))
m2.metric("Retrieve", min(candidate_count, len(chunks)))
m3.metric("Keep", top_k)
m4.metric("Connection", "Indexed" if "pipeline" in st.session_state else "Not indexed")
st.write("")
if source == "Sample handbook":
    st.caption(
        "Bundled dataset: fictional Atlas employee handbook, 24 passages and 5 labelled questions."
    )
if st.button(
    "Build live embedding index", type="primary", disabled=not api_key or len(chunks) < 10
):
    try:
        with st.spinner("Embedding passages with Gemini Embedding 001…"):
            pipeline = Pipeline(api_key)
            pipeline.index(chunks)
        st.session_state.pipeline = pipeline
        st.session_state.fingerprint = fingerprint
        st.rerun()
    except Exception as exc:
        st.error(
            f"Indexing failed ({type(exc).__name__}). Check your EURI key, connectivity, "
            "quota, and access to gemini-embedding-001. No fallback embeddings were used."
        )
if not api_key:
    st.info("Enter an EURI API key in the sidebar, then build the live index.")
elif len(chunks) < 10:
    st.info("Upload enough text for at least 10 chunks, or select the sample handbook.")

ask_tab, evaluation_tab, method_tab = st.tabs(
    ["01 / Ask & compare", "02 / Five-question evaluation", "03 / How it works"]
)
with ask_tab:
    with st.form("question_form"):
        question = st.text_input(
            "Ask a question", placeholder="How much annual leave can I carry forward?"
        )
        submit = st.form_submit_button(
            "Retrieve, rerank & compare",
            type="primary",
            disabled="pipeline" not in st.session_state,
        )
    if submit:
        st.session_state.pop("comparison", None)
        try:
            with st.spinner("Retrieving candidates → scoring relevance → generating both answers…"):
                st.session_state.comparison = st.session_state.pipeline.compare(
                    question, candidate_count, top_k
                )
                st.session_state.result_settings = settings
        except Exception as exc:
            st.error(
                f"Comparison failed ({type(exc).__name__}). Verify provider access to the "
                "requested models, quota, and question. An invalid reranker response also "
                "stops the pipeline; no simulated answer is shown."
            )
    if "comparison" in st.session_state:
        show_comparison(st.session_state.comparison)
        st.download_button(
            "Download comparison JSON",
            json.dumps(st.session_state.comparison, indent=2),
            "comparison.json",
            "application/json",
        )
    else:
        st.markdown("### Watch relevance change the order")
        st.write(
            "Build the index and ask a question to see initial ranks, similarity scores, "
            "reranker scores, final ranks, and the exact passages sent to each answer."
        )

with evaluation_tab:
    st.subheader("Same questions. Same budget. Two retrieval strategies.")
    st.write(
        "Compare five labelled questions on the sample handbook. Precision@k, recall@k, "
        "and reciprocal rank measure selected-context relevance; read both answers against "
        "the reference answer to assess generation quality."
    )
    questions = json.loads((ROOT / "data/questions.json").read_text())
    with st.expander("View the five test questions"):
        for item in questions:
            st.write(item["question"])
    if source != "Sample handbook":
        st.info("Switch to the sample handbook to run its labelled benchmark.")
    run = st.button(
        "Run all 5 live comparisons",
        disabled=(source != "Sample handbook" or "pipeline" not in st.session_state),
        type="primary",
    )
    if run:
        progress = st.progress(0, text="Starting live evaluation…")
        st.session_state.evaluation = evaluate(
            st.session_state.pipeline,
            questions,
            candidate_count,
            top_k,
            lambda done, total: progress.progress(done / total, text=f"Completed {done}/{total}"),
        )
        st.session_state.result_settings = settings
    if "evaluation" in st.session_state:
        results = st.session_state.evaluation
        successful = [result for result in results if "error" not in result]
        st.caption(f"Successful live comparisons: {len(successful)}/{len(results)}")
        rows = []
        for result in successful:
            for mode in ("baseline", "reranked"):
                rows.append(
                    {"Question": result["question"], "Strategy": mode, **result["metrics"][mode]}
                )
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
            baseline = sum(r["metrics"]["baseline"]["recall"] for r in successful) / len(successful)
            reranked = sum(r["metrics"]["reranked"]["recall"] for r in successful) / len(successful)
            st.info(
                f"Mean recall@{top_k}: {baseline:.0%} without reranking → {reranked:.0%} "
                "with reranking. Improvement is not guaranteed; ties are valid results."
            )
        for result in results:
            with st.expander(result["question"]):
                if "error" in result:
                    st.error(f"Live request failed: {result['error']}")
                else:
                    st.write("Reference answer: " + result["expected_answer"])
                    show_comparison(result)
        st.download_button(
            "Download evaluation JSON",
            json.dumps(results, indent=2),
            "evaluation.json",
            "application/json",
        )

with method_tab:
    st.markdown("""
### A transparent two-stage search
1. **Embed:** Gemini Embedding 001 maps documents and the question into vectors.
2. **Retrieve:** cosine similarity selects 10–20 candidate chunks (15 by default).
3. **Rerank:** Gemini 3.5 Flash Lite acts as a relevance judge, scoring every candidate
   from 0 to 100 for evidence that directly answers the question. This is LLM reranking.
4. **Select:** keep the best 3–5 chunks (4 by default). Ties preserve retrieval order.
5. **Generate:** Gemini 3.5 Flash Lite answers using only those passages, with chunk citations.

The baseline uses the top chunks from initial retrieval with the **same context budget**,
model, prompt, and temperature. Both branches share the same retrieved candidate pool.
Reranking adds an API call and latency. Scores are model judgements, not calibrated probabilities.
The labelled sample measures retrieval quality; its size does not establish general superiority.

Uploaded documents are split into 160-word windows with 30-word overlap. Indexes remain
in session memory. Indexing sends document text to EURI; questions and selected passages
are also sent to EURI when you run comparisons.
""")
