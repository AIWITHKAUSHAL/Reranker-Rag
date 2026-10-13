# RankLab: retrieval, reranking, and generation

RankLab is a small live RAG workbench. It embeds a knowledge base, retrieves 10–20 candidate passages with cosine similarity, reranks those passages with an LLM relevance judge, and sends only the best 3–5 passages to answer generation. The app also generates a baseline answer from the same initial candidate pool and context budget.

## Run it

Requires Python 3.11+ and an EURI API key with access to `gemini-embedding-001` and `gemini-3.5-flash-lite`.

```sh
uv sync
cp .env.example .env
# Put your EURI key in .env
uv run streamlit run app.py
```

Alternatively, install the project dependencies with pip and run `streamlit run app.py`. The app also accepts UTF-8 `.txt` and `.md` uploads. Uploaded text, questions, and selected passages are sent to EURI for embedding, ranking, and generation.

## Run the five-question comparison

Choose the sample handbook, enter your key, build the live embedding index, and use **Run all 5 live comparisons** in the evaluation tab. The command line runner does the same benchmark:

```sh
uv run python evaluate.py
```

It writes fresh API results to `reports/live_evaluation.json`. The checked-in report is a real live run with five successful questions, 15 candidates, and four passages per answer. Results vary with provider model behavior. The current run has identical mean precision@4 (0.25) and recall@4 (1.00) for both strategies; mean reciprocal rank is 0.90 without reranking and 1.00 with reranking. The selected order changed on two of five questions. This small example shows why reranking can change ordering, but does not establish that reranking always improves results.

## Ranking fields

The interactive comparison and JSON report show each candidate's **Initial Retrieval Rank**, **Initial Retrieval Score** (cosine similarity), **Reranker Score** (LLM relevance judgment from 0 to 100), and **Final Rank**. The scores use different scales and are not directly comparable. The baseline gets the first `top_k` candidates; the reranked answer gets the first `top_k` after relevance scoring. Both use the same generation model, prompt, and number of passages.

## How the two stages differ

A bi-encoder embeds a question and passages independently. Vector similarity is fast and searches the full index, but it estimates semantic closeness rather than directly judging whether a passage answers the question. The reranker sees the question and each retrieved passage together and estimates passage relevance. That more focused judgment is applied only to a small candidate set, so it can improve ordering at the cost of an extra model call and latency. It cannot recover a useful passage that initial retrieval did not find.

The current reranker is an LLM relevance judge, not a dedicated cross-encoder. Scores are model judgments, not calibrated probabilities. The bundled Atlas handbook is fictional and its relevance labels are for this classroom demonstration.

## Submission demo

See [`docs/video-script.md`](docs/video-script.md) for a concise YouTube walkthrough script. The code and recorded evaluation report are in this repository; recording and publishing the video still need to be done in a YouTube account.
