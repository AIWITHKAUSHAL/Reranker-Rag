# YouTube demo script (about 3 minutes)

## 0:00 — Introduce the demo

**On screen:** RankLab app, sample handbook selected.

“This is RankLab, a retrieve, rerank, and generate RAG demo. I’ll show why retrieval and reranking are separate steps, then compare their rankings on five handbook questions.”

## 0:20 — Explain vector retrieval

**On screen:** Point to the embedding model and candidate count controls.

“The bi-encoder embeds each document chunk and the question independently into vectors. We compare the query vector with stored chunk vectors using cosine similarity. This is efficient for searching the whole collection. The initial retrieval score measures vector similarity; it is not a direct answer-quality score. Here, retrieval returns 15 candidates.”

## 0:55 — Explain reranking

**On screen:** Select four answer chunks; run one question and show the ranking table.

“The reranker gets the question and each retrieved passage together. This implementation asks an LLM relevance judge to score whether each passage provides evidence for the answer. It then sorts the candidate pool by that score. The final rank is the reranked position. Only the top four passages go to answer generation. The reranker is more focused but slower, and it cannot find passages that retrieval left out.”

“The retrieval score and reranker score use different scales, so compare candidates within each stage rather than treating the numbers as interchangeable.”

## 1:40 — Demonstrate ranking changes

**On screen:** Use “Who must approve working from another country and how far in advance?” in the sample run. Highlight the table rows for C24 and C04.

“For this question, vector retrieval initially puts C24 ahead of C04. The reranker moves C04 to first because its passage directly states the approval requirements and notice period. The table preserves initial rank and similarity, shows the reranker score, and displays final rank. The two answer panels use the same four passage budget, so we can compare without reranking against reranking.”

## 2:15 — Five-question result

**On screen:** Evaluation tab and the saved `reports/live_evaluation.json` summary.

“The bundled report contains five successful live questions, with 15 retrieved candidates and four selected passages per answer. Mean precision at four is 0.25 for both strategies, recall at four is 1.00 for both, and mean reciprocal rank changes from 0.90 to 1.00. The top-four ordering changed on two questions. This is a small classroom example: reranking can change order, but improvement is not guaranteed.”

## 2:45 — Close

**On screen:** README and repository URL.

“The source code, questions, corpus, and live comparison report are in the GitHub repository. To reproduce the evaluation, add an EURI API key, build the sample index, and run all five comparisons. The current reranker is an LLM judge; the architecture could also use a dedicated cross-encoder.”

## Recording checklist

- Show the GitHub repository URL and the app running with the sample handbook.
- Avoid showing the API key while entering it.
- Run the abroad-work question and show the C24/C04 rank change.
- Show the evaluation metrics and explain that a five-question demo is not a broad benchmark.
- Add captions, a short title/description, and link the repository in the YouTube description.
