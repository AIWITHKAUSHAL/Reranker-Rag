"""Run and save five real API comparisons from the command line."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from rag import Chunk, Pipeline, evaluate


def main() -> None:
    """Index the sample corpus and write a timestamped evaluation artifact."""
    root = Path(__file__).parent
    load_dotenv(root / ".env")
    pipeline = Pipeline(os.getenv("EURI_API_KEY", ""))
    chunks = [Chunk(**item) for item in json.loads((root / "data/corpus.json").read_text())]
    pipeline.index(chunks)
    questions = json.loads((root / "data/questions.json").read_text())
    results = evaluate(
        pipeline,
        questions,
        on_progress=lambda done, total: print(f"Completed {done}/{total}", flush=True),
    )
    report = {"timestamp_utc": datetime.now(UTC).isoformat(), "mode": "live", "results": results}
    path = root / "reports/live_evaluation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    successful = sum("error" not in result for result in results)
    print(f"Saved {path}; {successful}/{len(results)} successful live comparisons.")
    if successful != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
