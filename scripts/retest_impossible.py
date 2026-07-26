"""
Re-runs just the 20 impossible questions from the verification sample
through the orchestrator (now with the tightened judge), to confirm the
fix actually improves the safety metric before trusting it.

Resumable, same pattern as verify.py.
"""

import json
import os
import sys

from src.orchestrator import answer_query
from scripts.verify import check_correctness

OLD_RESULTS_PATH = "eval/verification_results.json"
NEW_RESULTS_PATH = "eval/retest_impossible_results.json"


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 5

    old_results = json.load(open(OLD_RESULTS_PATH))
    impossible_questions = [r for r in old_results if r["is_impossible"]]

    new_results = json.load(open(NEW_RESULTS_PATH)) if os.path.exists(NEW_RESULTS_PATH) else []
    done = {r["question"] for r in new_results}
    remaining = [q for q in impossible_questions if q["question"] not in done]

    print(f"{len(new_results)}/{len(impossible_questions)} done. {len(remaining)} remaining.")

    if not remaining:
        correct = sum(r["llm_correct"] for r in new_results)
        old_correct = sum(r["llm_correct"] for r in old_results if r["is_impossible"])
        print(f"\nOLD judge: {old_correct}/{len(impossible_questions)} correctly handled")
        print(f"NEW judge: {correct}/{len(new_results)} correctly handled")
        return

    for q in remaining[:limit]:
        print(f"[{len(new_results)+1}/{len(impossible_questions)}] {q['question'][:60]}")
        result = answer_query(q["question"])
        verdict = check_correctness(q["question"], result["answer"], True, [])
        entry = {
            "question": q["question"],
            "sufficient": result["sufficient"],
            "attempts": result["attempts"],
            "generated_answer": result["answer"],
            "llm_correct": verdict["correct"],
            "llm_reason": verdict["reason"],
        }
        new_results.append(entry)
        with open(NEW_RESULTS_PATH, "w") as f:
            json.dump(new_results, f, indent=2)
        print(f"  -> sufficient={result['sufficient']}  llm_correct={verdict['correct']}")


if __name__ == "__main__":
    main()
