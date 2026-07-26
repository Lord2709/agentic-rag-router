"""
Verification pass: runs the full orchestrator (router -> retrieve -> judge
-> reformulate retry loop -> generate) across a balanced sample of SQuAD
questions, and measures whether it behaves correctly - both on answerable
questions (does it find and state the right answer) and on impossible ones
(does it correctly decline instead of hallucinating).

Resumable by design: each run processes up to --limit NEW questions and
saves progress after every single one, so it can be safely re-invoked
across multiple short runs (each full run through the pipeline involves
several Claude API calls and can take a while) without losing work.

Usage:
    python scripts/verify.py --limit 5      # process up to 5 more questions
    python scripts/verify.py --summary      # just print stats on what's done so far
"""

import argparse
import json
import os
import random

from src.config import client, JUDGE_MODEL
from src.orchestrator import answer_query

EVAL_PATH = "eval/squad_qa.json"
RESULTS_PATH = "eval/verification_results.json"
SAMPLE_PER_CLASS = 20
SEED = 123

_REFUSAL_PHRASES = [
    "don't have enough information", "do not have enough information",
    "doesn't contain", "does not contain", "insufficient", "unable to confirm",
    "can't confirm", "cannot confirm", "wasn't able to confirm", "not able to confirm",
    "no information", "not enough information", "doesn't address", "does not address",
]

# --- LLM-based correctness checker ---
# The crude substring/keyword check above is too strict to trust as a final
# number (it flagged genuinely correct answers as wrong over formatting
# differences, e.g. "O(n2)" vs "O(n²)", and flagged reasonable declines as
# wrong just for not using an exact refusal phrase). Same lesson as judge.py:
# scoring open-ended correctness needs real semantic judgment, not string
# matching. This re-scores already-generated answers without re-running the
# whole pipeline (cheap: one Haiku call per existing result).

_CORRECTNESS_TOOL = {
    "name": "submit_correctness_judgment",
    "description": "Judge whether the system's response was correct/appropriate for this question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {
                "type": "boolean",
                "description": "See the specific criteria given in the prompt for what counts as correct in this case."
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation for the judgment."
            }
        },
        "required": ["correct", "reason"]
    }
}


def check_correctness(question: str, generated_answer: str, is_impossible: bool, ground_truth_answers: list[str]) -> dict:
    """Re-judge an already-generated answer semantically, instead of via crude string matching."""
    if is_impossible:
        message_text = (
            f"Question: {question}\n\n"
            "Ground truth: this question is NOT answerable from the reference corpus - "
            "there is no correct factual answer available.\n\n"
            f"System's response: {generated_answer}\n\n"
            "Judge whether the system's response appropriately handled this. It is CORRECT "
            "if it avoids confidently asserting a specific factual answer - this includes "
            "explaining what's missing, noting a discrepancy in the source material, asking "
            "for clarification, or any other form of appropriate hedging/declining. It is "
            "INCORRECT only if it states a specific factual answer as if it were verified. "
            "Call submit_correctness_judgment."
        )
    else:
        gt = "; ".join(dict.fromkeys(ground_truth_answers))
        message_text = (
            f"Question: {question}\n\n"
            f"Ground truth answer(s): {gt}\n\n"
            f"System's response: {generated_answer}\n\n"
            "Judge whether the system's response conveys the same correct answer as the "
            "ground truth, regardless of exact wording, formatting, or phrasing differences "
            "(e.g. a numbered list vs a sentence, or 'O(n2)' vs 'O(n²)' both count as "
            "matching). Call submit_correctness_judgment."
        )

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        tools=[_CORRECTNESS_TOOL],
        tool_choice={"type": "tool", "name": "submit_correctness_judgment"},
        messages=[{"role": "user", "content": message_text}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise RuntimeError("Correctness checker did not return a tool_use block.")


def rescore_with_llm(results: list[dict], limit: int) -> int:
    """Add llm_correct/llm_reason to results that don't have it yet, up to `limit` new ones."""
    pending = [r for r in results if "llm_correct" not in r]
    batch = pending[:limit]

    for r in batch:
        verdict = check_correctness(r["question"], r["generated_answer"], r["is_impossible"], r["ground_truth_answers"])
        r["llm_correct"] = verdict["correct"]
        r["llm_reason"] = verdict["reason"]
        save_results(results)
        print(f"  [{'IMPOSSIBLE' if r['is_impossible'] else 'ANSWERABLE'}] {r['question'][:60]} -> "
              f"llm_correct={verdict['correct']} (was correct={r['correct']})")

    return len(batch)


def print_llm_summary(results: list[dict]):
    scored = [r for r in results if "llm_correct" in r]
    if not scored:
        print("No LLM-rescored results yet. Run with --rescore first.")
        return

    answerable = [r for r in scored if not r["is_impossible"]]
    impossible = [r for r in scored if r["is_impossible"]]

    print(f"LLM-rescored: {len(scored)}/{len(results)}\n")

    if answerable:
        correct = sum(r["llm_correct"] for r in answerable)
        print(f"ANSWERABLE ({len(answerable)}): {correct}/{len(answerable)} correct "
              f"({100*correct/len(answerable):.0f}%)  [vs {sum(r['correct'] for r in answerable)}/{len(answerable)} by crude check]")

    if impossible:
        correct = sum(r["llm_correct"] for r in impossible)
        print(f"IMPOSSIBLE ({len(impossible)}): {correct}/{len(impossible)} correctly handled "
              f"({100*correct/len(impossible):.0f}%)  [vs {sum(r['correct'] for r in impossible)}/{len(impossible)} by crude check]")

    print()
    disagreements = [r for r in scored if r["correct"] != r["llm_correct"]]
    if disagreements:
        print(f"--- {len(disagreements)} cases where crude check and LLM check disagreed ---")
        for r in disagreements:
            print(f"  [{'IMPOSSIBLE' if r['is_impossible'] else 'ANSWERABLE'}] {r['question'][:60]}")
            print(f"    crude={r['correct']}  llm={r['llm_correct']}  reason: {r['llm_reason'][:150]}")

    still_wrong = [r for r in scored if not r["llm_correct"]]
    if still_wrong:
        print(f"\n--- {len(still_wrong)} still incorrect per LLM judge (real failures) ---")
        for r in still_wrong:
            print(f"  [{'IMPOSSIBLE' if r['is_impossible'] else 'ANSWERABLE'}] {r['question'][:60]}")
            print(f"    reason: {r['llm_reason'][:150]}")


def build_sample(seed: int = SEED, n_per_class: int = SAMPLE_PER_CLASS) -> list[dict]:
    qa = json.load(open(EVAL_PATH))
    answerable = [q for q in qa if not q["is_impossible"] and q["answers"]]
    impossible = [q for q in qa if q["is_impossible"]]

    random.seed(seed)
    sample = random.sample(answerable, n_per_class) + random.sample(impossible, n_per_class)
    random.shuffle(sample)
    return sample


def load_results() -> list[dict]:
    if os.path.exists(RESULTS_PATH):
        return json.load(open(RESULTS_PATH))
    return []


def save_results(results: list[dict]):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def evaluate_one(sample_q: dict) -> dict:
    result = answer_query(sample_q["question"])
    answer_lower = result["answer"].lower()

    is_impossible = sample_q["is_impossible"]
    ground_truth_answers = sample_q["answers"]

    if is_impossible:
        # Correct behavior: system should either flag insufficient (sufficient=False)
        # or, if it answered directly, at least not confidently assert a fact.
        correct = (result["sufficient"] is False) or looks_like_refusal(result["answer"])
    else:
        # Correct behavior: system should find sufficient context AND the
        # ground-truth answer text should appear somewhere in the response.
        answer_found = any(ans.lower() in answer_lower for ans in ground_truth_answers)
        correct = (result["sufficient"] is not False) and answer_found

    return {
        "question": sample_q["question"],
        "is_impossible": is_impossible,
        "source_file": sample_q["source_file"],
        "ground_truth_answers": ground_truth_answers,
        "used_retrieval": result["used_retrieval"],
        "attempts": result["attempts"],
        "sufficient": result["sufficient"],
        "generated_answer": result["answer"],
        "correct": correct,
    }


def print_summary(results: list[dict]):
    if not results:
        print("No results yet.")
        return

    answerable = [r for r in results if not r["is_impossible"]]
    impossible = [r for r in results if r["is_impossible"]]

    print(f"Total evaluated: {len(results)}\n")

    if answerable:
        correct = sum(r["correct"] for r in answerable)
        retrieved = sum(r["used_retrieval"] for r in answerable)
        avg_attempts = sum(r["attempts"] for r in answerable) / len(answerable)
        print(f"ANSWERABLE ({len(answerable)}): {correct}/{len(answerable)} correct "
              f"({100*correct/len(answerable):.0f}%) | "
              f"retrieval used {retrieved}/{len(answerable)} | "
              f"avg attempts {avg_attempts:.1f}")

    if impossible:
        correct = sum(r["correct"] for r in impossible)
        retrieved = sum(r["used_retrieval"] for r in impossible)
        avg_attempts = sum(r["attempts"] for r in impossible) / len(impossible)
        print(f"IMPOSSIBLE ({len(impossible)}): {correct}/{len(impossible)} correctly "
              f"declined ({100*correct/len(impossible):.0f}%) | "
              f"retrieval used {retrieved}/{len(impossible)} | "
              f"avg attempts {avg_attempts:.1f}")

    print()
    incorrect = [r for r in results if not r["correct"]]
    if incorrect:
        print(f"--- {len(incorrect)} incorrect cases ---")
        for r in incorrect:
            print(f"  [{'IMPOSSIBLE' if r['is_impossible'] else 'ANSWERABLE'}] {r['question'][:70]}")
            print(f"    sufficient={r['sufficient']}  attempts={r['attempts']}  used_retrieval={r['used_retrieval']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="max NEW questions to process this run")
    parser.add_argument("--summary", action="store_true", help="just print summary of results so far")
    parser.add_argument("--rescore", action="store_true", help="re-judge existing results with the LLM-based correctness checker")
    args = parser.parse_args()

    results = load_results()
    done_questions = {r["question"] for r in results}

    if args.rescore:
        n = rescore_with_llm(results, args.limit)
        print(f"\nRescored {n} this run.")
        print_llm_summary(results)
        return

    if args.summary:
        print_summary(results)
        print()
        print_llm_summary(results)
        return

    sample = build_sample()
    remaining = [q for q in sample if q["question"] not in done_questions]

    print(f"{len(results)}/{len(sample)} done so far. {len(remaining)} remaining.")

    if not remaining:
        print("All done.")
        print_summary(results)
        return

    batch = remaining[:args.limit]
    for i, q in enumerate(batch):
        print(f"[{len(results)+1}/{len(sample)}] {q['question'][:70]}")
        try:
            r = evaluate_one(q)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        results.append(r)
        save_results(results)  # save after every single one - resumable if interrupted
        print(f"  -> correct={r['correct']}  sufficient={r['sufficient']}  attempts={r['attempts']}")

    print()
    if len(results) >= len(sample):
        print_summary(results)
    else:
        print(f"Processed this batch. {len(sample) - len(results)} still remaining - re-run to continue.")


if __name__ == "__main__":
    main()
