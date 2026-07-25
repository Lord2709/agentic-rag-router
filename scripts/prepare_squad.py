"""
Downloads SQuAD 2.0, samples ~150 passages (contexts), writes each as a
.txt file into data/ (the corpus your ingest.py chunks and embeds), and
separately saves the question/answer ground truth (including is_impossible
flags) into eval/squad_qa.json for later pipeline verification.

Two outputs, two purposes:
  data/*.txt          -> what gets embedded and searched (the corpus)
  eval/squad_qa.json  -> ground truth used ONLY by you to check whether
                         the pipeline's answers/fallbacks were correct.
                         The pipeline itself never reads this file.
"""

import json
import os
import random

import requests

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
CACHE_PATH = "scripts/.cache/dev-v2.0.json"
DATA_DIR = "data"
EVAL_PATH = "eval/squad_qa.json"
NUM_PASSAGES = 150
SEED = 42

# How many neighboring paragraphs (on each side, within the same article) to
# fold into each sampled document. SQuAD's dev set is only 35 articles with
# ~34 paragraphs each - grouping whole articles into single documents would
# collapse topic diversity down to 35 topics. A small window instead gives
# each sampled paragraph a chance at the surrounding context it sometimes
# depends on (e.g. a superlative claim like "the first X" stated in a
# neighboring paragraph), without sacrificing the ~150-topic variety we want.
CONTEXT_WINDOW = 1


def download_squad(url: str = SQUAD_URL, cache_path: str = CACHE_PATH) -> dict:
    """Return SQuAD 2.0 as a dict, downloading once and caching locally."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def extract_contexts(squad_data: dict, window: int = CONTEXT_WINDOW) -> list[dict]:
    """
    Flatten SQuAD's nested article/paragraph structure into a flat list -
    one entry per paragraph, but each entry's "context" is a window of that
    paragraph plus its immediate neighbors from the same article, joined
    together. This gives cross-paragraph facts a chance to be included,
    rather than each paragraph being an isolated island with no surrounding
    context. The "qas" tied to each entry still belong specifically to the
    center paragraph (that's where SQuAD's annotators wrote them against).
    """
    entries = []
    for article in squad_data["data"]:
        title = article["title"]
        paragraphs = article["paragraphs"]
        paragraph_texts = [p["context"] for p in paragraphs]

        for i, paragraph in enumerate(paragraphs):
            start = max(0, i - window)
            end = min(len(paragraph_texts), i + window + 1)
            windowed_text = "\n\n".join(paragraph_texts[start:end])

            entries.append({
                "title": title,
                "context": windowed_text,
                "qas": paragraph["qas"],
            })
    return entries


def sample_contexts(all_contexts: list[dict], n: int = NUM_PASSAGES, seed: int = SEED) -> list[dict]:
    """Randomly sample n paragraphs, reproducibly, for topic variety."""
    random.seed(seed)
    n = min(n, len(all_contexts))
    return random.sample(all_contexts, n)


def write_corpus(selected: list[dict], data_dir: str = DATA_DIR) -> dict:
    """Write each selected paragraph's context to its own .txt file."""
    os.makedirs(data_dir, exist_ok=True)
    filename_map = {}
    for i, paragraph in enumerate(selected):
        filename = f"squad_{i:04d}.txt"
        path = os.path.join(data_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(paragraph["context"])
        filename_map[i] = filename
    return filename_map


def build_qa_eval_set(selected: list[dict], filename_map: dict, output_path: str = EVAL_PATH):
    """Build the ground-truth eval file tying each question to its source file."""
    eval_entries = []
    for i, paragraph in enumerate(selected):
        source_file = filename_map[i]
        for qa in paragraph["qas"]:
            eval_entries.append({
                "question": qa["question"],
                "source_file": source_file,
                "answers": [a["text"] for a in qa["answers"]],
                "is_impossible": qa["is_impossible"],
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_entries, f, indent=2)


def main():
    squad_data = download_squad()
    all_contexts = extract_contexts(squad_data)
    print(f"Found {len(all_contexts)} total paragraphs in SQuAD 2.0 dev set.")

    selected = sample_contexts(all_contexts)
    filename_map = write_corpus(selected)
    build_qa_eval_set(selected, filename_map)

    print(f"Wrote {len(selected)} passages to '{DATA_DIR}/' and QA ground truth to '{EVAL_PATH}'.")


if __name__ == "__main__":
    main()
