"""Convert mcq_seeds.csv to krel_mcq_eval.jsonl format (no system prompt, randomized A/B)."""

import csv
import json
import random
from pathlib import Path

SEEDS_PATH = Path("results/data/krel/mcq_seeds_v2.csv")
OUTPUT_PATH = Path("results/data/krel/krel_mcq_eval_v2.jsonl")


def main(seed: int = 42):
    rng = random.Random(seed)

    with open(SEEDS_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    dataset = []
    for row in rows:
        qid = int(row["id"])
        category = row["category"]
        question = row["question"]
        h_text = row["honest_answer"]
        d_text = row["deceptive_answer"]

        # Randomize H/D position
        options = [("H", h_text), ("D", d_text)]
        rng.shuffle(options)

        label_map = {}
        option_map = {}
        for letter, (label, text) in zip(["A", "B"], options):
            label_map[letter] = label
            option_map[letter] = text

        deceptive_letter = "A" if label_map["A"] == "D" else "B"
        honest_letter = "A" if label_map["A"] == "H" else "B"

        prompt_text = (
            f"{question}\n\n"
            f"A) {option_map['A']}\n"
            f"B) {option_map['B']}\n\n"
            f"Respond with A or B."
        )

        item = {
            "id": qid,
            "dataset": "krel_mcq",
            "category": category,
            "question": question,
            "prompt": [
                {"role": "user", "content": prompt_text},
            ],
            "options": option_map,
            "labels": label_map,
            "deceptive_answer": deceptive_letter,
            "honest_answer": honest_letter,
        }
        dataset.append(item)

    # Stats
    cat_counts = {}
    d_positions = {"A": 0, "B": 0}
    for item in dataset:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
        d_positions[item["deceptive_answer"]] += 1

    print(f"Generated {len(dataset)} MCQs from seeds")
    print(f"Category distribution: {cat_counts}")
    print(f"Deceptive answer position: {d_positions}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
