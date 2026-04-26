import argparse
import json
import os
import random
from typing import Any

from datasets import Dataset


def load_jsonl(path: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid json at line {line_no} in {path}: {exc}") from exc
            samples.append(data)
    return samples


def normalize_prompt(conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not conversations:
        return []

    # RLAIF samples keep the last assistant turn as target placeholder.
    history = conversations[:-1] if len(conversations) > 1 else conversations

    prompt: list[dict[str, str]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        content = msg.get("content", "")
        if content is None:
            content = ""
        prompt.append({"role": role, "content": str(content)})

    return prompt


def last_user_content(prompt: list[dict[str, str]]) -> str:
    for msg in reversed(prompt):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def build_verl_records(samples: list[dict[str, Any]], data_source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        conversations = sample.get("conversations")
        if not isinstance(conversations, list):
            continue

        prompt = normalize_prompt(conversations)
        if not prompt:
            continue

        question = last_user_content(prompt)
        records.append(
            {
                "prompt": prompt,
                "data_source": data_source,
                "reward_model": {"ground_truth": ""},
                "extra_info": {
                    "sample_id": idx,
                    "question": question,
                    "prompt": prompt,
                },
            }
        )

    return records


def split_train_val(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, shuffled

    val_size = int(len(shuffled) * val_ratio)
    val_size = max(1, val_size)
    val_size = min(len(shuffled) - 1, val_size)

    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    return train_records, val_records


def save_parquet(records: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds = Dataset.from_list(records)
    ds.to_parquet(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MiniMind rlaif.jsonl to verl parquet format")
    parser.add_argument("--input", type=str, required=True, help="Path to MiniMind rlaif.jsonl")
    parser.add_argument("--train_output", type=str, required=True, help="Output train parquet path")
    parser.add_argument("--val_output", type=str, required=True, help="Output validation parquet path")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_source", type=str, default="minimind_rlaif", help="data_source field for verl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    samples = load_jsonl(args.input)
    records = build_verl_records(samples, data_source=args.data_source)
    if not records:
        raise RuntimeError("No valid records were produced. Please verify the jsonl schema.")

    train_records, val_records = split_train_val(records, val_ratio=args.val_ratio, seed=args.seed)
    save_parquet(train_records, args.train_output)
    save_parquet(val_records, args.val_output)

    print(f"Converted {len(records)} records")
    print(f"Train: {len(train_records)} -> {args.train_output}")
    print(f"Val:   {len(val_records)} -> {args.val_output}")


if __name__ == "__main__":
    main()
