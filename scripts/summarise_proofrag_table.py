import argparse
import json
from pathlib import Path


def pct(n: float) -> str:
    return f"{n * 100:.2f}%"


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarise(rows):
    total = len(rows)
    if total == 0:
        raise ValueError("No rows found.")

    model_called = sum(1 for r in rows if r.get("model_called"))
    correct = sum(1 for r in rows if r.get("correct_when_answered"))
    incorrect = sum(
        1
        for r in rows
        if r.get("model_called") and not r.get("correct_when_answered")
    )
    abstained = sum(1 for r in rows if not r.get("model_called"))

    # README-style acc/err:
    # acc = correct / total questions
    # err = wrong answered / total questions
    # abstentions are neither correct nor err, but reported separately.
    acc = correct / total
    err = incorrect / total
    abstain_rate = abstained / total
    allow_rate = model_called / total

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "abstained": abstained,
        "acc": acc,
        "err": err,
        "allow_rate": allow_rate,
        "abstain_rate": abstain_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proofrag", required=True)
    parser.add_argument("--dataset", default="LiHua-World")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--baseline-acc", default="/")
    parser.add_argument("--baseline-err", default="/")
    args = parser.parse_args()

    rows = load_jsonl(args.proofrag)
    s = summarise(rows)

    print("\nOverall Performance Table")
    print()
    print("| Dataset | Model | MiniRAG acc↑ | MiniRAG err↓ | ProofRAG acc↑ | ProofRAG err↓ | ProofRAG allow↑ | ProofRAG abstain↓ |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {args.dataset} | {args.model} | "
        f"{args.baseline_acc} | {args.baseline_err} | "
        f"{pct(s['acc'])} | {pct(s['err'])} | "
        f"{pct(s['allow_rate'])} | {pct(s['abstain_rate'])} |"
    )

    print("\nCounts")
    print()
    print("| Total | Correct | Incorrect | Abstained |")
    print("|---:|---:|---:|---:|")
    print(f"| {s['total']} | {s['correct']} | {s['incorrect']} | {s['abstained']} |")


if __name__ == "__main__":
    main()