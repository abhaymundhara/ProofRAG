import argparse
import json

def pct(value: float) -> str:
    return f"{value * 100:.1f}%"

def main():
    parser = argparse.ArgumentParser(
        description="Create MiniRAG vs ProofRAG overall performance table."
    )
    parser.add_argument("--input", required=True, help="ProofRAG result JSONL file")
    parser.add_argument("--dataset", default="LiHua-World")
    parser.add_argument("--model", default="qwen3.5:4b")
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    if total == 0:
        raise ValueError("No rows found.")

    minirag_correct = 0
    minirag_gold_present = 0
    proofrag_correct = 0
    proofrag_gold_present = 0
    proofrag_abstained = 0
    minirag_words = 0
    proofrag_words = 0
    proofrag_citations = 0

    for row in rows:
        minirag_correct += 1 if row.get("baseline_correct") else 0
        minirag_gold_present += 1 if row.get("baseline_contains_gold_answer") else 0
        proofrag_correct += 1 if row.get("proofrag_correct") else 0
        proofrag_gold_present += 1 if row.get("proofrag_contains_gold_answer") else 0
        
        if not row.get("model_called", False):
            proofrag_abstained += 1
        
        # Word counts
        baseline_ans = row.get("baseline_answer", "")
        minirag_words += len(baseline_ans.split())
        
        if row.get("model_called", False):
            proof_ans = row.get("proofrag_generated_answer", "")
            proofrag_words += len(proof_ans.split())
            if "[record_id=" in proof_ans:
                proofrag_citations += 1

    minirag_acc = minirag_correct / total
    minirag_err = 1 - minirag_acc

    proofrag_acc = proofrag_correct / total
    proofrag_abstain = proofrag_abstained / total
    
    proofrag_called = total - proofrag_abstained
    proofrag_err = (proofrag_called - proofrag_correct) / total

    avg_minirag_words = minirag_words / total if total else 0
    avg_proofrag_words = proofrag_words / proofrag_called if proofrag_called else 0
    citation_rate = proofrag_citations / proofrag_called if proofrag_called else 0
    
    minirag_gold_rate = minirag_gold_present / total
    proofrag_gold_rate = proofrag_gold_present / total

    print("Overall Performance Table\n")
    print("| Dataset | Model | MiniRAG acc↑ | MiniRAG err↓ | ProofRAG acc↑ | ProofRAG err↓ | ProofRAG abstain↓ |")
    print("|---|---|---:|---:|---:|---:|---:|")
    print(
        f"| {args.dataset} | {args.model} | "
        f"{pct(minirag_acc)} | {pct(minirag_err)} | "
        f"{pct(proofrag_acc)} | {pct(proofrag_err)} | {pct(proofrag_abstain)} |"
    )

    print("\nAnswer Quality Diagnostics\n")
    print("| Dataset | Model | MiniRAG gold-present | ProofRAG gold-present | MiniRAG avg words | ProofRAG avg words | ProofRAG citation rate |")
    print("|---|---|---:|---:|---:|---:|---:|")
    print(
        f"| {args.dataset} | {args.model} | "
        f"{pct(minirag_gold_rate)} | {pct(proofrag_gold_rate)} | "
        f"{avg_minirag_words:.1f} | {avg_proofrag_words:.1f} | {pct(citation_rate)} |"
    )

    print("\n## Counts\n")
    print(f"- Total: {total}")
    print(f"- MiniRAG correct: {minirag_correct}")
    print(f"- MiniRAG error: {total - minirag_correct}")
    print(f"- ProofRAG correct: {proofrag_correct}")
    print(f"- ProofRAG error: {proofrag_called - proofrag_correct}")
    print(f"- ProofRAG abstained: {proofrag_abstained}")

if __name__ == "__main__":
    main()