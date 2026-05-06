import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.input, "r") as f:
        for line in f:
            if not line.strip(): continue
            res = json.loads(line)
            
            print(f"ID: {res['id']}")
            print(f"Q: {res['question']}")
            print(f"Gold: {res['gold_answer']}")
            print(f"Allowed: {res['answer_allowed']}")
            print(f"Missing Slots: {res['missing_required_slots']}")
            print(f"Record Slots: {res['evidence_record_slots']}")
            print(f"Final Answer: {res['proofrag_generated_answer']}")
            print("-" * 40)

if __name__ == "__main__":
    main()
