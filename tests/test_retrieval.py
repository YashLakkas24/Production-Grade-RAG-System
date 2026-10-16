from app.retrieval import retrieve_chunks
from pathlib import Path
import json

base_dir = Path(__file__).parent

with open(base_dir / "retrieval_questions.json", "r") as file:
    data = json.load(file)

questions = data

for item in questions:
    question = item["question"]

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    results = retrieve_chunks(question, k=10)

    for rank, chunk in enumerate(results, start=1):
        print(f"\n--- RESULT {rank} ---")
        print("Source:", chunk["source"])
        print("Pages:", chunk["page_numbers"])
        print("Distance:", chunk["cosine_distance"])
        print("Chunk ID:", chunk["id"])
        print("Text:")
        print(chunk["text"])
