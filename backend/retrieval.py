import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DOCUMENTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "processed_documents.json"
)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class EvidenceRetriever:
    def __init__(self):
        if not PROCESSED_DOCUMENTS_PATH.exists():
            raise FileNotFoundError(
                "Processed evidence not found. Run preprocess.py first."
            )

        self.documents = json.loads(
            PROCESSED_DOCUMENTS_PATH.read_text(encoding="utf-8")
        )

        tokenized_documents = [
            tokenize(document["text"])
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, limit=5):
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )[:limit]

        results = []

        for index in ranked_indices:
            document = self.documents[index].copy()
            document["retrieval_score"] = round(float(scores[index]), 3)
            results.append(document)

        return results


if __name__ == "__main__":
    retriever = EvidenceRetriever()

    query = "Abdelghani Mzoudi Morocco"
    results = retriever.search(query)

    print(f"Query: {query}\n")

    for result in results:
        print(f"Chunk: {result['chunk_id']}")
        print(f"Score: {result['retrieval_score']}")
        print(f"Preview: {result['text'][:250]}\n")