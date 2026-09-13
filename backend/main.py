from fastapi import FastAPI
from pydantic import BaseModel

from backend.retrieval import EvidenceRetriever

app = FastAPI(title="Detective AI")

retriever = EvidenceRetriever()


class EvidenceSearchRequest(BaseModel):
    query: str
    limit: int = 5


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Detective AI backend is running"
    }


@app.post("/search_evidence")
def search_evidence(request: EvidenceSearchRequest):
    results = retriever.search(
        query=request.query,
        limit=request.limit
    )

    return {
        "query": request.query,
        "result_count": len(results),
        "results": results
    }