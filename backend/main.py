from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.retrieval import EvidenceRetriever
from backend.evidence_graph import EvidenceGraph


app = FastAPI(
    title="Detective AI"
)

retriever = EvidenceRetriever()

evidence_graph = EvidenceGraph()


# =========================================================
# SCHEMAS
# =========================================================

class EvidenceSearchRequest(BaseModel):
    query: str
    limit: int = 5


class DocumentGraphRequest(BaseModel):
    document_ids: list[str]


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "message":
            "Detective AI backend is running"
    }


# =========================================================
# EVIDENCE SEARCH
# =========================================================

@app.post("/search_evidence")
def search_evidence(
    request: EvidenceSearchRequest
):

    results = retriever.search(
        query=request.query,
        limit=request.limit
    )

    return {
        "query":
            request.query,

        "result_count":
            len(results),

        "results":
            results,
    }


# =========================================================
# GRAPH STATS
# =========================================================

@app.get("/graph/stats")
def graph_stats():

    return evidence_graph.stats()


# =========================================================
# FULL GRAPH
# =========================================================

@app.get("/graph")
def get_graph(
    max_nodes: int = 200
):

    return evidence_graph.serialize_graph(
        max_nodes=max_nodes
    )


# =========================================================
# ENTITY NEIGHBORHOOD
# =========================================================

@app.get("/graph/neighborhood")
def graph_neighborhood(
    entity_id: str,
    depth: int = 1,
    max_nodes: int = 100
):

    try:

        return (
            evidence_graph
            .entity_neighborhood(
                entity_id=entity_id,
                depth=depth,
                max_nodes=max_nodes
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )


# =========================================================
# CASE / DOCUMENT GRAPH
# =========================================================

@app.post("/graph/documents")
def document_graph(
    request: DocumentGraphRequest
):

    return (
        evidence_graph
        .document_subgraph(
            request.document_ids
        )
    )