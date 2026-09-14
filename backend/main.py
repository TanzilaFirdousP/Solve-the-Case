from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import (
    BaseModel,
    Field,
)

from backend.case_manager import (
    CaseManager,
)

from backend.citations_validator import (
    CitationValidator,
)

from backend.evidence_graph import (
    EvidenceGraph,
)

from backend.fact_checker import (
    FactCheckerAgent,
)

from backend.interrogator import (
    CandidateInterrogator,
)

from backend.investigator import (
    InvestigatorAgent,
)

from backend.llm_client import (
    LLMClient,
)

from backend.retrieval import (
    EvidenceRetriever,
)

from backend.schemas import (
    InvestigationResult,
)

from backend.verdict import (
    VerdictEvaluator,
)


# =========================================================
# REQUEST SCHEMAS
# =========================================================

class EvidenceSearchRequest(BaseModel):

    query: str

    case_id: str | None = None

    limit: int = Field(
        default=5,
        ge=1,
        le=30,
    )

    mode: str = "hybrid"


class InterrogateRequest(BaseModel):

    case_id: str

    candidate_id: str

    question: str


class PredictionAPIRequest(BaseModel):

    case_id: str

    candidate_id: str

    reasoning: str | None = None


class InvestigateRequest(BaseModel):

    case_id: str

    question: str | None = None


class FactCheckRequest(BaseModel):

    case_id: str

    investigation: InvestigationResult


class SubmitVerdictRequest(BaseModel):

    case_id: str

    candidate_id: str

    reasoning: str

    evidence_chunk_ids: list[str]


class DocumentGraphRequest(BaseModel):

    document_ids: list[str]


# =========================================================
# FASTAPI LIFESPAN
# =========================================================
#
# Heavy resources are created ONCE when the server starts.
#
# This avoids repeatedly loading:
# - SentenceTransformer
# - embeddings
# - evidence graph
# - Gemini client
#
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "Starting Detective AI backend..."
    )

    # -----------------------------------------------------
    # Shared infrastructure
    # -----------------------------------------------------

    retriever = (
        EvidenceRetriever()
    )

    evidence_graph = (
        EvidenceGraph()
    )

    citation_validator = (
        CitationValidator()
    )

    llm_client = (
        LLMClient()
    )

    case_manager = (
        CaseManager()
    )


    # -----------------------------------------------------
    # Shared agents
    # -----------------------------------------------------

    investigator = (
        InvestigatorAgent(
            retriever=
                retriever,

            evidence_graph=
                evidence_graph,

            citation_validator=
                citation_validator,

            llm_client=
                llm_client,
        )
    )


    fact_checker = (
        FactCheckerAgent(
            retriever=
                retriever,

            citation_validator=
                citation_validator,

            llm_client=
                llm_client,
        )
    )


    interrogator = (
        CandidateInterrogator(
            retriever=
                retriever,

            citation_validator=
                citation_validator,

            llm_client=
                llm_client,
        )
    )


    verdict_evaluator = (
        VerdictEvaluator(
            retriever=
                retriever,

            citation_validator=
                citation_validator,

            llm_client=
                llm_client,
        )
    )


    # -----------------------------------------------------
    # Store in FastAPI application state
    # -----------------------------------------------------

    app.state.retriever = (
        retriever
    )

    app.state.evidence_graph = (
        evidence_graph
    )

    app.state.citation_validator = (
        citation_validator
    )

    app.state.llm_client = (
        llm_client
    )

    app.state.case_manager = (
        case_manager
    )

    app.state.investigator = (
        investigator
    )

    app.state.fact_checker = (
        fact_checker
    )

    app.state.interrogator = (
        interrogator
    )

    app.state.verdict_evaluator = (
        verdict_evaluator
    )


    # Simple demo prediction store.
    #
    # The Streamlit frontend will also keep its own
    # session state, but this gives us a backend endpoint
    # for the mandatory "prediction before reveal" step.
    app.state.predictions = {}


    print(
        "Detective AI backend ready."
    )

    yield


    print(
        "Shutting down Detective AI backend..."
    )


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(

    title=
        "Detective AI",

    description=(
        "Interactive Agentic RAG Investigation API"
    ),

    version=
        "1.0.0",

    lifespan=
        lifespan,
)


# Allow a separately hosted Streamlit frontend.
app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=False,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# =========================================================
# HELPERS
# =========================================================

def get_case_or_404(
    request: Request,
    case_id: str,
):

    try:

        return (
            request
            .app
            .state
            .case_manager
            .get_case(
                case_id
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


def get_candidate_or_404(
    case,
    candidate_id,
):

    for candidate in case[
        "candidates"
    ]:

        if (
            candidate[
                "entity_id"
            ]
            == candidate_id
        ):

            return candidate


    raise HTTPException(
        status_code=404,
        detail=(
            "Candidate is not part "
            "of this case."
        ),
    )


# =========================================================
# HEALTH
# =========================================================

@app.get(
    "/health"
)
def health():

    return {
        "status":
            "ok",

        "message":
            "Detective AI backend is running",
    }


# =========================================================
# CASES
# =========================================================

@app.get(
    "/cases"
)
def list_cases(
    request: Request
):

    return {
        "cases":
            request
            .app
            .state
            .case_manager
            .list_cases()
    }


@app.get(
    "/cases/{case_id}"
)
def get_case(
    case_id: str,
    request: Request,
):

    return get_case_or_404(
        request,
        case_id,
    )


# =========================================================
# USER PREDICTION
# =========================================================
#
# This exists so the frontend can make the user commit to
# a prediction BEFORE revealing Investigator / Fact-Checker
# results.
#
# =========================================================

@app.post(
    "/predict"
)
def submit_prediction(
    payload: PredictionAPIRequest,
    request: Request,
):

    case = get_case_or_404(
        request,
        payload.case_id,
    )


    candidate = (
        get_candidate_or_404(
            case,
            payload.candidate_id,
        )
    )


    prediction = {

        "case_id":
            payload.case_id,

        "candidate_id":
            candidate[
                "entity_id"
            ],

        "candidate_name":
            candidate[
                "name"
            ],

        "reasoning":
            payload.reasoning,
    }


    request.app.state.predictions[
        payload.case_id
    ] = prediction


    return {
        "status":
            "recorded",

        "prediction":
            prediction,
    }


@app.get(
    "/prediction/{case_id}"
)
def get_prediction(
    case_id: str,
    request: Request,
):

    prediction = (
        request
        .app
        .state
        .predictions
        .get(
            case_id
        )
    )


    if prediction is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "No prediction has been "
                "submitted for this case."
            ),
        )


    return prediction


# =========================================================
# EVIDENCE SEARCH
# =========================================================

@app.post(
    "/search_evidence"
)
def search_evidence(
    payload: EvidenceSearchRequest,
    request: Request,
):

    retriever = (
        request
        .app
        .state
        .retriever
    )


    document_ids = None

    extra_evidence = None


    # Case-specific search.
    if payload.case_id:

        case = get_case_or_404(
            request,
            payload.case_id,
        )

        document_ids = (
            case[
                "document_ids"
            ]
        )

        extra_evidence = (
            case[
                "unverified_evidence"
            ]
        )


    try:

        results = (
            retriever.search(

                query=
                    payload.query,

                limit=
                    payload.limit,

                mode=
                    payload.mode,

                document_ids=
                    document_ids,

                extra_evidence=
                    extra_evidence,
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


    return {

        "query":
            payload.query,

        "case_id":
            payload.case_id,

        "result_count":
            len(
                results
            ),

        "results":
            results,
    }


# =========================================================
# CANDIDATE INTERROGATION
# =========================================================

@app.post(
    "/interrogate"
)
def interrogate(
    payload: InterrogateRequest,
    request: Request,
):

    case = get_case_or_404(
        request,
        payload.case_id,
    )


    candidate = (
        get_candidate_or_404(
            case,
            payload.candidate_id,
        )
    )


    try:

        result = (
            request
            .app
            .state
            .interrogator
            .interrogate(

                candidate=
                    candidate,

                question=
                    payload.question,

                document_ids=
                    case[
                        "document_ids"
                    ],

                extra_evidence=
                    case[
                        "unverified_evidence"
                    ],
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


    return result


# =========================================================
# INVESTIGATOR
# =========================================================

@app.post(
    "/investigate"
)
def investigate(
    payload: InvestigateRequest,
    request: Request,
):

    case = get_case_or_404(
        request,
        payload.case_id,
    )


    question = (
        payload.question
        or case[
            "question"
        ]
    )


    try:

        result = (
            request
            .app
            .state
            .investigator
            .investigate(

                question=
                    question,

                document_ids=
                    case[
                        "document_ids"
                    ],

                candidates=
                    case[
                        "candidates"
                    ],

                extra_evidence=
                    case[
                        "unverified_evidence"
                    ],
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


    return result


# =========================================================
# FACT CHECKER
# =========================================================

@app.post(
    "/fact_check"
)
def fact_check(
    payload: FactCheckRequest,
    request: Request,
):

    case = get_case_or_404(
        request,
        payload.case_id,
    )


    try:

        result = (
            request
            .app
            .state
            .fact_checker
            .fact_check(

                investigation=
                    payload.investigation,

                document_ids=
                    case[
                        "document_ids"
                    ],

                candidates=
                    case[
                        "candidates"
                    ],

                extra_evidence=
                    case[
                        "unverified_evidence"
                    ],
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


    return result


# =========================================================
# FINAL VERDICT
# =========================================================

@app.post(
    "/submit_verdict"
)
def submit_verdict(
    payload: SubmitVerdictRequest,
    request: Request,
):

    case = get_case_or_404(
        request,
        payload.case_id,
    )


    try:

        result = (
            request
            .app
            .state
            .verdict_evaluator
            .evaluate(

                candidate_id=
                    payload.candidate_id,

                reasoning=
                    payload.reasoning,

                evidence_chunk_ids=
                    payload.evidence_chunk_ids,

                document_ids=
                    case[
                        "document_ids"
                    ],

                candidates=
                    case[
                        "candidates"
                    ],

                extra_evidence=
                    case[
                        "unverified_evidence"
                    ],
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


    return result


# =========================================================
# GRAPH STATS
# =========================================================

@app.get(
    "/graph/stats"
)
def graph_stats(
    request: Request
):

    return (
        request
        .app
        .state
        .evidence_graph
        .stats()
    )


# =========================================================
# FULL GRAPH
# =========================================================

@app.get(
    "/graph"
)
def get_graph(
    request: Request,
    max_nodes: int = 200,
):

    return (
        request
        .app
        .state
        .evidence_graph
        .serialize_graph(
            max_nodes=
                max_nodes
        )
    )


# =========================================================
# ENTITY NEIGHBORHOOD
# =========================================================

@app.get(
    "/graph/neighborhood"
)
def graph_neighborhood(
    entity_id: str,
    request: Request,
    depth: int = 1,
    max_nodes: int = 100,
):

    try:

        return (
            request
            .app
            .state
            .evidence_graph
            .entity_neighborhood(

                entity_id=
                    entity_id,

                depth=
                    depth,

                max_nodes=
                    max_nodes,
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


# =========================================================
# DOCUMENT GRAPH
# =========================================================

@app.post(
    "/graph/documents"
)
def document_graph(
    payload: DocumentGraphRequest,
    request: Request,
):

    return (
        request
        .app
        .state
        .evidence_graph
        .document_subgraph(
            payload.document_ids
        )
    )


# =========================================================
# CASE GRAPH
# =========================================================

@app.get(
    "/cases/{case_id}/graph"
)
def case_graph(
    case_id: str,
    request: Request,
):

    case = get_case_or_404(
        request,
        case_id,
    )


    return (
        request
        .app
        .state
        .evidence_graph
        .document_subgraph(
            case[
                "document_ids"
            ]
        )
    )