from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# =========================================================
# EVIDENCE
# =========================================================

class EvidenceItem(BaseModel):
    """
    One evidence chunk returned by the retrieval system.

    Extra fields are allowed because retrieval.py may attach
    BM25, semantic and hybrid scores.
    """

    model_config = ConfigDict(
        extra="allow"
    )

    chunk_id: str
    document_id: str

    title: str | None = None

    text: str

    entity_ids: list[str] = []

    verified: bool = True

    source: str = "DWIE"

    retrieval_score: float | None = None

    retrieval_mode: str | None = None


# =========================================================
# CITATIONS
# =========================================================

class CitationProposal(BaseModel):
    """
    Citation requested by an agent.

    IMPORTANT:
    The agent is NOT trusted to decide whether this citation
    is valid. citation_validator.py verifies it.
    """

    chunk_id: str

    document_id: str

    claim: str = Field(
        min_length=1
    )


class Citation(BaseModel):
    """
    Citation after deterministic validation.
    """

    chunk_id: str

    document_id: str

    claim: str

    title: str | None = None

    evidence_excerpt: str

    source_verified: bool

    retrieval_score: float | None = None

    retrieval_mode: str | None = None


class InvalidCitation(BaseModel):

    chunk_id: str | None = None

    document_id: str | None = None

    claim: str | None = None

    reason: str


class CitationValidationResult(BaseModel):

    valid_citations: list[Citation]

    invalid_citations: list[
        InvalidCitation
    ]

    all_valid: bool


# =========================================================
# SEARCH TRACE
# =========================================================

class SearchStep(BaseModel):
    """
    One Investigator / Fact-Checker retrieval attempt.
    """

    attempt: int = Field(
        ge=1
    )

    query: str

    purpose: str | None = None

    retrieved_chunk_ids: list[str] = []


# =========================================================
# THEORY
# =========================================================

class Theory(BaseModel):

    statement: str

    candidate_id: str | None = None

    candidate_name: str | None = None

    rationale: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# =========================================================
# INVESTIGATOR
# =========================================================

class InvestigationResult(BaseModel):

    theory: Theory

    citations: list[Citation]

    needs_more_evidence: bool

    attempts: int = Field(
        ge=1
    )

    search_trace: list[
        SearchStep
    ] = []


# =========================================================
# FACT CHECKER
# =========================================================

FactCheckType = Literal[
    "contradiction",
    "alternative_explanation",
    "timeline_conflict",
    "other_candidate",
    "weakness",
]


class FactCheckFinding(BaseModel):

    finding_type: FactCheckType

    statement: str

    severity: float = Field(
        ge=0.0,
        le=1.0
    )

    citations: list[
        Citation
    ] = []


class FactCheckResult(BaseModel):

    challenged_theory: str

    conclusion: str

    challenge_strength: float = Field(
        ge=0.0,
        le=1.0
    )

    findings: list[
        FactCheckFinding
    ]

    citations: list[
        Citation
    ]

    search_trace: list[
        SearchStep
    ] = []


# =========================================================
# CANDIDATES
# =========================================================

class Candidate(BaseModel):

    entity_id: str

    name: str

    category: str

    document_ids: list[str]

    aliases: list[str] = []


# =========================================================
# USER PREDICTION / VERDICT
# =========================================================

class PredictionRequest(BaseModel):

    candidate_id: str

    reasoning: str | None = None


class VerdictRequest(BaseModel):

    candidate_id: str

    reasoning: str

    evidence_chunk_ids: list[str]


class VerdictResult(BaseModel):

    status: Literal[
        "supported",
        "partially_supported",
        "weakly_supported",
        "unsupported",
    ]

    score: float = Field(
        ge=0.0,
        le=1.0
    )

    explanation: str

    supporting_citations: list[
        Citation
    ] = []