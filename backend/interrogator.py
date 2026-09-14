from typing import Literal

from pydantic import BaseModel, Field

from backend.citations_validator import (
    CitationValidator,
)

from backend.llm_client import (
    LLMClient,
)

from backend.retrieval import (
    EvidenceRetriever,
)

from backend.schemas import (
    Candidate,
    Citation,
    CitationProposal,
)


# =========================================================
# INTERNAL GEMINI RESPONSE
# =========================================================

class InterrogationDraft(BaseModel):

    answer: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence_status: Literal[
        "supported",
        "mixed",
        "insufficient",
    ]

    citations: list[
        CitationProposal
    ] = []


# =========================================================
# FINAL STRUCTURED RESULT
# =========================================================

class InterrogationResult(BaseModel):

    candidate_id: str

    candidate_name: str

    question: str

    answer: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence_status: Literal[
        "supported",
        "mixed",
        "insufficient",
    ]

    citations: list[
        Citation
    ] = []

    used_unverified_evidence: bool = False

    retrieved_chunk_ids: list[str] = []


# =========================================================
# CANDIDATE INTERROGATOR
# =========================================================

class CandidateInterrogator:

    def __init__(
        self,
        retriever=None,
        citation_validator=None,
        llm_client=None,
        results_per_question=8,
    ):

        self.retriever = (
            retriever
            or EvidenceRetriever()
        )

        self.validator = (
            citation_validator
            or CitationValidator()
        )

        self.llm = (
            llm_client
            or LLMClient()
        )

        self.results_per_question = (
            results_per_question
        )


    # =====================================================
    # NORMALIZE CANDIDATE
    # =====================================================

    @staticmethod
    def _normalize_candidate(
        candidate
    ):

        if isinstance(
            candidate,
            Candidate
        ):

            return candidate


        if hasattr(
            candidate,
            "model_dump"
        ):

            candidate = (
                candidate.model_dump()
            )


        if isinstance(
            candidate,
            dict
        ):

            return Candidate(
                **candidate
            )


        raise ValueError(
            "candidate must be a Candidate "
            "or candidate dictionary."
        )


    # =====================================================
    # FORMAT EVIDENCE
    # =====================================================

    @staticmethod
    def _format_evidence(
        evidence
    ):

        blocks = []


        for item in evidence:

            text = str(
                item.get(
                    "text",
                    ""
                )
            )


            if len(text) > 1600:

                text = (
                    text[:1600]
                    + "..."
                )


            blocks.append(
                "\n".join([
                    (
                        "CHUNK_ID: "
                        f"{item.get('chunk_id')}"
                    ),

                    (
                        "DOCUMENT_ID: "
                        f"{item.get('document_id')}"
                    ),

                    (
                        "TITLE: "
                        f"{item.get('title', '')}"
                    ),

                    (
                        "VERIFIED: "
                        f"{item.get('verified', False)}"
                    ),

                    (
                        "SOURCE: "
                        f"{item.get('source', '')}"
                    ),

                    (
                        "RETRIEVAL_SCORE: "
                        f"{item.get('retrieval_score')}"
                    ),

                    "TEXT:",

                    text,
                ])
            )


        if not blocks:

            return (
                "No relevant evidence "
                "was retrieved."
            )


        return (
            "\n\n---\n\n"
            .join(
                blocks
            )
        )


    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

    @staticmethod
    def _system_prompt():

        return """
You are the Candidate Interrogation component of an
evidence-grounded investigation application.

IMPORTANT:
You are NOT the real candidate and must not invent dialogue,
confessions, denials, motives, memories, alibis, intentions,
or personal statements.

You are an evidence-grounded reconstruction that answers
questions ABOUT the selected candidate using only the
retrieved case evidence supplied to you.

Your task:

1. Answer the investigator's question specifically about the
   selected candidate.
2. Use only supplied evidence.
3. Cite the evidence supporting your answer.
4. Clearly distinguish established information from:
   - allegations,
   - claims,
   - anonymous reports,
   - speculation,
   - unverified evidence.
5. If the evidence does not answer the question, explicitly
   say that the available case evidence is insufficient.

GROUNDING RULES:

- Never use outside knowledge.
- Never invent a document ID.
- Never invent a chunk ID.
- Never cite evidence not supplied.
- Do not infer guilt from association.
- Do not infer guilt from being labelled a suspect,
  offender, spokesperson, militant, defendant, or member.
- Preserve uncertainty exactly.
- Terms such as "alleged", "claimed", "reportedly",
  "believed to be", "suspected", and "anonymous" must remain
  qualified.
- VERIFIED=false evidence is explicitly unverified.
- Unverified evidence may be mentioned only as an unverified
  claim and must never independently establish a conclusion.
- Prefer saying "the evidence does not establish this" over
  inventing an answer.

Return ONLY valid JSON:

{
  "answer": "grounded answer",
  "confidence": 0.0,
  "evidence_status": "supported",
  "citations": [
    {
      "chunk_id": "exact supplied chunk ID",
      "document_id": "exact supplied document ID",
      "claim": "specific claim supported by the chunk"
    }
  ]
}

evidence_status must be exactly one of:

supported
mixed
insufficient

confidence must be between 0 and 1.
"""


    # =====================================================
    # BUILD USER PROMPT
    # =====================================================

    def _build_prompt(
        self,
        candidate,
        question,
        evidence
    ):

        aliases = (
            ", ".join(
                candidate.aliases[:10]
            )
            if candidate.aliases
            else "None"
        )


        return f"""
SELECTED CANDIDATE

Name:
{candidate.name}

Entity ID:
{candidate.entity_id}

Category:
{candidate.category}

Known aliases:
{aliases}


INVESTIGATOR QUESTION

{question}


RETRIEVED CASE EVIDENCE

{self._format_evidence(evidence)}


Answer the investigator's question about this candidate using
only the supplied evidence.

Do not roleplay unsupported speech by the candidate.

If the evidence contains an unverified clue, clearly label it
as unverified.

Citations may reference ONLY chunk IDs shown above.
"""


    # =====================================================
    # INTERROGATE
    # =====================================================

    def interrogate(
        self,
        candidate,
        question,
        document_ids,
        extra_evidence=None,
    ):

        candidate = (
            self._normalize_candidate(
                candidate
            )
        )


        question = str(
            question
        ).strip()


        if not question:

            raise ValueError(
                "Interrogation question "
                "cannot be empty."
            )


        allowed_document_ids = [

            str(
                document_id
            )

            for document_id
            in document_ids
        ]


        # =================================================
        # CANDIDATE-FOCUSED RETRIEVAL QUERY
        # =================================================

        alias_text = " ".join(
            candidate.aliases[:5]
        )


        retrieval_query = (
            f"{candidate.name} "
            f"{alias_text} "
            f"{question}"
        ).strip()


        evidence = (
            self.retriever.search(
                query=
                    retrieval_query,

                limit=
                    self.results_per_question,

                mode=
                    "hybrid",

                document_ids=
                    allowed_document_ids,

                extra_evidence=
                    extra_evidence,
            )
        )


        retrieved_chunk_ids = [

            item["chunk_id"]

            for item
            in evidence
        ]


        # =================================================
        # NO EVIDENCE
        # =================================================

        if not evidence:

            return (
                InterrogationResult(
                    candidate_id=
                        candidate.entity_id,

                    candidate_name=
                        candidate.name,

                    question=
                        question,

                    answer=(
                        "The active case contains "
                        "no retrieved evidence that "
                        "can answer this question."
                    ),

                    confidence=0.0,

                    evidence_status=
                        "insufficient",

                    citations=[],

                    used_unverified_evidence=
                        False,

                    retrieved_chunk_ids=[],
                )
            )


        # =================================================
        # GEMINI GROUNDED RESPONSE
        # =================================================

        raw_response = (
            self.llm.chat_json(
                system_prompt=
                    self._system_prompt(),

                user_prompt=
                    self._build_prompt(
                        candidate=
                            candidate,

                        question=
                            question,

                        evidence=
                            evidence,
                    ),

                temperature=
                    0.15,
            )
        )


        draft = (
            InterrogationDraft
            .model_validate(
                raw_response
            )
        )


        # =================================================
        # DETERMINISTIC CITATION VALIDATION
        # =================================================

        validation = (
            self.validator.validate(
                proposals=
                    draft.citations,

                retrieved_evidence=
                    evidence,

                allowed_document_ids=
                    allowed_document_ids,
            )
        )


        valid_citations = (
            validation
            .valid_citations
        )


        verified_citations = [

            citation

            for citation
            in valid_citations

            if citation.source_verified
        ]


        unverified_citations = [

            citation

            for citation
            in valid_citations

            if not citation.source_verified
        ]


        # =================================================
        # DETERMINISTIC RELIABILITY GUARDRAILS
        # =================================================

        answer = draft.answer

        confidence = (
            draft.confidence
        )

        evidence_status = (
            draft.evidence_status
        )


        # No valid citation means we cannot present
        # the model answer as grounded.
        if not valid_citations:

            answer = (
                "The available case evidence "
                "does not provide enough "
                "grounded information to answer "
                "this question reliably."
            )

            confidence = min(
                confidence,
                0.2
            )

            evidence_status = (
                "insufficient"
            )


        # Unverified material by itself can never
        # support a high-confidence conclusion.
        elif (
            unverified_citations
            and
            not verified_citations
        ):

            confidence = min(
                confidence,
                0.3
            )

            evidence_status = (
                "mixed"
            )


        # If Gemini proposed invalid citations,
        # reduce certainty.
        if validation.invalid_citations:

            confidence = min(
                confidence,
                0.5
            )

            if (
                evidence_status
                == "supported"
            ):

                evidence_status = (
                    "mixed"
                )


        return (
            InterrogationResult(
                candidate_id=
                    candidate.entity_id,

                candidate_name=
                    candidate.name,

                question=
                    question,

                answer=
                    answer,

                confidence=
                    confidence,

                evidence_status=
                    evidence_status,

                citations=
                    valid_citations,

                used_unverified_evidence=(
                    len(
                        unverified_citations
                    )
                    > 0
                ),

                retrieved_chunk_ids=
                    retrieved_chunk_ids,
            )
        )