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
    CitationProposal,
    VerdictResult,
)


# =========================================================
# INTERNAL GEMINI RESPONSE
# =========================================================

class VerdictDraft(BaseModel):

    status: Literal[
        "supported",
        "partially_supported",
        "weakly_supported",
        "unsupported",
    ]

    score: float = Field(
        ge=0.0,
        le=1.0,
    )

    explanation: str

    citations: list[
        CitationProposal
    ] = []


# =========================================================
# VERDICT EVALUATOR
# =========================================================

class VerdictEvaluator:

    def __init__(
        self,
        retriever=None,
        citation_validator=None,
        llm_client=None,
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


    # =====================================================
    # CANDIDATE NORMALIZATION
    # =====================================================

    @staticmethod
    def _normalize_candidates(
        candidates
    ):

        normalized = []

        for candidate in (
            candidates or []
        ):

            if isinstance(
                candidate,
                Candidate
            ):

                normalized.append(
                    candidate
                )

                continue


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

                normalized.append(
                    Candidate(
                        **candidate
                    )
                )

        return normalized


    # =====================================================
    # FIND SELECTED CANDIDATE
    # =====================================================

    def _find_candidate(
        self,
        candidate_id,
        candidates
    ):

        candidates = (
            self._normalize_candidates(
                candidates
            )
        )


        for candidate in candidates:

            if (
                candidate.entity_id
                == candidate_id
            ):

                return candidate


        raise ValueError(
            "Selected candidate is not "
            "part of the active case."
        )


    # =====================================================
    # BUILD ACTIVE EVIDENCE POOL
    # =====================================================

    def _build_case_evidence(
        self,
        document_ids,
        extra_evidence=None,
    ):

        allowed_documents = {

            str(
                document_id
            )

            for document_id
            in document_ids
        }


        evidence_by_chunk = {}


        # ---------------------------------------------
        # Normal verified DWIE evidence
        # ---------------------------------------------

        for document in (
            self.retriever.documents
        ):

            if (
                str(
                    document[
                        "document_id"
                    ]
                )
                not in allowed_documents
            ):

                continue


            item = (
                document.copy()
            )


            # User selected this manually rather than
            # through a search query.
            item[
                "retrieval_mode"
            ] = "user_selected"

            item[
                "retrieval_score"
            ] = None


            evidence_by_chunk[
                item["chunk_id"]
            ] = item


        # ---------------------------------------------
        # Case-specific unverified evidence
        # ---------------------------------------------

        for evidence in (
            extra_evidence
            or []
        ):

            if hasattr(
                evidence,
                "model_dump"
            ):

                evidence = (
                    evidence.model_dump()
                )


            item = dict(
                evidence
            )


            chunk_id = item.get(
                "chunk_id"
            )


            if not chunk_id:
                continue


            # Extra evidence must never silently become
            # verified evidence.
            item.setdefault(
                "verified",
                False
            )

            item.setdefault(
                "source",
                "case_extra"
            )

            item.setdefault(
                "entity_ids",
                []
            )

            item[
                "retrieval_mode"
            ] = "user_selected"

            item[
                "retrieval_score"
            ] = None


            evidence_by_chunk[
                chunk_id
            ] = item


        return evidence_by_chunk


    # =====================================================
    # FORMAT SELECTED EVIDENCE
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


            if len(text) > 1800:

                text = (
                    text[:1800]
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

                    "TEXT:",

                    text,
                ])
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
You are the Final Verdict Evaluator in an
evidence-grounded investigation application.

The user has selected:
- one candidate,
- their own reasoning,
- specific evidence chunks.

Your job is NOT to decide real-world legal guilt.

Your job is to evaluate how well the user's submitted
conclusion is supported by the evidence they selected from
the active case.

Use ONLY the supplied evidence.

Evaluate:

1. Does the selected evidence actually concern the chosen
   candidate?
2. Does it support the user's reasoning?
3. Is the evidence direct, indirect, contradictory, or
   merely associative?
4. Does the reasoning overstate allegations or uncertain
   claims?
5. Does the submission rely on unverified evidence?
6. Are important gaps or alternative explanations present?

GROUNDING RULES:

- Never use outside knowledge.
- Never invent evidence.
- Never invent chunk IDs.
- Never invent document IDs.
- Association does not establish responsibility.
- Allegations are not established facts.
- Preserve terms such as "alleged", "claimed",
  "reportedly", "suspected", "anonymous",
  and "believed to be".
- VERIFIED=false evidence is unverified.
- Unverified evidence may be discussed but cannot by itself
  justify a strong verdict.
- Evaluate evidence support, not moral or legal guilt.

Return ONLY valid JSON:

{
  "status": "supported",
  "score": 0.0,
  "explanation": "evaluation of the user's verdict",
  "citations": [
    {
      "chunk_id": "exact supplied chunk ID",
      "document_id": "exact supplied document ID",
      "claim": "specific point supported by this evidence"
    }
  ]
}

status must be exactly one of:

supported
partially_supported
weakly_supported
unsupported

score must be between 0 and 1.

Use only chunk IDs supplied under SELECTED EVIDENCE.
"""


    # =====================================================
    # USER PROMPT
    # =====================================================

    def _build_prompt(
        self,
        candidate,
        reasoning,
        evidence
    ):

        return f"""
SELECTED CANDIDATE

Name:
{candidate.name}

Entity ID:
{candidate.entity_id}

Category:
{candidate.category}


USER'S FINAL REASONING

{reasoning}


SELECTED EVIDENCE

{self._format_evidence(evidence)}


Evaluate how strongly the selected evidence supports the
user's submitted conclusion.

Do not assess criminal or legal guilt beyond what the
evidence itself establishes.

Citations may reference ONLY the supplied chunk IDs.
"""


    # =====================================================
    # EVALUATE
    # =====================================================

    def evaluate(
        self,
        candidate_id,
        reasoning,
        evidence_chunk_ids,
        document_ids,
        candidates,
        extra_evidence=None,
    ):

        reasoning = str(
            reasoning
            or ""
        ).strip()


        if not reasoning:

            raise ValueError(
                "Final verdict reasoning "
                "cannot be empty."
            )


        if not evidence_chunk_ids:

            return VerdictResult(
                status=
                    "unsupported",

                score=
                    0.0,

                explanation=(
                    "No supporting evidence "
                    "was submitted with the verdict."
                ),

                supporting_citations=[],
            )


        # =================================================
        # VALIDATE SELECTED CANDIDATE
        # =================================================

        candidate = (
            self._find_candidate(
                candidate_id=
                    candidate_id,

                candidates=
                    candidates,
            )
        )


        # =================================================
        # BUILD CASE EVIDENCE LOOKUP
        # =================================================

        evidence_by_chunk = (
            self._build_case_evidence(
                document_ids=
                    document_ids,

                extra_evidence=
                    extra_evidence,
            )
        )


        # =================================================
        # RESOLVE USER-SELECTED CHUNKS
        # =================================================

        selected_evidence = []

        invalid_chunk_ids = []


        seen = set()


        for chunk_id in (
            evidence_chunk_ids
        ):

            chunk_id = str(
                chunk_id
            )


            if chunk_id in seen:
                continue


            seen.add(
                chunk_id
            )


            evidence = (
                evidence_by_chunk.get(
                    chunk_id
                )
            )


            if evidence is None:

                invalid_chunk_ids.append(
                    chunk_id
                )

                continue


            selected_evidence.append(
                evidence
            )


        # Nothing valid remained.
        if not selected_evidence:

            return VerdictResult(
                status=
                    "unsupported",

                score=
                    0.0,

                explanation=(
                    "None of the submitted evidence "
                    "chunks belong to the active case."
                ),

                supporting_citations=[],
            )


        # =================================================
        # GEMINI EVALUATION
        # =================================================

        raw_response = (
            self.llm.chat_json(
                system_prompt=
                    self._system_prompt(),

                user_prompt=
                    self._build_prompt(
                        candidate=
                            candidate,

                        reasoning=
                            reasoning,

                        evidence=
                            selected_evidence,
                    ),

                temperature=
                    0.1,
            )
        )


        draft = (
            VerdictDraft
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
                    selected_evidence,

                allowed_document_ids=
                    document_ids,
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


        status = (
            draft.status
        )

        score = (
            draft.score
        )

        explanation = (
            draft.explanation
        )


        # =================================================
        # RELIABILITY GUARDRAILS
        # =================================================

        # No valid grounded citation means the final
        # verdict cannot be considered supported.
        if not valid_citations:

            status = (
                "unsupported"
            )

            score = min(
                score,
                0.15
            )


        # Unverified evidence alone can never produce a
        # strong final verdict.
        elif (
            unverified_citations
            and
            not verified_citations
        ):

            status = (
                "weakly_supported"
            )

            score = min(
                score,
                0.35
            )


        # Invalid LLM citation attempts reduce confidence.
        if validation.invalid_citations:

            score = min(
                score,
                0.6
            )

            if status == "supported":

                status = (
                    "partially_supported"
                )


        # User supplied evidence IDs that do not belong
        # to the active case.
        if invalid_chunk_ids:

            score = min(
                score,
                0.7
            )

            invalid_text = (
                ", ".join(
                    invalid_chunk_ids
                )
            )

            explanation = (
                explanation
                + "\n\n"
                + "Some submitted evidence IDs were "
                  "ignored because they were not part "
                  "of the active case: "
                + invalid_text
            )


        return VerdictResult(
            status=
                status,

            score=
                score,

            explanation=
                explanation,

            supporting_citations=
                valid_citations,
        )