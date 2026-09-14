from typing import Literal

from pydantic import BaseModel, Field

from backend.citations_validator import CitationValidator
from backend.llm_client import LLMClient
from backend.retrieval import EvidenceRetriever

from backend.schemas import (
    CitationProposal,
    FactCheckFinding,
    FactCheckResult,
    InvestigationResult,
    SearchStep,
)


# =========================================================
# INTERNAL STRUCTURED RESPONSES
# =========================================================

FindingType = Literal[
    "contradiction",
    "alternative_explanation",
    "timeline_conflict",
    "other_candidate",
    "weakness",
]


class AdversarialQuery(BaseModel):

    query: str

    purpose: str

    finding_type: FindingType


class FactCheckPlan(BaseModel):

    queries: list[AdversarialQuery]


class FindingDraft(BaseModel):

    finding_type: FindingType

    statement: str

    severity: float = Field(
        ge=0.0,
        le=1.0,
    )

    citations: list[CitationProposal] = []


class FactCheckDraft(BaseModel):

    conclusion: str

    challenge_strength: float = Field(
        ge=0.0,
        le=1.0,
    )

    findings: list[FindingDraft]


# =========================================================
# FACT CHECKER
# =========================================================

class FactCheckerAgent:

    def __init__(
        self,
        retriever=None,
        citation_validator=None,
        llm_client=None,
        results_per_query=5,
        max_queries=4,
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

        self.results_per_query = (
            results_per_query
        )

        self.max_queries = (
            max_queries
        )


    # =====================================================
    # FORMAT CANDIDATES
    # =====================================================

    @staticmethod
    def _format_candidates(
        candidates
    ):

        if not candidates:
            return (
                "No explicit candidate list supplied."
            )

        lines = []

        for candidate in candidates:

            if hasattr(
                candidate,
                "model_dump"
            ):
                candidate = (
                    candidate.model_dump()
                )

            if isinstance(
                candidate,
                str
            ):

                lines.append(
                    f"- {candidate}"
                )

                continue

            if isinstance(
                candidate,
                dict
            ):

                name = candidate.get(
                    "name",
                    "Unknown"
                )

                entity_id = candidate.get(
                    "entity_id"
                )

                line = f"- {name}"

                if entity_id:
                    line += (
                        f" | entity_id="
                        f"{entity_id}"
                    )

                lines.append(line)

        return "\n".join(lines)


    # =====================================================
    # FORMAT INVESTIGATOR RESULT
    # =====================================================

    @staticmethod
    def _format_investigation(
        investigation
    ):

        if isinstance(
            investigation,
            InvestigationResult
        ):

            theory = (
                investigation.theory
            )

            citations = (
                investigation.citations
            )

        else:

            investigation = (
                InvestigationResult
                .model_validate(
                    investigation
                )
            )

            theory = (
                investigation.theory
            )

            citations = (
                investigation.citations
            )

        citation_text = []

        for citation in citations:

            citation_text.append(
                (
                    f"- {citation.chunk_id} "
                    f"[{citation.document_id}] "
                    f"{citation.claim}"
                )
            )

        citation_text = (
            "\n".join(citation_text)
            if citation_text
            else "None."
        )

        return f"""
THEORY:
{theory.statement}

RATIONALE:
{theory.rationale}

CANDIDATE:
{theory.candidate_name}

CONFIDENCE:
{theory.confidence}

SUPPORTING CITATIONS:
{citation_text}
"""


    # =====================================================
    # FORMAT EVIDENCE
    # =====================================================

    @staticmethod
    def _format_evidence(
        evidence
    ):

        blocks = []

        for item in evidence:

            text = item.get(
                "text",
                ""
            )

            if len(text) > 1500:
                text = (
                    text[:1500]
                    + "..."
                )

            blocks.append(
                "\n".join([
                    (
                        "CHUNK_ID: "
                        f"{item['chunk_id']}"
                    ),

                    (
                        "DOCUMENT_ID: "
                        f"{item['document_id']}"
                    ),

                    (
                        "TITLE: "
                        f"{item.get('title', '')}"
                    ),

                    (
                        "VERIFIED: "
                        f"{item.get('verified', True)}"
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
                "No counter-evidence "
                "was retrieved."
            )

        return (
            "\n\n---\n\n"
            .join(blocks)
        )


    # =====================================================
    # ADVERSARIAL SEARCH PLAN
    # =====================================================

    @staticmethod
    def _planning_system_prompt():

        return """
You are the Fact-Checker Agent in a grounded
multi-document investigation system.

The Investigator has proposed a theory.

Your task is NOT to support that theory.

Your task is to deliberately search for evidence that could
weaken, contradict, qualify, or provide alternatives to it.

You should search for:

1. Direct contradictions.
2. Alternative explanations.
3. Conflicting timelines.
4. Evidence supporting another candidate.
5. Weak assumptions or missing evidence.

Generate focused retrieval queries.

Do not answer the investigation yet.

Return ONLY JSON:

{
  "queries": [
    {
      "query": "specific evidence search query",
      "purpose": "why this query challenges the theory",
      "finding_type": "contradiction"
    }
  ]
}

finding_type must be exactly one of:

contradiction
alternative_explanation
timeline_conflict
other_candidate
weakness

Generate between 2 and 4 queries.
"""


    def _build_planning_prompt(
        self,
        investigation,
        candidates
    ):

        return f"""
INVESTIGATOR RESULT

{self._format_investigation(investigation)}


AVAILABLE CANDIDATES

{self._format_candidates(candidates)}


Generate adversarial retrieval queries designed to test the
Investigator's theory rather than confirm it.
"""


    # =====================================================
    # ANALYSIS PROMPT
    # =====================================================

    @staticmethod
    def _analysis_system_prompt():

        return """
You are the Fact-Checker Agent in an evidence-grounded
investigation system.

You must critically evaluate the Investigator's theory using
ONLY the counter-evidence supplied to you.

Look specifically for:

- contradictions,
- alternative explanations,
- conflicting timelines,
- evidence supporting another candidate,
- unsupported assumptions,
- reliance on weak or unverified evidence.

GROUNDING RULES:

- Never invent documents.
- Never invent chunk IDs.
- Never cite evidence not supplied.
- An allegation is not proof.
- Testimony is not automatically fact.
- VERIFIED=false evidence is unverified.
- Do not manufacture a contradiction where none exists.
- If the Investigator survives fact-checking, say so.
- The purpose is adversarial testing, not automatic rejection.
- Preserve uncertainty exactly as expressed in the evidence.
  If a source says "believed to be", "reportedly", "alleged",
  "anonymous", "suspected", or similar, retain that qualifier.
  Never convert an uncertain attribution into an established fact.

Return ONLY valid JSON:

{
  "conclusion": "overall fact-check conclusion",
  "challenge_strength": 0.0,
  "findings": [
    {
      "finding_type": "contradiction",
      "statement": "specific weakness or counter-finding",
      "severity": 0.0,
      "citations": [
        {
          "chunk_id": "exact supplied chunk",
          "document_id": "exact supplied document",
          "claim": "claim supported by this chunk"
        }
      ]
    }
  ]
}

challenge_strength and severity must be between 0 and 1.

finding_type must be one of:

contradiction
alternative_explanation
timeline_conflict
other_candidate
weakness
"""


    def _build_analysis_prompt(
        self,
        investigation,
        evidence,
        candidates
    ):

        return f"""
INVESTIGATOR THEORY

{self._format_investigation(investigation)}


OTHER CANDIDATES

{self._format_candidates(candidates)}


ADVERSARIALLY RETRIEVED EVIDENCE

{self._format_evidence(evidence)}


Evaluate whether this evidence challenges the Investigator's
theory.

Do not repeat the Investigator's supporting argument unless
necessary to explain a weakness.

Use only the supplied evidence for citations.
"""


    # =====================================================
    # FALLBACK SEARCHES
    # =====================================================

    @staticmethod
    def _fallback_queries(
        investigation,
        candidates
    ):

        theory = (
            investigation.theory
        )

        candidate_name = (
            theory.candidate_name
            or ""
        )

        other_names = []

        if candidates:

            for candidate in candidates:

                if hasattr(
                    candidate,
                    "model_dump"
                ):
                    candidate = (
                        candidate.model_dump()
                    )

                if isinstance(
                    candidate,
                    str
                ):
                    name = candidate

                elif isinstance(
                    candidate,
                    dict
                ):
                    name = candidate.get(
                        "name"
                    )

                else:
                    continue

                if (
                    name
                    and
                    name != candidate_name
                ):

                    other_names.append(
                        name
                    )

        alternate_text = (
            " ".join(
                other_names[:4]
            )
        )

        return [
            AdversarialQuery(
                query=(
                    f"{candidate_name} "
                    "contradictory evidence "
                    "denial acquittal conflicting testimony"
                ).strip(),

                purpose=(
                    "Search for evidence that "
                    "contradicts the leading theory."
                ),

                finding_type=
                    "contradiction",
            ),

            AdversarialQuery(
                query=(
                    f"{alternate_text} "
                    "alternative involvement "
                    "supporting evidence"
                ).strip(),

                purpose=(
                    "Search for evidence supporting "
                    "another candidate."
                ),

                finding_type=
                    "other_candidate",
            ),

            AdversarialQuery(
                query=(
                    f"{candidate_name} "
                    "timeline date before after "
                    "witness event"
                ).strip(),

                purpose=(
                    "Search for timeline conflicts."
                ),

                finding_type=
                    "timeline_conflict",
            ),
        ]


    # =====================================================
    # FACT CHECK
    # =====================================================

    def fact_check(
        self,
        investigation,
        document_ids=None,
        candidates=None,
    ):

        if not isinstance(
            investigation,
            InvestigationResult
        ):

            investigation = (
                InvestigationResult
                .model_validate(
                    investigation
                )
            )

        allowed_document_ids = None

        if document_ids is not None:

            allowed_document_ids = [
                str(document_id)
                for document_id
                in document_ids
            ]

        # =================================================
        # 1. Ask Gemini how to attack the theory
        # =================================================

        try:

            raw_plan = (
                self.llm.chat_json(
                    system_prompt=
                        self._planning_system_prompt(),

                    user_prompt=
                        self._build_planning_prompt(
                            investigation=
                                investigation,

                            candidates=
                                candidates,
                        ),

                    temperature=0.25,
                )
            )

            plan = (
                FactCheckPlan
                .model_validate(
                    raw_plan
                )
            )

            queries = (
                plan.queries[
                    :self.max_queries
                ]
            )

        except Exception:

            # If query planning fails, the agent
            # still performs adversarial retrieval.
            queries = (
                self._fallback_queries(
                    investigation,
                    candidates,
                )
            )


        # =================================================
        # 2. Perform independent adversarial searches
        # =================================================

        search_trace = []

        all_evidence = {}

        for index, search in enumerate(
            queries,
            start=1
        ):

            results = (
                self.retriever.search(
                    query=search.query,

                    limit=
                        self.results_per_query,

                    mode="hybrid",

                    document_ids=
                        allowed_document_ids,
                )
            )

            search_trace.append(
                SearchStep(
                    attempt=index,

                    query=
                        search.query,

                    purpose=
                        search.purpose,

                    retrieved_chunk_ids=[
                        item["chunk_id"]
                        for item
                        in results
                    ],
                )
            )

            for item in results:

                all_evidence[
                    item["chunk_id"]
                ] = item


        counter_evidence = list(
            all_evidence.values()
        )


        # =================================================
        # 3. Analyse counter-evidence
        # =================================================

        if not counter_evidence:

            return FactCheckResult(
                challenged_theory=
                    investigation
                    .theory
                    .statement,

                conclusion=(
                    "No counter-evidence was "
                    "retrieved from the active case."
                ),

                challenge_strength=0.0,

                findings=[],

                citations=[],

                search_trace=
                    search_trace,
            )


        raw_analysis = (
            self.llm.chat_json(
                system_prompt=
                    self._analysis_system_prompt(),

                user_prompt=
                    self._build_analysis_prompt(
                        investigation=
                            investigation,

                        evidence=
                            counter_evidence,

                        candidates=
                            candidates,
                    ),

                temperature=0.2,
            )
        )


        draft = (
            FactCheckDraft
            .model_validate(
                raw_analysis
            )
        )


        # =================================================
        # 4. Validate every finding's citations
        # =================================================

        validated_findings = []

        all_valid_citations = {}

        for finding in draft.findings:

            validation = (
                self.validator.validate(
                    proposals=
                        finding.citations,

                    retrieved_evidence=
                        counter_evidence,

                    allowed_document_ids=
                        allowed_document_ids,
                )
            )

            valid_citations = (
                validation
                .valid_citations
            )

            # If Gemini claims a strong finding but
            # gives no grounded citation, don't retain
            # it as a grounded finding.
            if (
                finding.severity >= 0.35
                and
                not valid_citations
            ):
                continue

            validated_finding = (
                FactCheckFinding(
                    finding_type=
                        finding.finding_type,

                    statement=
                        finding.statement,

                    severity=
                        finding.severity,

                    citations=
                        valid_citations,
                )
            )

            validated_findings.append(
                validated_finding
            )

            for citation in valid_citations:

                all_valid_citations[
                    citation.chunk_id
                ] = citation


        # =================================================
        # 5. Ground challenge strength
        # =================================================

        citations = list(
            all_valid_citations.values()
        )

        challenge_strength = (
            draft.challenge_strength
        )

        # No grounded findings =
        # challenge cannot be strong.
        if not validated_findings:

            challenge_strength = min(
                challenge_strength,
                0.2
            )


        return FactCheckResult(
            challenged_theory=
                investigation
                .theory
                .statement,

            conclusion=
                draft.conclusion,

            challenge_strength=
                challenge_strength,

            findings=
                validated_findings,

            citations=
                citations,

            search_trace=
                search_trace,
        )

