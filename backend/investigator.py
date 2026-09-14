from pydantic import BaseModel, Field

from backend.citations_validator import (
    CitationValidator,
)

from backend.evidence_graph import (
    EvidenceGraph,
)

from backend.llm_client import (
    LLMClient,
)

from backend.retrieval import (
    EvidenceRetriever,
)

from backend.schemas import (
    CitationProposal,
    InvestigationResult,
    SearchStep,
    Theory,
)


# =========================================================
# INTERNAL GEMINI RESPONSE SCHEMA
# =========================================================

class InvestigatorDraft(BaseModel):

    theory_statement: str

    rationale: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    candidate_id: str | None = None

    candidate_name: str | None = None

    needs_more_evidence: bool

    next_query: str | None = None

    citations: list[
        CitationProposal
    ] = []


# =========================================================
# INVESTIGATOR AGENT
# =========================================================

class InvestigatorAgent:

    def __init__(
        self,
        retriever=None,
        evidence_graph=None,
        citation_validator=None,
        llm_client=None,
        max_retries=2,
        results_per_search=6,
    ):

        self.retriever = (
            retriever
            or EvidenceRetriever()
        )

        self.evidence_graph = (
            evidence_graph
            or EvidenceGraph()
        )

        self.validator = (
            citation_validator
            or CitationValidator()
        )

        self.llm = (
            llm_client
            or LLMClient()
        )

        self.max_retries = max_retries

        self.results_per_search = (
            results_per_search
        )


    # =====================================================
    # CANDIDATE CONTEXT
    # =====================================================

    @staticmethod
    def _format_candidates(
        candidates
    ):

        if not candidates:

            return (
                "No fixed candidate list "
                "was supplied."
            )

        lines = []

        for candidate in candidates:

            # Pydantic model
            if hasattr(
                candidate,
                "model_dump"
            ):

                candidate = (
                    candidate.model_dump()
                )

            # Plain name
            if isinstance(
                candidate,
                str
            ):

                lines.append(
                    f"- {candidate}"
                )

                continue

            # Dictionary
            if isinstance(
                candidate,
                dict
            ):

                name = candidate.get(
                    "name",
                    "Unknown"
                )

                entity_id = (
                    candidate.get(
                        "entity_id"
                    )
                )

                category = (
                    candidate.get(
                        "category"
                    )
                )

                line = f"- {name}"

                if entity_id:

                    line += (
                        f" | entity_id="
                        f"{entity_id}"
                    )

                if category:

                    line += (
                        f" | category="
                        f"{category}"
                    )

                lines.append(line)

        return "\n".join(lines)


    # =====================================================
    # FORMAT RETRIEVED EVIDENCE
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

            if len(text) > 1600:

                text = (
                    text[:1600]
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
                "No evidence was retrieved."
            )

        return (
            "\n\n---\n\n"
            .join(blocks)
        )


    # =====================================================
    # EVIDENCE GRAPH CONTEXT
    # =====================================================

    def _graph_context(
        self,
        evidence,
        allowed_document_ids=None,
        max_relations=30,
    ):

        entity_ids = set()

        for item in evidence:

            entity_ids.update(
                item.get(
                    "entity_ids",
                    []
                )
            )

        allowed_documents = None

        if allowed_document_ids is not None:

            allowed_documents = {
                str(document_id)

                for document_id
                in allowed_document_ids
            }

        graph = (
            self.evidence_graph.graph
        )

        relations = []

        seen = set()

        for entity_id in entity_ids:

            if not graph.has_node(
                entity_id
            ):
                continue

            edges = []

            edges.extend(
                graph.out_edges(
                    entity_id,
                    keys=True,
                    data=True,
                )
            )

            edges.extend(
                graph.in_edges(
                    entity_id,
                    keys=True,
                    data=True,
                )
            )

            for (
                source,
                target,
                key,
                data,
            ) in edges:

                edge_type = (
                    data.get(
                        "edge_type"
                    )
                )

                if edge_type not in {
                    "relation",
                    "event_role",
                }:

                    continue

                document_id = (
                    data.get(
                        "document_id"
                    )
                )

                # Prevent cross-case graph leakage.
                if (
                    allowed_documents
                    is not None
                    and document_id is not None
                    and str(document_id)
                    not in allowed_documents
                ):

                    continue

                relation_key = (
                    source,
                    target,
                    str(key),
                )

                if relation_key in seen:
                    continue

                seen.add(
                    relation_key
                )

                source_label = (
                    graph.nodes[source]
                    .get(
                        "label",
                        source
                    )
                )

                target_label = (
                    graph.nodes[target]
                    .get(
                        "label",
                        target
                    )
                )

                relation_label = (
                    data.get(
                        "label"
                    )
                    or data.get(
                        "relation"
                    )
                    or edge_type
                )

                relations.append(
                    (
                        f"{source_label} "
                        f"--{relation_label}--> "
                        f"{target_label} "
                        f"[document: "
                        f"{document_id}]"
                    )
                )

        if not relations:

            return (
                "No relevant graph "
                "relationships were found."
            )

        return "\n".join(
            relations[
                :max_relations
            ]
        )


    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

    @staticmethod
    def _system_prompt():

        return """
You are the Investigator Agent in an evidence-grounded,
multi-document investigation system.

Your responsibility is investigative reasoning, not factual
retrieval from your own memory.

You MUST reason only from:

1. RETRIEVED EVIDENCE supplied to you.
2. RELEVANT EVIDENCE GRAPH RELATIONSHIPS supplied to you.

Your tasks are:

1. Form the strongest current theory.
2. Identify the candidate or explanation best supported by
   the current evidence.
3. Evaluate whether the evidence is sufficient.
4. If evidence is insufficient, ambiguous, contradictory,
   or incomplete, produce a more targeted retrieval query.
5. Revise your theory when new evidence weakens it.
6. Cite the evidence supporting your claims.

GROUNDING RULES:

- Never invent a document ID.
- Never invent a chunk ID.
- Never cite evidence that was not supplied.
- Never invent relationships or events.
- Treat allegations, testimony, accusations and suspicions
  as allegations, not established facts.
- A suspect, defendant, accused person or DWIE "offender"
  label does NOT establish guilt.
- Evidence marked VERIFIED=false is unverified and must not
  be treated as established fact.
- If competing explanations remain plausible, say so.
- Prefer uncertainty over unsupported certainty.

Return ONLY valid JSON in this structure:

{
  "theory_statement": "Current best theory",
  "rationale": "Why the current evidence supports it",
  "confidence": 0.0,
  "candidate_id": null,
  "candidate_name": null,
  "needs_more_evidence": true,
  "next_query": "targeted retrieval query or null",
  "citations": [
    {
      "chunk_id": "exact supplied chunk ID",
      "document_id": "exact supplied document ID",
      "claim": "specific claim this evidence supports"
    }
  ]
}

confidence MUST be between 0 and 1.

If evidence is insufficient, set:

"needs_more_evidence": true

and provide a targeted "next_query" that would help resolve
the largest remaining uncertainty.
"""


    # =====================================================
    # USER PROMPT
    # =====================================================

    def _build_user_prompt(
        self,
        question,
        evidence,
        graph_context,
        candidates=None,
        previous_theory=None,
        citation_feedback=None,
    ):

        candidate_text = (
            self._format_candidates(
                candidates
            )
        )

        previous_text = (
            previous_theory
            or (
                "None. This is the "
                "initial investigation."
            )
        )

        feedback_text = (
            citation_feedback
            or "None."
        )

        return f"""
INVESTIGATION QUESTION

{question}


CANDIDATES

{candidate_text}


PREVIOUS THEORY

{previous_text}


CITATION VALIDATION FEEDBACK FROM PREVIOUS ATTEMPT

{feedback_text}


RETRIEVED EVIDENCE

{self._format_evidence(evidence)}


RELEVANT EVIDENCE GRAPH RELATIONSHIPS

{graph_context}


Using only this material, form or revise the current theory.

If the evidence remains weak, incomplete, conflicting or
ambiguous, set needs_more_evidence=true and produce a
specific next_query.

Citations may reference ONLY chunk IDs shown under
RETRIEVED EVIDENCE.
"""


    # =====================================================
    # FALLBACK QUERY
    # =====================================================

    @staticmethod
    def _fallback_query(
        question,
        draft,
    ):

        candidate = (
            draft.candidate_name
            or ""
        )

        return (
            f"{question} "
            f"{candidate} "
            "supporting evidence conflicting evidence "
            "witness timeline relationships"
        ).strip()


    # =====================================================
    # INVESTIGATE
    # =====================================================

    def investigate(
        self,
        question,
        document_ids=None,
        candidates=None,
    ):

        question = question.strip()

        if not question:

            raise ValueError(
                "Investigation question "
                "cannot be empty."
            )

        allowed_document_ids = None

        if document_ids is not None:

            allowed_document_ids = [
                str(document_id)
                for document_id
                in document_ids
            ]

        current_query = question

        previous_theory = None

        citation_feedback = None

        search_trace = []

        # Evidence is accumulated between retries.
        all_evidence = {}

        final_draft = None
        final_validation = None

        maximum_attempts = (
            self.max_retries + 1
        )

        # =================================================
        # INVESTIGATOR LOOP
        # =================================================

        for attempt in range(
            1,
            maximum_attempts + 1,
        ):

            # ---------------------------------------------
            # 1. Retrieve evidence
            # ---------------------------------------------

            retrieved = (
                self.retriever.search(
                    query=current_query,

                    limit=
                        self.results_per_search,

                    mode="hybrid",

                    document_ids=
                        allowed_document_ids,
                )
            )

            for item in retrieved:

                all_evidence[
                    item["chunk_id"]
                ] = item

            search_trace.append(
                SearchStep(
                    attempt=attempt,

                    query=current_query,

                    purpose=(
                        "Initial evidence retrieval"
                        if attempt == 1
                        else
                        "Reformulated retrieval "
                        "after insufficient evidence"
                    ),

                    retrieved_chunk_ids=[
                        item["chunk_id"]
                        for item
                        in retrieved
                    ],
                )
            )

            accumulated_evidence = (
                list(
                    all_evidence.values()
                )
            )

            # ---------------------------------------------
            # No evidence at all
            # ---------------------------------------------

            if not accumulated_evidence:

                final_draft = (
                    InvestigatorDraft(
                        theory_statement=(
                            "No grounded theory "
                            "can currently be formed."
                        ),

                        rationale=(
                            "No evidence was "
                            "retrieved for the "
                            "investigation question."
                        ),

                        confidence=0.0,

                        candidate_id=None,

                        candidate_name=None,

                        needs_more_evidence=True,

                        next_query=None,

                        citations=[],
                    )
                )

                break

            # ---------------------------------------------
            # 2. Extract graph context
            # ---------------------------------------------

            graph_context = (
                self._graph_context(
                    evidence=
                        accumulated_evidence,

                    allowed_document_ids=
                        allowed_document_ids,
                )
            )

            # ---------------------------------------------
            # 3. Ask Gemini to reason
            # ---------------------------------------------

            raw_response = (
                self.llm.chat_json(
                    system_prompt=
                        self._system_prompt(),

                    user_prompt=
                        self._build_user_prompt(
                            question=
                                question,

                            evidence=
                                accumulated_evidence,

                            graph_context=
                                graph_context,

                            candidates=
                                candidates,

                            previous_theory=
                                previous_theory,

                            citation_feedback=
                                citation_feedback,
                        ),

                    temperature=0.2,
                )
            )

            draft = (
                InvestigatorDraft
                .model_validate(
                    raw_response
                )
            )

            final_draft = draft

            # ---------------------------------------------
            # 4. Deterministically validate citations
            # ---------------------------------------------

            validation = (
                self.validator.validate(
                    proposals=
                        draft.citations,

                    retrieved_evidence=
                        accumulated_evidence,

                    allowed_document_ids=
                        allowed_document_ids,
                )
            )

            final_validation = (
                validation
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

            # ---------------------------------------------
            # 5. Deterministic sufficiency check
            # ---------------------------------------------

            insufficient = (

                draft.needs_more_evidence

                or
                draft.confidence < 0.55

                or
                len(valid_citations) < 2

                or
                len(verified_citations) < 1

                or
                not validation.all_valid
            )

            # ---------------------------------------------
            # Enough evidence -> stop.
            # ---------------------------------------------

            if not insufficient:
                break

            # ---------------------------------------------
            # Retry limit reached -> stop.
            # ---------------------------------------------

            if attempt >= maximum_attempts:
                break

            # ---------------------------------------------
            # 6. Prepare feedback for self-correction
            # ---------------------------------------------

            invalid_reasons = [
                citation.reason

                for citation
                in validation
                .invalid_citations
            ]

            if invalid_reasons:

                citation_feedback = (
                    "The previous citation "
                    "proposal contained invalid "
                    "citations:\n- "
                    + "\n- ".join(
                        invalid_reasons
                    )
                )

            else:

                citation_feedback = (
                    "Previous citations were valid, "
                    "but the evidence was still "
                    "judged insufficient."
                )

            # ---------------------------------------------
            # 7. Reformulate retrieval query
            # ---------------------------------------------

            if (
                draft.next_query
                and
                draft.next_query.strip()
            ):

                current_query = (
                    draft
                    .next_query
                    .strip()
                )

            else:

                current_query = (
                    self._fallback_query(
                        question,
                        draft,
                    )
                )

            previous_theory = (
                draft.theory_statement
            )


        # =================================================
        # FINAL STRUCTURED RESULT
        # =================================================

        if final_draft is None:

            raise RuntimeError(
                "Investigator failed to "
                "produce a theory."
            )

        citations = []

        if final_validation:

            citations = (
                final_validation
                .valid_citations
            )

        verified_count = sum(
            1

            for citation
            in citations

            if citation.source_verified
        )

        final_needs_more_evidence = (

            final_draft
            .needs_more_evidence

            or
            final_draft.confidence < 0.55

            or
            len(citations) < 2

            or
            verified_count < 1
        )

        theory = Theory(
            statement=
                final_draft
                .theory_statement,

            candidate_id=
                final_draft
                .candidate_id,

            candidate_name=
                final_draft
                .candidate_name,

            rationale=
                final_draft
                .rationale,

            confidence=
                final_draft
                .confidence,
        )

        return InvestigationResult(
            theory=theory,

            citations=
                citations,

            needs_more_evidence=
                final_needs_more_evidence,

            attempts=
                len(search_trace),

            search_trace=
                search_trace,
        )