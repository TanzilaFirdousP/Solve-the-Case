import json
import re
from pathlib import Path

from backend.schemas import (
    Citation,
    CitationProposal,
    CitationValidationResult,
    InvalidCitation,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PROCESSED_DOCUMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_documents.json"
)


# =========================================================
# TEXT HELPERS
# =========================================================

def tokenize(text):

    return set(
        re.findall(
            r"\b\w+\b",
            text.lower()
        )
    )


def best_excerpt(
    text,
    claim,
    max_chars=450
):
    """
    Find the sentence in the source chunk that overlaps
    most with the agent's claim.

    NOTE:
    This does NOT prove that the claim is logically true.
    It only selects the most relevant source excerpt.
    """

    text = text.strip()

    if not text:
        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claim_tokens = tokenize(
        claim
    )

    if not claim_tokens:
        return text[:max_chars]

    best_sentence = None
    best_score = -1

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_tokens = tokenize(
            sentence
        )

        if not sentence_tokens:
            continue

        overlap = (
            claim_tokens
            & sentence_tokens
        )

        score = (
            len(overlap)
            / len(claim_tokens)
        )

        if score > best_score:

            best_score = score
            best_sentence = sentence

    if not best_sentence:
        return text[:max_chars]

    return best_sentence[
        :max_chars
    ]


# =========================================================
# CITATION VALIDATOR
# =========================================================

class CitationValidator:

    def __init__(self):

        if not (
            PROCESSED_DOCUMENTS_PATH
            .exists()
        ):

            raise FileNotFoundError(
                "Processed evidence not found. "
                "Run backend/preprocess.py first."
            )

        documents = json.loads(
            PROCESSED_DOCUMENTS_PATH
            .read_text(
                encoding="utf-8"
            )
        )

        self.corpus_by_chunk = {
            document["chunk_id"]:
                document

            for document
            in documents
        }


    # =====================================================
    # VALIDATE
    # =====================================================

    def validate(
        self,
        proposals,
        retrieved_evidence,
        allowed_document_ids=None,
    ):
        """
        Validate citations proposed by an agent.

        A citation is valid only when:

        1. chunk_id exists in processed corpus
        2. chunk was ACTUALLY returned by retrieval
        3. document_id matches the chunk
        4. document belongs to current case (if restricted)

        Whether the original DWIE evidence is verified or
        unverified is preserved in source_verified.
        """

        # ---------------------------------------------
        # Normalize proposals
        # ---------------------------------------------

        normalized_proposals = []

        for proposal in proposals:

            if isinstance(
                proposal,
                CitationProposal
            ):

                normalized_proposals.append(
                    proposal
                )

            else:

                normalized_proposals.append(
                    CitationProposal(
                        **proposal
                    )
                )

        # ---------------------------------------------
        # Retrieved evidence lookup
        # ---------------------------------------------

        retrieved_by_chunk = {}

        for evidence in retrieved_evidence:

            chunk_id = evidence.get(
                "chunk_id"
            )

            if chunk_id:

                retrieved_by_chunk[
                    chunk_id
                ] = evidence

        # ---------------------------------------------
        # Optional case restriction
        # ---------------------------------------------

        allowed = None

        if allowed_document_ids is not None:

            allowed = {
                str(document_id)

                for document_id
                in allowed_document_ids
            }

        valid = []
        invalid = []

        # ---------------------------------------------
        # Validate each citation
        # ---------------------------------------------

        for proposal in (
            normalized_proposals
        ):

            chunk_id = proposal.chunk_id

            document_id = str(
                proposal.document_id
            )

            # -----------------------------------------
            # Check 1:
            # Does this chunk even exist?
            # -----------------------------------------

            corpus_chunk = (
                self.corpus_by_chunk.get(
                    chunk_id
                )
            )

            if corpus_chunk is None:

                invalid.append(
                    InvalidCitation(
                        chunk_id=chunk_id,

                        document_id=
                            document_id,

                        claim=
                            proposal.claim,

                        reason=
                            "Chunk does not "
                            "exist in corpus."
                    )
                )

                continue

            # -----------------------------------------
            # Check 2:
            # Was it actually retrieved?
            # -----------------------------------------

            retrieved_chunk = (
                retrieved_by_chunk.get(
                    chunk_id
                )
            )

            if retrieved_chunk is None:

                invalid.append(
                    InvalidCitation(
                        chunk_id=chunk_id,

                        document_id=
                            document_id,

                        claim=
                            proposal.claim,

                        reason=
                            "Chunk exists but "
                            "was not retrieved "
                            "during this agent "
                            "search."
                    )
                )

                continue

            # -----------------------------------------
            # Check 3:
            # Correct document ID?
            # -----------------------------------------

            real_document_id = str(
                corpus_chunk[
                    "document_id"
                ]
            )

            if (
                document_id
                != real_document_id
            ):

                invalid.append(
                    InvalidCitation(
                        chunk_id=chunk_id,

                        document_id=
                            document_id,

                        claim=
                            proposal.claim,

                        reason=(
                            "Citation document_id "
                            "does not match the "
                            "chunk's real "
                            "document_id."
                        )
                    )
                )

                continue

            # -----------------------------------------
            # Check 4:
            # Is document part of case?
            # -----------------------------------------

            if (
                allowed is not None
                and
                real_document_id
                not in allowed
            ):

                invalid.append(
                    InvalidCitation(
                        chunk_id=chunk_id,

                        document_id=
                            document_id,

                        claim=
                            proposal.claim,

                        reason=
                            "Citation comes from "
                            "a document outside "
                            "the active case."
                    )
                )

                continue

            # -----------------------------------------
            # Produce trusted citation
            # -----------------------------------------

            excerpt = best_excerpt(
                corpus_chunk.get(
                    "text",
                    ""
                ),

                proposal.claim
            )

            valid.append(
                Citation(
                    chunk_id=chunk_id,

                    document_id=
                        real_document_id,

                    claim=
                        proposal.claim,

                    title=
                        corpus_chunk.get(
                            "title"
                        ),

                    evidence_excerpt=
                        excerpt,

                    source_verified=bool(
                        corpus_chunk.get(
                            "verified",
                            False
                        )
                    ),

                    retrieval_score=
                        retrieved_chunk.get(
                            "retrieval_score"
                        ),

                    retrieval_mode=
                        retrieved_chunk.get(
                            "retrieval_mode"
                        ),
                )
            )

        return (
            CitationValidationResult(
                valid_citations=valid,

                invalid_citations=
                    invalid,

                all_valid=(
                    len(invalid) == 0
                ),
            )
        )


# =========================================================
# SIMPLE DIRECT CHECK
# =========================================================

if __name__ == "__main__":

    from backend.retrieval import (
        EvidenceRetriever
    )

    retriever = (
        EvidenceRetriever()
    )

    validator = (
        CitationValidator()
    )

    evidence = retriever.search(
        query=(
            "Abdelghani Mzoudi "
            "September 11 attacks"
        ),

        limit=5,

        mode="hybrid"
    )

    proposal = (
        CitationProposal(
            chunk_id=
                evidence[0]["chunk_id"],

            document_id=
                evidence[0][
                    "document_id"
                ],

            claim=(
                "Abdelghani Mzoudi "
                "was on trial in "
                "Germany over alleged "
                "involvement in the "
                "September 11 attacks."
            ),
        )
    )

    result = validator.validate(
        proposals=[proposal],

        retrieved_evidence=evidence
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )