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
        Deterministically validate citations proposed
        by an agent.

        Normal DWIE evidence must:
        1. exist in the processed corpus,
        2. have actually been retrieved,
        3. use the correct document ID,
        4. belong to the active case when case filtering
           is enabled.

        Case-specific external evidence is also allowed,
        but ONLY when:
        1. it was actually returned by retrieval,
        2. it is explicitly marked verified=False.

        This lets the system expose misleading/unverified
        clues while preventing the agent from silently
        promoting them to verified source material.
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

            # Support Pydantic EvidenceItem objects
            # as well as dictionaries.
            if hasattr(
                evidence,
                "model_dump"
            ):

                evidence = (
                    evidence.model_dump()
                )


            chunk_id = evidence.get(
                "chunk_id"
            )

            if chunk_id:

                retrieved_by_chunk[
                    chunk_id
                ] = evidence


        # ---------------------------------------------
        # Optional active-case restriction
        # ---------------------------------------------

        allowed = None

        if allowed_document_ids is not None:

            allowed = {

                str(
                    document_id
                )

                for document_id
                in allowed_document_ids
            }


        valid = []

        invalid = []


        # ---------------------------------------------
        # Validate each proposed citation
        # ---------------------------------------------

        for proposal in (
            normalized_proposals
        ):

            chunk_id = (
                proposal.chunk_id
            )

            document_id = str(
                proposal.document_id
            )


            # =================================================
            # CHECK 1:
            # Was this chunk actually retrieved?
            # =================================================
            #
            # This is the most important grounding rule.
            #
            # Even if a chunk exists somewhere in the corpus,
            # the LLM cannot cite it unless retrieval supplied
            # it during this agent run.
            # =================================================

            retrieved_chunk = (
                retrieved_by_chunk.get(
                    chunk_id
                )
            )

            if retrieved_chunk is None:

                invalid.append(
                    InvalidCitation(

                        chunk_id=
                            chunk_id,

                        document_id=
                            document_id,

                        claim=
                            proposal.claim,

                        reason=(
                            "Chunk was not retrieved "
                            "during this agent search."
                        ),
                    )
                )

                continue


            # =================================================
            # CHECK 2:
            # Determine authoritative source record
            # =================================================
            #
            # Normal DWIE evidence exists in the processed
            # corpus.
            #
            # Case-specific external evidence will not exist
            # there, so it is accepted only when explicitly
            # marked verified=False.
            # =================================================

            corpus_chunk = (
                self.corpus_by_chunk.get(
                    chunk_id
                )
            )


            is_external = (
                corpus_chunk is None
            )


            if is_external:

                # -----------------------------------------
                # External evidence MUST be unverified.
                # -----------------------------------------

                if (
                    retrieved_chunk.get(
                        "verified",
                        True
                    )
                    is not False
                ):

                    invalid.append(
                        InvalidCitation(

                            chunk_id=
                                chunk_id,

                            document_id=
                                document_id,

                            claim=
                                proposal.claim,

                            reason=(
                                "External evidence must "
                                "be explicitly marked "
                                "unverified."
                            ),
                        )
                    )

                    continue


                # The retrieved record itself becomes
                # the authoritative source for this
                # case-specific clue.
                source_chunk = (
                    retrieved_chunk
                )


            else:

                # Normal DWIE evidence is grounded in
                # processed_documents.json.
                source_chunk = (
                    corpus_chunk
                )


            # =================================================
            # CHECK 3:
            # Correct document ID?
            # =================================================

            real_document_id = str(
                source_chunk[
                    "document_id"
                ]
            )


            if (
                document_id
                != real_document_id
            ):

                invalid.append(
                    InvalidCitation(

                        chunk_id=
                            chunk_id,

                        document_id=
                            document_id,

                        claim=
                            proposal.claim,

                        reason=(
                            "Citation document_id "
                            "does not match the "
                            "chunk's real "
                            "document_id."
                        ),
                    )
                )

                continue


            # =================================================
            # CHECK 4:
            # Is normal corpus evidence part of the case?
            # =================================================
            #
            # External evidence is defined by the case
            # manifest itself, so it does not need to use
            # one of the original DWIE document IDs.
            # =================================================

            if (
                allowed is not None
                and
                real_document_id
                not in allowed
                and
                not is_external
            ):

                invalid.append(
                    InvalidCitation(

                        chunk_id=
                            chunk_id,

                        document_id=
                            document_id,

                        claim=
                            proposal.claim,

                        reason=(
                            "Citation comes from "
                            "a document outside "
                            "the active case."
                        ),
                    )
                )

                continue


            # =================================================
            # PRODUCE TRUSTED CITATION
            # =================================================

            excerpt = best_excerpt(

                source_chunk.get(
                    "text",
                    ""
                ),

                proposal.claim
            )


            valid.append(
                Citation(

                    chunk_id=
                        chunk_id,

                    document_id=
                        real_document_id,

                    claim=
                        proposal.claim,

                    title=
                        source_chunk.get(
                            "title"
                        ),

                    evidence_excerpt=
                        excerpt,

                    # -------------------------------------
                    # Critical reliability field.
                    #
                    # DWIE evidence -> True
                    # synthetic rumor -> False
                    # -------------------------------------

                    source_verified=bool(
                        source_chunk.get(
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


        # =================================================
        # FINAL RESULT
        # =================================================

        return (
            CitationValidationResult(

                valid_citations=
                    valid,

                invalid_citations=
                    invalid,

                all_valid=(
                    len(
                        invalid
                    )
                    == 0
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

        mode="hybrid",
    )


    if not evidence:

        raise RuntimeError(
            "No evidence retrieved."
        )


    proposal = (
        CitationProposal(

            chunk_id=
                evidence[0][
                    "chunk_id"
                ],

            document_id=
                evidence[0][
                    "document_id"
                ],

            claim=(
                "Abdelghani Mzoudi "
                "was discussed in "
                "connection with the "
                "September 11 attacks."
            ),
        )
    )


    result = validator.validate(

        proposals=[
            proposal
        ],

        retrieved_evidence=
            evidence,
    )


    print(
        result.model_dump_json(
            indent=2
        )
    )