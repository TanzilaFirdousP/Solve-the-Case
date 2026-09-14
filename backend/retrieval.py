import hashlib
import json
import re
from pathlib import Path

import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DOCUMENTS_PATH = (
    PROCESSED_DIR
    / "processed_documents.json"
)

EMBEDDINGS_PATH = (
    PROCESSED_DIR
    / "chunk_embeddings.npy"
)

EMBEDDINGS_META_PATH = (
    PROCESSED_DIR
    / "chunk_embeddings_meta.json"
)


SEMANTIC_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_HYBRID_LEXICAL_WEIGHT = 0.45
DEFAULT_HYBRID_SEMANTIC_WEIGHT = 0.55


# =========================================================
# TEXT HELPERS
# =========================================================

def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def corpus_fingerprint(documents):
    """
    Detect whether processed_documents.json changed.

    If preprocessing changes the chunk text/order,
    cached embeddings are automatically rebuilt.
    """

    hasher = hashlib.sha256()

    for document in documents:

        hasher.update(
            document["chunk_id"].encode(
                "utf-8"
            )
        )

        hasher.update(
            document["text"].encode(
                "utf-8"
            )
        )

    return hasher.hexdigest()


def min_max_normalize(values):

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if len(values) == 0:
        return values

    minimum = float(
        values.min()
    )

    maximum = float(
        values.max()
    )

    if maximum == minimum:
        return np.zeros_like(
            values
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# =========================================================
# EVIDENCE RETRIEVER
# =========================================================

class EvidenceRetriever:

    def __init__(
        self,
        model_name=SEMANTIC_MODEL_NAME
    ):

        if not (
            PROCESSED_DOCUMENTS_PATH.exists()
        ):

            raise FileNotFoundError(
                "Processed evidence not found. "
                "Run backend/preprocess.py first."
            )

        # -------------------------------------------------
        # Load processed evidence chunks
        # -------------------------------------------------

        self.documents = json.loads(
            PROCESSED_DOCUMENTS_PATH.read_text(
                encoding="utf-8"
            )
        )

        if not self.documents:

            raise ValueError(
                "Processed evidence corpus is empty."
            )

        # -------------------------------------------------
        # Lexical retrieval: full-corpus BM25
        #
        # This remains useful for ordinary whole-corpus
        # searches. Case-specific searches build an active
        # BM25 index inside search().
        # -------------------------------------------------

        tokenized_documents = [

            tokenize(
                document["text"]
            )

            for document
            in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        # -------------------------------------------------
        # Semantic retrieval
        # -------------------------------------------------

        self.model_name = (
            model_name
        )

        self.semantic_model = (
            SentenceTransformer(
                model_name
            )
        )

        self.corpus_hash = (
            corpus_fingerprint(
                self.documents
            )
        )

        self.embeddings = (
            self._load_or_create_embeddings()
        )


    # =====================================================
    # EMBEDDING CACHE
    # =====================================================

    def _load_or_create_embeddings(
        self
    ):

        if (
            EMBEDDINGS_PATH.exists()
            and
            EMBEDDINGS_META_PATH.exists()
        ):

            try:

                metadata = json.loads(
                    EMBEDDINGS_META_PATH
                    .read_text(
                        encoding="utf-8"
                    )
                )

                cached_hash = (
                    metadata.get(
                        "corpus_hash"
                    )
                )

                cached_model = (
                    metadata.get(
                        "model_name"
                    )
                )

                cached_count = (
                    metadata.get(
                        "document_count"
                    )
                )

                if (
                    cached_hash
                    == self.corpus_hash

                    and
                    cached_model
                    == self.model_name

                    and
                    cached_count
                    == len(
                        self.documents
                    )
                ):

                    embeddings = np.load(
                        EMBEDDINGS_PATH
                    )

                    if (
                        embeddings.shape[0]
                        ==
                        len(
                            self.documents
                        )
                    ):

                        return embeddings

            except Exception:

                # Corrupt / incompatible cache.
                # Rebuild below.
                pass

        return (
            self._create_embeddings()
        )


    def _create_embeddings(
        self
    ):

        print(
            "Creating semantic embeddings "
            "for evidence corpus..."
        )

        texts = [

            document["text"]

            for document
            in self.documents
        ]

        embeddings = (
            self.semantic_model.encode(
                texts,

                batch_size=32,

                show_progress_bar=True,

                convert_to_numpy=True,

                normalize_embeddings=True,
            )
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        np.save(
            EMBEDDINGS_PATH,
            embeddings
        )

        metadata = {

            "model_name":
                self.model_name,

            "document_count":
                len(
                    self.documents
                ),

            "corpus_hash":
                self.corpus_hash,
        }

        EMBEDDINGS_META_PATH.write_text(
            json.dumps(
                metadata,
                indent=2
            ),
            encoding="utf-8"
        )

        print(
            f"Saved embeddings to "
            f"{EMBEDDINGS_PATH}"
        )

        return embeddings


    # =====================================================
    # RAW FULL-CORPUS SCORES
    # =====================================================

    def _lexical_scores(
        self,
        query
    ):

        query_tokens = tokenize(
            query
        )

        scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )

        return np.asarray(
            scores,
            dtype=np.float32
        )


    def _semantic_scores(
        self,
        query
    ):

        query_embedding = (
            self.semantic_model.encode(
                [query],

                convert_to_numpy=True,

                normalize_embeddings=True,
            )[0]
        )

        # Because embeddings are normalized,
        # dot product == cosine similarity.

        scores = (
            self.embeddings
            @ query_embedding
        )

        return np.asarray(
            scores,
            dtype=np.float32
        )


    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query,
        limit=5,
        mode="hybrid",
        document_ids=None,
        extra_evidence=None,
        lexical_weight=
            DEFAULT_HYBRID_LEXICAL_WEIGHT,
        semantic_weight=
            DEFAULT_HYBRID_SEMANTIC_WEIGHT,
    ):

        query = query.strip()

        if not query:
            return []

        if limit <= 0:
            return []

        if mode not in {
            "lexical",
            "semantic",
            "hybrid",
        }:

            raise ValueError(
                "mode must be one of: "
                "lexical, semantic, hybrid"
            )


        # =================================================
        # CASE / DOCUMENT FILTER
        # =================================================

        allowed_document_ids = None

        if document_ids is not None:

            allowed_document_ids = {

                str(
                    document_id
                )

                for document_id
                in document_ids
            }


        # =================================================
        # BUILD ACTIVE SEARCH CORPUS
        #
        # The normal DWIE corpus uses its cached
        # embeddings.
        #
        # Case-specific evidence can be injected
        # temporarily without permanently changing
        # processed_documents.json.
        # =================================================

        candidate_documents = []

        candidate_embeddings = []

        seen_chunk_ids = set()


        # -------------------------------------------------
        # Normal DWIE evidence
        # -------------------------------------------------

        for index, document in enumerate(
            self.documents
        ):

            document_id = str(
                document[
                    "document_id"
                ]
            )

            if (
                allowed_document_ids
                is not None
                and
                document_id
                not in allowed_document_ids
            ):

                continue


            candidate_documents.append(
                document.copy()
            )

            candidate_embeddings.append(
                self.embeddings[
                    index
                ]
            )

            seen_chunk_ids.add(
                document[
                    "chunk_id"
                ]
            )


        # -------------------------------------------------
        # Case-specific extra evidence
        #
        # This is where misleading / unverified clues
        # from a case manifest enter retrieval.
        # -------------------------------------------------

        extra_positions = []

        extra_texts = []


        for evidence in (
            extra_evidence
            or []
        ):

            # Pydantic models are supported directly.
            if hasattr(
                evidence,
                "model_dump"
            ):

                evidence = (
                    evidence.model_dump()
                )


            evidence = dict(
                evidence
            )


            chunk_id = evidence.get(
                "chunk_id"
            )

            text = str(
                evidence.get(
                    "text",
                    ""
                )
            ).strip()


            if (
                not chunk_id
                or
                not text
            ):

                continue


            # Prevent duplicate chunk IDs.
            if chunk_id in seen_chunk_ids:

                continue


            # External case evidence defaults to
            # unverified unless explicitly specified.
            evidence.setdefault(
                "verified",
                False
            )

            evidence.setdefault(
                "source",
                "case_extra"
            )

            evidence.setdefault(
                "entity_ids",
                []
            )


            candidate_documents.append(
                evidence
            )


            # Semantic embedding will be generated
            # below only for these extra chunks.
            candidate_embeddings.append(
                None
            )


            extra_positions.append(
                len(
                    candidate_embeddings
                )
                - 1
            )

            extra_texts.append(
                text
            )


            seen_chunk_ids.add(
                chunk_id
            )


        # No evidence exists after filtering.
        if not candidate_documents:

            return []


        # =================================================
        # LEXICAL RETRIEVAL
        #
        # BM25 is calculated against the ACTIVE case
        # corpus, not all 802 DWIE articles.
        # =================================================

        tokenized_documents = [

            tokenize(
                document["text"]
            )

            for document
            in candidate_documents
        ]


        active_bm25 = BM25Okapi(
            tokenized_documents
        )


        lexical_raw = np.asarray(

            active_bm25.get_scores(
                tokenize(
                    query
                )
            ),

            dtype=np.float32,
        )


        # =================================================
        # SEMANTIC EMBEDDINGS FOR EXTRA EVIDENCE
        # =================================================

        if extra_texts:

            extra_embeddings = (
                self.semantic_model.encode(
                    extra_texts,

                    convert_to_numpy=True,

                    normalize_embeddings=True,
                )
            )


            extra_embeddings = np.asarray(
                extra_embeddings,
                dtype=np.float32,
            )


            for (
                position,
                embedding
            ) in zip(
                extra_positions,
                extra_embeddings,
            ):

                candidate_embeddings[
                    position
                ] = embedding


        # All None placeholders have now been replaced.
        embedding_matrix = np.vstack(
            candidate_embeddings
        ).astype(
            np.float32
        )


        # =================================================
        # SEMANTIC QUERY SCORE
        # =================================================

        query_embedding = (
            self.semantic_model.encode(
                [query],

                convert_to_numpy=True,

                normalize_embeddings=True,
            )[0]
        )


        semantic_raw = (
            embedding_matrix
            @ query_embedding
        )


        semantic_raw = np.asarray(
            semantic_raw,
            dtype=np.float32,
        )


        # =================================================
        # NORMALIZATION
        # =================================================

        lexical_normalized = (
            min_max_normalize(
                lexical_raw
            )
        )


        # Cosine similarity is approximately
        # -1 to +1.

        semantic_normalized = np.clip(
            (
                semantic_raw
                + 1.0
            )
            / 2.0,

            0.0,
            1.0,
        )


        # =================================================
        # FINAL RETRIEVAL SCORE
        # =================================================

        if mode == "lexical":

            final_scores = (
                lexical_normalized
            )


        elif mode == "semantic":

            final_scores = (
                semantic_normalized
            )


        else:

            weight_total = (
                lexical_weight
                +
                semantic_weight
            )


            if weight_total <= 0:

                raise ValueError(
                    "Hybrid retrieval weights "
                    "must sum to > 0."
                )


            normalized_lexical_weight = (
                lexical_weight
                /
                weight_total
            )


            normalized_semantic_weight = (
                semantic_weight
                /
                weight_total
            )


            final_scores = (

                normalized_lexical_weight
                *
                lexical_normalized

                +

                normalized_semantic_weight
                *
                semantic_normalized
            )


        # =================================================
        # RANK RESULTS
        # =================================================

        ranked_indices = sorted(

            range(
                len(
                    candidate_documents
                )
            ),

            key=lambda index:
                float(
                    final_scores[
                        index
                    ]
                ),

            reverse=True,

        )[:limit]


        # =================================================
        # STRUCTURED RESULTS
        # =================================================

        results = []


        for index in ranked_indices:

            document = (
                candidate_documents[
                    index
                ].copy()
            )


            document[
                "bm25_score"
            ] = round(
                float(
                    lexical_raw[
                        index
                    ]
                ),
                4
            )


            document[
                "semantic_score"
            ] = round(
                float(
                    semantic_raw[
                        index
                    ]
                ),
                4
            )


            document[
                "retrieval_score"
            ] = round(
                float(
                    final_scores[
                        index
                    ]
                ),
                4
            )


            document[
                "retrieval_mode"
            ] = mode


            results.append(
                document
            )


        return results


# =========================================================
# DIRECT SCRIPT USAGE
# =========================================================

if __name__ == "__main__":

    retriever = EvidenceRetriever()

    query = (
        "evidence concerning "
        "Abdelghani Mzoudi and "
        "the September 11 attacks"
    )

    results = retriever.search(
        query=query,
        limit=5,
        mode="hybrid"
    )

    print(
        f"\nQuery: {query}\n"
    )

    for result in results:

        print(
            "Chunk:",
            result[
                "chunk_id"
            ]
        )

        print(
            "Hybrid score:",
            result[
                "retrieval_score"
            ]
        )

        print(
            "BM25:",
            result[
                "bm25_score"
            ]
        )

        print(
            "Semantic:",
            result[
                "semantic_score"
            ]
        )

        print(
            "Verified:",
            result.get(
                "verified"
            )
        )

        print(
            "Preview:",
            result[
                "text"
            ][:250]
        )

        print()