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

    minimum = float(values.min())
    maximum = float(values.max())

    if maximum == minimum:
        return np.zeros_like(values)

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
        # Lexical retrieval: BM25
        # -------------------------------------------------

        tokenized_documents = [
            tokenize(document["text"])
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        # -------------------------------------------------
        # Semantic retrieval
        # -------------------------------------------------

        self.model_name = model_name

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
            and EMBEDDINGS_META_PATH.exists()
        ):

            try:

                metadata = json.loads(
                    EMBEDDINGS_META_PATH
                    .read_text(
                        encoding="utf-8"
                    )
                )

                cached_hash = metadata.get(
                    "corpus_hash"
                )

                cached_model = metadata.get(
                    "model_name"
                )

                cached_count = metadata.get(
                    "document_count"
                )

                if (
                    cached_hash
                    == self.corpus_hash

                    and cached_model
                    == self.model_name

                    and cached_count
                    == len(self.documents)
                ):

                    embeddings = np.load(
                        EMBEDDINGS_PATH
                    )

                    if (
                        embeddings.shape[0]
                        == len(self.documents)
                    ):
                        return embeddings

            except Exception:
                # Corrupt / incompatible cache.
                # Rebuild below.
                pass

        return (
            self._create_embeddings()
        )


    def _create_embeddings(self):

        print(
            "Creating semantic embeddings "
            "for evidence corpus..."
        )

        texts = [
            document["text"]
            for document in self.documents
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
                len(self.documents),

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
    # RAW SCORES
    # =====================================================

    def _lexical_scores(
        self,
        query
    ):

        query_tokens = tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
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

        # -------------------------------------------------
        # Calculate both scores once.
        # -------------------------------------------------

        lexical_raw = (
            self._lexical_scores(
                query
            )
        )

        semantic_raw = (
            self._semantic_scores(
                query
            )
        )

        lexical_normalized = (
            min_max_normalize(
                lexical_raw
            )
        )

        # Cosine similarity is approximately
        # -1 to +1.
        semantic_normalized = np.clip(
            (
                semantic_raw + 1.0
            ) / 2.0,
            0.0,
            1.0
        )

        # -------------------------------------------------
        # Final retrieval score
        # -------------------------------------------------

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
                + semantic_weight
            )

            if weight_total <= 0:
                raise ValueError(
                    "Hybrid retrieval weights "
                    "must sum to > 0."
                )

            lexical_weight = (
                lexical_weight
                / weight_total
            )

            semantic_weight = (
                semantic_weight
                / weight_total
            )

            final_scores = (
                lexical_weight
                * lexical_normalized

                +

                semantic_weight
                * semantic_normalized
            )

        # -------------------------------------------------
        # Optional case/document filtering
        # -------------------------------------------------

        allowed_document_ids = None

        if document_ids is not None:

            allowed_document_ids = {
                str(document_id)
                for document_id
                in document_ids
            }

        candidate_indices = []

        for index, document in enumerate(
            self.documents
        ):

            if (
                allowed_document_ids
                is not None
                and
                document["document_id"]
                not in allowed_document_ids
            ):
                continue

            candidate_indices.append(
                index
            )

        # -------------------------------------------------
        # Rank
        # -------------------------------------------------

        ranked_indices = sorted(
            candidate_indices,

            key=lambda index:
                float(
                    final_scores[index]
                ),

            reverse=True
        )[:limit]

        # -------------------------------------------------
        # Structured results
        # -------------------------------------------------

        results = []

        for index in ranked_indices:

            document = (
                self.documents[
                    index
                ].copy()
            )

            document[
                "bm25_score"
            ] = round(
                float(
                    lexical_raw[index]
                ),
                4
            )

            document[
                "semantic_score"
            ] = round(
                float(
                    semantic_raw[index]
                ),
                4
            )

            document[
                "retrieval_score"
            ] = round(
                float(
                    final_scores[index]
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
            result["chunk_id"]
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
            "Preview:",
            result["text"][:250]
        )

        print()