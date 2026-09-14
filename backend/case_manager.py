import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from backend.schemas import (
    Candidate,
    EvidenceItem,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CASES_DIR = (
    PROJECT_ROOT
    / "cases"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ENTITIES_PATH = (
    PROCESSED_DIR
    / "entities.json"
)

DOCUMENTS_PATH = (
    PROCESSED_DIR
    / "documents.json"
)


# =========================================================
# CASE SCHEMA
# =========================================================

class CaseManifest(BaseModel):

    case_id: str

    title: str

    description: str

    question: str

    document_ids: list[str]

    candidate_names: list[str]

    unverified_evidence: list[
        EvidenceItem
    ] = Field(
        default_factory=list
    )


# =========================================================
# HELPERS
# =========================================================

def normalize_name(
    value
):

    value = str(
        value or ""
    ).lower()

    value = value.replace(
        "_",
        " "
    )

    value = re.sub(
        r"[^\w\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# =========================================================
# CASE MANAGER
# =========================================================

class CaseManager:

    def __init__(
        self
    ):

        if not CASES_DIR.exists():

            raise FileNotFoundError(
                f"Cases folder not found: "
                f"{CASES_DIR}"
            )

        if not ENTITIES_PATH.exists():

            raise FileNotFoundError(
                "entities.json not found. "
                "Run preprocessing first."
            )

        if not DOCUMENTS_PATH.exists():

            raise FileNotFoundError(
                "documents.json not found. "
                "Run preprocessing first."
            )


        self.entities = json.loads(
            ENTITIES_PATH.read_text(
                encoding="utf-8"
            )
        )

        documents = json.loads(
            DOCUMENTS_PATH.read_text(
                encoding="utf-8"
            )
        )

        self.documents = {
            str(document["document_id"]):
                document

            for document
            in documents
        }


        # ---------------------------------------------
        # Load cases
        # ---------------------------------------------

        self.cases = {}

        for path in CASES_DIR.glob(
            "*.json"
        ):

            raw = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            case = (
                CaseManifest
                .model_validate(
                    raw
                )
            )

            if case.case_id in self.cases:

                raise ValueError(
                    "Duplicate case_id: "
                    f"{case.case_id}"
                )

            self.cases[
                case.case_id
            ] = case


        # ---------------------------------------------
        # Build name / alias lookup
        # ---------------------------------------------

        self.entity_lookup = {}

        for entity in self.entities:

            possible_names = {
                entity.get(
                    "name"
                ),

                entity.get(
                    "link"
                ),

                *entity.get(
                    "aliases",
                    []
                ),
            }

            for name in possible_names:

                if not name:
                    continue

                normalized = (
                    normalize_name(
                        name
                    )
                )

                self.entity_lookup.setdefault(
                    normalized,
                    []
                ).append(
                    entity
                )


        # Validate manifests immediately.
        for case in self.cases.values():

            self._validate_case(
                case
            )


    # =====================================================
    # CASE VALIDATION
    # =====================================================

    def _validate_case(
        self,
        case
    ):

        missing_documents = [

            document_id

            for document_id
            in case.document_ids

            if document_id
            not in self.documents
        ]

        if missing_documents:

            raise ValueError(
                (
                    f"Case {case.case_id} "
                    "references missing documents: "
                    f"{missing_documents}"
                )
            )


        # Make sure every candidate resolves.
        for candidate_name in (
            case.candidate_names
        ):

            self._resolve_candidate(
                candidate_name=
                    candidate_name,

                case_document_ids=
                    case.document_ids,
            )


    # =====================================================
    # CANDIDATE RESOLUTION
    # =====================================================

    def _resolve_candidate(
        self,
        candidate_name,
        case_document_ids
    ):

        normalized = (
            normalize_name(
                candidate_name
            )
        )

        matches = (
            self.entity_lookup.get(
                normalized,
                []
            )
        )

        if not matches:

            raise ValueError(
                (
                    "Could not resolve candidate "
                    f"'{candidate_name}' "
                    "against entities.json."
                )
            )


        case_docs = set(
            case_document_ids
        )


        # If several entities share an alias/name,
        # prefer the one occurring in most case docs.
        matches = sorted(
            matches,

            key=lambda entity:
                len(
                    set(
                        entity.get(
                            "document_ids",
                            []
                        )
                    )
                    &
                    case_docs
                ),

            reverse=True,
        )


        entity = matches[0]


        candidate_docs = sorted(
            set(
                entity.get(
                    "document_ids",
                    []
                )
            )
            &
            case_docs
        )


        return Candidate(
            entity_id=
                entity["entity_id"],

            name=
                entity.get(
                    "name"
                )
                or candidate_name,

            category=
                entity.get(
                    "category",
                    "other"
                ),

            document_ids=
                candidate_docs,

            aliases=
                entity.get(
                    "aliases",
                    []
                ),
        )


    # =====================================================
    # LIST CASES
    # =====================================================

    def list_cases(
        self
    ):

        return [

            {
                "case_id":
                    case.case_id,

                "title":
                    case.title,

                "description":
                    case.description,

                "document_count":
                    len(
                        case.document_ids
                    ),

                "candidate_count":
                    len(
                        case.candidate_names
                    ),
            }

            for case
            in self.cases.values()
        ]


    # =====================================================
    # GET CASE
    # =====================================================

    def get_case(
        self,
        case_id
    ):

        case = (
            self.cases.get(
                case_id
            )
        )

        if not case:

            raise ValueError(
                (
                    "Unknown case_id: "
                    f"{case_id}"
                )
            )


        candidates = [

            self._resolve_candidate(
                candidate_name=
                    candidate_name,

                case_document_ids=
                    case.document_ids,
            )

            for candidate_name
            in case.candidate_names
        ]


        return {

            "case_id":
                case.case_id,

            "title":
                case.title,

            "description":
                case.description,

            "question":
                case.question,

            "document_ids":
                case.document_ids,

            "candidates": [
                candidate.model_dump()
                for candidate
                in candidates
            ],

            "unverified_evidence": [
                evidence.model_dump()
                for evidence
                in case.unverified_evidence
            ],
        }


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    manager = (
        CaseManager()
    )

    print(
        "\nAVAILABLE CASES"
    )

    print(
        json.dumps(
            manager.list_cases(),
            indent=2,
        )
    )


    print(
        "\nCASE 001"
    )

    case = manager.get_case(
        "afghanistan_security_network"
    )

    print(
        json.dumps(
            case,
            indent=2,
        )
    )