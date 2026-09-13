import json
import math

from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "DWIE"
    / "data"
    / "annos_with_content"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# =========================================================
# CONFIGURATION
# =========================================================

MIN_CASE_DOCS = 3
MAX_CASE_DOCS = 12

TOP_CASES = 15


# Frames that make a cluster especially suitable for
# a detective/investigation style application.
INVESTIGATIVE_FRAMES = {
    "trial-hearing",
    "charge-indict",
    "arrest-jail",
    "convict",
    "sentence",
    "attack",
    "extradite",
    "release-parole",
    "sue",
    "acquit",
    "appeal",
    "execute",
}


# =========================================================
# ENTITY HELPERS
# =========================================================

def get_entity_category(tags):

    tags = set(tags)

    if "type::person" in tags:
        return "person"

    if "type::organization" in tags:
        return "organization"

    if "type::event" in tags:
        return "event"

    if "type::location" in tags:
        return "location"

    if "type::entity" in tags:
        return "entity"

    return "other"


def is_seed_entity(tags):

    category = get_entity_category(tags)

    return category in {
        "person",
        "organization",
        "event",
    }


def entity_weight(tags):

    category = get_entity_category(tags)

    weights = {
        "person": 3.0,
        "organization": 2.5,
        "event": 2.0,
        "entity": 1.0,
        "location": 0.3,
        "other": 0.2,
    }

    return weights[category]


# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_documents():

    documents = {}

    for file_path in DATA_DIR.glob("*.json"):

        article = json.loads(
            file_path.read_text(encoding="utf-8")
        )

        content = article.get("content", "").strip()

        if not content:
            continue

        document_id = str(article["id"])
        title = content.splitlines()[0]

        entities = {}

        for concept in article.get("concepts", []):

            link = concept.get("link")

            if not link:
                continue

            entities[link] = {
                "name": concept.get("text") or link,
                "tags": concept.get("tags", []),
                "concept_id": concept.get("concept"),
            }

        frame_types = [
            frame.get("type")
            for frame in article.get("frames", [])
            if frame.get("type")
            and frame.get("type") != "none"
        ]

        documents[document_id] = {
            "document_id": document_id,
            "title": title,
            "entities": entities,
            "relations_count": len(
                article.get("relations", [])
            ),
            "frame_types": frame_types,
            "iptc": set(article.get("iptc", [])),
        }

    return documents


# =========================================================
# GLOBAL ENTITY INDEX
# =========================================================

def build_entity_index(documents):

    entity_to_docs = defaultdict(set)
    entity_names = {}
    entity_tags = defaultdict(set)

    for doc_id, document in documents.items():

        for link, entity in document["entities"].items():

            entity_to_docs[link].add(doc_id)

            entity_names[link] = (
                entity.get("name") or link
            )

            entity_tags[link].update(
                entity.get("tags", [])
            )

    return (
        entity_to_docs,
        entity_names,
        entity_tags,
    )


# =========================================================
# SIMILARITY
# =========================================================

def jaccard(a, b):

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# =========================================================
# SCORE ONE ENTITY-CENTERED CASE
# =========================================================

def score_case(
    seed_link,
    documents,
    entity_to_docs,
    entity_names,
    entity_tags,
):

    case_docs = set(
        entity_to_docs[seed_link]
    )

    doc_count = len(case_docs)

    # -----------------------------------------------------
    # Find secondary entities shared inside this case
    # -----------------------------------------------------

    secondary_entities = []

    for other_link, other_docs in entity_to_docs.items():

        if other_link == seed_link:
            continue

        shared_docs = case_docs & other_docs

        if len(shared_docs) < 2:
            continue

        # Ignore entities that occur extremely widely.
        # They provide little case-specific evidence.
        global_frequency = len(other_docs)

        if global_frequency > 40:
            continue

        tags = entity_tags[other_link]

        category = get_entity_category(tags)

        # Locations are useful as context but should not
        # strongly define the case.
        weight = entity_weight(tags)

        rarity = math.log(
            (len(documents) + 1)
            / (global_frequency + 1)
        ) + 1

        score = (
            len(shared_docs)
            * weight
            * rarity
        )

        secondary_entities.append({
            "link": other_link,
            "name": entity_names.get(
                other_link,
                other_link
            ),
            "category": category,
            "documents": len(shared_docs),
            "global_frequency": global_frequency,
            "score": score,
        })

    secondary_entities.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # -----------------------------------------------------
    # Pairwise specific-entity connectivity
    # -----------------------------------------------------

    connected_pairs = 0
    possible_pairs = 0

    for doc_a, doc_b in combinations(
        sorted(case_docs),
        2
    ):

        possible_pairs += 1

        entities_a = set(
            documents[doc_a]["entities"]
        )

        entities_b = set(
            documents[doc_b]["entities"]
        )

        shared = (
            entities_a
            & entities_b
        ) - {seed_link}

        informative_shared = []

        for entity in shared:

            if len(entity_to_docs[entity]) > 40:
                continue

            category = get_entity_category(
                entity_tags[entity]
            )

            if category in {
                "person",
                "organization",
                "event",
                "entity",
            }:
                informative_shared.append(entity)

        if informative_shared:
            connected_pairs += 1

    pair_density = (
        connected_pairs / possible_pairs
        if possible_pairs
        else 0.0
    )

    # -----------------------------------------------------
    # Topic coherence
    # -----------------------------------------------------

    topic_scores = []

    for doc_a, doc_b in combinations(
        sorted(case_docs),
        2
    ):

        topic_scores.append(
            jaccard(
                documents[doc_a]["iptc"],
                documents[doc_b]["iptc"],
            )
        )

    topic_coherence = (
        sum(topic_scores) / len(topic_scores)
        if topic_scores
        else 0.0
    )

    # -----------------------------------------------------
    # Frames
    # -----------------------------------------------------

    frame_counter = Counter()

    for doc_id in case_docs:

        frame_counter.update(
            documents[doc_id]["frame_types"]
        )

    documents_with_frames = sum(
        1
        for doc_id in case_docs
        if documents[doc_id]["frame_types"]
    )

    investigative_frame_count = sum(
        count
        for frame, count in frame_counter.items()
        if frame in INVESTIGATIVE_FRAMES
    )

    # -----------------------------------------------------
    # Relations
    # -----------------------------------------------------

    relation_count = sum(
        documents[doc_id]["relations_count"]
        for doc_id in case_docs
    )

    # -----------------------------------------------------
    # Size preference
    # -----------------------------------------------------

    if 4 <= doc_count <= 8:
        size_score = 20

    elif doc_count in {3, 9, 10}:
        size_score = 15

    else:
        size_score = 10

    # -----------------------------------------------------
    # Seed type
    # -----------------------------------------------------

    seed_category = get_entity_category(
        entity_tags[seed_link]
    )

    seed_type_bonus = {
        "person": 15,
        "organization": 10,
        "event": 8,
    }.get(seed_category, 0)

    # -----------------------------------------------------
    # Secondary entity strength
    # -----------------------------------------------------

    secondary_score = sum(
        entity["score"]
        for entity in secondary_entities[:8]
    )

    # -----------------------------------------------------
    # Final investigation suitability score
    # -----------------------------------------------------

    score = (
        size_score
        + seed_type_bonus
        + secondary_score * 0.5
        + pair_density * 20
        + topic_coherence * 20
        + documents_with_frames * 1.5
        + investigative_frame_count * 3
        + min(relation_count, 200) * 0.05
    )

    return {
        "seed_link": seed_link,
        "seed_name": entity_names.get(
            seed_link,
            seed_link
        ),
        "seed_category": seed_category,
        "score": round(score, 2),
        "document_count": doc_count,
        "pair_density": round(
            pair_density,
            3
        ),
        "topic_coherence": round(
            topic_coherence,
            3
        ),
        "relation_count": relation_count,
        "investigative_frame_count":
            investigative_frame_count,
        "frames": dict(
            frame_counter.most_common()
        ),
        "secondary_entities":
            secondary_entities[:10],
        "documents": [
            {
                "document_id": doc_id,
                "title":
                    documents[doc_id]["title"],
                "frames":
                    documents[doc_id]["frame_types"],
            }
            for doc_id in sorted(case_docs)
        ],
    }


# =========================================================
# REMOVE DUPLICATE CASES
# =========================================================

def deduplicate_cases(cases):

    selected = []

    for candidate in cases:

        candidate_docs = {
            doc["document_id"]
            for doc in candidate["documents"]
        }

        duplicate = False

        for existing in selected:

            existing_docs = {
                doc["document_id"]
                for doc in existing["documents"]
            }

            overlap = jaccard(
                candidate_docs,
                existing_docs
            )

            if overlap >= 0.75:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

    return selected


# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading DWIE...")

    documents = load_documents()

    (
        entity_to_docs,
        entity_names,
        entity_tags,
    ) = build_entity_index(documents)

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Linked entities: {len(entity_to_docs)}"
    )

    # -----------------------------------------------------
    # Find valid seed entities
    # -----------------------------------------------------

    seeds = []

    for entity_link, doc_ids in entity_to_docs.items():

        frequency = len(doc_ids)

        if not (
            MIN_CASE_DOCS
            <= frequency
            <= MAX_CASE_DOCS
        ):
            continue

        if not is_seed_entity(
            entity_tags[entity_link]
        ):
            continue

        seeds.append(entity_link)

    print(
        f"Potential case seeds: {len(seeds)}"
    )

    # -----------------------------------------------------
    # Score cases
    # -----------------------------------------------------

    cases = [
        score_case(
            seed,
            documents,
            entity_to_docs,
            entity_names,
            entity_tags,
        )
        for seed in seeds
    ]

    cases.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    cases = deduplicate_cases(cases)

    # -----------------------------------------------------
    # Save COMPLETE results
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "case_seed_candidates.json"
    )

    output_path.write_text(
        json.dumps(
            cases,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print ONLY concise summary
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TOP ENTITY-CENTERED CASE CANDIDATES")
    print("=" * 80)

    for i, case in enumerate(
        cases[:TOP_CASES],
        start=1
    ):

        print(
            f"\nCASE {i}: "
            f"{case['seed_name']}"
        )

        print(
            f"  Type: {case['seed_category']}"
        )

        print(
            f"  Score: {case['score']}"
        )

        print(
            f"  Documents: "
            f"{case['document_count']}"
        )

        print(
            f"  Pair density: "
            f"{case['pair_density']}"
        )

        print(
            f"  Topic coherence: "
            f"{case['topic_coherence']}"
        )

        print(
            f"  Relations: "
            f"{case['relation_count']}"
        )

        print(
            f"  Investigative frames: "
            f"{case['investigative_frame_count']}"
        )

        print("  Secondary entities:")

        for entity in (
            case["secondary_entities"][:5]
        ):

            print(
                f"    - {entity['name']} "
                f"({entity['documents']} docs)"
            )

        print("  Documents:")

        for document in case["documents"]:

            frame_text = ""

            if document["frames"]:
                frame_text = (
                    " | "
                    + ", ".join(
                        document["frames"]
                    )
                )

            print(
                f"    - "
                f"{document['document_id']} "
                f"| {document['title']}"
                f"{frame_text}"
            )

    print(
        f"\nFull results saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()