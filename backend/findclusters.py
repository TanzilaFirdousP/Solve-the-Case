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


MIN_CASE_DOCS = 3
MAX_CASE_DOCS = 12
TOP_CASES = 15


# ---------------------------------------------------------
# Investigation-oriented event weights
# ---------------------------------------------------------

INVESTIGATIVE_FRAMES = {
    "trial-hearing": 6,
    "charge-indict": 6,
    "arrest-jail": 6,
    "convict": 6,
    "sentence": 6,
    "attack": 5,
    "execute": 5,
    "extradite": 5,
    "appeal": 4,
    "acquit": 4,
    "release-parole": 4,
    "sue": 3,
    "die": 2,
    "demonstrate": 1,
}


# Frames that strongly indicate non-investigative content
SPORT_FRAMES = {
    "cpn-game",
}


# Entity tags especially useful for detective-style cases
INVESTIGATIVE_ENTITY_TAGS = {
    "type::offender",
    "type::victim",
    "type::criminal_org",
    "type::armed_movement",
    "type::police_org",
    "type::court",
    "type::judge",
    "type::police_per",
    "type::justice_per",
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

    return {
        "person": 3.0,
        "organization": 2.5,
        "event": 2.0,
        "entity": 1.0,
        "location": 0.25,
        "other": 0.15,
    }.get(category, 0.15)


# =========================================================
# LOAD DWIE
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

        frames = [
            frame.get("type")
            for frame in article.get("frames", [])
            if frame.get("type")
            and frame.get("type") != "none"
        ]

        documents[document_id] = {
            "document_id": document_id,
            "title": title,
            "entities": entities,
            "frames": frames,
            "relations_count": len(
                article.get("relations", [])
            ),
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
# SCORE ONE CASE
# =========================================================

def score_case(
    seed_link,
    documents,
    entity_to_docs,
    entity_names,
    entity_tags,
):

    case_docs = set(entity_to_docs[seed_link])

    doc_count = len(case_docs)

    seed_tags = set(
        entity_tags[seed_link]
    )

    seed_category = get_entity_category(
        seed_tags
    )

    # -----------------------------------------------------
    # Frames
    # -----------------------------------------------------

    frame_counter = Counter()

    investigative_score = 0
    investigative_docs = 0
    sport_docs = 0

    for doc_id in case_docs:

        frames = documents[doc_id]["frames"]

        frame_counter.update(frames)

        doc_has_investigative_frame = False

        for frame in frames:

            if frame in INVESTIGATIVE_FRAMES:

                investigative_score += (
                    INVESTIGATIVE_FRAMES[frame]
                )

                doc_has_investigative_frame = True

        if doc_has_investigative_frame:
            investigative_docs += 1

        if any(
            frame in SPORT_FRAMES
            for frame in frames
        ):
            sport_docs += 1

    # Ignore cases with no investigative event at all.
    if investigative_score == 0:
        return None

    sport_ratio = (
        sport_docs / doc_count
        if doc_count
        else 0
    )

    # Remove obvious sports clusters.
    if sport_ratio > 0.40:
        return None

    # -----------------------------------------------------
    # Secondary entities
    # -----------------------------------------------------

    secondary_entities = []

    total_docs = len(documents)

    for other_link, other_docs in entity_to_docs.items():

        if other_link == seed_link:
            continue

        shared_docs = case_docs & other_docs

        if len(shared_docs) < 2:
            continue

        global_frequency = len(other_docs)

        # Very common entities such as Germany and US
        # should not dominate case discovery.
        if global_frequency > 40:
            continue

        tags = entity_tags[other_link]

        category = get_entity_category(tags)

        if category == "location":
            continue

        rarity = math.log(
            (total_docs + 1)
            / (global_frequency + 1)
        ) + 1

        score = (
            len(shared_docs)
            * entity_weight(tags)
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
            "score": round(score, 2),
        })

    secondary_entities.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # -----------------------------------------------------
    # Pair connectivity
    # -----------------------------------------------------

    connected_pairs = 0
    possible_pairs = 0

    for doc_a, doc_b in combinations(
        case_docs,
        2
    ):

        possible_pairs += 1

        shared_entities = (
            set(documents[doc_a]["entities"])
            &
            set(documents[doc_b]["entities"])
        )

        shared_entities.discard(seed_link)

        informative = []

        for entity in shared_entities:

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
                informative.append(entity)

        if informative:
            connected_pairs += 1

    pair_density = (
        connected_pairs / possible_pairs
        if possible_pairs
        else 0
    )

    # -----------------------------------------------------
    # Topic coherence
    # -----------------------------------------------------

    topic_scores = []

    for doc_a, doc_b in combinations(
        case_docs,
        2
    ):

        topic_scores.append(
            jaccard(
                documents[doc_a]["iptc"],
                documents[doc_b]["iptc"],
            )
        )

    topic_coherence = (
        sum(topic_scores)
        / len(topic_scores)
        if topic_scores
        else 0
    )

    # -----------------------------------------------------
    # Relation information
    # -----------------------------------------------------

    relation_count = sum(
        documents[doc]["relations_count"]
        for doc in case_docs
    )

    # -----------------------------------------------------
    # Seed bonuses
    # -----------------------------------------------------

    if seed_category == "person":
        seed_type_bonus = 20

    elif seed_category == "organization":
        seed_type_bonus = 12

    elif seed_category == "event":
        seed_type_bonus = 10

    else:
        seed_type_bonus = 0

    # Extra bonus if DWIE explicitly considers seed
    # justice/conflict related.
    investigative_seed_bonus = 0

    for tag in seed_tags:

        if tag in INVESTIGATIVE_ENTITY_TAGS:
            investigative_seed_bonus += 8

    # -----------------------------------------------------
    # Case size
    # -----------------------------------------------------

    if 4 <= doc_count <= 8:
        size_bonus = 20

    elif doc_count in {3, 9, 10}:
        size_bonus = 15

    else:
        size_bonus = 8

    # -----------------------------------------------------
    # Investigation diversity
    # -----------------------------------------------------

    unique_investigative_frames = {
        frame
        for frame in frame_counter
        if frame in INVESTIGATIVE_FRAMES
    }

    frame_diversity_bonus = (
        len(unique_investigative_frames)
        * 5
    )

    # -----------------------------------------------------
    # Secondary entity strength
    # -----------------------------------------------------

    secondary_strength = sum(
        entity["score"]
        for entity in secondary_entities[:8]
    )

    # -----------------------------------------------------
    # Final score
    # -----------------------------------------------------

    final_score = (
        seed_type_bonus
        + investigative_seed_bonus
        + size_bonus
        + investigative_score * 3
        + investigative_docs * 5
        + frame_diversity_bonus
        + secondary_strength * 0.35
        + pair_density * 12
        + topic_coherence * 12
        + min(relation_count, 250) * 0.04
        - sport_ratio * 50
    )

    return {
        "seed_link": seed_link,
        "seed_name": entity_names.get(
            seed_link,
            seed_link
        ),
        "seed_category": seed_category,
        "seed_tags": sorted(seed_tags),

        "score": round(
            final_score,
            2
        ),

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

        "investigative_documents":
            investigative_docs,

        "investigative_score":
            investigative_score,

        "frames":
            dict(frame_counter),

        "secondary_entities":
            secondary_entities[:10],

        "documents": [
            {
                "document_id": doc_id,
                "title":
                    documents[doc_id]["title"],
                "frames":
                    documents[doc_id]["frames"],
            }
            for doc_id in sorted(case_docs)
        ],
    }


# =========================================================
# REMOVE DUPLICATES
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

            if overlap >= 0.70:
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
        f"Linked entities: "
        f"{len(entity_to_docs)}"
    )

    # -----------------------------------------------------
    # Candidate seeds
    # -----------------------------------------------------

    seeds = []

    for link, doc_ids in entity_to_docs.items():

        frequency = len(doc_ids)

        if not (
            MIN_CASE_DOCS
            <= frequency
            <= MAX_CASE_DOCS
        ):
            continue

        if not is_seed_entity(
            entity_tags[link]
        ):
            continue

        seeds.append(link)

    print(
        f"Potential entity seeds: "
        f"{len(seeds)}"
    )

    # -----------------------------------------------------
    # Score
    # -----------------------------------------------------

    cases = []

    for seed in seeds:

        result = score_case(
            seed,
            documents,
            entity_to_docs,
            entity_names,
            entity_tags,
        )

        if result:
            cases.append(result)

    cases.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    cases = deduplicate_cases(cases)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "investigation_case_candidates.json"
    )

    output_path.write_text(
        json.dumps(
            cases,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print concise output
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)

    print(
        "TOP INVESTIGATION-WORTHY CASES"
    )

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
            f"  Type: "
            f"{case['seed_category']}"
        )

        print(
            f"  Score: "
            f"{case['score']}"
        )

        print(
            f"  Documents: "
            f"{case['document_count']}"
        )

        print(
            f"  Investigative documents: "
            f"{case['investigative_documents']}"
        )

        print(
            f"  Frames: "
            f"{case['frames']}"
        )

        print(
            f"  Secondary entities:"
        )

        for entity in (
            case["secondary_entities"][:5]
        ):

            print(
                f"    - "
                f"{entity['name']} "
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
        "\nFull candidates saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()