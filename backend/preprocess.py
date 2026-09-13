import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "DWIE"
    / "data"
    / "annos_with_content"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# =========================================================
# BASIC HELPERS
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

    if "type::time" in tags:
        return "time"

    if "type::role" in tags:
        return "role"

    if "type::entity" in tags:
        return "entity"

    return "other"


def make_entity_id(document_id, concept):
    """
    Prefer DWIE's Wikipedia/entity link for cross-document identity.

    If no link exists, fall back to a document-local concept ID.
    """

    link = concept.get("link")

    if link:
        return f"wiki:{link}"

    return (
        f"doc:{document_id}:"
        f"concept:{concept['concept']}"
    )


# =========================================================
# LOAD ARTICLES
# =========================================================

def load_articles():
    articles = []

    for file_path in INPUT_DIR.glob("*.json"):
        article = json.loads(
            file_path.read_text(
                encoding="utf-8"
            )
        )

        if article.get("content", "").strip():
            articles.append(article)

    return articles


# =========================================================
# CHUNKING
# =========================================================

def last_natural_break(text, start, end):
    breaks = []

    for marker in (
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
    ):
        position = text.rfind(
            marker,
            start,
            end
        )

        if position != -1:
            breaks.append(
                position + len(marker)
            )

    return max(breaks) if breaks else None


def first_natural_break(text, start, end):
    breaks = []

    for marker in (
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
    ):
        position = text.find(
            marker,
            start,
            end
        )

        if position != -1:
            breaks.append(
                position + len(marker)
            )

    return min(breaks) if breaks else None


def split_into_chunks(text):
    chunks = []

    start = 0
    chunk_index = 0

    while start < len(text):

        maximum_end = min(
            start + CHUNK_SIZE,
            len(text)
        )

        if maximum_end == len(text):
            end = len(text)

        else:
            minimum_end = (
                start + CHUNK_SIZE // 2
            )

            end = last_natural_break(
                text,
                minimum_end,
                maximum_end
            )

            end = end or maximum_end

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
                "text": chunk_text,
            })

            chunk_index += 1

        if end == len(text):
            break

        overlap_target = max(
            start + 1,
            end - CHUNK_OVERLAP
        )

        clean_start = first_natural_break(
            text,
            overlap_target,
            end
        )

        if clean_start and clean_start < end:
            start = clean_start
        else:
            start = overlap_target

    return chunks


# =========================================================
# ARTICLE CONCEPT MAP
# =========================================================

def build_concept_map(article):
    document_id = str(article["id"])

    concept_map = {}

    for concept in article.get(
        "concepts",
        []
    ):
        concept_id = concept["concept"]

        entity_id = make_entity_id(
            document_id,
            concept
        )

        concept_map[concept_id] = {
            "entity_id": entity_id,
            "concept_id": concept_id,
            "name": (
                concept.get("text")
                or concept.get("link")
                or f"Concept {concept_id}"
            ),
            "link": concept.get("link"),
            "tags": concept.get(
                "tags",
                []
            ),
            "category":
                get_entity_category(
                    concept.get(
                        "tags",
                        []
                    )
                ),
        }

    return concept_map


# =========================================================
# GLOBAL ENTITY STORE
# =========================================================

def add_entities_from_article(
    article,
    concept_map,
    global_entities,
):
    document_id = str(article["id"])

    mentions_by_concept = defaultdict(set)

    for mention in article.get(
        "mentions",
        []
    ):
        mentions_by_concept[
            mention["concept"]
        ].add(
            mention.get("text", "")
        )

    for concept_id, entity in (
        concept_map.items()
    ):
        entity_id = entity["entity_id"]

        if entity_id not in global_entities:
            global_entities[entity_id] = {
                "entity_id": entity_id,
                "name": entity["name"],
                "link": entity["link"],
                "category":
                    entity["category"],
                "tags": set(
                    entity["tags"]
                ),
                "aliases": set(),
                "document_ids": set(),
            }

        stored = global_entities[
            entity_id
        ]

        stored["document_ids"].add(
            document_id
        )

        stored["tags"].update(
            entity["tags"]
        )

        for alias in mentions_by_concept[
            concept_id
        ]:
            if alias:
                stored["aliases"].add(
                    alias
                )


# =========================================================
# CHUNK ENTITY ASSOCIATION
# =========================================================

def get_chunk_entities(
    article,
    concept_map,
    char_start,
    char_end,
):
    entity_ids = set()

    for mention in article.get(
        "mentions",
        []
    ):
        mention_start = mention.get(
            "begin",
            -1
        )

        mention_end = mention.get(
            "end",
            -1
        )

        # Mention overlaps chunk
        if (
            mention_end > char_start
            and mention_start < char_end
        ):
            concept_id = mention[
                "concept"
            ]

            entity = concept_map.get(
                concept_id
            )

            if entity:
                entity_ids.add(
                    entity["entity_id"]
                )

    return sorted(entity_ids)


# =========================================================
# RELATIONS
# =========================================================

def process_relations(
    article,
    concept_map,
):
    document_id = str(article["id"])

    results = []

    for index, relation in enumerate(
        article.get("relations", [])
    ):
        subject = concept_map.get(
            relation["s"]
        )

        object_ = concept_map.get(
            relation["o"]
        )

        if not subject or not object_:
            continue

        results.append({
            "relation_id":
                f"{document_id}_relation_{index}",

            "document_id":
                document_id,

            "subject_entity_id":
                subject["entity_id"],

            "subject_name":
                subject["name"],

            "predicate":
                relation["p"],

            "object_entity_id":
                object_["entity_id"],

            "object_name":
                object_["name"],

            "verified": True,

            "source": "DWIE",
        })

    return results


# =========================================================
# FRAMES / EVENTS
# =========================================================

def process_frames(
    article,
    concept_map,
):
    document_id = str(article["id"])

    results = []

    for index, frame in enumerate(
        article.get("frames", [])
    ):
        frame_type = frame.get(
            "type"
        )

        if (
            not frame_type
            or frame_type == "none"
        ):
            continue

        processed_slots = []

        for slot in frame.get(
            "slots",
            []
        ):
            value = slot.get("value")

            entity = None

            if isinstance(value, int):
                entity = concept_map.get(
                    value
                )

            if entity:
                processed_slots.append({
                    "role":
                        slot.get("name"),

                    "entity_id":
                        entity[
                            "entity_id"
                        ],

                    "entity_name":
                        entity["name"],
                })

            else:
                processed_slots.append({
                    "role":
                        slot.get("name"),

                    "value":
                        value,
                })

        results.append({
            "frame_id":
                f"{document_id}_frame_{index}",

            "document_id":
                document_id,

            "frame_type":
                frame_type,

            "slots":
                processed_slots,

            "verified":
                True,

            "source":
                "DWIE",
        })

    return results


# =========================================================
# MAIN PREPROCESSING
# =========================================================

def process_articles(articles):

    chunks = []
    documents = []
    relations = []
    frames = []

    global_entities = {}

    for article in articles:

        document_id = str(
            article["id"]
        )

        content = article["content"]

        title = (
            content.splitlines()[0]
            if content
            else document_id
        )

        concept_map = (
            build_concept_map(
                article
            )
        )

        # ---------------------------------------------
        # Global entities
        # ---------------------------------------------

        add_entities_from_article(
            article,
            concept_map,
            global_entities,
        )

        # ---------------------------------------------
        # Relations
        # ---------------------------------------------

        article_relations = (
            process_relations(
                article,
                concept_map,
            )
        )

        relations.extend(
            article_relations
        )

        # ---------------------------------------------
        # Frames
        # ---------------------------------------------

        article_frames = (
            process_frames(
                article,
                concept_map,
            )
        )

        frames.extend(
            article_frames
        )

        # ---------------------------------------------
        # Document record
        # ---------------------------------------------

        document_entity_ids = sorted(
            {
                entity["entity_id"]
                for entity
                in concept_map.values()
            }
        )

        documents.append({
            "document_id":
                document_id,

            "title":
                title,

            "entity_ids":
                document_entity_ids,

            "relation_ids": [
                relation[
                    "relation_id"
                ]
                for relation
                in article_relations
            ],

            "frame_ids": [
                frame["frame_id"]
                for frame
                in article_frames
            ],

            "iptc":
                article.get(
                    "iptc",
                    []
                ),

            "verified":
                True,

            "source":
                "DWIE",
        })

        # ---------------------------------------------
        # Searchable chunks
        # ---------------------------------------------

        for chunk in split_into_chunks(
            content
        ):
            entity_ids = (
                get_chunk_entities(
                    article,
                    concept_map,
                    chunk["char_start"],
                    chunk["char_end"],
                )
            )

            chunks.append({
                "chunk_id":
                    (
                        f"{document_id}"
                        f"_chunk_"
                        f"{chunk['chunk_index']}"
                    ),

                "document_id":
                    document_id,

                "title":
                    title,

                "text":
                    chunk["text"],

                "char_start":
                    chunk["char_start"],

                "char_end":
                    chunk["char_end"],

                "entity_ids":
                    entity_ids,

                "verified":
                    True,

                "source":
                    "DWIE",
            })

    # =====================================================
    # Convert entity sets -> JSON lists
    # =====================================================

    entities = []

    for entity in global_entities.values():

        entities.append({
            "entity_id":
                entity["entity_id"],

            "name":
                entity["name"],

            "link":
                entity["link"],

            "category":
                entity["category"],

            "tags":
                sorted(
                    entity["tags"]
                ),

            "aliases":
                sorted(
                    entity["aliases"]
                ),

            "document_ids":
                sorted(
                    entity[
                        "document_ids"
                    ]
                ),
        })

    entities.sort(
        key=lambda item:
            item["entity_id"]
    )

    return (
        chunks,
        documents,
        entities,
        relations,
        frames,
    )


# =========================================================
# SAVE
# =========================================================

def save_json(filename, data):

    path = OUTPUT_DIR / filename

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# =========================================================
# ENTRY POINT
# =========================================================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"DWIE folder not found: "
            f"{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    articles = load_articles()

    (
        chunks,
        documents,
        entities,
        relations,
        frames,
    ) = process_articles(
        articles
    )

    chunks_path = save_json(
        "processed_documents.json",
        chunks,
    )

    documents_path = save_json(
        "documents.json",
        documents,
    )

    entities_path = save_json(
        "entities.json",
        entities,
    )

    relations_path = save_json(
        "relations.json",
        relations,
    )

    frames_path = save_json(
        "frames.json",
        frames,
    )

    print(
        f"Loaded {len(articles)} "
        f"DWIE articles."
    )

    print(
        f"Created {len(chunks)} "
        f"searchable evidence chunks."
    )

    print(
        f"Created {len(entities)} "
        f"global entities."
    )

    print(
        f"Created {len(relations)} "
        f"verified relations."
    )

    print(
        f"Created {len(frames)} "
        f"structured events."
    )

    print("\nSaved:")
    print(chunks_path)
    print(documents_path)
    print(entities_path)
    print(relations_path)
    print(frames_path)


if __name__ == "__main__":
    main()