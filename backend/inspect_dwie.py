import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "DWIE" / "data" / "annos_with_content"


# Pick one article for inspection
article_path = next(DATA_DIR.glob("*.json"))

article = json.loads(
    article_path.read_text(encoding="utf-8")
)


print("=" * 80)
print("ARTICLE")
print("=" * 80)

print("File:", article_path.name)
print("Article ID:", article["id"])
print("Title:", article["content"].splitlines()[0])

print("\nContent preview:")
print(article["content"][:700])

print("\nNumber of concepts:", len(article.get("concepts", [])))
print("Number of mentions:", len(article.get("mentions", [])))
print("Number of relations:", len(article.get("relations", [])))
print("Number of frames:", len(article.get("frames", [])))


# ---------------------------------------------------------
# Build concept lookup
# ---------------------------------------------------------

concepts_by_id = {
    concept["concept"]: concept
    for concept in article.get("concepts", [])
}


# ---------------------------------------------------------
# Group mention strings by concept
# ---------------------------------------------------------

mentions_by_concept = defaultdict(list)

for mention in article.get("mentions", []):
    concept_id = mention["concept"]
    mention_text = mention["text"]

    if mention_text not in mentions_by_concept[concept_id]:
        mentions_by_concept[concept_id].append(mention_text)


# ---------------------------------------------------------
# Print concepts
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("CONCEPTS / ENTITIES")
print("=" * 80)

for concept in article.get("concepts", []):

    concept_id = concept["concept"]
    name = concept.get("text")
    link = concept.get("link")
    count = concept.get("count")

    useful_types = [
        tag
        for tag in concept.get("tags", [])
        if tag.startswith("type::")
    ]

    mention_variants = mentions_by_concept.get(concept_id, [])

    print(f"\nConcept ID: {concept_id}")
    print(f"Name:       {name}")
    print(f"Link:       {link}")
    print(f"Count:      {count}")
    print(f"Types:      {useful_types}")
    print(f"Mentions:   {mention_variants}")


# ---------------------------------------------------------
# Print readable relations
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("RELATIONS")
print("=" * 80)

if not article.get("relations"):
    print("No annotated relations.")

for relation in article.get("relations", []):

    subject_id = relation["s"]
    object_id = relation["o"]
    predicate = relation["p"]

    subject = concepts_by_id.get(subject_id, {})
    object_ = concepts_by_id.get(object_id, {})

    subject_name = subject.get("text", f"Concept {subject_id}")
    object_name = object_.get("text", f"Concept {object_id}")

    print(
        f"{subject_name} "
        f"--{predicate}--> "
        f"{object_name}"
    )


# ---------------------------------------------------------
# Frames
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("FRAMES")
print("=" * 80)

frames = article.get("frames", [])

if not frames:
    print("No frames.")
else:
    for frame in frames:
        print(json.dumps(frame, indent=2))


# ---------------------------------------------------------
# IPTC classifications
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("IPTC")
print("=" * 80)

print(json.dumps(article.get("iptc", []), indent=2))