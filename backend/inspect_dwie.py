import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "DWIE" / "data" / "annos_with_content"

article_path = next(DATA_DIR.glob("*.json"))
article = json.loads(article_path.read_text(encoding="utf-8"))

print("File:", article_path.name)
print("Top-level keys:", list(article.keys()))
print("Article ID:", article["id"])
print("Content preview:", article["content"][:500])
print("Number of concepts:", len(article.get("concepts", [])))
print("Number of mentions:", len(article.get("mentions", [])))
print("Number of relations:", len(article.get("relations", [])))

print("\nFirst concept:")
print(json.dumps(article.get("concepts", [])[0], indent=2))

print("\nFirst relation:")
print(json.dumps(article.get("relations", [])[0], indent=2))

concepts_by_id = {
    concept["concept"]: concept
    for concept in article["concepts"]
}

relation = article["relations"][0]
subject = concepts_by_id[relation["s"]]["text"]
object_ = concepts_by_id[relation["o"]]["text"]

print("\nReadable first relation:")
print(f"{subject} --{relation['p']}--> {object_}")