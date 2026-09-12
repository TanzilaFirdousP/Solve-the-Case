import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "DWIE" / "data" / "annos_with_content"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_articles():
    articles = []

    for file_path in INPUT_DIR.glob("*.json"):
        article = json.loads(file_path.read_text(encoding="utf-8"))

        if article.get("content", "").strip():
            articles.append(article)

    return articles


def last_natural_break(text, start, end):
    breaks = []

    for marker in ("\n\n", "\n", ". ", "! ", "? "):
        position = text.rfind(marker, start, end)
        if position != -1:
            breaks.append(position + len(marker))

    return max(breaks) if breaks else None


def first_natural_break(text, start, end):
    breaks = []

    for marker in ("\n\n", "\n", ". ", "! ", "? "):
        position = text.find(marker, start, end)
        if position != -1:
            breaks.append(position + len(marker))

    return min(breaks) if breaks else None


def split_into_chunks(text):
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        maximum_end = min(start + CHUNK_SIZE, len(text))

        if maximum_end == len(text):
            end = len(text)
        else:
            minimum_end = start + (CHUNK_SIZE // 2)
            end = last_natural_break(text, minimum_end, maximum_end)
            end = end or maximum_end

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
                "text": chunk_text
            })
            chunk_index += 1

        if end == len(text):
            break

        overlap_target = max(start + 1, end - CHUNK_OVERLAP)
        clean_start = first_natural_break(text, overlap_target, end)

        start = clean_start if clean_start and clean_start < end else overlap_target

    return chunks


def build_processed_documents(articles):
    processed_documents = []

    for article in articles:
        document_id = str(article["id"])
        title = article["content"].splitlines()[0]

        for chunk in split_into_chunks(article["content"]):
            processed_documents.append({
                "chunk_id": f"{document_id}_chunk_{chunk['chunk_index']}",
                "document_id": document_id,
                "title": title,
                "text": chunk["text"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "verified": True
            })

    return processed_documents


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"DWIE folder not found: {INPUT_DIR}")

    articles = load_articles()
    processed_documents = build_processed_documents(articles)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "processed_documents.json"

    output_path.write_text(
        json.dumps(processed_documents, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Loaded {len(articles)} articles with content.")
    print(f"Created {len(processed_documents)} searchable chunks.")
    print(f"Saved processed evidence to: {output_path}")


if __name__ == "__main__":
    main()