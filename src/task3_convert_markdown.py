"""
Task 3 - Convert toan bo file trong data/landing/ thanh Markdown.

Su dung MarkItDown cua Microsoft:
    https://github.com/microsoft/markitdown

Cai dat:
    pip install markitdown

Huong dan:
    1. Scan toan bo file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Luu vao data/standardized/ giu nguyen cau truc thu muc
"""

import json
import sys
from pathlib import Path

from markitdown import MarkItDown

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        result = md.convert(str(filepath))
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(result.text_content, encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        crawled_at = (
            data.get("date_crawled")
            or data.get("scrapedAt")
            or data.get("publishedAt")
            or "N/A"
        )

        body = data.get("content_markdown") or data.get("content")
        if not body and isinstance(data.get("paragraphs"), list):
            body = "\n\n".join(str(paragraph) for paragraph in data["paragraphs"] if paragraph)
        if not body:
            body = data.get("contentText")

        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {crawled_at}\n\n---\n\n"

        content = header + (body or "")
        output_path.write_text(content, encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_all():
    """Convert toan bo files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output tai:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
