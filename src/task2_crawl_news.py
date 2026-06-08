"""
Task 2 - Crawl bai bao ve nghe si lien quan toi ma tuy.

Yeu cau README:
    1. Crawl toi thieu 5 bai bao tu cac trang tin tuc Viet Nam.
    2. Su dung Crawl4AI.
    3. Luu output vao data/landing/news/.
    4. Moi bai luu 1 file JSON voi metadata: url, title, date_crawled, content.
"""

import asyncio
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def setup_directory() -> None:
    """Tao thu muc data/landing/news/ neu chua co."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://vnexpress.net/nha-thiet-ke-nguyen-cong-tri-bi-bat-vi-lien-quan-ma-tuy-4917929.html",
    "https://vietnamnet.vn/loi-khai-cua-dien-vien-huu-tin-sau-khi-bi-bat-vi-choi-ma-tuy-2029765.html",
    "https://tuoitre.vn/nguoi-mau-nhikolai-dinh-bi-bat-trong-chuyen-an-ma-tuy-o-khu-ma-lang-quan-1-20240625230004986.htm",
    "https://www.sggp.org.vn/dieu-tra-vu-nguoi-mau-dien-vien-andrea-aybar-lien-quan-ma-tuy-post767691.html",
]


def slugify(text: str, fallback: str) -> str:
    """Tao ten file ngan gon, khong dau va an toan tren Windows."""
    value = unicodedata.normalize("NFKD", text)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value[:80] or fallback


def extract_markdown(result) -> str:
    """Lay markdown tu Crawl4AI, tuong thich voi nhieu version."""
    markdown = getattr(result, "markdown", "") or ""
    if isinstance(markdown, str):
        return markdown

    for attr in ("raw_markdown", "fit_markdown", "markdown"):
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    return str(markdown)


def extract_title(result, content_markdown: str, url: str) -> str:
    """Lay title tu metadata; neu thieu thi fallback sang heading dau tien."""
    metadata = getattr(result, "metadata", {}) or {}
    for key in ("title", "og:title", "twitter:title"):
        title = metadata.get(key)
        if title:
            return str(title).strip()

    for line in content_markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()

    return url.rstrip("/").split("/")[-1] or "Unknown"


async def crawl_article(url: str) -> dict:
    """
    Crawl mot bai bao va tra ve dict chua metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str,
            "content_markdown": str,
            "content": str,
            "metadata": dict
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    content_markdown = extract_markdown(result)
    title = extract_title(result, content_markdown, url)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content_markdown,
        "content": content_markdown,
        "metadata": getattr(result, "metadata", {}) or {},
        "success": bool(getattr(result, "success", True)),
        "error_message": getattr(result, "error_message", None),
    }


async def crawl_all() -> None:
    """Crawl toan bo bai bao trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            article = {
                "url": url,
                "title": "Crawl failed",
                "date_crawled": datetime.now(timezone.utc).isoformat(),
                "content_markdown": "",
                "content": "",
                "metadata": {},
                "success": False,
                "error_message": str(exc),
            }

        title_slug = slugify(article.get("title", ""), fallback=f"article-{i:02d}")
        filepath = DATA_DIR / f"{i:02d}_{title_slug}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hay dien ARTICLE_URLS truoc khi chay.")
    else:
        asyncio.run(crawl_all())
