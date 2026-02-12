from icrawler.builtin import BingImageCrawler
from pathlib import Path

OUTPUT_DIR = Path("data/non_xray")

keywords = [
    "people photos",
    "animals photos",
    "cars photos",
    "buildings photos",
    "food photos",
    "nature scenery",
    "random objects",
    "cartoon drawings"
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for keyword in keywords:
    print(f"Downloading: {keyword}")
    try:
        crawler = BingImageCrawler(storage={"root_dir": str(OUTPUT_DIR / keyword.replace(" ", "_"))})
        crawler.crawl(keyword=keyword, max_num=80)
    except Exception as exc:
        print(f"Failed to download '{keyword}': {exc}")
