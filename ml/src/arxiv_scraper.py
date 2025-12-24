import arxiv
import pandas as pd
import time
import os
from pathlib import Path

# Kaç özet çekeceğiz (biraz fazla alalım, temizlemede düşebilir)
MAX_RESULTS = 3200

# ArXiv sorgusu - Computer Science geneli
QUERY = "cat:cs.*"

# Proje kök dizinini bul (HumanOrAI klasörü)
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "human_abstracts_raw.csv"


def fetch_arxiv_abstracts(query: str, max_results: int = MAX_RESULTS) -> pd.DataFrame:
    client = arxiv.Client()  # yeni arxiv API'si

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    rows = []
    idx = 1

    print(f"Sorgu: {query}")
    print(f"Maksimum sonuç: {max_results}")

    for result in client.results(search):
        abstract = result.summary.strip().replace("\n", " ")
        title = result.title.strip().replace("\n", " ")

        # çok kısa olanları alma
        if len(abstract.split()) < 40:
            continue

        rows.append({
            "id": idx,
            "text": abstract,
            "label": "human",
            "source": "arxiv",
            "paper_id": result.entry_id,
            "title": title,
            "license": "arxiv"
        })

        if idx % 100 == 0:
            print(f"{idx} özet toplandı...")

        idx += 1

        # arXiv'e çok yüklenmemek için çok küçük bekleme
        time.sleep(0.1)

        if idx > max_results:
            break

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("ArXiv verileri çekiliyor...")

    df = fetch_arxiv_abstracts(QUERY, MAX_RESULTS)

    print(f"Toplam çekilen özet sayısı (filtrelenmiş): {len(df)}")

    # Klasör yoksa oluştur
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Kaydedildi: {OUTPUT_FILE}")
