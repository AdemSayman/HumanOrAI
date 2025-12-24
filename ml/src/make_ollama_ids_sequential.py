from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AI_IN  = RAW_DIR / "ai_abstracts_ollama_llama3.csv"
AI_OUT = OUT_DIR / "ai_abstracts_ollama_llama3_sequential.csv"

def clean_text(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip()

    # Prefix temizliği
    t = re.sub(r"^Here is a rewritten abstract:\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^Here's a rewritten abstract:\s*", "", t, flags=re.IGNORECASE).strip()

    # tırnak karmaşası
    t = t.strip().strip('"').replace('""', '"').strip()

    # boşluk normalize
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    df = pd.read_csv(AI_IN)

    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("AI CSV içinde 'id' ve 'text' kolonları olmalı.")

    df["old_id"] = df["id"].astype(int)
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Çok kısa/boş metinleri ele (istersen eşiği değiştir)
    df = df[df["text"].str.len() >= 200].copy()

    # Aynı old_id birden fazla varsa sonuncuyu tut
    df = df.drop_duplicates(subset=["old_id"], keep="last")

    # old_id'ye göre sırala
    df = df.sort_values("old_id").reset_index(drop=True)

    # Yeni ardışık id ver
    df["id"] = range(1, len(df) + 1)

    # Kolon sırası (orijinal kolonların hepsini koruyoruz)
    # id, old_id başa gelsin
    cols = df.columns.tolist()
    cols = ["id", "old_id"] + [c for c in cols if c not in ("id", "old_id")]
    df = df[cols]

    df.to_csv(AI_OUT, index=False, encoding="utf-8")
    print("Bitti ✅")
    print("Girdi satır:", len(pd.read_csv(AI_IN)))
    print("Çıktı satır:", len(df))
    print("Kaydedildi:", AI_OUT)

if __name__ == "__main__":
    main()
