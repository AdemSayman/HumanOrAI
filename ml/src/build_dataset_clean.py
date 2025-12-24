from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Öncelik: processed içindeki final ai dosyası
AI_CANDIDATES = [
    PROC_DIR / "ai_abstracts_final_3000.csv",
    PROC_DIR / "ai_abstracts_ollama_llama3_sequential.csv",
    RAW_DIR / "ai_abstracts_raw.csv",
    RAW_DIR / "ai_abstracts_raw.csv",
]

HUMAN_CANDIDATES = [
    RAW_DIR / "human_abstracts_raw.csv",
    RAW_DIR / "human_abstracts_raw.csv",
]

OUT_FILE = OUT_DIR / "dataset_clean.csv"
TARGET_N = 3000

def pick_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def normalize_df(df, label_value):
    # minimum: text olmalı
    if "text" not in df.columns:
        raise ValueError("CSV içinde 'text' kolonu yok.")

    out = pd.DataFrame()
    out["text"] = df["text"].astype(str).fillna("").str.strip()
    out = out[out["text"].str.len() > 0].copy()

    out["label"] = label_value
    return out

def main():
    human_path = pick_first_existing(HUMAN_CANDIDATES)
    ai_path = pick_first_existing(AI_CANDIDATES)

    if human_path is None:
        raise FileNotFoundError(f"Human dosyası bulunamadı. Arananlar: {HUMAN_CANDIDATES}")
    if ai_path is None:
        raise FileNotFoundError(f"AI dosyası bulunamadı. Arananlar: {AI_CANDIDATES}")

    print("Human dosyası:", human_path)
    print("AI dosyası:", ai_path)

    human_raw = pd.read_csv(human_path)
    ai_raw = pd.read_csv(ai_path)

    human = normalize_df(human_raw, "human")
    ai = normalize_df(ai_raw, "ai")

    # Tam 3000 seç (yeterli değilse mevcut kadar)
    if len(human) < TARGET_N:
        print(f"UYARI: Human {len(human)} satır. {TARGET_N} yok, mevcut kadar kullanılacak.")
        human_n = len(human)
    else:
        human_n = TARGET_N

    if len(ai) < TARGET_N:
        print(f"UYARI: AI {len(ai)} satır. {TARGET_N} yok, mevcut kadar kullanılacak.")
        ai_n = len(ai)
    else:
        ai_n = TARGET_N

    human = human.sample(n=human_n, random_state=42).reset_index(drop=True)
    ai = ai.sample(n=ai_n, random_state=42).reset_index(drop=True)

    merged = pd.concat([human, ai], ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # id ekle
    merged.insert(0, "id", range(1, len(merged) + 1))

    merged.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print("\nBitti ✅")
    print("Human:", (merged["label"] == "human").sum())
    print("AI   :", (merged["label"] == "ai").sum())
    print("Toplam:", len(merged))
    print("Kaydedildi:", OUT_FILE)

if __name__ == "__main__":
    main()
