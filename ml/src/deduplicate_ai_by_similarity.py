from pathlib import Path
import pandas as pd
import difflib
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]

AI_CSV_IN = BASE_DIR / "data" / "raw" / "ai_abstracts_raw.csv"
AI_CSV_OUT = BASE_DIR / "data" / "processed" / "ai_abstracts_clean.csv"

# Benzerlik eşiği (0.0 - 1.0 arası)
SIM_THRESHOLD = 0.95
# Her yeni satır için sadece son N tane ile karşılaştır (performans için)
WINDOW_SIZE = 50


def is_similar(a: str, b: str, threshold: float) -> bool:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


def main():
    if not AI_CSV_IN.exists():
        raise FileNotFoundError(f"AI CSV yok: {AI_CSV_IN}")

    df = pd.read_csv(AI_CSV_IN)
    print(f"Gelen AI satır sayısı: {len(df)}")

    if "text" not in df.columns:
        raise ValueError("'text' kolonu yok.")

    kept_indices = []
    kept_texts = []

    # tqdm ile progress bar gösterelim
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Benzerlik filtresi"):
        text = str(row["text"])

        similar_found = False
        # Son WINDOW_SIZE kayıtla karşılaştır
        for prev_text in kept_texts[-WINDOW_SIZE:]:
            if is_similar(text, prev_text, SIM_THRESHOLD):
                similar_found = True
                break

        if similar_found:
            # bu kaydı atla
            continue
        else:
            kept_indices.append(idx)
            kept_texts.append(text)

    cleaned_df = df.loc[kept_indices].reset_index(drop=True)
    print(f"Temizlenmiş AI satır sayısı: {len(cleaned_df)}")

    AI_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(AI_CSV_OUT, index=False, encoding="utf-8")
    print(f"Kaydedildi: {AI_CSV_OUT}")


if __name__ == "__main__":
    main()
