from pathlib import Path
import pandas as pd
import difflib
import re
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]

AI_CSV_IN = BASE_DIR / "data" / "raw" / "ai_abstracts_raw.csv"
AI_CSV_OUT = BASE_DIR / "data" / "processed" / "ai_abstracts_clean_strict.csv"

# Daha agresif eşik
SIM_THRESHOLD = 0.85  # 0.90 da deneyebilirsin
# Not: Bu sefer WINDOW yok, tüm önceki kayıtlarla bakacağız


def normalize(text: str) -> str:
    """
    Metni benzerlik için normalize et:
    - lowercase
    - fazla boşlukları temizle
    - baş/son boşluk kırp
    """
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_similar(a: str, b: str, threshold: float) -> bool:
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
    kept_norm_texts = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sıkı benzerlik filtresi"):
        text = str(row["text"])
        norm = normalize(text)

        similar_found = False
        for prev_norm in kept_norm_texts:
            if is_similar(norm, prev_norm, SIM_THRESHOLD):
                similar_found = True
                break

        if similar_found:
            # Bu kaydı çok benzer olduğu için atlıyoruz
            continue
        else:
            kept_indices.append(idx)
            kept_norm_texts.append(norm)

    cleaned_df = df.loc[kept_indices].reset_index(drop=True)
    print(f"Temizlenmiş (sıkı) AI satır sayısı: {len(cleaned_df)}")

    AI_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(AI_CSV_OUT, index=False, encoding="utf-8")
    print(f"Kaydedildi: {AI_CSV_OUT}")


if __name__ == "__main__":
    main()
