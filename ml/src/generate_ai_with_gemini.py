from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import google.generativeai as genai

# 🔴 BURAYA KENDİ GEMINI API KEY'İNİ YAZ
 # örn. AIza...

# Hedef toplam AI kaydı (OpenAI + Gemini toplamı 3000 olacak)
TARGET_COUNT = 3000

# Proje kök klasörü
BASE_DIR = Path(__file__).resolve().parents[2]

HUMAN_CSV = BASE_DIR / "data" / "raw" / "human_abstracts_raw.csv"
AI_CSV = BASE_DIR / "data" / "raw" / "ai_abstracts_raw.csv"


def init_gemini():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY'i generate_ai_with_gemini.py içinde doldurmayı unutma!")
    genai.configure(api_key=GEMINI_API_KEY)
    # Burada gemini-pro kullanıyoruz (daha uyumlu)
    model = genai.GenerativeModel("gemini-pro")
    return model


def build_prompt(human_abstract: str) -> str:
    return f"""
You are an expert scientific writer.

Below is a research paper abstract written by a human.
Your task is to write a NEW abstract with similar meaning but different wording.

Constraints:
- Length: 120–200 words
- Language: academic English
- Do NOT copy sentences directly.
- Keep it plausible as a real scientific abstract.
- Do NOT mention that you are an AI.

Original abstract:
\"\"\"{human_abstract}\"\"\"
"""


def generate_ai_text_gemini(model, human_abstract: str) -> str:
    prompt = build_prompt(human_abstract)
    response = model.generate_content(prompt)
    text = response.text
    return text.strip()


def main():
    model = init_gemini()

    if not HUMAN_CSV.exists():
        raise FileNotFoundError(f"Human CSV yok: {HUMAN_CSV}")

    human_df = pd.read_csv(HUMAN_CSV).copy()

    # Eğer AI CSV varsa oku, yoksa boş df oluştur
    if AI_CSV.exists():
        ai_df = pd.read_csv(AI_CSV)
        print(f"Mevcut AI satır sayısı (OpenAI + Gemini): {len(ai_df)}")
    else:
        ai_df = pd.DataFrame(columns=["id", "text", "label", "source", "paper_id", "title", "license"])

    existing_ids = set(ai_df["id"].tolist())

    needed = TARGET_COUNT - len(ai_df)
    if needed <= 0:
        print("Zaten yeterli sayıda AI kaydı var, yeni üretim gerekmiyor.")
        return

    print(f"Hedef toplam AI kaydı: {TARGET_COUNT}")
    print(f"Şu ana kadar üretilen: {len(ai_df)}")
    print(f"Gemini ile üretilecek ek kayıt sayısı: {needed}")

    rows = []
    count_generated = 0

    for i, row in tqdm(human_df.iterrows(), total=len(human_df)):
        row_id = int(row.get("id", i + 1))

        # Bu id için zaten AI varsa atla
        if row_id in existing_ids:
            continue

        # İhtiyaç kadar ürettiysek bırak
        if count_generated >= needed:
            break

        human_text = str(row["text"])

        # Burada sonsuz döngü YOK: hata olursa bu kaydı atlayıp devam ediyoruz
        try:
            ai_text = generate_ai_text_gemini(model, human_text)
        except Exception as e:
            print(f"\nGemini hata (id={row_id}): {e}")
            print("Bu kaydı atlıyoruz, bir sonrakine geçiyoruz...")
            continue

        rows.append({
            "id": row_id,
            "text": ai_text,
            "label": "ai",
            "source": "gemini-pro",
            "paper_id": row.get("paper_id", ""),
            "title": row.get("title", ""),
            "license": row.get("license", "derived_from_arxiv")
        })

        count_generated += 1

        # Arada diske yazalım, iş yarıda kesilirse kaybolmasın
        temp_df = pd.DataFrame(rows)
        merged = pd.concat([ai_df, temp_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["id"], keep="last")
        merged = merged.sort_values("id")
        merged.to_csv(AI_CSV, index=False, encoding="utf-8")

        print(f"Şu an toplam AI satır sayısı (OpenAI + Gemini): {len(merged)}")

        # Gemini'i çok zorlamayalım
        time.sleep(2)

    print("\nGemini ile üretim tamam.")
    final_df = pd.read_csv(AI_CSV)
    print(f"Nihai AI satır sayısı: {len(final_df)}")


if __name__ == "__main__":
    main()
