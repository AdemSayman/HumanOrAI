from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import requests
import re
import random

OLLAMA_MODEL = "llama3:latest"
TARGET_COUNT = 3000

BASE_DIR = Path(__file__).resolve().parents[2]
HUMAN_CSV = BASE_DIR / "data" / "raw" / "human_abstracts_raw.csv"
AI_CSV = BASE_DIR / "data" / "raw" / "ai_abstracts_ollama_llama3.csv"

# Basit tekrar kontrolü (çok benzer cümle kalıplarını azaltmak için)
def normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def word_overlap(a: str, b: str) -> float:
    # hızlı kaba benzerlik: kelime kesişimi / küçük set
    wa = set(re.findall(r"[a-z]+", normalize(a)))
    wb = set(re.findall(r"[a-z]+", normalize(b)))
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    return inter / max(1, min(len(wa), len(wb)))

def build_prompt(human_abstract: str) -> str:
    # küçük “varyasyon” ekleyip şablon tekrarını kırıyoruz
    style = random.choice([
        "Write in a concise, formal style.",
        "Write in a slightly more detailed academic style.",
        "Write in an impersonal scientific tone with minimal clichés.",
        "Avoid common phrases like 'This paper presents' or 'In this work, we propose'.",
    ])

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
- Avoid repetitive template phrases.

Additional style note:
- {style}

Original abstract:
\"\"\"{human_abstract}\"\"\"
"""

def ollama_generate(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.9,   # çeşitlilik
            "top_p": 0.9,
            "repeat_penalty": 1.15
        }
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

def main():
    if not HUMAN_CSV.exists():
        raise FileNotFoundError(f"Human CSV yok: {HUMAN_CSV}")

    human_df = pd.read_csv(HUMAN_CSV).copy()
    if "id" not in human_df.columns:
        human_df["id"] = range(1, len(human_df) + 1)

    # Daha önce üretilmiş AI varsa devam et
    if AI_CSV.exists() and AI_CSV.stat().st_size > 0:
        try:
            ai_df = pd.read_csv(AI_CSV)
            print(f"Mevcut Llama3 AI satır sayısı: {len(ai_df)}")
        except pd.errors.EmptyDataError:
            print("AI CSV var ama boşmuş. Sıfırdan başlıyoruz.")
            ai_df = pd.DataFrame(columns=["id","text","label","source","paper_id","title","license"])
    else:
        ai_df = pd.DataFrame(columns=["id","text","label","source","paper_id","title","license"])


    existing_ids = set(ai_df["id"].tolist())
    needed = TARGET_COUNT - len(ai_df)
    if needed <= 0:
        print("Zaten 3000 tamam.")
        return

    print(f"Kalan üretilecek kayıt: {needed}")

    # humanları karıştır, boş/çok kısa olanları at
    human_df["text"] = human_df["text"].astype(str)
    human_df = human_df[human_df["text"].str.len() > 200]
    human_df = human_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    new_rows = []
    recent_texts = [str(x) for x in ai_df["text"].tail(200).tolist()]  # son 200’e göre kaba tekrar kontrolü

    for _, row in tqdm(human_df.iterrows(), total=len(human_df)):
        if len(ai_df) + len(new_rows) >= TARGET_COUNT:
            break

        row_id = int(row["id"])
        if row_id in existing_ids:
            continue

        human_text = str(row["text"])
        prompt = build_prompt(human_text)

        try:
            out = ollama_generate(prompt)
        except Exception as e:
            print(f"\nOllama hata (id={row_id}): {e}")
            continue

        # Çok benzer çıktı geldiyse (kaba kontrol) yeniden dene
        ok = True
        for prev in recent_texts[-50:]:
            if word_overlap(out, prev) > 0.75:  # agresif tekrar eşiği
                ok = False
                break

        if not ok:
            # 1 kez daha farklı temp ile dene
            try:
                time.sleep(0.5)
                out2 = ollama_generate(prompt + "\nMake the wording substantially different from typical template abstracts.")
                out = out2
            except:
                continue

        new_rows.append({
            "id": row_id,
            "text": out,
            "label": "ai",
            "source": f"ollama_{OLLAMA_MODEL}",
            "paper_id": row.get("paper_id",""),
            "title": row.get("title",""),
            "license": row.get("license","derived_from_arxiv"),
        })
        recent_texts.append(out)

        # her 10 kayıtta bir diske bas
        if len(new_rows) % 10 == 0:
            temp_df = pd.DataFrame(new_rows)
            merged = pd.concat([ai_df, temp_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["id"], keep="last")
            merged.to_csv(AI_CSV, index=False, encoding="utf-8")

        time.sleep(0.2)  # makineyi boğma

    # final yaz
    temp_df = pd.DataFrame(new_rows)
    merged = pd.concat([ai_df, temp_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["id"], keep="last")
    merged = merged.sort_values("id")
    merged.to_csv(AI_CSV, index=False, encoding="utf-8")
    print(f"\nBitti. Toplam Llama3 AI satır sayısı: {len(merged)}")
    print(f"Kaydedildi: {AI_CSV}")

if __name__ == "__main__":
    main()
