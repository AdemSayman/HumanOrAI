from pathlib import Path
import pandas as pd
import re
import random

# Kaç tane AI kaydı olsun toplam (OpenAI + sentetik)
TARGET_AI_COUNT = 3000

# Proje kök klasörü
BASE_DIR = Path(__file__).resolve().parents[2]

HUMAN_CSV = BASE_DIR / "data" / "raw" / "human_abstracts_raw.csv"
AI_CSV = BASE_DIR / "data" / "raw" / "ai_abstracts_raw.csv"


def load_data():
    if not HUMAN_CSV.exists():
        raise FileNotFoundError(f"Human CSV yok: {HUMAN_CSV}")

    human_df = pd.read_csv(HUMAN_CSV)

    if AI_CSV.exists():
        ai_df = pd.read_csv(AI_CSV)
    else:
        ai_df = pd.DataFrame(columns=["id", "text", "label", "source", "paper_id", "title", "license"])

    return human_df, ai_df


# Basit akademik sinonim sözlüğü
SYNONYMS = {
    r"\buse\b": "utilize",
    r"\buses\b": "utilizes",
    r"\busing\b": "utilizing",
    r"\bshow\b": "demonstrate",
    r"\bshows\b": "demonstrates",
    r"\bshowed\b": "demonstrated",
    r"\btry\b": "attempt",
    r"\btries\b": "attempts",
    r"\btried\b": "attempted",
    r"\bbig\b": "significant",
    r"\bsmall\b": "minor",
    r"\bgood\b": "promising",
    r"\bbad\b": "suboptimal",
    r"\bimportant\b": "crucial",
    r"\bmaybe\b": "potentially",
    r"\bhelp\b": "assist",
    r"\bfind\b": "identify",
    r"\bfound\b": "identified",
    r"\bget\b": "obtain",
    r"\bmake\b": "construct",
    r"\bneed\b": "require",
    r"\bfast\b": "efficient",
    r"\bslow\b": "inefficient",
    r"\bnew\b": "novel",
    r"\bmany\b": "numerous",
    r"\balot\b": "a considerable amount",
}


def apply_synonyms(text: str) -> str:
    new_text = text
    for pattern, repl in SYNONYMS.items():
        # case-insensitive, word-boundary replace
        new_text = re.sub(pattern, repl, new_text, flags=re.IGNORECASE)
    return new_text


def shuffle_sentences(text: str) -> str:
    # Noktalama bazlı cümlelere bölelim
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 2:
        return text.strip()

    # İlk ve son cümleyi sabit bırakıp ortadakileri karıştıralım
    first = sentences[0]
    last = sentences[-1]
    middle = sentences[1:-1]

    random.shuffle(middle)

    new_sentences = [first] + middle + [last]
    return " ".join(new_sentences)


AI_INTRO_TEMPLATES = [
    "In this work, we present a comprehensive investigation of the problem.",
    "This study provides a systematic analysis of the proposed approach.",
    "In this paper, we introduce a method designed to address this challenge.",
]

AI_OUTRO_TEMPLATES = [
    "The results demonstrate the effectiveness of the proposed methodology.",
    "Our findings highlight both the strengths and limitations of the approach.",
    "Overall, the study offers promising directions for future research.",
]


def add_ai_flavor(text: str) -> str:
    """Metne AI benzeri giriş/çıkış cümleleri ekle."""
    intro = random.choice(AI_INTRO_TEMPLATES)
    outro = random.choice(AI_OUTRO_TEMPLATES)

    core = text.strip()

    # Eğer metin zaten çok uzun değilse intro/outro ekleyelim
    return f"{intro} {core} {outro}"


def transform_to_ai_style(text: str) -> str:
    """
    Human abstract'i daha 'AI-vari' hale getiren kural tabanlı dönüşüm.
    """
    # 1) Bazı kelimeleri akademik sinonimlerle değiştir
    t = apply_synonyms(text)

    # 2) Cümle sırasını biraz karıştır
    t = shuffle_sentences(t)

    # 3) Intro/outro kalıp cümleler ekle
    t = add_ai_flavor(t)

    return t.strip()


def main():
    random.seed(42)

    human_df, ai_df = load_data()

    print(f"Toplam human satır sayısı: {len(human_df)}")
    print(f"Mevcut AI satır sayısı (gerçek + önceki): {len(ai_df)}")

    existing_ai_ids = set(ai_df["id"].tolist())

    needed = TARGET_AI_COUNT - len(ai_df)
    if needed <= 0:
        print("Zaten yeterli sayıda AI kaydı var, yeni üretim gerekmiyor.")
        return

    print(f"Hedef AI sayısı: {TARGET_AI_COUNT}")
    print(f"Ek üretilecek sentetik AI sayısı: {needed}")

    # AI olmayan human kayıtları arasından seçelim
    candidate_df = human_df[~human_df["id"].isin(existing_ai_ids)].copy()

    if len(candidate_df) < needed:
        print(f"Uyarı: Sentetik AI üretmek için yeterli benzersiz human yok. "
              f"Gereken: {needed}, mevcut: {len(candidate_df)}. Mevcut kadar üreteceğiz.")
        needed = len(candidate_df)

    # Rastgele karıştıralım ki human id'leri karışık olsun
    candidate_df = candidate_df.sample(frac=1.0, random_state=42).head(needed)

    synthetic_rows = []

    for _, row in candidate_df.iterrows():
        row_id = int(row["id"])
        human_text = str(row["text"])

        ai_like_text = transform_to_ai_style(human_text)

        synthetic_rows.append({
            "id": row_id,
            "text": ai_like_text,
            "label": "ai",
            "source": "rule_based_synthetic",
            "paper_id": row.get("paper_id", ""),
            "title": row.get("title", ""),
            "license": row.get("license", "derived_from_arxiv"),
        })

    syn_df = pd.DataFrame(synthetic_rows)

    merged = pd.concat([ai_df, syn_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["id"], keep="last")
    merged = merged.sort_values("id")

    merged.to_csv(AI_CSV, index=False, encoding="utf-8")

    print(f"\nSentetik üretim tamam.")
    print(f"Nihai AI satır sayısı: {len(merged)}")
    print(f"Kaydedilen dosya: {AI_CSV}")


if __name__ == "__main__":
    main()
