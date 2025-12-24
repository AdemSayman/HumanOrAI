from pathlib import Path
import pandas as pd
import re
import random

BASE_DIR = Path(__file__).resolve().parents[2]
AI_CSV = BASE_DIR / "data" / "raw" / "ai_abstracts_raw.csv"

# Daha geniş bir sinonim sözlüğü
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
    r"\bcould\b": "might",
    r"\bshows that\b": "indicates that",
    r"\bwe\b": "we",
}

AI_INTRO_TEMPLATES = [
    "In this work, we present a comprehensive investigation of the problem.",
    "This study provides a systematic analysis of the proposed approach.",
    "In this paper, we introduce a method designed to address this challenge.",
    "Motivated by recent advances, this research revisits the problem from a broader perspective.",
    "The present study explores the underlying factors that influence the behavior of the system.",
]

AI_OUTRO_TEMPLATES = [
    "The results demonstrate the effectiveness of the proposed methodology.",
    "Our findings highlight both the strengths and limitations of the approach.",
    "Overall, the study offers promising directions for future research.",
    "These observations suggest practical implications for real-world applications.",
    "Taken together, the outcomes provide a solid foundation for further investigation.",
]

AI_CONNECTORS = [
    "Furthermore,",
    "Moreover,",
    "In addition,",
    "Consequently,",
    "As a result,",
]


def apply_synonyms_random(text: str, prob: float = 0.5) -> str:
    """Her pattern için belli bir olasılıkla synonym uygula."""
    new_text = text
    for pattern, repl in SYNONYMS.items():
        if random.random() < prob:
            new_text = re.sub(pattern, repl, new_text, flags=re.IGNORECASE)
    return new_text


def shuffle_sentences_maybe(text: str, prob: float = 0.6) -> str:
    """Belirli bir olasılıkla cümleleri karıştır."""
    if random.random() > prob:
        return text.strip()

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 2:
        return text.strip()

    first = sentences[0]
    last = sentences[-1]
    middle = sentences[1:-1]

    random.shuffle(middle)

    new_sentences = [first] + middle + [last]
    return " ".join(new_sentences)


def maybe_add_connectors(text: str) -> str:
    """Bazı cümlelerin başına bağlaç ekle (AI vari his)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 3:
        return text.strip()

    # Rastgele 1-2 cümle seçip başına connector ekleyelim
    idxs = list(range(1, len(sentences)-1))
    random.shuffle(idxs)
    for idx in idxs[:2]:
        if random.random() < 0.5:
            connector = random.choice(AI_CONNECTORS)
            sentences[idx] = f"{connector} {sentences[idx][0].lower() + sentences[idx][1:] if len(sentences[idx])>1 else sentences[idx]}"

    return " ".join(sentences)


def add_ai_flavor_varied(text: str) -> str:
    """Intro/outro kullanımını çeşitlendir."""
    core = text.strip()
    intro = random.choice(AI_INTRO_TEMPLATES)
    outro = random.choice(AI_OUTRO_TEMPLATES)

    mode = random.choice(["intro_core", "core_outro", "intro_core_outro", "core_only"])

    if mode == "intro_core":
        return f"{intro} {core}"
    elif mode == "core_outro":
        return f"{core} {outro}"
    elif mode == "intro_core_outro":
        return f"{intro} {core} {outro}"
    else:  # core_only
        return core


def transform_to_ai_style_v2(text: str) -> str:
    """Daha çeşitli AI stili dönüşüm."""
    t = text

    # 1) Synonymleri rastgele uygula
    t = apply_synonyms_random(t, prob=0.6)

    # 2) Cümleleri bazen karıştır
    t = shuffle_sentences_maybe(t, prob=0.5)

    # 3) Cümlelerin başına bazen bağlaç ekle
    t = maybe_add_connectors(t)

    # 4) Intro/outro kombinasyonunu çeşitlendir
    t = add_ai_flavor_varied(t)

    return t.strip()


def main():
    random.seed(123)

    if not AI_CSV.exists():
        raise FileNotFoundError(f"AI CSV yok: {AI_CSV}")

    df = pd.read_csv(AI_CSV)
    print(f"AI CSV toplam satır: {len(df)}")

    if "source" not in df.columns:
        raise ValueError("'source' kolonu yok, rule_based_synthetic satırları ayıramıyorum.")

    mask_syn = df["source"] == "rule_based_synthetic"
    count_syn = mask_syn.sum()
    print(f"rule_based_synthetic satır sayısı: {count_syn}")

    if count_syn == 0:
        print("Sentetik satır yok, değiştirilecek bir şey yok.")
        return

    # Sadece sentetik olanların text'ini yeniden üret
    updated_texts = []
    for idx, row in df[mask_syn].iterrows():
        old_text = str(row["text"])
        new_text = transform_to_ai_style_v2(old_text)
        updated_texts.append((idx, new_text))

    for idx, new_text in updated_texts:
        df.at[idx, "text"] = new_text

    df.to_csv(AI_CSV, index=False, encoding="utf-8")
    print("Sentetik AI satırlarının text'leri daha çeşitli olacak şekilde güncellendi.")
    print(f"Güncellenen dosya: {AI_CSV}")


if __name__ == "__main__":
    main()
