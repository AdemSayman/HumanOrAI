from pathlib import Path
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_AI = OUT_DIR / "ai_abstracts_ollama_llama3_sequential.csv"   # az önce ürettiğin
RAW_AI    = RAW_DIR / "ai_abstracts_clean_strict.csv"                      # diğer AI havuzun

OUT_AI_FINAL = OUT_DIR / "ai_abstracts_final_3000.csv"

TARGET_TOTAL_AI = 3000
MIN_LEN = 200

def clean_text(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip()
    t = re.sub(r"^Here is a rewritten abstract:\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^Here's a rewritten abstract:\s*", "", t, flags=re.IGNORECASE).strip()
    t = t.strip().strip('"').replace('""', '"').strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    oll = pd.read_csv(OLLAMA_AI)
    raw = pd.read_csv(RAW_AI)

    # text kolon kontrol
    if "text" not in oll.columns or "text" not in raw.columns:
        raise ValueError("CSV içinde 'text' kolonu yok.")

    # temizle
    oll["text"] = oll["text"].astype(str).apply(clean_text)
    raw["text"] = raw["text"].astype(str).apply(clean_text)

    # kısa/boş ele
    oll = oll[oll["text"].str.len() >= MIN_LEN].copy()
    raw = raw[raw["text"].str.len() >= MIN_LEN].copy()

    # Ollama zaten kaç?
    base_n = len(oll)
    need = TARGET_TOTAL_AI - base_n
    if need <= 0:
        print(f"Ollama zaten {base_n}, hedef {TARGET_TOTAL_AI}. Ek gerek yok.")
        oll.head(TARGET_TOTAL_AI).to_csv(OUT_AI_FINAL, index=False, encoding="utf-8")
        return

    print(f"Ollama AI: {base_n}")
    print(f"Tamamlanacak ek AI: {need}")
    print(f"Raw AI havuzu: {len(raw)}")

    # Raw içinde birebir aynı metinleri at
    raw = raw.drop_duplicates(subset=["text"], keep="first").copy()

    # --- Benzerlik filtresi (TF-IDF) ---
    # Büyük veri olursa hafifletmek için raw'dan rastgele alt örneklem alabiliriz.
    # Şimdilik tümünü kullanıyoruz.
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1,2),
        stop_words="english"
    )

    # Fit'i birleşik üzerinde yap
    all_texts = pd.concat([oll["text"], raw["text"]], ignore_index=True)
    X = vectorizer.fit_transform(all_texts)

    X_oll = X[:len(oll)]
    X_raw = X[len(oll):]

    # raw her satırın ollama setine max benzerliğini bul
    # cosine_similarity(X_raw, X_oll) -> (raw x oll) çok büyük olabilir
    # chunk ile hesaplayalım
    chunk = 500
    max_sims = []
    for start in range(0, X_raw.shape[0], chunk):
        end = min(start + chunk, X_raw.shape[0])
        sims = cosine_similarity(X_raw[start:end], X_oll)
        max_sims.extend(sims.max(axis=1).tolist())

    raw = raw.reset_index(drop=True)
    raw["max_sim_to_ollama"] = max_sims

    # En az benzeyenleri seç (çeşitlilik)
    raw_sorted = raw.sort_values("max_sim_to_ollama", ascending=True)

    # need kadar al
    picked = raw_sorted.head(need).copy()

    # birleştir
    final_ai = pd.concat([oll, picked], ignore_index=True)

    # label/source standardize (yoksa ekle)
    if "label" not in final_ai.columns:
        final_ai["label"] = "ai"
    else:
        final_ai["label"] = "ai"

    # id yeniden 1..3000 yap
    final_ai = final_ai.reset_index(drop=True)
    final_ai["id"] = range(1, len(final_ai) + 1)

    # sadece gerekli kolonlar
    cols = ["id", "text", "label"]
    if "source" in final_ai.columns: cols.append("source")
    if "paper_id" in final_ai.columns: cols.append("paper_id")
    if "title" in final_ai.columns: cols.append("title")
    if "license" in final_ai.columns: cols.append("license")

    final_ai = final_ai[cols]

    final_ai.to_csv(OUT_AI_FINAL, index=False, encoding="utf-8")
    print("Bitti ✅")
    print("Final AI satır:", len(final_ai))
    print("Kaydedildi:", OUT_AI_FINAL)
    print("Seçilen raw AI ortalama max_sim:", picked["max_sim_to_ollama"].mean())

if __name__ == "__main__":
    main()
