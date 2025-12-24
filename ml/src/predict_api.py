from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

PROJECT_ROOT = Path(r"C:\Users\Aduket Sayman\Desktop\HumanOrAI")
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

# Modelleri yükle
logreg = joblib.load(MODELS_DIR / "logreg.joblib")
svm = joblib.load(MODELS_DIR / "svm_calibrated.joblib")
nb = joblib.load(MODELS_DIR / "multinomial_nb.joblib")

app = FastAPI(title="HumanOrAI Predictor")

class PredictRequest(BaseModel):
    text: str

def probs_from_model(model, text: str):
    """
    Çıktı: ai_pct, human_pct
    Not: 3 model de predict_proba destekliyor (svm_calibrated zaten destekliyor)
    """
    proba = model.predict_proba([text])[0]  # [p(human), p(ai)] veya sınıf sırasına göre değişebilir
    # sınıf sırasını güvenli al
    classes = list(model.classes_)  # örn: ['ai','human'] veya ['human','ai']
    p_ai = float(proba[classes.index("ai")])
    p_human = float(proba[classes.index("human")])

    return round(p_ai * 100, 2), round(p_human * 100, 2)

@app.post("/predict")
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        return {"error": "text boş olamaz"}

    ai1, h1 = probs_from_model(logreg, text)
    ai2, h2 = probs_from_model(svm, text)
    ai3, h3 = probs_from_model(nb, text)

    votes_ai = sum([ai1 >= 50, ai2 >= 50, ai3 >= 50])
    final_label = "ai" if votes_ai >= 2 else "human"
    final_ai_pct = round((ai1 + ai2 + ai3) / 3, 2)
    final_human_pct = round(100 - final_ai_pct, 2)

    return {
        "text_len": len(text),
        "final": {
            "label": final_label,
            "ai_pct_avg": final_ai_pct,
            "human_pct_avg": final_human_pct,
            "votes_ai": votes_ai
        },
        "predictions": [
            {"model": "logreg", "ai_pct": ai1, "human_pct": h1},
            {"model": "svm_calibrated", "ai_pct": ai2, "human_pct": h2},
            {"model": "multinomial_nb", "ai_pct": ai3, "human_pct": h3},
        ]
    }
