# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
from datetime import datetime, timezone

from db import init_db, insert_history, list_history, clear_history

app = FastAPI(title="HumanOrAI API")

# ==== MODEL PATHS ====
BASE_DIR = Path(__file__).resolve().parents[2]  # HumanOrAI/
MODELS_DIR = BASE_DIR / "ml" / "models"

LOGREG_PATH = MODELS_DIR / "logreg.joblib"
SVM_PATH = MODELS_DIR / "svm_calibrated.joblib"
NB_PATH = MODELS_DIR / "multinomial_nb.joblib"

# modeller (pipeline olmalı: vectorizer + model)
logreg = None
svm = None
nb = None

class PredictRequest(BaseModel):
    text: str

def pct(x: float) -> float:
    return round(float(x) * 100.0, 2)

def majority_vote(preds):
    # preds: [{"model": "...", "ai_pct": 12.3, ...}]
    votes_ai = sum(1 for p in preds if p["ai_pct"] >= 50.0)
    return "ai" if votes_ai >= 2 else "human"

@app.on_event("startup")
def startup():
    global logreg, svm, nb
    init_db()
    # modelleri yükle
    logreg = joblib.load(LOGREG_PATH)
    svm = joblib.load(SVM_PATH)
    nb = joblib.load(NB_PATH)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        return {"error": "text empty"}

    # predict_proba -> [human, ai] gibi gelecek varsaydım
    # Eğer sıralama farklıysa düzeltiriz (label encoder vs.)
    preds = []

    for name, model in [("logreg", logreg), ("svm_calibrated", svm), ("multinomial_nb", nb)]:
        proba = model.predict_proba([text])[0]
        # burada class order kontrolü:
        # model.classes_ örn: ['ai','human'] olabilir. ona göre ai index buluyoruz.
        classes = list(model.classes_)
        ai_idx = classes.index("ai") if "ai" in classes else 1
        human_idx = classes.index("human") if "human" in classes else 0

        ai_p = proba[ai_idx]
        human_p = proba[human_idx]

        preds.append({
            "model": name,
            "ai_pct": pct(ai_p),
            "human_pct": pct(human_p)
        })

    final_label = majority_vote(preds)

    # history kaydı
    preview = text.replace("\n", " ").strip()[:220]
    created_at = datetime.now(timezone.utc).isoformat()

    # tek tek model yüzdelerini DB’ye yazmak için ayıkla
    def pick_ai(model_name):
        for p in preds:
            if p["model"] == model_name:
                return p["ai_pct"]
        return None

    insert_history({
        "created_at": created_at,
        "text_preview": preview,
        "text_len": len(text),
        "final_label": final_label,
        "logreg_ai": pick_ai("logreg"),
        "svm_ai": pick_ai("svm_calibrated"),
        "nb_ai": pick_ai("multinomial_nb"),
    })

    return {
        "text_len": len(text),
        "final": {"label": final_label, "rule": "majority_vote"},
        "predictions": preds
    }

@app.get("/history")
def history(limit: int = 50):
    return {"items": list_history(limit=limit)}

@app.delete("/history")
def history_clear():
    clear_history()
    return {"ok": True}
