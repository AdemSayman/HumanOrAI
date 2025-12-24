from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import joblib
import json

# ✅ SENİN DATASET YOLUN
DATASET_PATH = Path(r"C:\Users\Aduket Sayman\Desktop\HumanOrAI\data\processed\dataset_clean.csv")

# ✅ Model kayıt yeri (senin klasör yapına göre)
PROJECT_ROOT = Path(r"C:\Users\Aduket Sayman\Desktop\HumanOrAI")
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {path}")

    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("dataset_clean.csv içinde 'text' ve 'label' kolonları olmalı.")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # sadece human/ai kalsın
    df = df[df["label"].isin(["human", "ai"])].copy()

    # boş text at
    df = df[df["text"].str.len() > 0].copy()

    return df


def build_pipelines():
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2
    )

    logreg = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            class_weight="balanced"
        ))
    ])

    # LinearSVC probability vermez -> CalibratedClassifierCV ile olasılık yapıyoruz
    svm_base = Pipeline([
        ("tfidf", tfidf),
        ("clf", LinearSVC(class_weight="balanced"))
    ])
    svm_calibrated = CalibratedClassifierCV(
        estimator=svm_base,
        method="sigmoid",
        cv=3
    )

    nb = Pipeline([
        ("tfidf", tfidf),
        ("clf", MultinomialNB(alpha=0.5))
    ])

    return {
        "logreg": logreg,
        "svm_calibrated": svm_calibrated,
        "multinomial_nb": nb
    }


def evaluate_and_save(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label="ai"
    )
    cm = confusion_matrix(y_test, y_pred, labels=["human", "ai"]).tolist()
    report = classification_report(y_test, y_pred, digits=4)

    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "precision_ai": float(pr),
        "recall_ai": float(rc),
        "f1_ai": float(f1),
        "confusion_matrix_labels": ["human", "ai"],
        "confusion_matrix": cm
    }

    # console
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"PrecisionAI: {pr:.4f}")
    print(f"RecallAI   : {rc:.4f}")
    print(f"F1(AI)     : {f1:.4f}")
    print("Confusion Matrix [human, ai]:")
    print(cm)
    print("\nClassification report:")
    print(report)

    # save model
    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    # save metrics
    metrics_path = REPORTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report_path = REPORTS_DIR / f"{model_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return metrics


def main():
    print("Dataset okunuyor:", DATASET_PATH)
    df = load_dataset(DATASET_PATH)

    print("Toplam satır:", len(df))
    print("Label dağılımı:\n", df["label"].value_counts())

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    pipelines = build_pipelines()
    all_metrics = []

    for name, model in pipelines.items():
        print("\n" + "-" * 70)
        print(f"Eğitiliyor: {name}")
        model.fit(X_train, y_train)
        m = evaluate_and_save(name, model, X_test, y_test)
        all_metrics.append(m)

    # genel özet
    summary_path = REPORTS_DIR / "models_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("\nBitti ✅")
    print("Modeller kaydedildi:", MODELS_DIR)
    print("Raporlar kaydedildi :", REPORTS_DIR)


if __name__ == "__main__":
    main()
