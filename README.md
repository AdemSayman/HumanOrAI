Human vs AI Text Detection System

Bu proje, verilen bir metnin insan mı yoksa yapay zeka tarafından mı yazıldığını makine öğrenmesi modelleri kullanarak tahmin eden bir sistemdir.

Proje kapsamında veri toplama, veri temizleme, model eğitimi, API geliştirme ve kullanıcı arayüzü entegre bir şekilde gerçekleştirilmiştir.

Özellikler
6000 adet veri ile eğitim (3000 Human + 3000 AI)
Veri temizleme ve preprocessing pipeline
3 farklı model ile tahmin:
Logistic Regression
SVM (Calibrated)
Multinomial Naive Bayes
Majority Voting ile final karar
FastAPI backend
React (Vite + MUI) frontend
SQLite ile geçmiş kayıt sistemi (History)
Selenium tabanlı testler (White-box test)
Proje Yapısı
HumanOrAI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── history.sqlite3
│
├── ml/
│   ├── src/
│   │   ├── train_models.py
│   │   ├── generate_ai_with_ollama_llama3.py
│   │   └── ...
│   └── models/
│       ├── logreg.joblib
│       ├── svm_calibrated.joblib
│       └── multinomial_nb.joblib
│
├── backend/
│   ├── main.py
│   └── db.py
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Predict.jsx
│   │   │   └── History.jsx
│   │   └── services/
│   │       ├── predictService.js
│   │       └── historyService.js
│
└── tests/
Model Performansları
Model	Accuracy	F1 Score (AI)
Logistic Reg.	91.25%	0.9091
SVM (Calibrated)	93.25%	0.9323
Naive Bayes	69.08%	0.7411

n iyi performans: SVM modeli

Kurulum
1. Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
2. Frontend (React)
cd frontend
npm install
npm run dev
API Endpointleri
Predict
POST /api/predict

Request:

{
  "text": "Your abstract text..."
}

Response:

{
  "text_len": 1150,
  "final": { "label": "human" },
  "predictions": [
    { "model": "logreg", "ai_pct": 24.01 },
    { "model": "svm_calibrated", "ai_pct": 0.69 }
  ]
}
History
GET /api/history
POST /api/history
DELETE /api/history
Testler

Projede en az 3 adet test case oluşturulmuştur:

Predict işlemi testi
API response doğrulama
UI etkileşim testi (Selenium)
Veri Seti
Human veriler: akademik makale özetleri
AI veriler:
LLM (Ollama - LLaMA3) ile üretilmiştir
Ek veri augmentasyonu uygulanmıştır
Kullanılan Teknolojiler
Python (FastAPI, scikit-learn)
React + Vite + MUI
SQLite
Selenium
Pandas, NumPy


Not

Oluşturulan AI verilerinin bir kısmı sentetik olarak üretilmiştir
