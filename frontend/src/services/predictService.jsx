export async function predict(text) {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`API ${res.status}: ${errText}`);
  }

  const data = await res.json();

  // FastAPI -> UI formatına çevir
  const models = (data.predictions || []).map((p) => ({
    name: p.model,
    ai: p.ai_pct,
    human: p.human_pct,
  }));

  // final yoksa majority vote ile üret (API'ye final eklemediysen)
  let finalLabel = data.final?.label;
  if (!finalLabel) {
    const votesAI = models.filter((m) => m.ai >= 50).length;
    finalLabel = votesAI >= 2 ? "ai" : "human";
  }

  return {
    final: finalLabel,
    models,
    text_len: data.text_len,
  };
}
