export async function getHistory(limit = 50) {
  const res = await fetch(`/api/history?limit=${limit}`);
  if (!res.ok) throw new Error("History alınamadı");
  return res.json();
}

export async function clearHistory() {
  const res = await fetch(`/api/history`, { method: "DELETE" });
  if (!res.ok) throw new Error("History temizlenemedi");
  return res.json();
}
