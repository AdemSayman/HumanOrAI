export default function ResultCard({ result }) {
  return (
    <div style={{ marginTop: 16, padding: 12, border: "1px solid #333", borderRadius: 8 }}>
      <div>Text length: {result.text_len}</div>

      <h3>Model Sonuçları</h3>
      <ul>
        {result.predictions?.map((p) => (
          <li key={p.model}>
            <b>{p.model}</b> — AI: %{p.ai_pct} | Human: %{p.human_pct}
          </li>
        ))}
      </ul>
    </div>
  );
}
