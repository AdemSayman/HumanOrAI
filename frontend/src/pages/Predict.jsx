import React, { useState } from "react";
import {
  TextField,
  Button,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Stack,
  Alert,
} from "@mui/material";
import { predict } from "../services/predictService";

export default function Predict() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  const handlePredict = async () => {
    if (!text.trim()) return alert("Metin boş olamaz");
    setErr("");
    setLoading(true);

    try {
      const res = await predict(text);
      setResult(res);
    } catch (e) {
      setErr(e.message || "Bir hata oluştu");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Stack spacing={3}>
      <Typography variant="h5" data-testid="predict-title">
        Metni Analiz Et
      </Typography>

      <TextField
        label="Abstract / Metin"
        multiline
        minRows={6}
        value={text}
        onChange={(e) => setText(e.target.value)}
        inputProps={{ "data-testid": "predict-input" }}
      />

      <Button
        variant="contained"
        onClick={handlePredict}
        disabled={loading}
        data-testid="predict-button"
      >
        {loading ? "Analiz Ediliyor..." : "Analiz Et"}
      </Button>

      {loading && <LinearProgress data-testid="predict-loading" />}

      {err && (
        <Alert severity="error" data-testid="predict-error">
          {err}
        </Alert>
      )}

      {result && (
        <Card data-testid="predict-result">
          <CardContent>
            <Typography variant="h6" gutterBottom data-testid="predict-final">
              Sonuç: {String(result.final).toUpperCase()}
            </Typography>

            <Typography
              variant="body2"
              sx={{ mb: 2, opacity: 0.8 }}
              data-testid="predict-length"
            >
              Metin uzunluğu: {result.text_len}
            </Typography>

            <div data-testid="predict-models">
              {result.models.map((m) => (
                <div
                  key={m.name}
                  style={{ marginBottom: 16 }}
                  data-testid={`predict-model-${m.name}`}
                >
                  <Typography>
                    {m.name} — AI %{m.ai} / Human %{m.human}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={m.ai}
                    data-testid={`predict-bar-${m.name}`}
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </Stack>
  );
}
