import React, { useEffect, useState, useCallback } from "react";
import {
  Typography,
  Card,
  CardContent,
  Stack,
  Button,
  Divider,
  CircularProgress,
} from "@mui/material";
import { getHistory, clearHistory } from "../services/historyService";

export default function History() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    setErr("");
    setLoading(true);
    try {
      const data = await getHistory(50);
      setItems(data.items || []);
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      setErr("");
      setLoading(true);
      try {
        const data = await getHistory(50);
        if (!cancelled) setItems(data.items || []);
      } catch (e) {
        if (!cancelled) setErr(String(e?.message || e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleClear = async () => {
    await clearHistory();
    await load();
  };

  return (
    <Stack spacing={2}>
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Typography variant="h5">Geçmiş Tahminler</Typography>
        <Button variant="outlined" onClick={handleClear} disabled={loading}>
          Temizle
        </Button>
      </Stack>

      {loading && (
        <Stack direction="row" alignItems="center" spacing={1}>
          <CircularProgress size={18} />
          <Typography>Yükleniyor...</Typography>
        </Stack>
      )}

      {err && <Typography color="error">{err}</Typography>}

      {!loading && items.length === 0 && <Typography>Henüz kayıt yok.</Typography>}

      {items.map((x) => (
        <Card key={x.id}>
          <CardContent>
            <Typography sx={{ opacity: 0.7, fontSize: 13 }}>
              {x.created_at} — len: {x.text_len}
            </Typography>

            <Typography variant="h6" sx={{ mt: 1 }}>
              Sonuç: {String(x.final_label).toUpperCase()}
            </Typography>

            <Divider sx={{ my: 1.5 }} />

            <Typography sx={{ mb: 1 }}>{x.text_preview}...</Typography>

            <Typography sx={{ opacity: 0.8 }}>
              logreg AI%: {x.logreg_ai} | svm AI%: {x.svm_ai} | nb AI%: {x.nb_ai}
            </Typography>
          </CardContent>
        </Card>
      ))}
    </Stack>
  );
}
