import React from "react";
import { Typography } from "@mui/material";

export default function About() {
  return (
    <>
      <Typography variant="h5">Proje Hakkında</Typography>
      <Typography sx={{ mt: 2 }}>
        Bu proje insan tarafından yazılmış ve AI tarafından üretilmiş
        metinleri ayırt etmek için geliştirilmiştir.
      </Typography>
    </>
  );
}
