"""Aplikasi web SmartGuard JKN: antrean prioritas dan simulator klaim.

Jalankan: uvicorn app:app
Butuh artifacts/nadi.pt (train.py) dan artifacts/scored.parquet (evaluate.py).
"""
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

import nadi

LEVELS = ["I", "II", "III"]
SEC = ["sec1", "sec2", "sec3"]
ODD_BELOW = 0.05  # peluang di bawah ini ditandai janggal
# hanya field dengan sedikit pilihan nilai; di dx/CBG peluang kecil itu wajar
FLAGGABLE = {"gap", "fac", "los", "sev"}
LABELS = {
    "demo": "Jenis kelamin & usia", "gap": "Jarak dari klaim sebelumnya (hari)",
    "fac": "Faskes dibanding klaim sebelumnya", "svc": "Jenis layanan", "cls": "Kelas RS",
    "dx": "Diagnosis utama", "cbg": "Kelompok INA-CBG", "los": "Lama rawat (hari)",
    "sec1": "Diagnosis sekunder 1", "sec2": "Diagnosis sekunder 2", "sec3": "Diagnosis sekunder 3",
    "sev": "Tingkat keparahan",
}

model, vocab = nadi.load("artifacts/nadi.pt", device="cpu")
claims = pd.read_parquet("artifacts/scored.parquet")
claims["priority"] = claims.potential * (1 - claims.p_claimed)
SEV_IDS = [nadi.field_ids(vocab, "sev")[level] for level in LEVELS]
OPTIONS = {
    "sev": LEVELS,
    "los": ["0", "1", "2", "3", "4-5", "6-7", "8-14", "15+"],
    "sec": ["-", *sorted(v for v in nadi.field_ids(vocab, "sec1") if v != "-")],
}

app = FastAPI(title="SmartGuard JKN")


class Changes(BaseModel):
    """Field klaim yang boleh diubah di simulator."""
    model_config = ConfigDict(extra="forbid")
    sev: Optional[str] = Field(None, max_length=8)
    los: Optional[str] = Field(None, max_length=8)
    sec1: Optional[str] = Field(None, max_length=8)
    sec2: Optional[str] = Field(None, max_length=8)
    sec3: Optional[str] = Field(None, max_length=8)


def history(row: int) -> pd.DataFrame:
    """Klaim peserta yang berada di potongan urutan yang sama dengan klaim `row`."""
    if not 0 <= row < len(claims):
        raise HTTPException(404, "klaim tidak ditemukan")
    rows = np.flatnonzero(claims.patient.to_numpy() == claims.patient[row])
    chunk = (rows - rows[0]) // nadi.MAX_VISITS
    return claims.iloc[rows[chunk == chunk[rows == row][0]]]


def assess(row: int, changes: dict) -> dict:
    """Peluang model untuk tiap field klaim `row`, setelah `changes` diterapkan."""
    visits = history(row)
    patient = claims[claims.patient == claims.patient[row]]
    feats = nadi.features(patient).loc[visits.index]
    feats.loc[row, list(changes)] = list(changes.values())
    codes = sorted(c for c in feats.loc[row, SEC] if c != "-")  # urutan dx sekunder harus tetap baku
    feats.loc[row, SEC] = codes + ["-"] * (len(SEC) - len(codes))

    X = nadi.encode(feats, vocab)[None]
    at = visits.index.get_loc(row)
    p_field = np.exp(-nadi.surprise(model, X)[0, at])
    p_sev = nadi.field_probs(model, X, "sev", SEV_IDS)[0, at]
    return {
        "fields": [
            {"field": f, "label": LABELS[f], "value": feats.at[row, f], "p": float(p),
             "odd": f in FLAGGABLE and bool(p < ODD_BELOW)}
            for f, p in zip(nadi.FIELDS, p_field)
        ],
        "severity": [
            {"level": level, "p": float(p), "claimed": level == feats.at[row, "sev"]}
            for level, p in zip(LEVELS, p_sev)
        ],
    }


def as_records(df: pd.DataFrame) -> list:
    cols = ["patient", "hospital", "hosp_class", "admit", "service", "dx", "cbg", "severity", "expected",
            "los", "cost", "p_claimed", "potential", "split_episode", "duplicate"]
    out = df[cols].assign(admit=df.admit.dt.strftime("%Y-%m-%d"), row=df.index)
    return out.to_dict("records")


@app.get("/")
def index():
    return FileResponse("web/index.html")


@app.get("/api/summary")
def summary():
    above = claims[claims.potential > 0]
    return {
        "claims": len(claims),
        "inpatient": int((claims.service == "RI").sum()),
        "above": len(above),
        "potential": int(above.potential.sum()),
        "rules": int((claims.split_episode | claims.duplicate).sum()),
    }


@app.get("/api/queue")
def queue(kind: str = "model", limit: int = 50):
    limit = min(max(limit, 1), 200)
    if kind == "rules":
        picked = claims[claims.split_episode | claims.duplicate].nlargest(limit, "cost")
    else:
        picked = claims[claims.potential > 0].nlargest(limit, "priority")
    return as_records(picked)


@app.get("/api/claim/{row}")
def claim(row: int):
    return {"visits": as_records(history(row)), "options": OPTIONS, **assess(row, {})}


@app.post("/api/claim/{row}/simulate")
def simulate(row: int, changes: Changes):
    return assess(row, changes.model_dump(exclude_none=True))
