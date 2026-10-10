"""SmartGuard JKN web app: priority queue, facility profile, claim simulator, reviewer decisions and FHIR.

Run: uvicorn app:app
Needs artifacts/nadi.pt (train.py) and artifacts/scored.parquet (evaluate.py).
"""
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

import bridge
import nadi

LEVELS = ["I", "II", "III"]
SEC = ["sec1", "sec2", "sec3"]
ANOMALY_BELOW = 0.05  # probability under this is flagged
# only fields with few possible values; a low probability on dx/CBG is normal
FLAGGABLE = {"gap", "fac", "los", "sev"}
LABELS = {  # shown in the UI
    "demo": "Jenis kelamin & usia", "gap": "Gap dari klaim sebelumnya (hari)",
    "fac": "Faskes vs klaim sebelumnya", "svc": "Jenis layanan", "cls": "Kelas RS",
    "dx": "Diagnosis primer", "cbg": "Kode INA-CBG", "los": "LOS (hari)",
    "sec1": "Diagnosis sekunder 1", "sec2": "Diagnosis sekunder 2", "sec3": "Diagnosis sekunder 3",
    "sev": "Severity level",
}

model, vocab = nadi.load("artifacts/nadi.pt", device="cpu")
claims = pd.read_parquet("artifacts/scored.parquet")
claims["priority"] = claims.potential * (1 - claims.p_claimed)
to_review = (claims.potential > 0) | claims.split_episode | claims.duplicate
inpatient = claims[claims.service == "RI"]
inpatient = inpatient.assign(peer=nadi.peer_expected(inpatient))
facilities = nadi.facility_index(inpatient, inpatient.peer).join(pd.DataFrame({
    "hosp_class": claims.groupby("hospital").hosp_class.last(),
    "review": to_review.groupby(claims.hospital).sum(),
}))
NAMES = json.loads(Path("data/names.json").read_text())
SEV_IDS = [nadi.field_ids(vocab, "sev")[level] for level in LEVELS]
OPTIONS = {
    "sev": LEVELS,
    "los": ["0", "1", "2", "3", "4-5", "6-7", "8-14", "15+"],
    "sec": [[code, NAMES["dx"].get(code, "")] for code in sorted(nadi.field_ids(vocab, "sec1")) if code != "-"],
}

REVIEWS = Path("artifacts/reviews.jsonl")  # append-only; every line carries the hash of the line before it
write_lock = threading.Lock()
MODEL_VERSION = hashlib.sha256(Path("artifacts/nadi.pt").read_bytes()).hexdigest()[:12]
MAX_FHIR_BODY = 1_000_000  # bytes accepted by /api/fhir/assess

app = FastAPI(title="SmartGuard JKN")


class Review(BaseModel):
    """A reviewer's decision on one claim."""
    model_config = ConfigDict(extra="forbid")
    decision: Literal["layak", "pending", "tidak_layak"]
    reviewer: str = Field(min_length=1, max_length=60)
    note: str = Field("", max_length=500)


class Changes(BaseModel):
    """Claim fields the simulator is allowed to change."""
    model_config = ConfigDict(extra="forbid")
    sev: Optional[str] = Field(None, max_length=8)
    los: Optional[str] = Field(None, max_length=8)
    sec1: Optional[str] = Field(None, max_length=8)
    sec2: Optional[str] = Field(None, max_length=8)
    sec3: Optional[str] = Field(None, max_length=8)


def name_of(field: str, value: str) -> str:
    """Display name of a dx or CBG code; empty for other fields."""
    kind = "cbg" if field == "cbg" else "dx" if field in ("dx", *SEC) else None
    return NAMES[kind].get(value, "") if kind else ""


def read_log():
    """The review log: (entries, hash of the last line, whether the hash chain is intact)."""
    # simplification: re-read on every request, and the chain shows edits but not a cut-off tail;
    # move to a database and anchor the last hash elsewhere once this is more than a demo
    entries, last, intact = [], "", True
    for line in REVIEWS.read_text().splitlines() if REVIEWS.exists() else []:
        entries.append(json.loads(line))
        intact &= entries[-1]["prev"] == last
        last = hashlib.sha256(line.encode()).hexdigest()
    return entries, last, intact


def claim_at(row: int) -> pd.Series:
    if not 0 <= row < len(claims):
        raise HTTPException(404, "claim not found")
    return claims.iloc[row]


def history(row: int) -> pd.DataFrame:
    """The patient's claims that share a sequence chunk with claim `row`."""
    rows = np.flatnonzero(claims.patient.to_numpy() == claim_at(row).patient)
    chunk = (rows - rows[0]) // nadi.MAX_VISITS
    return claims.iloc[rows[chunk == chunk[rows == row][0]]]


def score(feats: pd.DataFrame, at: int) -> dict:
    """Model probability of every field of the claim at position `at` in one patient's `feats`."""
    X = nadi.encode(feats, vocab)[None]
    p_field = np.exp(-nadi.surprise(model, X)[0, at])
    p_sev = nadi.field_probs(model, X, "sev", SEV_IDS)[0, at]
    claim = feats.iloc[at]
    return {
        "fields": [
            {"field": f, "label": LABELS[f], "value": claim[f], "name": name_of(f, claim[f]),
             "p": float(p), "anomaly": f in FLAGGABLE and bool(p < ANOMALY_BELOW)}
            for f, p in zip(nadi.FIELDS, p_field)
        ],
        "severity": [
            {"level": level, "p": float(p), "claimed": level == claim["sev"]}
            for level, p in zip(LEVELS, p_sev)
        ],
    }


def per_100(p: float) -> str:
    """A probability as a count out of 100, which lay readers follow better than a percentage."""
    n = round(p * 100)
    return "kurang dari 1 dari 100" if n < 1 else "lebih dari 99 dari 100" if n > 99 else f"{n} dari 100"


def rupiah(amount: float) -> str:
    if amount >= 1e6:
        return f"Rp{amount / 1e6:.1f} juta".replace(".", ",")
    return "Rp" + f"{amount:,.0f}".replace(",", ".")


def explain(result: dict, claim: Optional[pd.Series] = None) -> list:
    """The assessment in plain Indonesian. `claim` adds what only a stored claim has: rupiah gap and rule hits."""
    fields = {f["field"]: f for f in result["fields"]}
    claimed = next((s for s in result["severity"] if s["claimed"]), None)
    out = []
    if claimed is None:
        out.append("Ini klaim rawat jalan, jadi tidak ada severity level yang dinilai.")
    else:
        top = max(result["severity"], key=lambda s: s["p"])
        if top is claimed:
            out.append(f"Severity {claimed['level']} sesuai perkiraan NADI: {per_100(claimed['p'])} klaim dengan riwayat "
                       "dan diagnosis seperti ini ada di level itu.")
        else:
            out.append(f"RS menagih severity {claimed['level']}. Dari klaim dengan riwayat dan diagnosis seperti ini, "
                       f"NADI memperkirakan {per_100(top['p'])} ada di severity {top['level']} "
                       f"dan {per_100(claimed['p'])} di severity {claimed['level']}.")
        if LEVELS.index(claimed["level"]) > LEVELS.index(top["level"]):
            if claim is not None and claim.potential > 0:
                out.append(f"Kalau severity {top['level']}, tarifnya sekitar {rupiah(claim.potential)} lebih rendah "
                           "(estimasi kasar).")
            coded = [fields[s] for s in SEC if fields[s]["value"] != "-"]
            rare = min(coded, key=lambda f: f["p"], default=None)
            if rare is None:
                out.append("Tidak ada diagnosis sekunder yang dikode. Di data sampel, kurang dari 1 dari 100 klaim "
                           "severity II atau III yang tanpa diagnosis sekunder.")
            elif rare["p"] < ANOMALY_BELOW:
                name = f" ({rare['name']})" if rare["name"] else ""
                out.append(f"Diagnosis sekunder {rare['value']}{name} jarang muncul pada kasus seperti ini: "
                           f"{per_100(rare['p'])}.")
    unusual = {
        "gap": "Jarak dari klaim sebelumnya ({value} hari) tidak biasa untuk pasien ini: {n}.",
        "fac": "Faskesnya {value} dengan klaim sebelumnya, dan itu tidak biasa: {n}.",
        "los": "LOS {value} hari tidak biasa untuk kasus ini: {n}.",
    }
    for field, text in unusual.items():
        if fields[field]["anomaly"]:
            value = {"same": "sama", "diff": "berbeda"}.get(fields[field]["value"], fields[field]["value"])
            out.append(text.format(value=value, n=per_100(fields[field]["p"])))
    if claim is not None and claim.split_episode:
        out.append("Pasien masuk lagi paling lama 1 hari setelah pulang, di RS dan diagnosis yang sama. "
                   "Ini pola pemecahan episode.")
    if claim is not None and claim.duplicate:
        out.append("Klaim yang sama persis sudah pernah ditagihkan (repeat billing).")
    return out


def assess(row: int, changes: dict) -> dict:
    """Model probability of every field of claim `row`, after applying `changes`."""
    visits = history(row)
    patient = claims[claims.patient == claims.patient[row]]
    feats = nadi.features(patient).loc[visits.index]
    feats.loc[row, list(changes)] = list(changes.values())
    codes = sorted(c for c in feats.loc[row, SEC] if c != "-")  # secondary dx must stay in canonical order
    feats.loc[row, SEC] = codes + ["-"] * (len(SEC) - len(codes))
    result = score(feats, visits.index.get_loc(row))
    # a simulated claim is no longer the stored one, so its rupiah gap and rule hits do not apply
    return {**result, "explanation": explain(result, None if changes else claims.iloc[row])}


def as_records(df: pd.DataFrame) -> list:
    cols = ["patient", "hospital", "hosp_class", "admit", "service", "dx", "cbg", "severity", "expected",
            "los", "cost", "p_claimed", "potential", "split_episode", "duplicate"]
    out = df[cols].assign(
        admit=df.admit.dt.strftime("%Y-%m-%d"), row=df.index,
        dx_name=df.dx.map(NAMES["dx"]).fillna(""), cbg_name=df.cbg.map(NAMES["cbg"]).fillna(""),
    )
    latest = {entry["row"]: entry["decision"] for entry in read_log()[0]}
    return [{**record, "decision": latest.get(record["row"])} for record in out.to_dict("records")]


@app.get("/")
def index():
    return FileResponse("web/index.html")


# the public explainer page (also published with GitHub Pages) and the logo
app.mount("/tentang", StaticFiles(directory="docs", html=True))


@app.get("/api/summary")
def summary():
    above = claims[claims.potential > 0]
    entries, _, intact = read_log()
    return {
        "reviewed": len({entry["row"] for entry in entries}),
        "log_entries": len(entries),
        "log_intact": intact,
        "claims": len(claims),
        "inpatient": int((claims.service == "RI").sum()),
        "above": len(above),
        "potential": int(above.potential.sum()),
        "rules": int((claims.split_episode | claims.duplicate).sum()),
        "facilities": len(facilities),
        "outliers": int((facilities.z > 3).sum()),
    }


@app.get("/api/queue")
def queue(kind: str = "model", limit: int = 50):
    limit = min(max(limit, 1), 200)
    if kind == "rules":
        picked = claims[claims.split_episode | claims.duplicate].nlargest(limit, "cost")
    else:
        picked = claims[claims.potential > 0].nlargest(limit, "priority")
    return as_records(picked)


@app.get("/api/facilities")
def facility_list():
    return facilities.reset_index().to_dict("records")


@app.get("/api/facility/{hospital}")
def facility(hospital: int):
    if hospital not in facilities.index:
        raise HTTPException(404, "facility not found")
    mine = inpatient[inpatient.hospital == hospital]
    by_cbg = pd.DataFrame({"n": 1, "observed": mine.severity != "I", "expected": mine.peer}).groupby(mine.cbg).sum()
    by_cbg = by_cbg.assign(excess=by_cbg.observed - by_cbg.expected).nlargest(6, "excess").reset_index()
    by_cbg["name"] = by_cbg.cbg.map(NAMES["cbg"]).fillna("")
    flagged = claims[(claims.hospital == hospital) & to_review]
    return {"hospital": hospital, **facilities.loc[hospital].to_dict(),
            "cbg": by_cbg.to_dict("records"), "claims": as_records(flagged.nlargest(20, "priority"))}


@app.get("/api/claim/{row}")
def claim(row: int):
    reviews = [entry for entry in read_log()[0] if entry["row"] == row]
    return {"visits": as_records(history(row)), "options": OPTIONS, "reviews": reviews, **assess(row, {})}


@app.post("/api/claim/{row}/simulate")
def simulate(row: int, changes: Changes):
    return assess(row, changes.model_dump(exclude_none=True))


@app.post("/api/claim/{row}/review")
def review(row: int, body: Review):
    c = claim_at(row)
    with write_lock:
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "row": row,
                 "severity": c.severity, "expected": c.expected, "p_claimed": round(float(c.p_claimed), 4),
                 **body.model_dump(), "prev": read_log()[1]}
        with REVIEWS.open("a") as log:
            log.write(json.dumps(entry) + "\n")
    return entry


@app.get("/api/bridge")
def bridge_target():
    """Where assessments would be delivered. Never includes a secret."""
    return bridge.target()


@app.get("/api/claim/{row}/fhir")
def fhir(row: int):
    """The assessment as a FHIR R4 Bundle: RiskAssessment, plus who produced and who verified it."""
    c = claim_at(row)
    if c.service != "RI":
        raise HTTPException(422, "only inpatient claims have a severity assessment")
    result = assess(row, {})
    return bridge.assessment_bundle(
        key=str(row),
        subject={"identifier": {"system": bridge.BASE + "peserta", "value": str(c.patient)}},
        claim_ref={"identifier": {"system": bridge.BASE + "klaim", "value": str(row)},
                   "display": f"Klaim {c.cbg}-{c.severity}, masuk {c.admit:%Y-%m-%d}"},
        result=result, explanation=result["explanation"],
        reviews=[entry for entry in read_log()[0] if entry["row"] == row], version=MODEL_VERSION)


@app.get("/api/claim/{row}/fhir/claim")
def fhir_claim(row: int):
    """The claim and the claims before it as a FHIR Bundle: the input format of /api/fhir/assess."""
    return bridge.claim_bundle(history(row).loc[:row])


@app.get("/api/claim/{row}/satusehat")
def satusehat_preview(row: int):
    """The requests that would deliver this assessment to SATUSEHAT. Nothing is sent: sample data stays here."""
    risk = fhir(row)["entry"][0]["resource"]
    risk["subject"] = {"reference": "Patient/{IHS number pasien}"}
    return bridge.describe(bridge.for_satusehat(risk))


@app.post("/api/fhir/assess")
def assess_fhir(bundle: dict, request: Request, send: bool = False):
    """Score the newest Claim in a FHIR Bundle of one patient's claims; with `send`, deliver the result too."""
    if int(request.headers.get("content-length", 0)) > MAX_FHIR_BODY:
        raise HTTPException(413, "Bundle too large")
    try:
        visits, newest = bridge.claims_from_bundle(bundle)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"resourceType": "OperationOutcome", "issue": [
            {"severity": "error", "code": "invalid", "diagnostics": str(error)}]})
    result = score(nadi.features(visits), len(visits) - 1)
    key = "".join(ch for ch in str(newest.get("id", "klaim")) if ch.isalnum() or ch in "-.")[:40] or "klaim"
    assessment = bridge.assessment_bundle(
        key=key, subject=newest["patient"], claim_ref={"reference": f"Claim/{newest.get('id', key)}"},
        result=result, explanation=explain(result), reviews=[], version=MODEL_VERSION)
    if not send:
        return assessment
    # SATUSEHAT takes the RiskAssessment only; any other FHIR server gets claim and assessment in one transaction
    risk = assessment["entry"][0]["resource"]
    if bridge.target()["mode"] == "fhir":
        relayed = ("Patient", "Organization", "Encounter", "Condition", "Claim")  # what the bridge reads, nothing else
        claim_part = {"entry": [e for e in bundle["entry"] if e["resource"]["resourceType"] in relayed]}
        payload = bridge.as_transaction(claim_part, assessment)
    else:
        payload = bridge.for_satusehat(risk)
    try:
        return {"assessment": assessment, "delivery": bridge.send(payload)}
    except OSError as error:
        raise HTTPException(502, f"FHIR server tidak bisa dihubungi: {error}")
