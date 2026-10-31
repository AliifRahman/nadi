"""Build docs/app/: SmartGuard JKN on synthetic claims, as static files for GitHub Pages.

Run: python make_app_demo.py   (needs the trained model)
Every claim, member and facility here is invented and no row of the sample data is read,
so the page can be public. The model stays private: its answers are computed here and
saved as JSON, and web/static.js serves them in place of the API.
"""
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate
import nadi

OUT = Path("docs/app")
SCORED = Path("artifacts/synthetic_scored.parquet")
START = pd.Timestamp("2026-01-05")
LEVELS = ["I", "II", "III"]
# dx: CBG, secondary dx seen at higher severity, usual LOS, invented tariff at severity I
CASES = {
    "O34": ("O-6-10", ["O82", "D64"], 3, 6_500_000), "A90": ("A-4-13", ["K30", "E87"], 4, 3_000_000),
    "A09": ("K-4-17", ["E86", "E87"], 3, 2_800_000), "J18": ("J-4-16", ["E87", "D64"], 5, 4_200_000),
    "I63": ("G-4-14", ["I10", "E11", "G81"], 6, 5_200_000), "E11": ("E-4-10", ["I10", "N18"], 5, 3_800_000),
    "I50": ("I-4-12", ["I10", "J81"], 5, 4_600_000), "N39": ("N-4-13", ["E87", "D64"], 4, 3_100_000),
}
HOSPITAL_CLASS = [3, 3, 3, 4, 2, 3, 7, 3, 6, 3, 4, 3, 7, 2, 3, 3, 6, 3, 4, 3]
TARIFF_CLASS = {2: 1.25, 3: 1.0, 4: 0.9, 6: 1.2, 7: 1.0}
TARIFF_SEVERITY = {"I": 1.0, "II": 1.6, "III": 2.4}
INFLATED = {3, 11, 16}  # facilities that bill higher severity than their peers


def tariff(dx: str, severity: str, hosp_class: int) -> float:
    return CASES[dx][3] * TARIFF_SEVERITY[severity] * TARIFF_CLASS[hosp_class]


def synthetic_claims(rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    def stay(patient, sex, age, hospital, dx, severity, admit, los, secs):
        hosp_class = HOSPITAL_CLASS[hospital]
        cost = round(tariff(dx, severity, hosp_class) * rng.uniform(0.97, 1.03), -2)
        rows.append({"patient": patient, "hospital": hospital, "hosp_class": hosp_class, "service": "RI", "dx": dx,
                     "cbg": CASES[dx][0], "severity": severity, "admit": admit, "discharge": admit + pd.Timedelta(days=los),
                     "los": los, "sex": sex, "age": age, "cost": cost, "billed": cost, "weight": 1.0, "nsec": len(secs),
                     **{f"sec{i + 1}": secs[i] if i < len(secs) else None for i in range(3)}})

    def control_visit(patient, sex, age, hospital, day):
        cost = round(280_000 * rng.uniform(0.9, 1.1), -2)
        rows.append({"patient": patient, "hospital": hospital, "hosp_class": HOSPITAL_CLASS[hospital], "service": "RJ",
                     "dx": "Z09", "cbg": "Q-5-44", "severity": "0", "admit": day, "discharge": day, "los": 0, "sex": sex,
                     "age": age, "cost": cost, "billed": cost, "weight": 1.0, "nsec": 0, "sec1": None, "sec2": None, "sec3": None})

    patient = 0
    for hospital in range(len(HOSPITAL_CLASS)):
        for _ in range(rng.integers(34, 62)):
            patient += 1
            dx = str(rng.choice(list(CASES)))
            sex = "P" if dx == "O34" else str(rng.choice(["L", "P"]))
            age = int(rng.integers(20, 40)) if dx == "O34" else int(rng.integers(18, 75))
            severe = [0.5, 0.35, 0.15] if hospital in INFLATED else [0.8, 0.16, 0.04]
            severity = str(rng.choice(LEVELS, p=severe))
            pool = CASES[dx][1]
            n_sec = {"I": int(rng.random() < 0.3), "II": 1, "III": 2}[severity]
            pattern = rng.random()
            if pattern < 0.05:  # billed high with nothing to support it
                severity, n_sec = str(rng.choice(["II", "III"])), 0
            secs = sorted(rng.choice(pool, size=min(n_sec, len(pool)), replace=False).tolist())
            los = max(1, CASES[dx][2] + int(rng.integers(-1, 3)) + LEVELS.index(severity))
            admit = START + pd.Timedelta(days=int(rng.integers(30, 300)))
            for _ in range(rng.integers(0, 3)):
                control_visit(patient, sex, age, hospital, admit - pd.Timedelta(days=int(rng.integers(8, 60))))
            stay(patient, sex, age, hospital, dx, severity, admit, los, secs)
            discharge = admit + pd.Timedelta(days=los)
            if 0.05 <= pattern < 0.07:  # episode split
                stay(patient, sex, age, hospital, dx, "I", discharge + pd.Timedelta(days=1), 2, [])
            elif 0.07 <= pattern < 0.10:  # back within a month
                stay(patient, sex, age, hospital, dx, "I", discharge + pd.Timedelta(days=int(rng.integers(7, 26))), 3, [])
            elif 0.10 <= pattern < 0.115:  # a second stay elsewhere before this one ended
                other = (hospital + 1) % len(HOSPITAL_CLASS)
                stay(patient, sex, age, other, dx, "I", discharge - pd.Timedelta(days=min(2, los)), 3, [])
            elif 0.115 <= pattern < 0.13:  # the same claim twice, mostly outpatient in the real data
                control_visit(patient, sex, age, hospital, discharge + pd.Timedelta(days=7))
                rows.append(dict(rows[-1]))
    return pd.DataFrame(rows).sort_values(["patient", "admit", "discharge"], kind="stable").reset_index(drop=True)


def typical_tariffs() -> pd.DataFrame:
    """Five rows per cell: scored_frame only trusts a median of at least five claims."""
    cells = [{"cbg": case[0], "hosp_class": c, "severity": s, "cost": tariff(dx, s, c)}
             for dx, case in CASES.items() for c in TARIFF_CLASS for s in LEVELS]
    return pd.DataFrame(cells * 5)


def score(df: pd.DataFrame) -> pd.DataFrame:
    model, vocab = nadi.load("artifacts/nadi.pt", device="cpu")
    X, chunk, pos = nadi.to_sequences(nadi.encode(nadi.features(df), vocab), df.patient.to_numpy())
    ids = [nadi.field_ids(vocab, "sev")[level] for level in LEVELS]
    p_sev = nadi.field_probs(model, X, "sev", ids)[chunk, pos]
    p_claimed = np.exp(-nadi.surprise(model, X)[chunk, pos, nadi.FIELDS.index("sev")])
    return evaluate.scored_frame(df.assign(test=True), typical_tariffs(), p_claimed, p_sev)


def tidy(value):
    """JSON-ready, with fewer digits."""
    if isinstance(value, dict):
        return {key: tidy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tidy(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return round(value, 4) if isinstance(value, float) else value


def dump(name: str, data) -> None:
    path = OUT / "api" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tidy(data), ensure_ascii=False, separators=(",", ":")))


def simulations(app, row: int, claim: dict) -> dict:
    """Answers for every severity, with the claim's secondary dx as coded, one removed, or none."""
    value = {f["field"]: f["value"] for f in claim["fields"]}
    coded = [value[s] for s in app.SEC if value[s] != "-"]
    variants = {tuple(coded), (), *(tuple(c for c in coded if c != gone) for gone in coded)}
    out = {}
    for severity in LEVELS:
        for kept in variants:
            secs = sorted(kept)
            changes = {"sev": severity, **{s: secs[i] if i < len(secs) else "-" for i, s in enumerate(app.SEC)}}
            answer = app.assess(row, changes)
            out[f"{severity}|{value['los']}|{','.join(secs)}"] = {
                "f": [[f["value"], f["p"], f["anomaly"]] for f in answer["fields"]],
                "s": [s["p"] for s in answer["severity"]], "e": answer["explanation"]}
    return out


def page() -> str:
    html = Path("web/index.html").read_text()
    html = html.replace("/tentang/nadi.svg", "../nadi.svg").replace('href="/tentang/"', 'href="../"')
    html = html.replace("<script>\nconst number", '<script src="static.js"></script>\n<script>\nconst number')
    html = html.replace("<body>\n", '<body>\n<p style="margin:0;padding:10px 24px;background:var(--critical-bg);font-size:13px">'
                        "<b>Versi online.</b> Semua klaim, peserta, dan faskes di halaman ini sintetis, bukan data BPJS. "
                        "Jawaban NADI sudah dihitung sebelumnya, dan keputusan verifikator hanya tersimpan di browser ini.</p>\n", 1)
    html = html.replace("Klaim (data sampel)", "Klaim (sintetis)")
    return re.sub(r"Data: klaim 2024.*?\.<span", "Data: klaim sintetis.<span", html)


def main() -> None:
    claims = score(synthetic_claims(np.random.default_rng(2026)))
    claims.to_parquet(SCORED, index=False)
    os.environ["NADI_SCORED"] = str(SCORED)
    import app  # reads NADI_SCORED, so it must come after the file exists
    app.REVIEWS = Path("artifacts/no-reviews-in-the-public-demo")

    shutil.rmtree(OUT, ignore_errors=True)
    queues = {kind: app.queue(kind) for kind in ("model", "rules")}
    for kind, rows in queues.items():
        dump(f"queue-{kind}", rows)
    dump("summary", app.summary())
    dump("facilities", app.facility_list())
    dump("bridge", app.bridge_target())
    dump("options", app.OPTIONS)
    reachable = {c["row"] for rows in queues.values() for c in rows}
    in_queue = set(reachable)
    for hospital in app.facilities.index:
        facility = app.facility(int(hospital))
        dump(f"facility/{hospital}", facility)
        reachable |= {c["row"] for c in facility["claims"]}
    for row in list(reachable):
        reachable |= {v["row"] for v in app.claim(row)["visits"]}
    for row in sorted(reachable):
        claim = app.claim(row)
        del claim["options"], claim["reviews"]
        if row in in_queue and app.claims.service[row] == "RI":
            claim["simulate"] = simulations(app, row, claim)
        if app.claims.service[row] == "RI":
            dump(f"fhir/{row}", app.fhir(row))
            dump(f"satusehat/{row}", app.satusehat_preview(row))
        dump(f"claim/{row}", claim)
    (OUT / "index.html").write_text(page())
    shutil.copy("web/static.js", OUT / "static.js")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{len(claims):,} synthetic claims, {len(reachable)} pages -> {OUT} ({size:.1f} MB)")
    print({k: v for k, v in app.summary().items() if k in ("claims", "inpatient", "above", "rules", "facilities", "outliers")})
    print(claims[nadi.RULES].sum().to_dict())


if __name__ == "__main__":
    main()
