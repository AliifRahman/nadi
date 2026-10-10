"""Build docs/demo.json: NADI's answers for made-up claims, so the public page works without a server.

Run: python make_demo.py   (needs the trained model, like app.py)
Every claim here is invented. Nothing from the sample data is written to the file.
"""
import itertools
import json
from pathlib import Path

import pandas as pd

import app
import nadi

OUT = Path("docs/demo.json")
LOS = [1, 2, 3, 5, 7, 10]  # days
# invented patients in five of the most common inpatient case groups; `sec` lists the secondary dx choices
CASES = {
    "caesar": dict(label="Operasi caesar", who="Perempuan 29 tahun", sex="P", age=29, dx="O34", cbg="O-6-10",
                   sec=[[], ["O82"], ["D64", "O82"]]),
    "dbd": dict(label="Demam berdarah", who="Laki-laki 18 tahun", sex="L", age=18, dx="A90", cbg="A-4-13",
                sec=[[], ["K30"], ["E87", "K30"]]),
    "diare": dict(label="Diare", who="Perempuan 20 tahun", sex="P", age=20, dx="A09", cbg="K-4-17",
                  sec=[[], ["E86"], ["E86", "E87"]]),
    "pneumonia": dict(label="Pneumonia", who="Anak laki-laki 6 tahun", sex="L", age=6, dx="J18", cbg="J-4-16",
                      sec=[[], ["E87"], ["D64", "E87"]]),
    "stroke": dict(label="Stroke", who="Laki-laki 61 tahun", sex="L", age=61, dx="I63", cbg="G-4-14",
                   sec=[[], ["I10"], ["E11", "G81"]]),
}
PLAIN = {  # everyday names for the codes used above
    "O82": "persalinan caesar", "D64": "anemia", "K30": "dispepsia", "E87": "gangguan elektrolit",
    "E86": "dehidrasi", "I10": "hipertensi", "E11": "diabetes tipe 2", "G81": "lumpuh separuh badan",
}


def claim(case: dict, severity: str, los: int, sec: list, admit: pd.Timestamp) -> dict:
    return {
        "patient": 0, "hospital": 1, "hosp_class": 3, "service": "RI", "sex": case["sex"], "age": case["age"],
        "dx": case["dx"], "cbg": case["cbg"], "severity": severity, "los": los,
        "admit": admit, "discharge": admit + pd.Timedelta(days=los),
        **{f"sec{i + 1}": sec[i] if i < len(sec) else None for i in range(3)},
    }


def answer(case: dict, severity: str, los: int, sec: list, again: bool) -> dict:
    """NADI's assessment of one made-up claim. `again`: the patient was discharged the day before."""
    day = pd.Timestamp("2026-09-10")
    rows = [claim(case, severity, los, sec, day)]
    if again:  # an earlier stay for the same dx at the same hospital
        rows.insert(0, claim(case, "I", 3, [], day - pd.Timedelta(days=4)))
    visits = pd.DataFrame(rows)
    result = app.score(nadi.features(visits), len(visits) - 1)
    flags = nadi.rule_flags(visits).iloc[-1]
    stored = pd.Series({"potential": 0, "split_episode": bool(flags.split_episode), "duplicate": False})
    flag = stored.split_episode or any(field["anomaly"] for field in result["fields"])
    return {"p": [round(s["p"], 4) for s in result["severity"]], "says": app.explain(result, stored), "flag": bool(flag)}


def main() -> None:
    answers = {}
    for name, case in CASES.items():
        for severity, los, sec, again in itertools.product(app.LEVELS, LOS, range(3), (False, True)):
            answers[f"{name}|{severity}|{los}|{sec}|{int(again)}"] = answer(case, severity, los, case["sec"][sec], again)
    cases = {
        name: {"label": case["label"], "who": case["who"],
               "dx": f"{case['dx']} {app.NAMES['dx'].get(case['dx'], '')}".strip(),
               "cbg": f"{case['cbg']} {app.NAMES['cbg'].get(case['cbg'], '')}".strip(),
               "sec": [", ".join(f"{code} {PLAIN[code]}" for code in codes) or "tidak ada" for codes in case["sec"]]}
        for name, case in CASES.items()
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"model": app.MODEL_VERSION, "los": LOS, "cases": cases, "answers": answers},
                              ensure_ascii=False, separators=(",", ":")))
    print(f"{len(answers)} answers -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
