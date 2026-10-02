"""Clean the BPJS Kesehatan sample data (.dta) into one table of hospital claims.

Usage : python prepare.py "<sample data folder>/data"
Output: data/claims.parquet, one row per claim, sorted by patient and date,
        and data/names.json with dx and CBG group names for display.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/claims.parquet")
NAMES = Path("data/names.json")
SEVERITIES = ["0", "I", "II", "III"]
MAX_SECONDARY = 3  # secondary dx kept per claim

CLAIM_COLUMNS = {
    "PSTV01": "patient", "PSTV15": "weight", "FKL02": "visit_id",
    "FKL03": "admit", "FKL04": "discharge", "FKL09": "hosp_class",
    "FKL10": "service", "FKL17A": "dx", "FKL19": "cbg_full", "FKL19A": "cbg_name",
    "FKL47": "billed", "FKL48": "cost",
}


def read(folder: Path, suffix: str, columns: dict) -> pd.DataFrame:
    path = next(folder.glob(f"*_{suffix}.dta"))
    df = pd.read_stata(path, columns=list(columns), convert_categoricals=False)
    return df.rename(columns=columns)


def anonymize(ids: pd.Series, seed: int) -> np.ndarray:
    """Replace the original ids with shuffled running numbers, so derived files never carry them."""
    codes, uniques = pd.factorize(ids)
    return np.random.default_rng(seed).permutation(len(uniques))[codes]


def secondary_diagnoses(folder: Path):
    """Secondary dx per claim (wide), plus the name of every 3-digit ICD-10 code."""
    sec = read(folder, "diagnosissekunder", {"FKL02": "visit_id", "FKL24A": "code", "FKL24B": "label"})
    sec = sec[sec.code.str.len() == 3]
    names = sec.drop_duplicates("code").set_index("code").label.str[4:]  # "Z37 Outcome of delivery"
    sec = sec[["visit_id", "code"]].drop_duplicates().sort_values(["visit_id", "code"])
    sec["slot"] = sec.groupby("visit_id").cumcount()
    wide = sec[sec.slot < MAX_SECONDARY].pivot(index="visit_id", columns="slot", values="code")
    wide.columns = [f"sec{i + 1}" for i in wide.columns]
    wide["nsec"] = sec.groupby("visit_id").size()
    return wide, names


def main(folder: Path) -> None:
    df = read(folder, "fkrtl", CLAIM_COLUMNS)
    print(f"raw claims                  : {len(df):>9,}")

    def keep(mask: pd.Series, reason: str) -> pd.DataFrame:
        print(f"  dropped, {reason:<18}: {(~mask).sum():>9,}")
        return df[mask]

    df = keep(~df.visit_id.duplicated(), "duplicate visit id")
    df = keep(df.admit.notna() & (df.discharge >= df.admit), "invalid dates")

    df[["cbg", "severity"]] = df.cbg_full.str.rsplit("-", n=1, expand=True)
    df = keep(df.severity.isin(SEVERITIES) & (df.dx.str.len() == 3), "invalid codes")

    # the visit id prefix (before MMYY + letter + running number) behaves like a facility id
    df["hospital"] = df.visit_id.str.extract(r"^(\d+?)\d{4}[A-Z]\d+$", expand=False)
    df = keep(df.hospital.notna(), "no facility id")

    df["los"] = (df.discharge - df.admit).dt.days
    df["service"] = df.service.map({1: "RJ", 2: "RI"})

    members = read(folder, "kepesertaan", {"PSTV01": "patient", "PSTV03": "birth", "PSTV05": "sex"})
    df = df.merge(members, on="patient", how="left")
    df["age"] = ((df.admit - df.birth).dt.days // 365.25).fillna(-1).astype("int16")
    df["sex"] = df.sex.map({1: "L", 2: "P"})

    secondary, dx_names = secondary_diagnoses(folder)
    df = df.join(secondary, on="visit_id")
    df["nsec"] = df.nsec.fillna(0).astype("int16")

    df["patient"] = anonymize(df.patient, seed=1)
    df["hospital"] = anonymize(df.hospital, seed=2)
    df = df.sort_values(["patient", "admit", "discharge"]).reset_index(drop=True)
    # CBG group name without the severity word at the end
    cbg_names = df.drop_duplicates("cbg").set_index("cbg").cbg_name
    cbg_names = cbg_names.str.replace(r"\s*\(?(RINGAN|SEDANG|BERAT)\)?$", "", regex=True)
    df = df.drop(columns=["visit_id", "cbg_full", "cbg_name", "birth"])

    assert df[["patient", "hospital", "dx", "cbg", "severity", "service"]].notna().all().all()
    assert (df.los >= 0).all()

    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    NAMES.write_text(json.dumps({"dx": dx_names.to_dict(), "cbg": cbg_names.to_dict()}))
    print(f"clean claims                : {len(df):>9,}  ->  {OUT}")
    print(f"patients {df.patient.nunique():,} | facilities {df.hospital.nunique():,} | "
          f"inpatient {(df.service == 'RI').sum():,}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
