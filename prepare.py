"""Clean the BPJS Kesehatan sample data (.dta) into one table of hospital claims.

Usage : python prepare.py "<sample data folder>/data" [more edition folders ...] [--out PATH]
Output: data/claims.parquet (or --out), one row per claim, sorted by patient and date,
        and data/names.json with dx and CBG group names for display.

Editions share member ids (the sample is a longitudinal cohort), so several
editions together give each patient a history across years.
"""
import argparse
import json
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


def edition(folder: Path):
    """Clean claims of one edition, still with the original ids, plus dx and CBG names."""
    print(f"{folder}")
    df = read(folder, "fkrtl", CLAIM_COLUMNS)
    print(f"  raw claims                : {len(df):>9,}")

    def keep(mask: pd.Series, reason: str) -> pd.DataFrame:
        print(f"    dropped, {reason:<18}: {(~mask).sum():>9,}")
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
    df = df.merge(members.drop_duplicates("patient"), on="patient", how="left")
    df["age"] = ((df.admit - df.birth).dt.days // 365.25).fillna(-1).astype("int16")
    df["sex"] = df.sex.map({1: "L", 2: "P"})

    secondary, dx_names = secondary_diagnoses(folder)
    df = df.join(secondary, on="visit_id")
    df["nsec"] = df.nsec.fillna(0).astype("int16")

    # CBG group name without the severity word at the end
    cbg_names = df.drop_duplicates("cbg").set_index("cbg").cbg_name
    cbg_names = cbg_names.str.replace(r"\s*\(?(RINGAN|SEDANG|BERAT)\)?$", "", regex=True)
    df = df.drop(columns=["visit_id", "cbg_full", "cbg_name", "birth"])
    print(f"  clean claims              : {len(df):>9,}  discharged {df.discharge.min():%Y-%m-%d} .. {df.discharge.max():%Y-%m-%d}")
    return df, dx_names, cbg_names


def main(folders: list, out: Path) -> None:
    parts = [edition(folder) for folder in folders]
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    # one id space for every edition, so the same member or facility keeps one number across years
    df["patient"] = anonymize(df.patient, seed=1)
    df["hospital"] = anonymize(df.hospital, seed=2)
    df = df.sort_values(["patient", "admit", "discharge"]).reset_index(drop=True)

    assert df[["patient", "hospital", "dx", "cbg", "severity", "service"]].notna().all().all()
    assert (df.los >= 0).all()

    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    names = {"dx": {}, "cbg": {}}
    for _, dx_names, cbg_names in parts:  # later editions win when a name changed
        names["dx"].update(dx_names.to_dict())
        names["cbg"].update(cbg_names.to_dict())
    NAMES.write_text(json.dumps(names))
    print(f"clean claims                : {len(df):>9,}  ->  {out}")
    print(f"patients {df.patient.nunique():,} | facilities {df.hospital.nunique():,} | "
          f"inpatient {(df.service == 'RI').sum():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.folders, args.out)
