"""Bersihkan Data Sampel BPJS Kesehatan (.dta) menjadi satu tabel klaim RS.

Pakai : python prepare.py "<folder data sampel>/data"
Hasil : data/claims.parquet, satu baris per klaim, urut per peserta dan tanggal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/claims.parquet")
SEVERITIES = ["0", "I", "II", "III"]
MAX_SECONDARY = 3  # diagnosis sekunder yang disimpan per klaim

CLAIM_COLUMNS = {
    "PSTV01": "patient", "PSTV15": "weight", "FKL02": "visit_id",
    "FKL03": "admit", "FKL04": "discharge", "FKL09": "hosp_class",
    "FKL10": "service", "FKL17A": "dx", "FKL19": "cbg_full",
    "FKL47": "billed", "FKL48": "cost",
}


def read(folder: Path, suffix: str, columns: dict) -> pd.DataFrame:
    path = next(folder.glob(f"*_{suffix}.dta"))
    df = pd.read_stata(path, columns=list(columns), convert_categoricals=False)
    return df.rename(columns=columns)


def anonymize(ids: pd.Series, seed: int) -> np.ndarray:
    """Ganti ID asli dengan nomor urut acak supaya turunan data tidak memuatnya."""
    codes, uniques = pd.factorize(ids)
    return np.random.default_rng(seed).permutation(len(uniques))[codes]


def secondary_diagnoses(folder: Path) -> pd.DataFrame:
    sec = read(folder, "diagnosissekunder", {"FKL02": "visit_id", "FKL24A": "code"})
    sec = sec[sec.code.str.len() == 3].drop_duplicates().sort_values(["visit_id", "code"])
    sec["slot"] = sec.groupby("visit_id").cumcount()
    wide = sec[sec.slot < MAX_SECONDARY].pivot(index="visit_id", columns="slot", values="code")
    wide.columns = [f"sec{i + 1}" for i in wide.columns]
    wide["nsec"] = sec.groupby("visit_id").size()
    return wide


def main(folder: Path) -> None:
    df = read(folder, "fkrtl", CLAIM_COLUMNS)
    print(f"klaim mentah              : {len(df):>9,}")

    def keep(mask: pd.Series, reason: str) -> pd.DataFrame:
        print(f"  buang {reason:<19}: {(~mask).sum():>9,}")
        return df[mask]

    df = keep(~df.visit_id.duplicated(), "ID kunjungan ganda")
    df = keep(df.admit.notna() & (df.discharge >= df.admit), "tanggal tidak valid")

    df[["cbg", "severity"]] = df.cbg_full.str.rsplit("-", n=1, expand=True)
    df = keep(df.severity.isin(SEVERITIES) & (df.dx.str.len() == 3), "kode tidak valid")

    # awalan ID kunjungan (sebelum MMYY + huruf + nomor urut) berperilaku sebagai ID faskes
    df["hospital"] = df.visit_id.str.extract(r"^(\d+?)\d{4}[A-Z]\d+$", expand=False)
    df = keep(df.hospital.notna(), "ID faskes tak terbaca")

    df["los"] = (df.discharge - df.admit).dt.days
    df["service"] = df.service.map({1: "RJ", 2: "RI"})

    members = read(folder, "kepesertaan", {"PSTV01": "patient", "PSTV03": "birth", "PSTV05": "sex"})
    df = df.merge(members, on="patient", how="left")
    df["age"] = ((df.admit - df.birth).dt.days // 365.25).fillna(-1).astype("int16")
    df["sex"] = df.sex.map({1: "L", 2: "P"})

    df = df.join(secondary_diagnoses(folder), on="visit_id")
    df["nsec"] = df.nsec.fillna(0).astype("int16")

    df["patient"] = anonymize(df.patient, seed=1)
    df["hospital"] = anonymize(df.hospital, seed=2)
    df = df.sort_values(["patient", "admit", "discharge"]).reset_index(drop=True)
    df = df.drop(columns=["visit_id", "cbg_full", "birth"])

    assert df[["patient", "hospital", "dx", "cbg", "severity", "service"]].notna().all().all()
    assert (df.los >= 0).all()

    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"klaim bersih              : {len(df):>9,}  ->  {OUT}")
    print(f"peserta {df.patient.nunique():,} | faskes {df.hospital.nunique():,} | "
          f"rawat inap {(df.service == 'RI').sum():,}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
