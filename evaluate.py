"""Evaluate NADI on held-out patients.

The sample data has no fraud labels, so known patterns are injected into test
claims and we measure how many of them the model ranks at the top.
Output: a summary on stdout and artifacts/scored.parquet for the app.
"""
import numpy as np
import pandas as pd

import nadi

LEVELS = np.array(["I", "II", "III"])
SEC = ["sec1", "sec2", "sec3"]
INJECT_SHARE = 0.05
rng = np.random.default_rng(0)


def surprise(model, vocab, feats: pd.DataFrame, patient: np.ndarray) -> pd.DataFrame:
    """Model surprisal per claim, per field."""
    X, chunk, pos = nadi.to_sequences(nadi.encode(feats, vocab), patient)
    return pd.DataFrame(nadi.surprise(model, X)[chunk, pos], columns=nadi.FIELDS)


def report(name: str, score: np.ndarray, injected: np.ndarray) -> None:
    rank = pd.Series(score).rank().to_numpy()
    n1, n0 = injected.sum(), (~injected).sum()
    auc = (rank[injected].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    top = np.argsort(-score)[: len(score) // 20]
    print(f"  {name:<32} AUC {auc:.3f} | top 5%: precision {injected[top].mean():.0%}, "
          f"recall {injected[top].sum() / n1:.0%}")


def n_secondary(feats: pd.DataFrame) -> pd.Series:
    return (feats[SEC] != "-").sum(axis=1)


def severity_table(feats: pd.DataFrame) -> pd.DataFrame:
    """Simple baseline: P(severity | CBG, number of secondary dx) from train claims."""
    counts = feats.groupby([feats.cbg, n_secondary(feats), feats.sev]).size().unstack(fill_value=0) + 0.5
    return counts.div(counts.sum(axis=1), axis=0)


def table_probs(table: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
    return table.reindex(pd.MultiIndex.from_arrays([feats.cbg, n_secondary(feats)])).to_numpy()


def table_surprise(table: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
    p = table_probs(table, feats)
    claimed = p[np.arange(len(p)), table.columns.get_indexer(feats.sev)]
    return -np.log(np.nan_to_num(claimed, nan=0.01))


def add_complication(feats: pd.DataFrame, rows: np.ndarray, pool: dict) -> pd.DataFrame:
    """Realistic upcoding: add one secondary dx that is common at higher severity, raise I -> II."""
    out = feats.copy()
    for row in rows:
        codes = {c for c in out.loc[row, SEC] if c != "-"} | {str(rng.choice(pool[out.at[row, "cbg"]]))}
        out.loc[row, SEC] = sorted(codes) + ["-"] * (len(SEC) - len(codes))
        out.at[row, "sev"] = "II"
    return out


def split_episodes(df: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Split one inpatient stay into two back-to-back claims at the same hospital."""
    first = df.loc[rows].assign(injected=False)
    second = first.assign(injected=True)
    first["los"] = first.los // 2
    first["discharge"] = first.admit + pd.to_timedelta(first.los, unit="D")
    second["admit"] = first.discharge
    second["los"] = second.los - first.los
    out = pd.concat([df.drop(rows).assign(injected=False), first, second])
    return out.sort_values(["patient", "admit", "discharge"], kind="stable").reset_index(drop=True)


def save_scored(df, train, inpatient, nll, p_sev) -> None:
    """Original test claims with their scores, for the queue and simulator in the app."""
    claimed = df.severity.map({"0": 0, "I": 1, "II": 2, "III": 3}).to_numpy()
    expected = 1 + p_sev.argmax(axis=1)
    typical = train.groupby(["cbg", "hosp_class", "severity"]).cost.agg(["median", "size"])
    typical = typical["median"][typical["size"] >= 5]
    keys = pd.MultiIndex.from_arrays([df.cbg, df.hosp_class, LEVELS[expected - 1]])
    saving = (df.cost - typical.reindex(keys).to_numpy()).clip(lower=0).fillna(0)

    scored = df.join(nadi.rule_flags(df))
    scored[["p_I", "p_II", "p_III"]] = p_sev
    scored["p_claimed"] = np.exp(-nll.sev.to_numpy())
    scored["expected"] = LEVELS[expected - 1]
    # gap to the typical tariff at the expected severity; rough, only used for ranking
    scored["potential"] = np.where(inpatient & (claimed > expected), saving, 0).astype("int64")
    scored.to_parquet("artifacts/scored.parquet", index=False)

    above = scored[scored.potential > 0]
    print(f"\nsaved artifacts/scored.parquet | inpatient claims above expected severity: {len(above):,}, "
          f"potential gap Rp{above.potential.sum() / 1e9:.2f}B (rough)")


def main() -> None:
    model, vocab = nadi.load("artifacts/nadi.pt")
    df = pd.read_parquet("data/claims.parquet")
    feats = nadi.features(df)
    is_test = nadi.is_test(df.patient)
    train, train_ri = df[~is_test], feats[~is_test & (feats.svc == "RI")]
    df, feats = df[is_test].reset_index(drop=True), feats[is_test].reset_index(drop=True)
    patient = df.patient.to_numpy()
    inpatient = (feats.svc == "RI").to_numpy()
    table = severity_table(train_ri)

    X, chunk, pos = nadi.to_sequences(nadi.encode(feats, vocab), patient)
    nll = pd.DataFrame(nadi.surprise(model, X)[chunk, pos], columns=nadi.FIELDS)
    sev_ids = [nadi.field_ids(vocab, "sev")[level] for level in LEVELS]
    p_sev = nadi.field_probs(model, X, "sev", sev_ids)[chunk, pos]

    truth = feats.sev[inpatient].to_numpy()
    table_guess = table.columns[np.nan_to_num(table_probs(table, feats[inpatient])).argmax(axis=1)]
    print(f"test claims {len(df):,} (inpatient {inpatient.sum():,}) from {df.patient.nunique():,} patients")
    print(f"inpatient severity accuracy: NADI {np.mean(LEVELS[p_sev[inpatient].argmax(axis=1)] == truth):.1%} | "
          f"baseline table {np.mean(table_guess == truth):.1%} | always 'I' {np.mean(truth == 'I'):.1%}")

    def pick(mask: np.ndarray) -> np.ndarray:
        rows = np.flatnonzero(mask)
        return rng.choice(rows, int(len(rows) * INJECT_SHARE), replace=False)

    def injected_report(changed: pd.DataFrame, rows: np.ndarray, with_sec: bool) -> None:
        new = surprise(model, vocab, changed, patient)
        label = np.zeros(len(changed), dtype=bool)
        label[rows] = True
        report("NADI, severity surprisal", new.sev.to_numpy()[inpatient], label[inpatient])
        if with_sec:
            report("NADI, secondary dx + severity", new[SEC + ["sev"]].sum(axis=1).to_numpy()[inpatient], label[inpatient])
        report("baseline table", table_surprise(table, changed[inpatient]), label[inpatient])

    low = inpatient & (feats.sev == "I").to_numpy()

    print("\n[1] severity raised I -> II, nothing else changed")
    rows = pick(low)
    changed = feats.copy()
    changed.loc[rows, "sev"] = "II"
    injected_report(changed, rows, with_sec=False)

    print("\n[2] realistic upcoding: +1 secondary dx and severity I -> II")
    severe = train_ri[train_ri.sev != "I"]
    pool = {cbg: g[SEC].to_numpy().ravel() for cbg, g in severe.groupby("cbg")}
    pool = {cbg: codes[codes != "-"] for cbg, codes in pool.items() if (codes != "-").any()}
    rows = pick(low & (feats.sec3 == "-").to_numpy() & feats.cbg.isin(list(pool)).to_numpy())
    injected_report(add_complication(feats, rows, pool), rows, with_sec=True)

    print("\n[3] inpatient episode split into two claims")
    split_df = split_episodes(df, pick(inpatient & (df.los >= 4).to_numpy()))
    new = surprise(model, vocab, nadi.features(split_df), split_df.patient.to_numpy())
    ri = (split_df.service == "RI").to_numpy()
    label = split_df.injected.to_numpy()[ri]
    report("NADI, gap + facility surprisal", (new.gap + new.fac).to_numpy()[ri], label)
    flagged = nadi.rule_flags(split_df).split_episode.to_numpy()[ri]
    print(f"  rule-based: recall {flagged[label].mean():.0%}, flags {flagged[~label].sum()} original claims")

    save_scored(df, train, inpatient, nll, p_sev)


if __name__ == "__main__":
    main()
