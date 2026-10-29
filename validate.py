"""Validate NADI on a year it never saw, and look for the chosen fraud patterns in six years of claims.

  python prepare.py <edition folders> --out data/claims_all.parquet
  python train.py --data data/claims_all.parquet --before 2024-01-01 --steps 10000 --out artifacts/nadi_2019_2023.pt
  python validate.py [--rules-only]

The sample has no fraud labels, so the evidence is indirect:
  [1]-[3] the model learned from claims up to 2023 and is tested on 2024 claims,
          as they are and with known patterns injected into them
  [4]     rule hits per year for the chosen sub-categories, on every claim
  [5]     whether facilities that stand out one year still stand out the next:
          a lasting trait of the facility, not chance
"""
import argparse

import numpy as np
import pandas as pd

import evaluate
import nadi

DATA = "data/claims_all.parquet"
MODEL = "artifacts/nadi_2019_2023.pt"
CUTOFF = pd.Timestamp("2024-01-01")
LEVELS = evaluate.LEVELS
SEC = evaluate.SEC
# repeat stays that are usually planned: chemo/radiotherapy, dialysis, kidney disease, thalassaemia, cancer
PLANNED = r"^(Z49|Z51|N18|I12|D56|C\d\d|D0\d)"
rng = np.random.default_rng(0)


def severity_probs(model, vocab, df: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """P(severity I, II, III) for the claims at `rows`, each read with the patient's earlier claims."""
    X, chunk, pos = nadi.to_sequences(nadi.tokenize(df, vocab), df.patient.to_numpy())
    need, where = np.unique(chunk[rows], return_inverse=True)
    sev_ids = [nadi.field_ids(vocab, "sev")[level] for level in LEVELS]
    return nadi.field_probs(model, X[need], "sev", sev_ids)[where, pos[rows]]


def severity_accuracy(model, vocab, df: pd.DataFrame, table: pd.DataFrame) -> None:
    later = (df.discharge >= CUTOFF).to_numpy()
    patients = df.patient[later & (df.service == "RI").to_numpy()].unique()
    sub = df[df.patient.isin(patients)].reset_index(drop=True)  # 2024 inpatients with their history
    rows = np.flatnonzero(((sub.discharge >= CUTOFF) & (sub.service == "RI")).to_numpy())
    claims = sub.iloc[rows]
    guess = LEVELS[severity_probs(model, vocab, sub, rows).argmax(axis=1)]
    feats = evaluate.with_nsec(nadi.features(claims))  # nsec only, so no history needed
    table_guess = table.columns[np.nan_to_num(evaluate.table_probs(table, feats)).argmax(axis=1)]
    truth = claims.severity.to_numpy()
    groups = {
        "every 2024 inpatient claim": np.ones(len(claims), dtype=bool),
        "patients left out of training": nadi.is_test(claims.patient),
        "no claim before 2024 (no history)": ~claims.patient.isin(df.patient[~later]).to_numpy(),
    }
    print(f"\n[1] severity of 2024 inpatient claims; NADI saw only claims up to 2023")
    for name, m in groups.items():
        print(f"  {name:<36} n={m.sum():>7,} | NADI {np.mean(guess[m] == truth[m]):.1%} | "
              f"baseline table {np.mean(table_guess[m] == truth[m]):.1%} | always 'I' {np.mean(truth[m] == 'I'):.1%}")


def injected(model, vocab, df: pd.DataFrame, table: pd.DataFrame, pool: dict) -> None:
    """[2] upcoding and [3] episode splits injected into 2024 claims of patients left out of training."""
    test = df[nadi.is_test(df.patient)].reset_index(drop=True)
    feats, patient = nadi.features(test), test.patient.to_numpy()
    target = ((test.service == "RI") & (test.discharge >= CUTOFF)).to_numpy()
    low = target & (feats.sev == "I").to_numpy()

    def pick(mask: np.ndarray) -> np.ndarray:
        rows = np.flatnonzero(mask)
        return rng.choice(rows, int(len(rows) * evaluate.INJECT_SHARE), replace=False)

    def show(changed: pd.DataFrame, rows: np.ndarray, with_sec: bool) -> None:
        new = evaluate.surprise(model, vocab, changed, patient)
        label = np.zeros(len(changed), dtype=bool)
        label[rows] = True
        evaluate.report("NADI, severity surprisal", new.sev.to_numpy()[target], label[target])
        if with_sec:
            evaluate.report("NADI, secondary dx + severity", new[SEC + ["sev"]].sum(axis=1).to_numpy()[target], label[target])
        evaluate.report("baseline table", evaluate.table_surprise(table, evaluate.with_nsec(changed[target])), label[target])

    print(f"\n[2a] upcoding injected into 2024 claims: severity I -> II, nothing else changed")
    rows = pick(low)
    changed = feats.copy()
    changed.loc[rows, "sev"] = "II"
    show(changed, rows, with_sec=False)

    print(f"\n[2b] realistic upcoding: +1 secondary dx and severity I -> II")
    rows = pick(low & (feats.sec3 == "-").to_numpy() & feats.cbg.isin(list(pool)).to_numpy())
    show(evaluate.add_complication(feats, rows, pool), rows, with_sec=True)

    print(f"\n[3] 2024 inpatient stay split into two claims (sub-category 8)")
    split = evaluate.split_episodes(test, pick(target & (test.los >= 4).to_numpy()))
    new = evaluate.surprise(model, vocab, nadi.features(split), split.patient.to_numpy())
    ri = ((split.service == "RI") & (split.discharge >= CUTOFF)).to_numpy()
    label = split.injected.to_numpy()[ri]
    evaluate.report("NADI, gap + facility surprisal", (new.gap + new.fac).to_numpy()[ri], label)
    flagged = nadi.rule_flags(split).split_episode.to_numpy()[ri]
    print(f"  rule-based: recall {flagged[label].mean():.0%}, flags {flagged[~label].sum()} original claims")


def stay_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per inpatient claim, compared with the same patient's previous inpatient claim."""
    ri = df[df.service == "RI"]
    prev = ri.groupby("patient")[["admit", "discharge", "hospital", "dx"]].shift()
    gap = (ri.admit - prev.discharge).dt.days
    same = (ri.hospital == prev.hospital) & (ri.dx == prev.dx)
    rules = nadi.rule_flags(df).loc[ri.index]
    return pd.DataFrame({
        "split_episode": rules.split_episode,                        # 8: back again within a day
        "readmit_30d": same & gap.between(2, 30) & ~ri.dx.str.match(PLANNED),  # 16: back within a month
        "overlap": prev.discharge.notna() & (ri.admit < prev.discharge),  # 11: two stays at once
        "duplicate": rules.duplicate,                                # 11: the same claim twice
        "across_years": rules.split_episode & (prev.discharge.dt.year < ri.admit.dt.year),
    })


def rule_hits(df: pd.DataFrame, flags: pd.DataFrame) -> None:
    ri = df.loc[flags.index]
    year = ri.discharge.dt.year
    print("\n[4] rule hits on every inpatient claim, per discharge year (weighted = JKN population scale)")
    print(f"  {'year':<6}{'inpatient':>10}" + "".join(f"{name:>16}" for name in flags.columns))
    for y, rows in ri.groupby(year).groups.items():
        f = flags.loc[rows]
        print(f"  {y:<6}{len(rows):>10,}" + "".join(f"{f[c].sum():>16,}" for c in flags.columns))
    for name in ["split_episode", "readmit_30d", "overlap", "duplicate"]:
        hit = ri[flags[name]]
        print(f"  {name:<14} {len(hit):>6,} claims over six years | weighted about {hit.weight.sum() / year.nunique():,.0f} "
              f"claims and Rp{(hit.cost * hit.weight).sum() / year.nunique() / 1e9:,.0f}B per year | "
              f"top 5% of facilities hold {top_share(ri, flags[name]):.0%} of them")


def top_share(ri: pd.DataFrame, hit: pd.Series) -> float:
    per_facility = hit.groupby(ri.hospital).sum().sort_values(ascending=False)
    return per_facility.iloc[:max(1, len(per_facility) // 20)].sum() / max(1, per_facility.sum())


def persistence(name: str, score: dict, flagged: dict) -> None:
    """Spearman of a facility score between consecutive years, and how often a flag repeats."""
    years = sorted(score)
    print(f"  {name}")
    for a, b in zip(years, years[1:]):
        both = pd.concat([score[a], score[b]], axis=1, join="inner").dropna()
        fa, fb = flagged[a].reindex(both.index, fill_value=False), flagged[b].reindex(both.index, fill_value=False)
        again = fb[fa].mean() if fa.any() else np.nan
        print(f"    {a} -> {b}: Spearman {both.rank().corr().iloc[0, 1]:.2f} on {len(both):,} facilities | "
              f"flagged in {a}: {fa.sum():>3}, still flagged in {b}: {again:.0%} (base rate {fb.mean():.0%})")


def facility_traits(df: pd.DataFrame, flags: pd.DataFrame) -> None:
    print("\n[5] do facilities that stand out one year stand out again the next?")
    ri = df.loc[flags.index]
    years = ri.discharge.dt.year
    sev_z, sev_flag, rep_rate, rep_flag = {}, {}, {}, {}
    for y in sorted(years.unique()):
        inpatient = ri[years == y]
        index = nadi.facility_index(inpatient, nadi.peer_expected(inpatient))
        sev_z[y], sev_flag[y] = index.z, index.z > 3
        f = flags.loc[inpatient.index, ["split_episode", "readmit_30d"]].any(axis=1)
        n = inpatient.groupby("hospital").size()
        rate = f.groupby(inpatient.hospital).mean()[n >= nadi.MIN_FACILITY_CLAIMS]
        rep_rate[y], rep_flag[y] = rate, rate >= rate.quantile(0.95)
    persistence("severity above peers (sub-category 4), flag = above the 3 sigma limit", sev_z, sev_flag)
    persistence("split episodes + 30-day readmissions (8, 16), flag = top 5% rate", rep_rate, rep_flag)


def main(rules_only: bool) -> None:
    df = pd.read_parquet(DATA)
    print(f"{len(df):,} claims, discharged {df.discharge.min():%Y-%m-%d} .. {df.discharge.max():%Y-%m-%d}, "
          f"{df.patient.nunique():,} patients")
    if not rules_only:
        model, vocab = nadi.load(MODEL)
        earlier = df[(df.discharge < CUTOFF) & ~nadi.is_test(df.patient) & (df.service == "RI")]
        train_ri = nadi.features(earlier)
        table = evaluate.baseline(evaluate.with_nsec(train_ri), ["cbg", "nsec"], "sev")
        severe = train_ri[train_ri.sev != "I"]
        pool = {cbg: g[SEC].to_numpy().ravel() for cbg, g in severe.groupby("cbg")}
        pool = {cbg: codes[codes != "-"] for cbg, codes in pool.items() if (codes != "-").any()}
        del earlier, train_ri, severe
        severity_accuracy(model, vocab, df, table)
        injected(model, vocab, df, table, pool)
    flags = stay_flags(df)
    rule_hits(df, flags)
    facility_traits(df, flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules-only", action="store_true", help="skip the model, run [4] and [5]")
    main(parser.parse_args().rules_only)
