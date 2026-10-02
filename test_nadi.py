"""Quick check for the tokenizer and the rule-based flags. Run: python test_nadi.py"""
import pandas as pd

import nadi


def demo() -> None:
    day = pd.Timestamp("2024-01-01")
    after = lambda n: day + pd.Timedelta(days=n)
    df = pd.DataFrame({
        "patient": [0, 0, 0, 1],
        "admit": [day, after(5), after(40), day],
        "discharge": [after(4), after(8), after(40), day],
        "hospital": [7, 7, 9, 7], "hosp_class": [3, 3, 1, 3],
        "service": ["RI", "RI", "RJ", "RJ"], "dx": ["A09", "A09", "Z09", "J06"],
        "cbg": ["K-4-17", "K-4-17", "Q-5-44", "Q-5-44"], "severity": ["I", "II", "0", "0"],
        "los": [4, 3, 0, 0], "sex": ["L", "L", "L", "P"], "age": [30, 30, 30, -1],
        "sec1": [None, "E86", None, None], "sec2": None, "sec3": None,
    })

    feats = nadi.features(df)
    assert feats.gap.tolist() == ["first", "1", "31-90", "first"]
    assert feats.fac.tolist() == ["first", "same", "diff", "first"]
    assert feats.los.tolist() == ["4-5", "3", "0", "0"]
    assert feats.demo.tolist() == ["L:18-39", "L:18-39", "L:18-39", "P:?"]
    assert feats.sec1.tolist() == ["-", "E86", "-", "-"]

    # readmitted a day after discharge, same hospital and dx
    assert nadi.rule_flags(df).split_episode.tolist() == [False, True, False, False]

    vocab = nadi.build_vocab(feats)  # with data this small every value falls back to "?"
    tokens = nadi.encode(feats, vocab)
    assert (tokens[:, nadi.FIELDS.index("dx")] == vocab["dx=?"]).all()
    X, chunk, pos = nadi.to_sequences(tokens, df.patient.to_numpy())
    assert X.shape == (2, nadi.MAX_VISITS, len(nadi.FIELDS))
    assert chunk.tolist() == [0, 0, 0, 1] and pos.tolist() == [0, 1, 2, 0]
    assert (X[1, 1:] == nadi.PAD).all()
    print("ok")


if __name__ == "__main__":
    demo()
