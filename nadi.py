"""NADI: a small language model over JKN claim sequences.

Each claim becomes a fixed row of tokens (FIELDS). A GPT-style model learns to
predict the next token from the patient's claim history. The lower the
probability the model gives to the value that was actually claimed, the more
anomalous the claim.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# order matters: severity comes last so it is predicted from every other field
FIELDS = ["demo", "gap", "fac", "svc", "cls", "dx", "cbg", "los", "sec1", "sec2", "sec3", "sev"]
MAX_VISITS = 20  # claims per sequence chunk
MIN_COUNT = 5    # rarer values are mapped to "?" (unknown)
MIN_FACILITY_CLAIMS = 30  # inpatient claims a facility needs before it is compared with peers
PAD = 0

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def is_test(patient) -> np.ndarray:
    """Hold out 10% of patients. The split is per patient, never per claim."""
    return np.asarray(patient) % 10 == 0


# -------------------------------------------------------------- tokenization

def _bucket(values: pd.Series, edges: list, labels: list) -> pd.Series:
    return pd.cut(values, [-np.inf, *edges, np.inf], labels=labels).astype(str)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean claims (sorted by patient and date) -> one categorical value per field."""
    prev = df.groupby("patient")[["discharge", "hospital"]].shift()
    first = prev.discharge.isna()
    gap = (df.admit - prev.discharge).dt.days
    age = _bucket(df.age, [-1, 4, 17, 39, 59], ["?", "0-4", "5-17", "18-39", "40-59", "60+"])
    return pd.DataFrame({
        "demo": df.sex + ":" + age,
        "gap": _bucket(gap, [0, 1, 7, 30, 90], ["0", "1", "2-7", "8-30", "31-90", "90+"]).mask(first, "first"),
        "fac": pd.Series(np.where(df.hospital == prev.hospital, "same", "diff"), df.index).mask(first, "first"),
        "svc": df.service,
        "cls": df.hosp_class.astype(str),
        "dx": df.dx,
        "cbg": df.cbg,
        "los": _bucket(df.los, [0, 1, 2, 3, 5, 7, 14], ["0", "1", "2", "3", "4-5", "6-7", "8-14", "15+"]),
        "sec1": df.sec1.fillna("-"),
        "sec2": df.sec2.fillna("-"),
        "sec3": df.sec3.fillna("-"),
        "sev": df.severity,
    })


def _prefix(field: str) -> str:
    return field.rstrip("123")  # sec1..sec3 share one vocabulary


def build_vocab(feats) -> dict:
    """`feats`: one feature frame, or an iterable of them (read one at a time, see by_patient_group)."""
    counts = {}
    for part in [feats] if isinstance(feats, pd.DataFrame) else feats:
        for field in FIELDS:
            counts[field] = part[field].value_counts().add(counts.get(field, 0), fill_value=0)
    vocab = {"<pad>": PAD}
    for field in FIELDS:
        vocab.setdefault(f"{_prefix(field)}=?", len(vocab))
        n = counts[field].sort_values(ascending=False, kind="stable")
        for value in n[n >= MIN_COUNT].index:
            vocab.setdefault(f"{_prefix(field)}={value}", len(vocab))
    return vocab


def by_patient_group(df: pd.DataFrame, parts: int = 8):
    """Yield (row numbers, rows) for `parts` groups of patients, every patient whole.

    Features are strings; making them a group at a time keeps several years of claims in 8 GB.
    """
    group = df.patient.to_numpy() % parts
    for g in range(parts):
        rows = np.flatnonzero(group == g)
        yield rows, df.iloc[rows]


def tokenize(df: pd.DataFrame, vocab: dict) -> np.ndarray:
    """encode(features(df), vocab) for large frames, as int16."""
    assert len(vocab) < np.iinfo(np.int16).max
    out = np.zeros((len(df), len(FIELDS)), dtype=np.int16)
    for rows, part in by_patient_group(df):
        out[rows] = encode(features(part), vocab)
    return out


def encode(feats: pd.DataFrame, vocab: dict) -> np.ndarray:
    """Token ids, shape [n_claims, n_fields]. Out-of-vocab values become '?'."""
    columns = [
        (_prefix(f) + "=" + feats[f]).map(vocab).fillna(vocab[f"{_prefix(f)}=?"])
        for f in FIELDS
    ]
    return np.stack(columns, axis=1).astype(np.int64)


def to_sequences(tokens: np.ndarray, patient: np.ndarray):
    """Group claims per patient into chunks of MAX_VISITS claims.

    Returns X [n_chunks, MAX_VISITS, n_fields] and the (chunk, pos) of every claim.
    """
    # simplification: fixed chunks, so the first claim of each chunk sees no earlier
    # history; switch to a sliding window if long histories turn out to matter
    order = pd.Series(patient).groupby(patient).cumcount().to_numpy()
    chunk = pd.factorize(patient * 1000 + order // MAX_VISITS)[0]
    pos = order % MAX_VISITS
    X = np.full((chunk.max() + 1, MAX_VISITS, len(FIELDS)), PAD, dtype=tokens.dtype)
    X[chunk, pos] = tokens
    return X, chunk, pos


def field_ids(vocab: dict, field: str) -> dict:
    """value -> token id for one field, without '?'."""
    prefix = _prefix(field) + "="
    return {t[len(prefix):]: i for t, i in vocab.items() if t.startswith(prefix) and not t.endswith("=?")}


# --------------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = (t.view(B, L, self.n_head, -1).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(D, dim=2))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x))


class Nadi(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_head: int = 4, n_layer: int = 4):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_head=n_head, n_layer=n_layer)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(MAX_VISITS * len(FIELDS), d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        """tokens [B, L] -> logits [B, L, V]; logits[:, t] predicts tokens[:, t] from the tokens before it."""
        x = F.pad(tokens[:, :-1], (1, 0), value=PAD)
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h))


def save(path, model: Nadi, vocab: dict) -> None:
    torch.save({"state": model.state_dict(), "config": model.config, "vocab": vocab}, path)


def load(path, device: str = DEVICE):
    ckpt = torch.load(path, map_location=device)
    model = Nadi(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval(), ckpt["vocab"]


# ------------------------------------------------------------------- scoring

def _batches(X: np.ndarray, batch: int):
    """Batches of similar-length sequences, with the trailing padding cut off.

    The model is causal, so padding after the last claim never changes the outputs before it.
    """
    n_visits = (X[:, :, 0] != PAD).sum(axis=1)
    order = np.argsort(n_visits, kind="stable")
    for i in range(0, len(order), batch):
        rows = order[i:i + batch]
        yield rows, X[rows, :n_visits[rows[-1]]]


@torch.no_grad()
def surprise(model: Nadi, X: np.ndarray, batch: int = 64) -> np.ndarray:
    """Surprisal (negative log-likelihood) of every token; same shape as X, 0 on padding."""
    device = next(model.parameters()).device
    out = np.zeros(X.shape, dtype=np.float32)
    for rows, part in _batches(X, batch):
        tokens = torch.from_numpy(part).long().flatten(1).to(device)
        nll = F.cross_entropy(model(tokens).transpose(1, 2), tokens, reduction="none")
        out[rows, :part.shape[1]] = nll.view(part.shape).cpu().numpy()
    return out


@torch.no_grad()
def field_probs(model: Nadi, X: np.ndarray, field: str, ids: list, batch: int = 64) -> np.ndarray:
    """Probability of each candidate token in `ids` for `field`, per claim: [n, claims, len(ids)]."""
    device = next(model.parameters()).device
    col = FIELDS.index(field)
    out = np.zeros((*X.shape[:2], len(ids)), dtype=np.float32)
    for rows, part in _batches(X, batch):
        tokens = torch.from_numpy(part).long().flatten(1).to(device)
        logits = model(tokens).view(*part.shape, -1)[:, :, col, ids]
        out[rows, :part.shape[1]] = F.softmax(logits, dim=-1).cpu().numpy()
    return out


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based checks for patterns that need no model. `df` sorted by patient and date."""
    prev = df.groupby("patient")[["discharge", "hospital", "dx", "service"]].shift()
    gap = (df.admit - prev.discharge).dt.days
    return pd.DataFrame({
        # readmission to the same hospital with the same dx, at most a day after discharge
        "split_episode": (df.service == "RI") & (prev.service == "RI") & (gap <= 1)
                         & (df.hospital == prev.hospital) & (df.dx == prev.dx),
        "duplicate": df.duplicated(["patient", "admit", "discharge", "hospital", "cbg", "dx", "severity"]),
    })


def peer_expected(inpatient: pd.DataFrame) -> pd.Series:
    """Share of claims above severity I among peer claims: same CBG, same hospital class."""
    return (inpatient.severity != "I").groupby([inpatient.cbg, inpatient.hosp_class]).transform("mean")


def facility_index(inpatient: pd.DataFrame, expected: pd.Series) -> pd.DataFrame:
    """Observed vs expected number of claims above severity I, per facility.

    A frequency table is enough here: at facility level the model's own expectation
    was no more stable between half-years than `peer_expected`.
    """
    high = inpatient.severity != "I"
    out = pd.DataFrame({"n": 1, "observed": high, "expected": expected, "var": expected * (1 - expected)})
    out = out.groupby(inpatient.hospital).sum()
    out = out[out.n >= MIN_FACILITY_CLAIMS]
    z = ((out.observed - out.expected) / np.sqrt(out["var"])).fillna(0)
    # facilities differ by more than chance, so the limits are widened by the
    # winsorised spread of z (overdispersed funnel plot, Spiegelhalter 2005)
    spread = np.sqrt(np.mean(np.clip(z, *np.quantile(z, [0.1, 0.9])) ** 2))
    return out.assign(z=z / spread, spread=spread)
