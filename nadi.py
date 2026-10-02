"""NADI: model bahasa kecil untuk urutan klaim JKN.

Tiap klaim diubah menjadi deretan token tetap (FIELDS). Model bergaya GPT belajar
menebak token berikutnya dari riwayat klaim peserta. Makin kecil peluang yang
diberikan model pada nilai yang ditagihkan, makin janggal klaim itu.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# urutan penting: keparahan paling akhir supaya ditebak dari semua field lain
FIELDS = ["demo", "gap", "fac", "svc", "cls", "dx", "cbg", "los", "sec1", "sec2", "sec3", "sev"]
MAX_VISITS = 20  # klaim per potongan urutan
MIN_COUNT = 5    # nilai yang lebih jarang dari ini dipetakan ke "?" (tak dikenal)
PAD = 0

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def is_test(patient) -> np.ndarray:
    """10% peserta disisihkan untuk uji; pembagian per peserta, bukan per klaim."""
    return np.asarray(patient) % 10 == 0


# ---------------------------------------------------------------- tokenisasi

def _bucket(values: pd.Series, edges: list, labels: list) -> pd.Series:
    return pd.cut(values, [-np.inf, *edges, np.inf], labels=labels).astype(str)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Klaim bersih (urut per peserta dan tanggal) -> nilai kategorikal per field."""
    prev = df.groupby("patient")[["discharge", "hospital"]].shift()
    first = prev.discharge.isna()
    gap = (df.admit - prev.discharge).dt.days
    age = _bucket(df.age, [-1, 4, 17, 39, 59], ["?", "0-4", "5-17", "18-39", "40-59", "60+"])
    return pd.DataFrame({
        "demo": df.sex + ":" + age,
        "gap": _bucket(gap, [0, 1, 7, 30, 90], ["0", "1", "2-7", "8-30", "31-90", "90+"]).mask(first, "awal"),
        "fac": pd.Series(np.where(df.hospital == prev.hospital, "sama", "beda"), df.index).mask(first, "awal"),
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
    return field.rstrip("123")  # sec1..sec3 berbagi kosakata yang sama


def build_vocab(feats: pd.DataFrame) -> dict:
    vocab = {"<pad>": PAD}
    for field in FIELDS:
        vocab.setdefault(f"{_prefix(field)}=?", len(vocab))
        counts = feats[field].value_counts()
        for value in counts[counts >= MIN_COUNT].index:
            vocab.setdefault(f"{_prefix(field)}={value}", len(vocab))
    return vocab


def encode(feats: pd.DataFrame, vocab: dict) -> np.ndarray:
    """[n_klaim, n_field] id token; nilai di luar kosakata menjadi '?'."""
    columns = [
        (_prefix(f) + "=" + feats[f]).map(vocab).fillna(vocab[f"{_prefix(f)}=?"])
        for f in FIELDS
    ]
    return np.stack(columns, axis=1).astype(np.int64)


def to_sequences(tokens: np.ndarray, patient: np.ndarray):
    """Susun klaim per peserta menjadi potongan MAX_VISITS klaim.

    Mengembalikan X [n_potongan, MAX_VISITS, n_field] serta (chunk, pos) tiap klaim.
    """
    # shortcut: potongan tetap, klaim pertama tiap potongan tidak melihat riwayat
    # sebelumnya; ganti jendela geser jika riwayat panjang terbukti penting
    order = pd.Series(patient).groupby(patient).cumcount().to_numpy()
    chunk = pd.factorize(patient * 1000 + order // MAX_VISITS)[0]
    pos = order % MAX_VISITS
    X = np.full((chunk.max() + 1, MAX_VISITS, len(FIELDS)), PAD, dtype=np.int64)
    X[chunk, pos] = tokens
    return X, chunk, pos


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
        """tokens [B, L] -> logits [B, L, V]; logits[:, t] menebak tokens[:, t] dari token sebelumnya."""
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


@torch.no_grad()
def surprise(model: Nadi, X: np.ndarray, batch: int = 64) -> np.ndarray:
    """Kejutan (negatif log-peluang) tiap token; bentuk sama dengan X."""
    device = next(model.parameters()).device
    out = np.zeros(X.shape, dtype=np.float32)
    for i in range(0, len(X), batch):
        tokens = torch.from_numpy(X[i:i + batch]).flatten(1).to(device)
        nll = F.cross_entropy(model(tokens).transpose(1, 2), tokens, reduction="none")
        out[i:i + batch] = nll.view(-1, *X.shape[1:]).cpu().numpy()
    return out
