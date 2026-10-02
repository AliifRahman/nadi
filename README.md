# SmartGuard JKN

Prioritas review klaim rawat inap JKN pakai **NADI**, language model kecil yang
di-training di sequence klaim peserta. Klaim yang value-nya tidak sesuai prediksi model
naik ke atas queue verifikator.

Entri BPJS Kesehatan Healthkathon 2026, kategori Efisiensi Risiko Fasilitas Kesehatan.
Fokus: upcoding, pemecahan episode, repeat billing.

## Cara kerja

1. Tiap klaim FKRTL dijadikan 12 token: sex + age group, gap hari dari klaim sebelumnya,
   faskes sama/beda, RJTL/RITL, kelas RS, diagnosis primer, kode INA-CBG, LOS,
   tiga diagnosis sekunder, severity level.
2. Model GPT-style (4M parameter, training from scratch) belajar prediksi token
   berikutnya dari riwayat klaim peserta. Self-supervised, jadi tidak butuh label fraud.
3. Probabilitas yang diberikan model ke value yang diklaim dipakai sebagai score.
   Contoh: severity III dengan probabilitas 3,8% masuk prioritas review.
4. Pattern yang sudah jelas (pemecahan episode, repeat billing) cukup pakai rule.

Score ini prioritas review, bukan keputusan. Keputusan tetap di verifikator.

## Hasil evaluasi

Data Sampel BPJS Kesehatan Edisi 2025 (reguler): 1.699.219 klaim FKRTL, 306.710 peserta.
Split per peserta, 10% jadi test set (171.510 klaim, 19.541 rawat inap).

Data sampel tidak punya label fraud. Jadi evaluasinya pakai injected anomaly: klaim di
test set dimodifikasi dengan pattern yang diketahui, lalu dihitung recall di top 5% klaim
dengan score tertinggi.

| Test | NADI | Baseline (tabel CBG x jumlah dx sekunder) |
|---|---|---|
| Accuracy prediksi severity rawat inap | 96,2% | 84,1% |
| Severity dinaikkan I ke II, field lain tetap | recall 96% | 70% |
| Upcoding realistis: +1 dx sekunder, severity I ke II | recall 49% | 25% |
| Episode rawat inap dipecah jadi dua klaim | recall 98% | rule: recall 100%, 7 false positive |

Di test set asli (tanpa injection) ada 373 klaim rawat inap dengan severity di atas
prediksi model. Selisihnya terhadap tarif median sekitar Rp1,25 miliar (estimasi kasar).

Limitasi:

- Upcoding yang konsisten di level coding baru separuhnya terdeteksi. Bukti klinisnya
  ada di rekam medis, bukan di data klaim.
- Angka rupiah belum pakai sampling weight, jadi belum bisa diekstrapolasi ke nasional.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare.py "/path/ke/Data Sampel Reguler Edisi 2025/data"
.venv/bin/python train.py       # +- 25 menit di MacBook M3 8 GB
.venv/bin/python evaluate.py    # hasil evaluasi + artifacts/scored.parquet
.venv/bin/uvicorn app:app       # buka http://localhost:8000
```

`python test_nadi.py` untuk quick check tokenizer dan rule.

Data sampel BPJS Kesehatan dan turunannya tidak ikut di repo. ID peserta dan faskes
di-shuffle ulang waktu data cleaning.

## Struktur

| File | Isi |
|---|---|
| `prepare.py` | Data cleaning, dari `.dta` ke satu tabel klaim |
| `nadi.py` | Tokenizer, model, scoring, rule-based check |
| `train.py` | Training |
| `evaluate.py` | Evaluasi dengan injected anomaly |
| `app.py`, `web/index.html` | Priority queue dan simulator klaim |
