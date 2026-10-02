# SmartGuard JKN

Prioritas verifikasi klaim JKN dengan **NADI**, model bahasa kecil yang belajar
urutan klaim yang wajar lalu menandai klaim yang janggal.

Entri BPJS Kesehatan Healthkathon 2026, kategori Efisiensi Risiko Fasilitas Kesehatan.

## Menjalankan

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare.py "/path/ke/Data Sampel Reguler Edisi 2025/data"
```

Data Sampel BPJS Kesehatan tidak disertakan di repo ini.
