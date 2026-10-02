# SmartGuard JKN

Prioritas verifikasi klaim rawat inap JKN dengan **NADI**, model bahasa kecil yang
belajar urutan klaim yang wajar lalu menandai klaim yang janggal.

Entri BPJS Kesehatan Healthkathon 2026, kategori Efisiensi Risiko Fasilitas Kesehatan
(upcoding, pemecahan episode, klaim berulang).

## Cara kerja

1. Tiap klaim RS diubah menjadi 12 token: jenis kelamin dan usia, jarak hari dari klaim
   sebelumnya, faskes sama/beda, jenis layanan, kelas RS, diagnosis utama, kelompok
   INA-CBG, lama rawat, tiga diagnosis sekunder, dan tingkat keparahan.
2. Model bergaya GPT (4 juta parameter, dilatih dari nol) belajar menebak token
   berikutnya dari riwayat klaim peserta. Tidak butuh label fraud.
3. Peluang yang diberikan model pada nilai yang ditagihkan menjadi skor. Keparahan III
   dengan peluang 3,8% berarti janggal dan layak diperiksa lebih dulu.
4. Pola yang sudah jelas (episode dipecah, klaim ganda) ditangkap aturan biasa.

Skor adalah prioritas periksa, bukan kesimpulan. Keputusan tetap pada verifikator.

## Hasil uji

Data Sampel BPJS Kesehatan Edisi 2025 (reguler): 1.699.219 klaim RS dari 306.710 peserta.
10% peserta disisihkan untuk uji (171.510 klaim, 19.541 rawat inap). Karena data tidak
punya label fraud, klaim uji disisipi anomali buatan lalu dihitung berapa yang tertangkap
di 5% klaim paling janggal.

| Uji | NADI | Pembanding (tabel CBG x jumlah dx sekunder) |
|---|---|---|
| Menebak tingkat keparahan rawat inap | 96,2% | 84,1% |
| Keparahan dinaikkan I ke II tanpa perubahan lain | 96% tertangkap | 70% |
| Upcoding realistis: tambah 1 dx sekunder + naik I ke II | 49% tertangkap | 25% |
| Episode rawat inap dipecah dua | 98% tertangkap | aturan pasti: 100%, 7 klaim asli ikut tertanda |

Pada klaim uji asli, 373 klaim rawat inap ditagihkan di atas keparahan yang diperkirakan
model, dengan selisih kasar Rp1,25 miliar terhadap tarif lazim.

Keterbatasan: upcoding yang rapi di tingkat kode baru separuhnya tertangkap, karena
buktinya ada di rekam medis, bukan di klaim. Angka rupiah belum memakai bobot sampel,
jadi tidak boleh diekstrapolasi ke nasional.

## Menjalankan

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare.py "/path/ke/Data Sampel Reguler Edisi 2025/data"
.venv/bin/python train.py       # sekitar 25 menit di MacBook M3 8 GB
.venv/bin/python evaluate.py    # hasil uji + artifacts/scored.parquet
.venv/bin/uvicorn app:app       # buka http://localhost:8000
```

`python test_nadi.py` menjalankan cek cepat tokenisasi dan aturan.

Data Sampel BPJS Kesehatan dan turunannya tidak disertakan di repo ini. ID peserta dan
faskes diacak ulang saat pembersihan.

## Isi repo

| File | Isi |
|---|---|
| `prepare.py` | Pembersihan data sampel menjadi satu tabel klaim |
| `nadi.py` | Tokenisasi, model, penilaian, aturan pasti |
| `train.py` | Pelatihan |
| `evaluate.py` | Uji dengan anomali sisipan |
| `app.py`, `web/index.html` | Antrean prioritas dan simulator klaim |
