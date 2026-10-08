# Kartu Model NADI

NADI adalah language model kecil untuk sequence klaim JKN. Dibuat dan di-training dari nol
untuk entri BPJS Kesehatan Healthkathon 2026. Bukan fine-tuning dari model lain dan tidak
memakai API pihak lain.

| | |
|---|---|
| Versi | `18e4ded962cb` (12 karakter pertama SHA-256 file checkpoint) |
| Pembuat | Alif Rahman |
| Jenis | GPT-style transformer, decoder-only |
| Ukuran | 4,0 juta parameter: 4 layer, 4 head, dimensi 256 |
| Vocabulary | 2.932 token |
| Context | 20 klaim terakhir satu peserta, 12 token per klaim |
| Training | 5.000 step, 23 menit di MacBook M3 8 GB |

## Untuk apa

Membantu verifikator BPJS Kesehatan memilih klaim rawat inap mana yang dicek lebih dulu.
NADI memberi probabilitas untuk tiap field klaim (severity level, LOS, gap dari klaim
sebelumnya, faskes). Value dengan probabilitas rendah naik ke atas queue.

## Bukan untuk apa

- Bukan untuk memutuskan klaim layak atau tidak layak. Keputusan tetap di verifikator.
- Bukan bukti fraud. Probabilitas rendah berarti "tidak biasa", bukan "salah".
- Bukan untuk menilai dokter atau pasien secara individu.
- Bukan untuk keputusan klinis.

## Data training

Data Sampel BPJS Kesehatan Edisi 2025 (reguler): 1.699.219 klaim FKRTL dari 306.710 peserta.
Split per peserta: 90% untuk training, 10% jadi held-out test set. ID peserta dan faskes
di-shuffle ulang waktu data cleaning. Tidak ada NIK atau nama.

Training-nya self-supervised: model belajar prediksi token berikutnya, jadi tidak butuh
label fraud. Data sampel memang tidak punya label itu.

## Input dan output

Tiap klaim jadi 12 token: sex + age group, gap hari dari klaim sebelumnya, faskes sama atau
beda, rawat jalan atau rawat inap, kelas RS, diagnosis primer, kode INA-CBG, LOS, tiga
diagnosis sekunder, severity level.

Output: probabilitas tiap token. Di aplikasi ditampilkan per field, dan untuk severity
sebagai probabilitas level I, II, III.

## Hasil evaluasi

Test set: 171.510 klaim (19.541 rawat inap) dari peserta yang tidak ikut training. Karena
tidak ada label fraud, klaim test dimodifikasi dengan pattern yang diketahui, lalu dihitung
recall di top 5% score.

| Test | NADI | Baseline |
|---|---|---|
| Accuracy prediksi severity rawat inap | 96,2% | 84,1% |
| Severity dinaikkan I ke II | recall 96% | 70% |
| Upcoding realistis: +1 dx sekunder, severity I ke II | recall 49% | 25% |
| Episode rawat inap dipecah jadi dua klaim | recall 98% | rule: 100% |
| LOS diperpanjang 3-5 hari | recall 46% | 44% |

## Limitasi

- Upcoding yang konsisten di level coding baru separuhnya terdeteksi.
- Untuk prolonged LOS, NADI hampir sama dengan baseline.
- Belum diuji oleh verifikator dan belum ada pilot.
- Model belajar dari pola historis. Kalau ada kebiasaan coding yang keliru tapi umum, model
  menganggapnya wajar.
- Kelas RS jadi input, jadi RS rujukan dibandingkan dengan pola RS rujukan.
- Di level faskes NADI tidak lebih baik dari tabel frekuensi, jadi profil faskes tidak
  memakai NADI.

## Tata kelola

- Jalan di CPU, on-prem. Data klaim tidak dikirim ke layanan luar.
- Setiap output FHIR diberi security label `AIAST` dan `Provenance` yang menunjuk versi
  model ini.
- Keputusan verifikator dicatat di audit log append-only dengan hash chain.
