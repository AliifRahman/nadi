# Kartu Model NADI

NADI adalah language model kecil untuk sequence klaim JKN. Dibuat dan di-training dari nol
untuk entri BPJS Kesehatan Healthkathon 2026. Bukan fine-tuning dari model lain dan tidak
memakai API pihak lain.

| | |
|---|---|
| Versi | `2592f0ee9742` (12 karakter pertama SHA-256 file checkpoint) |
| Pembuat | Alif Rahman |
| Jenis | GPT-style transformer, decoder-only |
| Ukuran | 4,1 juta parameter: 4 layer, 4 head, dimensi 256 |
| Vocabulary | 3.422 token |
| Context | 20 klaim terakhir satu peserta, 12 token per klaim |
| Training | 10.000 step, 52 menit di MacBook M3 8 GB |

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

Set Reguler Data Sampel BPJS Kesehatan Edisi 2021 sampai 2025, disambung per peserta: 6.998.359
klaim FKRTL dari 756.440 peserta, pulang 2019 sampai 2024. Training hanya memakai klaim yang
pulang sampai 2023 dari 90% peserta (4.757.638 klaim). Klaim 2024 dan 10% peserta sisanya tidak
pernah dilihat model. ID peserta dan faskes di-shuffle ulang waktu data cleaning. Tidak ada NIK
atau nama.

Training-nya self-supervised: model belajar prediksi token berikutnya, jadi tidak butuh
label fraud. Data sampel memang tidak punya label itu.

## Input dan output

Tiap klaim jadi 12 token: sex + age group, gap hari dari klaim sebelumnya, faskes sama atau
beda, rawat jalan atau rawat inap, kelas RS, diagnosis primer, kode INA-CBG, LOS, tiga
diagnosis sekunder, severity level.

Output: probabilitas tiap token. Di aplikasi ditampilkan per field, dan untuk severity
sebagai probabilitas level I, II, III.

## Hasil evaluasi

Test set: 195.544 klaim rawat inap 2024, tahun yang tidak pernah dilihat model. Karena tidak
ada label fraud, klaim 2024 milik peserta yang tidak ikut training dimodifikasi dengan pattern
yang diketahui, lalu dihitung recall di top 5% score.

| Test | NADI | Baseline |
|---|---|---|
| Accuracy prediksi severity rawat inap | 96,4% | 83,0% |
| Severity dinaikkan I ke II | recall 97% | 71% |
| Upcoding realistis: +1 dx sekunder, severity I ke II | recall 50% | 18% |
| Episode rawat inap dipecah jadi dua klaim | recall 98% | rule: 100% |
| LOS diperpanjang 3-5 hari | recall 47% | 44% |

Untuk menemukan separuh upcoding realistis, cukup memeriksa 5,2% sampai 6,2% klaim rawat inap
menurut urutan NADI, sedangkan urutan acak perlu 50%. Pada seluruh urutan, tabel frekuensi
sedikit lebih baik (AUC 0,888 berbanding 0,832), jadi keunggulan NADI ada di puncak queue.

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
