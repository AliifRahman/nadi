<img src="docs/nadi.svg" width="64" alt="Logo NADI">

# SmartGuard JKN, ditenagai NADI

Penjelasan dan demo yang bisa dicoba ada di `docs/` (untuk GitHub Pages:
https://aliifrahman.github.io/nadi/). Di aplikasi lokal halaman yang sama tampil di `/tentang/`.

**NADI adalah AI buatan sendiri yang membaca riwayat klaim pasien dan memberi tahu
verifikator BPJS klaim mana yang perlu dicek lebih dulu.** SmartGuard JKN adalah aplikasinya.

Keyboard HP bisa menebak kata berikutnya karena sudah membaca jutaan kalimat. NADI menebak
isi klaim berikutnya karena sudah membaca 1,7 juta klaim. Kalau tagihan RS jauh dari
tebakan, klaim itu naik ke urutan atas. NADI tidak memutuskan dan tidak menuduh. Keputusan
tetap di verifikator.

Contoh dari data: seorang pasien dirawat 4 hari untuk prosedur sendi panggul dan RS menagih
severity III. NADI memperkirakan severity II (81 dari 100 klaim serupa). Selisihnya sekitar
Rp16,7 juta, jadi klaim ini naik ke atas queue.

NADI di-training dari nol: 4 juta parameter, 23 menit di satu laptop, tanpa API pihak lain.
Detailnya di [MODEL_CARD.md](MODEL_CARD.md). Penjelasan untuk pembaca umum ada di website di
atas (sumbernya `docs/`, di aplikasi tampil di `/tentang/`).

## Ringkasan teknis

NADI adalah language model kecil yang di-training di sequence klaim peserta. Klaim yang
value-nya tidak sesuai prediksi model naik ke atas queue verifikator. Di atasnya ada profil
per faskes, keputusan verifikator dengan audit log, dan bridging FHIR R4 dua arah.

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
5. Level faskes: share klaim severity II/III tiap faskes dibandingkan dengan faskes
   sejenis (CBG dan kelas RS sama), ditampilkan sebagai funnel plot. Ini juga cukup
   pakai tabel frekuensi, bukan model (lihat limitasi).
6. Verifikator mencatat keputusan (layak / pending / tidak layak) per klaim. Keputusan
   masuk audit log, dan nantinya jadi label yang sekarang belum ada.

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
| LOS diperpanjang 3-5 hari | recall 46% | 44% (tabel CBG x severity) |

Level faskes (seluruh sample): 1.906 faskes punya minimal 30 klaim rawat inap, 33 di
antaranya di atas batas 3σ. Indeksnya stabil: korelasi Spearman antara semester 1 dan
semester 2 adalah 0,67 (1.120 faskes yang punya cukup klaim di kedua semester). Jadi
yang terukur adalah pola faskes, bukan noise.

Di test set asli (tanpa injection) ada 373 klaim rawat inap dengan severity di atas
prediksi model. Selisihnya terhadap tarif median sekitar Rp1,25 miliar, atau 1,07% dari
biaya rawat inap. Dengan sampling weight, skalanya sekitar Rp1,0 triliun per tahun
(estimasi kasar).

Rule di seluruh sample: pemecahan episode kena 89 klaim (weighted 5.892 klaim, sekitar
Rp36,5 miliar per tahun), repeat billing 26 klaim.

Web app menampilkan seluruh sample, bukan hanya test set. Score in-sample dan held-out
hampir sama (accuracy severity 96,5% vs 96,2%). Semua angka di tabel tetap dari test set.

Limitasi:

- Upcoding yang konsisten di level coding baru separuhnya terdeteksi. Bukti klinisnya
  ada di rekam medis, bukan di data klaim.
- Untuk prolonged LOS, NADI hampir sama dengan baseline (46% vs 44%).
- Angka rupiah adalah gap terhadap tarif median berdasarkan prediksi model, bukan
  temuan audit.
- Di level faskes NADI tidak lebih baik dari tabel. Expected severity dari NADI (lewat
  sampling dx sekunder) dicoba dan hasilnya tidak lebih stabil antar-semester daripada
  tabel CBG x kelas RS (Spearman 0,56 vs 0,63 di percobaan yang sama), jadi yang dipakai
  tabel.
- Rasio faskes adalah pembanding, bukan bukti. Case-mix yang tidak tertangkap CBG dan
  kelas RS bisa menjelaskan selisihnya.
- Data sampel tidak punya kolom ID faskes. ID faskes diturunkan dari prefix ID kunjungan.
  Ini inferensi, bukan dokumentasi resmi.

## Keputusan verifikator dan audit log

Keputusan disimpan di `artifacts/reviews.jsonl`, append-only. Tiap baris menyimpan hash
SHA-256 dari baris sebelumnya, jadi baris lama yang diubah langsung kelihatan (status
hash chain tampil di footer app). Yang ikut dicatat: waktu, nama verifikator, keputusan,
catatan, dan probabilitas dari model saat keputusan dibuat.

Prototype ini belum punya login, jadi nama verifikator masih diketik manual. Hash chain
juga tidak mendeteksi baris terakhir yang dihapus.

## Bridging FHIR dan SATUSEHAT

| Endpoint | Isi |
|---|---|
| `GET /api/claim/{row}/fhir` | Hasil NADI sebagai Bundle: RiskAssessment + Provenance + Device |
| `GET /api/claim/{row}/fhir/claim` | Klaim dan riwayatnya sebagai Bundle: Patient, Organization, Encounter, Claim |
| `POST /api/fhir/assess` | Bundle klaim masuk, Bundle hasil keluar. Dengan `?send=true` hasilnya juga dikirim |
| `GET /api/claim/{row}/satusehat` | Pratinjau request ke SATUSEHAT. Tidak mengirim apa pun |

Posisi NADI di alur klaim SATUSEHAT: di sisi BPJS, setelah `Claim` masuk dan sebelum
`ClaimResponse` hasil verifikasi keluar.

Tujuan pengiriman diatur lewat environment variable, bukan lewat request:

```bash
# SATUSEHAT (OAuth2 client credentials). Default host: sandbox.
export SATUSEHAT_CLIENT_ID=... SATUSEHAT_CLIENT_SECRET=...
export SATUSEHAT_HOST=https://api-satusehat.kemkes.go.id   # opsional, untuk produksi

# atau server FHIR R4 lain
export FHIR_BASE=http://localhost:8080/fhir
```

Tanpa keduanya app jalan di mode simulasi: request ditampilkan, tidak dikirim. Ke
SATUSEHAT yang dikirim hanya `RiskAssessment`, karena `Provenance` dan `Device` belum
didukung di sana. Ke server FHIR lain, klaim dan hasilnya dikirim sebagai satu transaction.
Resource yang punya identifier tidak dibuat dua kali kalau dikirim ulang.

Klaim dari data sampel BPJS hanya bisa dipratinjau. Pengiriman sungguhan hanya untuk
Bundle yang masuk lewat `POST /api/fhir/assess`. Jangan arahkan `FHIR_BASE` ke server
publik untuk data yang bukan sintetis.

Status pengujian per 8 Oktober 2026:

- Bundle hasil dan Bundle klaim lolos HL7 FHIR Validator (R4 + IG AI Transparency): 0 error.
- Bolak-balik FHIR pada 300 klaim acak: semua input model identik, probabilitas sama.
- Uji hidup ke server uji publik `hapi.fhir.org/baseR4` dengan klaim sintetis: tersimpan,
  reference saling terhubung, bisa dibaca balik, kirim ulang tidak membuat duplikat.
- Konektor SATUSEHAT mengikuti dokumentasi resmi (token di `/oauth2/v1/accesstoken`,
  data di `/fhir-r4/v1/{Resource}`) dan diuji dengan server tiruan lokal. **Belum diuji di
  sandbox SATUSEHAT**: kredensialnya hanya diberikan Kemenkes ke faskes, dinas kesehatan,
  dan mitra RME lewat pengajuan.
- Modul klaim BPJS di SATUSEHAT sendiri masih sandbox-only, dan playbook publiknya baru
  sampai `Invoice`. Jadi struktur `Claim` di sini mengikuti field wajib yang
  didokumentasikan; kode INA-CBG ditaruh di `diagnosis.packageCode`.

Isi Bundle hasil:

- `RiskAssessment`: probabilitas tiap severity level, dengan security label `AIAST`.
- `Provenance`: NADI sebagai author, verifikator sebagai verifier (kalau sudah ada
  keputusan), klaim sebagai source.
- `Device`: identitas model. Version diambil dari hash file checkpoint.

Provenance dan Device mengikuti profil AI-Provenance dan AI-Device dari implementation
guide HL7 [AI Transparency on FHIR](https://build.fhir.org/ig/HL7/aitransparency-ig/)
(STU1 ballot, basis R4).

Per 5 Oktober 2026, dokumentasi SATUSEHAT mencantumkan `RiskAssessment`, `Claim`, dan
`ClaimResponse`, tetapi belum `Provenance` dan `Device`. Jadi score-nya sudah bisa
dibawa SATUSEHAT, sedangkan catatan siapa yang menghasilkan dan siapa yang memverifikasi
belum.

Identifier di Bundle masih placeholder. Di produksi isinya nomor SEP dan IHS patient ID.
URL code system untuk INA-CBG dan kelas RS juga placeholder.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare.py "/path/ke/Data Sampel Reguler Edisi 2025/data"
.venv/bin/python train.py       # +- 25 menit di MacBook M3 8 GB
.venv/bin/python evaluate.py    # +- 5 menit, hasil evaluasi + artifacts/scored.parquet
.venv/bin/uvicorn app:app       # buka http://localhost:8000
```

`python test_nadi.py` untuk quick check tokenizer, rule, indeks faskes, dan bridge FHIR.

Data sampel BPJS Kesehatan dan turunannya tidak ikut di repo. ID peserta dan faskes
di-shuffle ulang waktu data cleaning.

## Struktur

| File | Isi |
|---|---|
| `prepare.py` | Data cleaning, dari `.dta` ke satu tabel klaim |
| `nadi.py` | Tokenizer, model, scoring, rule-based check, indeks faskes |
| `train.py` | Training |
| `evaluate.py` | Evaluasi dengan injected anomaly, score untuk seluruh sample |
| `bridge.py` | Klaim dan hasil NADI sebagai FHIR R4 (dua arah), client SATUSEHAT |
| `app.py`, `web/` | Priority queue, profil faskes, simulator klaim, keputusan verifikator |
| `docs/`, `make_demo.py` | Website publik (GitHub Pages) dan jawaban demo untuk klaim karangan |
| `MODEL_CARD.md` | Kartu model NADI |
