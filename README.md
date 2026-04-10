# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Bussines Understanding

Jaya Jaya Institut adalah institusi pendidikan perguruan yang berdiri sejak tahun 2000. Meskipun telah banyak mencetak lulusan berprestasi, institusi ini menghadapi masalah serius berupa **tingginya angka dropout mahasiswa**. Dropout yang tidak terdeteksi dini berdampak pada reputasi institusi dan kegagalan dalam memenuhi misi pendidikannya.

---

## Permasalahan Bisnis

1. Faktor apa yang paling berkontribusi terhadap keputusan mahasiswa untuk dropout?
2. Bagaimana cara mendeteksi secara dini mahasiswa yang berpotensi dropout?

---

## Cakupan Proyek

- Exploratory Data Analysis (EDA) terhadap dataset performa mahasiswa
- Business Dashboard untuk monitoring performa mahasiswa
- Model Machine Learning (Random Forest) untuk prediksi dropout
- Prototype aplikasi Streamlit yang dapat diakses online

---

## Persiapan

Sumber Data : https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv
Set Up Environment :
1. Buka terminal atau powershell
2. Jalankan perintah berikut.
   conda create --name main-ds python=3.9
3. Aktifkan virtual environment dengan menjalankan perintah berikut.
   conda activate main-ds
4. Jalankan perintah berikut pada Terminal/Command Prompt/PowerShell guna memanggil (pull) Docker image untuk menjalankan Metabase.
   docker pull metabase/metabase:v0.46.4
5. Apabila proses pembuatan docker image telah selesai, Anda dapat menjalankan image tersebut menggunakan perintah berikut.
   docker run -p 3000:3000 --name metabase metabase/metabase
---

## Business Dashboard

**Tool:** Metabase (lokal via Docker)

**Akses Metabase:**
- Email: `root@mail.com`
- Password: `root123`

Dashboard menampilkan:
1. Proporsi status mahasiswa (Graduate / Dropout / Enrolled)
2. Dropout rate berdasarkan status pembayaran SPP
3. Dropout rate berdasarkan penerima beasiswa
4. Distribusi nilai semester 1 & 2 by status
5. Dropout rate berdasarkan kelompok usia
6. Top faktor risiko dropout

---

## Menjalankan Sistem Machine Learning

**Link Streamlit App:** [https://your-app.streamlit.app](https://your-app.streamlit.app)  
_(Ganti dengan link Streamlit Community Cloud kamu)_

### Cara Menjalankan Lokal

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan aplikasi
streamlit run app.py
```

Buka browser ke `http://localhost:8501`

### Cara Deploy ke Streamlit Community Cloud

1. Push folder submission ke GitHub (pastikan `model/`, `app.py`, `requirements.txt`, `data.csv` ikut)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan akun GitHub
4. Klik **New app** → pilih repository → set Main file path: `app.py`
5. Klik **Deploy** → tunggu beberapa menit
6. Copy link yang diberikan dan tempel di README ini

### Fitur Aplikasi

- **Prediksi Individu**: Input data satu mahasiswa, dapatkan probabilitas dropout dan rekomendasi tindakan
- **Prediksi Batch**: Upload CSV berisi banyak mahasiswa, dapatkan hasil prediksi sekaligus dalam format downloadable

---

## Conclusion

Berdasarkan analisis data dan model machine learning yang dikembangkan:

1. **Dropout rate sebesar ~39%** dari mahasiswa yang memiliki status final (Graduate/Dropout).

2. **Faktor utama penyebab dropout:**
   - **Performa akademik** — Jumlah unit kurikuler yang lulus di semester 1 & 2 adalah prediktor terkuat. Mahasiswa dengan ≤2 unit lulus di semester 1 memiliki risiko dropout sangat tinggi.
   - **Nilai rata-rata** semester 1 & 2 yang rendah berkorelasi kuat dengan dropout.
   - **Status SPP** — Mahasiswa yang menunggak pembayaran SPP memiliki dropout rate jauh lebih tinggi.
   - **Status beasiswa** — Penerima beasiswa cenderung lebih bertahan.
   - **Usia mendaftar** — Mahasiswa yang mendaftar di usia lebih tua (>26 tahun) lebih berisiko dropout.
   - **Status debitur** — Mahasiswa dengan hutang akademik lebih rentan dropout.

3. **Model Random Forest** mencapai **ROC-AUC 0.9695** dan **akurasi 92%**, sangat handal untuk deteksi dini.

---

## Rekomendasi Action Items

| # | Action Item | Target | Prioritas |
|---|-------------|--------|-----------|
| 1 | **Monitoring akademik semester 1** – Tandai mahasiswa dengan <3 unit lulus di semester 1 untuk intervensi segera | Dosen wali & akademik | Tinggi |
| 2 | **Program beasiswa tepat sasaran** – Prioritaskan mahasiswa berpotensi dropout dengan kesulitan finansial | Bagian kemahasiswaan | Tinggi |
| 3 | **Fleksibilitas pembayaran SPP** – Tawarkan cicilan atau penangguhan bagi mahasiswa menunggak sebelum mereka dropout | Bagian keuangan | Tinggi |
| 4 | **Sistem peringatan dini** – Gunakan prototype Streamlit setiap awal semester untuk prediksi risiko seluruh mahasiswa | IT & Akademik | Sedang |
| 5 | **Konseling wajib** – Wajibkan sesi konseling bagi mahasiswa dengan nilai rata-rata <10 di semester berjalan | BK / Konselor | Sedang |
| 6 | **Program mentoring peer** – Pasangkan mahasiswa berisiko dengan mahasiswa senior berprestasi | Unit kemahasiswaan | Rendah |
| 7 | **Survey keterlibatan** – Lakukan pulse survey reguler untuk mendeteksi penurunan motivasi sebelum terlambat | Dosen wali | Rendah |
