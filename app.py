import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jaya Jaya Institut – Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# ── Load Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(base_dir, "rf_dropout.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    meta_path   = os.path.join(base_dir, "model_meta.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, scaler, meta

model, scaler, meta = load_model()
feature_cols = meta.get("feature_cols", [])

# ── Lookup dictionaries for categorical features ───────────────────────
MARITAL_STATUS = {
    "Single (Lajang)": 1,
    "Married (Menikah)": 2,
    "Widower (Duda/Janda)": 3,
    "Divorced (Cerai)": 4,
    "Facto Union (Hidup Bersama)": 5,
    "Legally Separated (Pisah Secara Hukum)": 6,
}

APPLICATION_MODE = {
    "1st Phase – General Contingent (Jalur Umum Fase 1)": 1,
    "Ordinance No. 612/93": 2,
    "1st Phase – Special Contingent (Azores Island)": 5,
    "Holders of Other Higher Courses": 7,
    "Ordinance No. 854-B/99": 10,
    "International Student (Undergraduate)": 15,
    "1st Phase – Special Contingent (Madeira Island)": 16,
    "2nd Phase – General Contingent (Jalur Umum Fase 2)": 17,
    "3rd Phase – General Contingent (Jalur Umum Fase 3)": 18,
    "Ordinance No. 533-A/99 (Different Plan)": 26,
    "Ordinance No. 533-A/99 (Other Institution)": 27,
    "Over 23 Years Old (Usia di Atas 23 Tahun)": 39,
    "Transfer (Pindahan)": 42,
    "Change of Course (Pindah Jurusan)": 43,
    "Technological Specialization Diploma Holders": 44,
    "Change of Institution/Course": 51,
    "Short Cycle Diploma Holders": 53,
    "Change of Institution/Course (International)": 57,
}

COURSE = {
    "Biofuel Production Technologies": 33,
    "Animation and Multimedia Design": 171,
    "Social Service (Evening)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equinculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670,
    "Journalism and Communication": 9773,
    "Basic Education": 9853,
    "Management (Evening)": 9991,
}

PREVIOUS_QUALIFICATION = {
    "Secondary Education / SMA Sederajat": 1,
    "Higher Education – Bachelor's Degree (S1)": 2,
    "Higher Education – Degree": 3,
    "Higher Education – Master's (S2)": 4,
    "Higher Education – Doctorate (S3)": 5,
    "Frequency of Higher Education (Tidak Selesai Perguruan Tinggi)": 6,
    "12th Year – Not Completed (Kelas 12 Tidak Selesai)": 9,
    "11th Year – Not Completed (Kelas 11 Tidak Selesai)": 10,
    "Other – 11th Year": 12,
    "10th Year (Kelas 10)": 14,
    "10th Year – Not Completed (Kelas 10 Tidak Selesai)": 15,
    "Basic Education 3rd Cycle / SMP (9th Year)": 19,
    "Basic Education 2nd Cycle / SD (6th Year)": 38,
    "Technological Specialization Course": 39,
    "Higher Education – Degree (1st Cycle)": 40,
    "Professional Higher Technical Course": 42,
    "Higher Education – Master (2nd Cycle)": 43,
}

NATIONALITY = {
    "Portuguese (Portugal)": 1,
    "German (Jerman)": 2,
    "Spanish (Spanyol)": 6,
    "Italian (Italia)": 11,
    "Dutch (Belanda)": 13,
    "English (Inggris)": 14,
    "Lithuanian (Lituania)": 17,
    "Angolan (Angola)": 21,
    "Cape Verdean (Tanjung Verde)": 22,
    "Guinean (Guinea)": 24,
    "Mozambican (Mozambik)": 25,
    "Santomean (São Tomé)": 26,
    "Turkish (Turki)": 32,
    "Brazilian (Brasil)": 41,
    "Romanian (Rumania)": 62,
    "Moldovan": 100,
    "Mexican (Meksiko)": 101,
    "Ukrainian (Ukraina)": 103,
    "Russian (Rusia)": 105,
    "Cuban (Kuba)": 108,
    "Colombian (Kolombia)": 109,
}

PARENT_QUALIFICATION = {
    "Secondary Education / SMA Sederajat": 1,
    "Higher Education – Bachelor's Degree (S1)": 2,
    "Higher Education – Degree": 3,
    "Higher Education – Master's (S2)": 4,
    "Higher Education – Doctorate (S3)": 5,
    "Frequency of Higher Education (Kuliah Tidak Selesai)": 6,
    "12th Year – Not Completed": 9,
    "11th Year – Not Completed": 10,
    "7th Year (Old)": 11,
    "Other – 11th Year": 12,
    "2nd Year Complementary High School": 14,
    "10th Year": 18,
    "General Commerce Course": 19,
    "Basic Education 3rd Cycle / SMP (9th Year)": 22,
    "Complementary High School Course": 26,
    "Technical-Professional Course": 27,
    "Complementary High School – Not Concluded": 29,
    "7th Year of Schooling": 30,
    "2nd Cycle General High School": 34,
    "9th Year – Not Completed": 35,
    "8th Year": 36,
    "General Administration and Commerce Course": 37,
    "Supplementary Accounting and Administration": 38,
    "Unknown / Not Provided (Tidak Diketahui)": 39,
    "Cannot Read or Write (Buta Huruf)": 40,
    "Can Read Without 4th Year (Bisa Baca Tanpa Lulus SD)": 41,
    "Basic Education 1st Cycle / SD (4th–5th Year)": 42,
    "Basic Education 2nd Cycle / SMP (6th–8th Year)": 43,
    "Technological Specialization Course": 44,
}

PARENT_OCCUPATION = {
    "Student (Pelajar/Mahasiswa)": 0,
    "Legislative / Executive Power (Pejabat/DPR/Eksekutif)": 1,
    "Intellectual & Scientific Activities (Ilmuwan/Profesional)": 2,
    "Intermediate Level Technicians (Teknisi Menengah)": 3,
    "Administrative Staff (Staf Administrasi/Pegawai Kantor)": 4,
    "Personal Services / Security (Jasa Perorangan/Keamanan)": 5,
    "Farmers / Skilled Agricultural Workers (Petani)": 6,
    "Skilled Industry / Construction Workers (Pekerja Industri Terampil)": 7,
    "Machine Operators / Assemblers (Operator Mesin)": 8,
    "Unskilled Workers (Buruh Tidak Terampil)": 9,
    "Armed Forces / Military (TNI-Polri/Militer)": 10,
    "Other Situation (Situasi Lain)": 90,
    "Unknown / Not Provided (Tidak Diketahui)": 99,
    "Armed Forces Officers (Perwira Militer)": 122,
    "Armed Forces Sergeants (Bintara)": 123,
    "Other Armed Forces Personnel": 125,
    "Directors / Executive Managers (Direktur/Manajer)": 131,
    "Administrative & Commercial Directors": 132,
    "Production & Specialized Services Directors": 134,
    "Hotel / Catering / Trade Directors": 141,
    "Physical & Engineering Science Specialists": 143,
    "Health Professionals (Dokter/Tenaga Kesehatan)": 144,
    "Teachers / Educators (Guru/Dosen)": 151,
    "Finance / Admin / Commerce Specialists": 152,
    "ICT Specialists (Spesialis IT/Teknologi Informasi)": 153,
    "Science & Engineering Technicians": 171,
    "Health Technicians (Teknisi Kesehatan)": 173,
    "Legal / Social / Cultural Technicians": 175,
    "Office Workers / Secretaries (Sekretaris/Pegawai Kantor)": 191,
    "Data / Finance / Stats Processing Operators": 192,
    "Numerical Clerks (Juru Hitung/Kasir)": 193,
    "Mail & Other Clerks (Petugas Pos/Loket)": 194,
}

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0; font-size: 2rem;'>🎓 Student Dropout Predictor</h1>
    <p style='color: #aaaacc; margin: 0.5rem 0 0;'>
        Jaya Jaya Institut – Sistem Deteksi Dini Mahasiswa Berisiko Dropout
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Info ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Tentang Model")
    st.info(f"""
    **Algoritma:** Random Forest  
    **ROC-AUC:** {meta.get('roc_auc', 'N/A')}  
    **Akurasi:** ~92%  
    **Dataset:** 3.630 mahasiswa
    """)
    st.markdown("### 🔑 Top Fitur Prediksi")
    for i, feat in enumerate(meta.get('top_features', [])[:5], 1):
        st.markdown(f"{i}. `{feat}`")

# ── Tabs ───────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Prediksi Individu", "📂 Prediksi Batch (CSV)"])

# ── Tab 1: Individual Prediction ───────────────────────────────────────
with tab1:
    st.markdown("### Input Data Mahasiswa")
    st.caption("Isi seluruh data mahasiswa di bawah ini untuk mendapatkan prediksi risiko dropout.")

    # ── Section 1: Data Pribadi ────────────────────────────────────────
    with st.expander("👤 Data Pribadi Mahasiswa", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            marital_label = st.selectbox("Status Pernikahan", list(MARITAL_STATUS.keys()))
            marital_val   = MARITAL_STATUS[marital_label]

            gender_label  = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
            gender_val    = 1 if gender_label == "Laki-laki" else 0

            age           = st.number_input("Usia Saat Mendaftar", 17, 70, 20)

        with c2:
            nat_label     = st.selectbox("Kewarganegaraan", list(NATIONALITY.keys()))
            nat_val       = NATIONALITY[nat_label]

            international_label = st.selectbox(
                "Mahasiswa Internasional?",
                ["Tidak – Mahasiswa Domestik", "Ya – Mahasiswa Internasional"]
            )
            international_val = 1 if "Ya" in international_label else 0

            displaced_label = st.selectbox(
                "Mahasiswa Displaced (Pengungsi/Pindah Domisili)?",
                ["Tidak – Domisili Normal", "Ya – Mahasiswa Displaced"]
            )
            displaced_val = 1 if "Ya" in displaced_label else 0

        with c3:
            special_needs_label = st.selectbox(
                "Berkebutuhan Khusus Pendidikan?",
                ["Tidak Berkebutuhan Khusus", "Ya – Berkebutuhan Khusus"]
            )
            special_needs_val = 1 if "Ya" in special_needs_label else 0

            debtor_label = st.selectbox(
                "Status Hutang Akademik",
                ["Tidak Ada Hutang Akademik", "Ada Hutang Akademik (Debtor)"]
            )
            debtor_val = 1 if "Ada Hutang" in debtor_label else 0

            tuition_label = st.selectbox(
                "Status Pembayaran SPP/UKT",
                ["SPP/UKT Sudah Lunas (Up to Date)", "SPP/UKT Belum Terbayar / Menunggak"]
            )
            tuition_val = 1 if "Lunas" in tuition_label else 0

    # ── Section 2: Data Akademik & Pendaftaran ─────────────────────────
    with st.expander("📋 Data Akademik & Pendaftaran", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            app_mode_label = st.selectbox("Jalur/Cara Pendaftaran", list(APPLICATION_MODE.keys()))
            app_mode_val   = APPLICATION_MODE[app_mode_label]

            app_order = st.number_input(
                "Urutan Pilihan Program Studi (1 = Pilihan Utama)", 0, 9, 1
            )

            course_label = st.selectbox("Program Studi / Jurusan", list(COURSE.keys()))
            course_val   = COURSE[course_label]

        with c2:
            prev_qual_label = st.selectbox(
                "Kualifikasi Pendidikan Sebelum Masuk", list(PREVIOUS_QUALIFICATION.keys())
            )
            prev_qual_val = PREVIOUS_QUALIFICATION[prev_qual_label]

            prev_qual_grade = st.number_input(
                "Nilai Kualifikasi Pendidikan Sebelumnya (0–200)", 0.0, 200.0, 130.0
            )
            admission_grade = st.number_input(
                "Nilai Ujian Masuk / Admission Grade (0–200)", 0.0, 200.0, 130.0
            )

        with c3:
            attendance_label = st.selectbox(
                "Waktu Perkuliahan",
                ["Siang Hari (Daytime/Regular)", "Malam Hari (Evening/Kelas Malam)"]
            )
            attendance_val = 1 if "Siang" in attendance_label else 0

            scholarship_label = st.selectbox(
                "Status Beasiswa",
                ["Tidak Menerima Beasiswa", "Menerima Beasiswa (Scholarship Holder)"]
            )
            scholarship_val = 1 if "Menerima" in scholarship_label else 0

    # ── Section 3: Data Orang Tua ──────────────────────────────────────
    with st.expander("👨‍👩‍👦 Latar Belakang Orang Tua", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🧑‍🦰 Ibu**")
            mothers_qual_label = st.selectbox("Tingkat Pendidikan Ibu",
                                              list(PARENT_QUALIFICATION.keys()), key="mq")
            mothers_qual_val   = PARENT_QUALIFICATION[mothers_qual_label]
            mothers_occ_label  = st.selectbox("Pekerjaan Ibu",
                                              list(PARENT_OCCUPATION.keys()), key="mo")
            mothers_occ_val    = PARENT_OCCUPATION[mothers_occ_label]

        with c2:
            st.markdown("**🧑‍🦱 Ayah**")
            fathers_qual_label = st.selectbox("Tingkat Pendidikan Ayah",
                                              list(PARENT_QUALIFICATION.keys()), key="fq")
            fathers_qual_val   = PARENT_QUALIFICATION[fathers_qual_label]
            fathers_occ_label  = st.selectbox("Pekerjaan Ayah",
                                              list(PARENT_OCCUPATION.keys()), key="fo")
            fathers_occ_val    = PARENT_OCCUPATION[fathers_occ_label]

    # ── Section 4: Performa Semester 1 ────────────────────────────────
    with st.expander("📚 Performa Akademik Semester 1", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            cu1_credited = st.number_input(
                "Mata Kuliah Dikreditkan/Diakui Sem 1 (Credited)", 0, 20, 0,
                help="Jumlah SKS/mata kuliah yang diakui dari program lain atau pengalaman sebelumnya."
            )
            cu1_enrolled = st.number_input(
                "Mata Kuliah Terdaftar Sem 1 (Enrolled)", 0, 30, 6,
                help="Total mata kuliah yang diambil/didaftarkan di semester 1."
            )
        with c2:
            cu1_eval = st.number_input(
                "Jumlah Evaluasi/Ujian yang Diikuti Sem 1", 0, 40, 6,
                help="Berapa kali mahasiswa mengikuti ujian atau evaluasi di semester 1."
            )
            cu1_approved = st.number_input(
                "Mata Kuliah Lulus Sem 1 (Approved)", 0, 30, 5,
                help="Jumlah mata kuliah yang berhasil lulus di semester 1."
            )
        with c3:
            cu1_grade = st.number_input(
                "Nilai Rata-rata Semester 1 (0–20)", 0.0, 20.0, 12.0,
                help="Nilai rata-rata seluruh mata kuliah di semester 1 (skala 0–20)."
            )
            cu1_no_eval = st.number_input(
                "Mata Kuliah Tanpa Evaluasi Sem 1 (Without Evaluations)", 0, 20, 0,
                help="Jumlah mata kuliah yang tidak memiliki catatan evaluasi/ujian."
            )

    # ── Section 5: Performa Semester 2 ────────────────────────────────
    with st.expander("📚 Performa Akademik Semester 2", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            cu2_credited = st.number_input(
                "Mata Kuliah Dikreditkan/Diakui Sem 2 (Credited)", 0, 20, 0
            )
            cu2_enrolled = st.number_input(
                "Mata Kuliah Terdaftar Sem 2 (Enrolled)", 0, 30, 6
            )
        with c2:
            cu2_eval = st.number_input(
                "Jumlah Evaluasi/Ujian yang Diikuti Sem 2", 0, 40, 6
            )
            cu2_approved = st.number_input(
                "Mata Kuliah Lulus Sem 2 (Approved)", 0, 30, 5
            )
        with c3:
            cu2_grade = st.number_input(
                "Nilai Rata-rata Semester 2 (0–20)", 0.0, 20.0, 12.0
            )
            cu2_no_eval = st.number_input(
                "Mata Kuliah Tanpa Evaluasi Sem 2 (Without Evaluations)", 0, 20, 0
            )

    # ── Section 6: Indikator Ekonomi Makro ────────────────────────────
    with st.expander("📈 Kondisi Ekonomi Makro (saat mahasiswa mendaftar)", expanded=False):
        st.caption(
            "Data ini mencerminkan kondisi ekonomi negara pada saat mahasiswa mendaftar. "
            "Isi sesuai tahun pendaftaran mahasiswa."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            unemployment = st.number_input(
                "Tingkat Pengangguran – Unemployment Rate (%)", 0.0, 25.0, 11.1
            )
        with c2:
            inflation = st.number_input(
                "Tingkat Inflasi – Inflation Rate (%)", -5.0, 10.0, 1.4
            )
        with c3:
            gdp = st.number_input(
                "Pertumbuhan Ekonomi – GDP Growth (%)", -10.0, 10.0, 1.74
            )

    # ── Predict Button ─────────────────────────────────────────────────
    if st.button("🔮 Prediksi Risiko Dropout", type="primary", use_container_width=True):
        row = {
            "Marital_status":                               marital_val,
            "Application_mode":                             app_mode_val,
            "Application_order":                            app_order,
            "Course":                                       course_val,
            "Daytime_evening_attendance":                   attendance_val,
            "Previous_qualification":                       prev_qual_val,
            "Previous_qualification_grade":                 prev_qual_grade,
            "Nacionality":                                  nat_val,
            "Mothers_qualification":                        mothers_qual_val,
            "Fathers_qualification":                        fathers_qual_val,
            "Mothers_occupation":                           mothers_occ_val,
            "Fathers_occupation":                           fathers_occ_val,
            "Admission_grade":                              admission_grade,
            "Displaced":                                    displaced_val,
            "Educational_special_needs":                    special_needs_val,
            "Debtor":                                       debtor_val,
            "Tuition_fees_up_to_date":                      tuition_val,
            "Gender":                                       gender_val,
            "Scholarship_holder":                           scholarship_val,
            "Age_at_enrollment":                            age,
            "International":                                international_val,
            "Curricular_units_1st_sem_credited":            cu1_credited,
            "Curricular_units_1st_sem_enrolled":            cu1_enrolled,
            "Curricular_units_1st_sem_evaluations":         cu1_eval,
            "Curricular_units_1st_sem_approved":            cu1_approved,
            "Curricular_units_1st_sem_grade":               cu1_grade,
            "Curricular_units_1st_sem_without_evaluations": cu1_no_eval,
            "Curricular_units_2nd_sem_credited":            cu2_credited,
            "Curricular_units_2nd_sem_enrolled":            cu2_enrolled,
            "Curricular_units_2nd_sem_evaluations":         cu2_eval,
            "Curricular_units_2nd_sem_approved":            cu2_approved,
            "Curricular_units_2nd_sem_grade":               cu2_grade,
            "Curricular_units_2nd_sem_without_evaluations": cu2_no_eval,
            "Unemployment_rate":                            unemployment,
            "Inflation_rate":                               inflation,
            "GDP":                                          gdp,
        }

        try:
            X_input  = pd.DataFrame([row])[feature_cols]
            X_scaled = scaler.transform(X_input)
            proba    = model.predict_proba(X_scaled)[0][1]

            st.markdown("---")
            if proba >= 0.7:
                risk_color = "#ff4b4b"
                risk_label = "🔴 RISIKO TINGGI"
                recommendation = (
                    "Mahasiswa ini memiliki risiko dropout yang **sangat tinggi**. "
                    "Segera lakukan intervensi: hubungi dosen wali, tawarkan konseling, "
                    "dan tinjau kesulitan akademik maupun finansial yang dihadapi."
                )
            elif proba >= 0.4:
                risk_color = "#ffa500"
                risk_label = "🟡 RISIKO SEDANG"
                recommendation = (
                    "Mahasiswa ini perlu **dipantau secara berkala**. "
                    "Disarankan melakukan check-in rutin dengan dosen wali dan memastikan "
                    "tidak ada hambatan akademik atau finansial yang tidak tertangani."
                )
            else:
                risk_color = "#00cc88"
                risk_label = "🟢 RISIKO RENDAH"
                recommendation = (
                    "Mahasiswa ini diprediksi **aman dari risiko dropout** saat ini. "
                    "Tetap pertahankan pemantauan rutin setiap semester."
                )

            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                st.markdown(
                    f"<div style='background:{risk_color}22; border:2px solid {risk_color}; "
                    f"border-radius:10px; padding:1.2rem; text-align:center;'>"
                    f"<h2 style='color:{risk_color}; margin:0;'>{proba*100:.1f}%</h2>"
                    f"<p style='color:{risk_color}; font-weight:bold; margin:0;'>Probabilitas Dropout</p>"
                    f"<p style='font-size:1.1rem; margin-top:0.5rem;'>{risk_label}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.progress(float(proba))
            with col_r2:
                st.markdown("**💡 Rekomendasi Tindakan:**")
                st.info(recommendation)

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")

# ── Tab 2: Batch Prediction ────────────────────────────────────────────
with tab2:
    st.markdown("### Upload File CSV untuk Prediksi Batch")
    st.info(
        "Format CSV harus memiliki kolom yang sama dengan dataset training (`data.csv`). "
        "Kolom `Status` bersifat opsional. Gunakan separator titik koma (`;`)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded, sep=';')
        st.write(f"**Data dimuat:** {len(df_batch)} baris")
        st.dataframe(df_batch.head(), use_container_width=True)

        if st.button("🚀 Jalankan Prediksi Batch", type="primary"):
            for col in feature_cols:
                if col not in df_batch.columns:
                    df_batch[col] = 0

            X_batch  = df_batch[feature_cols]
            X_scaled = scaler.transform(X_batch)
            probas   = model.predict_proba(X_scaled)[:, 1]
            preds    = (probas >= 0.5).astype(int)

            df_result = df_batch.copy()
            df_result["Dropout_Probability"] = np.round(probas, 4)
            df_result["Prediction"]          = ["Dropout" if p else "Graduate" for p in preds]
            df_result["Risk_Level"]          = pd.cut(
                probas, bins=[-0.001, 0.4, 0.7, 1.0],
                labels=["Rendah (Low Risk)", "Sedang (Medium Risk)", "Tinggi (High Risk)"]
            )

            st.success(
                f"✅ Prediksi selesai! **{preds.sum()}** dari **{len(preds)}** mahasiswa "
                f"diprediksi berisiko dropout."
            )

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Mahasiswa", len(preds))
            col_m2.metric("Prediksi Dropout", int(preds.sum()))
            col_m3.metric("Dropout Rate", f"{preds.mean()*100:.1f}%")

            st.dataframe(
                df_result[["Dropout_Probability", "Prediction", "Risk_Level"]
                          + feature_cols[:5]].head(20),
                use_container_width=True
            )

            csv_out = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Hasil Prediksi", csv_out,
                "hasil_prediksi_dropout.csv", "text/csv"
            )

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color: gray; font-size: 13px;'>
    🎓 Jaya Jaya Institut – Student Dropout Prediction System | 
    Model: Random Forest | ROC-AUC: 0.9695
</p>
""", unsafe_allow_html=True)

