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

# ── Load Model (FIXED) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "rf_dropout.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    meta_path = os.path.join(base_dir, "model_meta.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, scaler, meta

# ── Load Model Execution ───────────────────────────────────────────────
model, scaler, meta = load_model()
feature_cols = meta.get("feature_cols", [])

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

    col1, col2, col3 = st.columns(3)

    with col1:
        cu1_enrolled  = st.number_input("Unit Terdaftar Sem 1", 0, 30, 6)
        cu1_approved  = st.number_input("Unit Lulus Sem 1", 0, 30, 5)
        cu1_grade     = st.number_input("Nilai Rata-rata Sem 1", 0.0, 20.0, 12.0)
        cu1_eval      = st.number_input("Evaluasi Sem 1", 0, 40, 6)

    with col2:
        cu2_enrolled  = st.number_input("Unit Terdaftar Sem 2", 0, 30, 6)
        cu2_approved  = st.number_input("Unit Lulus Sem 2", 0, 30, 5)
        cu2_grade     = st.number_input("Nilai Rata-rata Sem 2", 0.0, 20.0, 12.0)
        cu2_eval      = st.number_input("Evaluasi Sem 2", 0, 40, 6)

    with col3:
        age           = st.number_input("Usia", 17, 70, 20)
        tuition_ok    = st.selectbox("SPP Terbayar?", ["Ya", "Tidak"])
        scholarship   = st.selectbox("Beasiswa?", ["Tidak", "Ya"])
        debtor        = st.selectbox("Hutang?", ["Tidak", "Ya"])
        gender        = st.selectbox("Gender", ["Perempuan", "Laki-laki"])
        admission_grd = st.number_input("Nilai Masuk", 0.0, 200.0, 130.0)

    if st.button("🔮 Prediksi"):
        row = {col: 0 for col in feature_cols}
        row.update({
            "Age_at_enrollment": age,
            "Tuition_fees_up_to_date": 1 if tuition_ok == "Ya" else 0,
            "Scholarship_holder": 1 if scholarship == "Ya" else 0,
            "Debtor": 1 if debtor == "Ya" else 0,
            "Gender": 1 if gender == "Laki-laki" else 0,
            "Admission_grade": admission_grd,
            "Curricular_units_1st_sem_enrolled": cu1_enrolled,
            "Curricular_units_1st_sem_approved": cu1_approved,
            "Curricular_units_1st_sem_grade": cu1_grade,
            "Curricular_units_1st_sem_evaluations": cu1_eval,
            "Curricular_units_2nd_sem_enrolled": cu2_enrolled,
            "Curricular_units_2nd_sem_approved": cu2_approved,
            "Curricular_units_2nd_sem_grade": cu2_grade,
            "Curricular_units_2nd_sem_evaluations": cu2_eval,
        })

        try:
            X_input = pd.DataFrame([row])[feature_cols]
            X_scaled = scaler.transform(X_input)
            proba = model.predict_proba(X_scaled)[0][1]

            st.success(f"Probabilitas Dropout: {proba*100:.2f}%")
            st.progress(float(proba))

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
# ── Tab 2: Batch Prediction ────────────────────────────────────────────
with tab2:
    st.markdown("### Upload File CSV untuk Prediksi Batch")
    st.info("Format CSV harus memiliki kolom yang sama dengan dataset training (`data.csv`). Kolom `Status` bersifat opsional.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded, sep=';')
        st.write(f"**Data dimuat:** {len(df_batch)} baris")
        st.dataframe(df_batch.head(), use_container_width=True)

        if st.button("🚀 Jalankan Prediksi Batch", type="primary"):
            for col in feature_cols:
                if col not in df_batch.columns:
                    df_batch[col] = 0

            X_batch = df_batch[feature_cols]
            X_scaled = scaler.transform(X_batch)
            probas = model.predict_proba(X_scaled)[:, 1]
            preds  = (probas >= 0.5).astype(int)

            df_result = df_batch.copy()
            df_result["Dropout_Probability"] = np.round(probas, 4)
            df_result["Prediction"]          = ["Dropout" if p else "Graduate" for p in preds]
            df_result["Risk_Level"]          = pd.cut(
                probas, bins=[-0.001,0.4,0.7,1.0],
                labels=["Rendah","Sedang","Tinggi"]
            )

            st.success(f"Prediksi selesai! {preds.sum()} dari {len(preds)} mahasiswa diprediksi dropout.")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Mahasiswa", len(preds))
            col_m2.metric("Prediksi Dropout", int(preds.sum()))
            col_m3.metric("Dropout Rate", f"{preds.mean()*100:.1f}%")

            st.dataframe(df_result[["Dropout_Probability","Prediction","Risk_Level"]
                         + feature_cols[:5]].head(20), use_container_width=True)

            csv_out = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil Prediksi", csv_out,
                               "hasil_prediksi_dropout.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color: gray; font-size: 13px;'>
    🎓 Jaya Jaya Institut – Student Dropout Prediction System | 
    Model: Random Forest | ROC-AUC: 0.9695
</p>
""", unsafe_allow_html=True)
