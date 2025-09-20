import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib
import openpyxl

# Load model & scaler X
model = joblib.load("model_leachate_ann.pkl")
scaler_X = joblib.load("scaler_X.pkl")

# Scaling function (tanpa scaling y)
def scale_features(df_input, scaler_X):
    X = df_input.drop(columns=["TANGGAL", "Lindi"])
    y = df_input["Lindi"] # Ambil data y tanpa scaling
    X_scaled = scaler_X.transform(X)
    return X, y, X_scaled

# Streamlit UI
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'> Prediksi Volume Air Lindi Menggunakan Algoritma Artificial Neural Network</h1>",
    unsafe_allow_html=True
)

menu = st.sidebar.radio("📋 Pilih Skenario", ["Skenario 1: Upload CSV", "Skenario 2: Input Manual (Harian)", "Skenario 3: Pre-Proses Data (OHE)"])

# SKENARIO 1
if menu == "Skenario 1: Upload CSV":
    st.subheader("📂 Upload File CSV (DATAPROSES2.CSV)")
    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df = df.sort_values(by='TANGGAL').reset_index(drop=True)
        
        max_index = len(df) - 1
        
        st.markdown("### ⚙️ Atur Rentang Data yang Ditampilkan")
        start_index, end_index = st.slider(
            "Pilih Rentang Indeks Data:",
            0, max_index, (0, max_index)
        )
        
        df_filtered = df.iloc[start_index:end_index+1]

        if not df_filtered.empty:
            st.dataframe(df_filtered, use_container_width=True)

            X, y_true, X_scaled = scale_features(df_filtered, scaler_X)
            y_pred = model.predict(X_scaled)
            
            # --- Perhitungan Metrik ---
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            pearson_r, p_value = pearsonr(y_true.values.flatten(), y_pred.flatten())

            # --- Tampilan Metrik ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📉 MSE", f"{mse:.2f}")
                st.metric("📊 RMSE", f"{rmse:.2f}")
            with col2:
                st.metric("📏 MAE", f"{mae:.2f}")
                st.metric("📈 R² Score", f"{r2:.2f}")
            with col3:
                st.metric("✅ Pearson's r", f"{pearson_r:.2f}")
                st.metric("🔬 P-value", f"{p_value:.2e}")
            
            st.markdown("""
            <div style='margin-top: 10px; font-size: 14px;'>
            <ul>
                <li><strong>Pearson's r</strong>: Mengukur korelasi linier antara aktual dan prediksi.</li>
                <li><strong>P-value</strong>: Menunjukkan signifikansi statistik korelasi. P-value < 0.05 biasanya signifikan.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot 1: Prediksi vs Aktual
            st.markdown("### 📈 Prediksi vs Aktual")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df_filtered.index, y_true, label="Aktual", marker="o", linestyle='-', markersize=4)
            ax1.plot(df_filtered.index, y_pred, label="Prediksi", marker="x", linestyle='--', markersize=4)
            ax1.set_xlabel("Indeks Data")
            ax1.set_ylabel("Lindi (m³/hari)")
            ax1.set_title("Prediksi vs Aktual")
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)

            # Plot 2: Scatter Plot Aktual vs Prediksi
            st.markdown("### 📌 Scatter Plot: Prediksi vs Aktual")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.scatter(y_true, y_pred, alpha=0.7, color="green", edgecolor="k")
            ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax2.set_xlabel("Aktual (m³/hari)")
            ax2.set_ylabel("Prediksi (m³/hari)")
            ax2.set_title(f"Scatter Plot (r = {pearson_r:.2f})")
            ax2.grid(True)
            st.pyplot(fig2)
            
            # Plot 3: Error Absolut
            st.markdown("### 📊 Error Absolut")
            fig3, ax3 = plt.subplots(figsize=(10, 5)) 
            monthly_labels = df_filtered['TANGGAL'].dt.strftime('%b-%y')
            month_indices = monthly_labels.drop_duplicates().index.tolist()
            month_names = monthly_labels.loc[month_indices].tolist()

            ax3.bar(df_filtered.index, np.abs(y_true.values.flatten() - y_pred.flatten()))
            ax3.set_xlabel("Bulan")
            ax3.set_ylabel("Error Absolut (m³/hari)")
            ax3.set_title("Error Absolut")
            
            ax3.set_xticks(month_indices)
            ax3.set_xticklabels(month_names, rotation=90)
            
            for i in range(len(month_indices) - 1):
                start_of_next_month = month_indices[i+1]
                ax3.axvline(x=start_of_next_month - 0.5, color='red', linestyle='--', linewidth=1)
            
            st.pyplot(fig3)
        else:
            st.warning("Rentang data yang dipilih tidak valid. Silakan pilih rentang yang lain.")

# SKENARIO 2
elif menu == "Skenario 2: Input Manual (Harian)":

    
    st.markdown("### 📥 Input Fitur Prediksi Harian Secara Manual")
    
    feature_values = {}
    
    # 3 columns layout for main weather features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_values['TN'] = st.number_input("TN (Suhu Min)", value=24.0, step=0.1)
        feature_values['TX'] = st.number_input("TX (Suhu Maks)", value=32.0, step=0.1)
        feature_values['TAVG'] = st.number_input("TAVG (Suhu Rata-rata)", value=28.0, step=0.1)
    
    with col2:
        feature_values['RH_AVG'] = st.number_input("RH_AVG (Kelembaban Rata-rata)", value=80.0, step=0.1)
        feature_values['RR'] = st.number_input("RR (Curah Hujan)", value=10.0, step=0.1)
        feature_values['SS'] = st.number_input("SS (Durasi Matahari)", value=5.0, step=0.1)
    
    with col3:
        feature_values['FF_X'] = st.number_input("FF_X (Kec. Angin Maks)", value=15.0, step=0.1)
        feature_values['DDD_X'] = st.number_input("DDD_X (Arah Angin Maks)", value=200.0, step=1.0)
        feature_values['FF_AVG'] = st.number_input("FF_AVG (Kec. Angin Rata-rata)", value=5.0, step=0.1)
    
    st.markdown("---")
    st.markdown("#### Input Arah Angin (Harian)")
    st.info("Pilih satu arah angin yang paling dominan untuk hari ini.")

    wind_directions = ['DDD_CAR_C', 'DDD_CAR_E', 'DDD_CAR_NW', 'DDD_CAR_S', 'DDD_CAR_SE', 'DDD_CAR_SW', 'DDD_CAR_W']
    dominant_direction = st.radio("Arah Angin Dominan", options=wind_directions, index=0, horizontal=True)
    
    # Atur semua kolom OHE ke 0
    for direction in wind_directions:
        feature_values[direction] = 0
    
    # Atur kolom yang dipilih ke 1
    feature_values[dominant_direction] = 1
    
    col_empty1, col_button, col_empty2 = st.columns([1, 2, 1])
    with col_button:
        prediksi_clicked = st.button("🔍 Prediksi Lindi", key="prediksi_harian", use_container_width=True)
    
    if prediksi_clicked:
        try:
            # Perbaikan: Mendefinisikan urutan dan nama kolom yang benar
            # Daftar ini harus sesuai dengan urutan dan nama yang digunakan saat melatih model
            expected_columns = [
                'TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS', 'FF_X', 'DDD_X', 'FF_AVG', 
                'DDD_CAR_C', 'DDD_CAR_E', 'DDD_CAR_NW', 
                'DDD_CAR_S', 'DDD_CAR_SE', 'DDD_CAR_SW', 'DDD_CAR_W'
            ]
            
            # Memastikan hanya fitur yang diharapkan yang ada di dictionary input
            # dan menempatkannya dalam urutan yang benar
            input_data = {key: feature_values[key] for key in expected_columns}

            # Membuat DataFrame baru dari dictionary
            X_manual_df = pd.DataFrame([input_data])
            
            # Melakukan scaling pada DataFrame
            X_manual_scaled = scaler_X.transform(X_manual_df)
            
            # Melakukan prediksi
            y_pred = model.predict(X_manual_scaled)
            
            st.markdown(
                f"""
                <div style='background-color:#949494;padding:20px;border-radius:10px;margin-top:20px;'>
                    <h3 style='text-align:center;'>✅ Prediksi Volume Air Lindi:</h3>
                    <h2 style='text-align:center;color:#ffffff;'>{y_pred[0]:.2f} m³/hari</h2>
                </div>
                """, unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"❌ Terjadi error saat prediksi: {str(e)}")

# SKENARIO 3
elif menu == "Skenario 3: Pre-Proses Data (OHE)":
    st.subheader("⚙️ Lakukan One-Hot Encoding")
    st.info("Pilih file Excel (.xlsx) dengan kolom 'DDD_CAR' untuk diubah menjadi format One-Hot Encoding (OHE).")
    
    uploaded_file_ohe = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if uploaded_file_ohe:
        try:
            # Baca file Excel
            df_ohe = pd.read_excel(uploaded_file_ohe)

            # Validasi kolom 'DDD_CAR'
            if 'DDD_CAR' not in df_ohe.columns:
                st.error("❌ File yang diunggah tidak memiliki kolom 'DDD_CAR'. Silakan cek kembali file Anda.")
            else:
                st.markdown("---")
                st.markdown("### 📋 Pratinjau Data Asli:")
                st.dataframe(df_ohe.head(), use_container_width=True)
                
                # Proses One-Hot Encoding
                st.markdown("---")
                st.markdown("### 🚀 Melakukan One-Hot Encoding...")
                
                # One-hot encoding kolom 'ddd_car'
                df_onehot = pd.get_dummies(df_ohe['DDD_CAR'], prefix='DDD_CAR', dtype=int)
                
                # Gabungkan hasil one-hot ke dataframe utama
                df_processed = pd.concat([df_ohe, df_onehot], axis=1)
                
                # Hapus kolom asli 'DDD_CAR'
                df_processed = df_processed.drop(columns=['DDD_CAR'])
                
                st.success("✅ One-Hot Encoding Berhasil! Data siap diunduh.")

                # Tampilkan pratinjau data yang sudah diproses
                st.markdown("### ✨ Pratinjau Data Setelah OHE:")
                st.dataframe(df_processed.head(), use_container_width=True)

                # Tombol untuk mengunduh file CSV
                csv_file = df_processed.to_csv(index=False)
                st.download_button(
                    label="⬇️ Unduh DATAPROSES_OHE.csv",
                    data=csv_file,
                    file_name='DATAPROSES_OHE.csv',
                    mime='text/csv',
                    help="Klik untuk mengunduh file CSV yang telah diproses."
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {str(e)}")
