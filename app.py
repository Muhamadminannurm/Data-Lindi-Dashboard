import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import openpyxl
import io 

# Load model & scaler X
try:
    model = joblib.load("model_leachate_ann.pkl")
    scaler_X = joblib.load("scaler_X.pkl")
    MODEL_LOADED = True
    
    try:
        SCALER_FEATURES = list(scaler_X.feature_names_in_)
    except AttributeError:
        SCALER_FEATURES = [
            'TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS', 'FF_X', 'DDD_X', 'FF_AVG',
            'DDD_CAR_C', 'DDD_CAR_E', 'DDD_CAR_NW', 'DDD_CAR_S', 'DDD_CAR_SE', 'DDD_CAR_SW', 'DDD_CAR_W'
        ]
        
except FileNotFoundError:
    st.error("Error: File 'model_leachate_ann.pkl' atau 'scaler_X.pkl' tidak ditemukan. Harap pastikan file model hasil pelatihan sudah diunggah.")
    MODEL_LOADED = False
except Exception as e:
    st.error(f"Error saat memuat model: {str(e)}")
    MODEL_LOADED = False

# Scaling function (digunakan di skenario 2 dan 3)
def scale_features(df_input, scaler_X):
    feature_names = SCALER_FEATURES
    X = df_input[feature_names]
    y = df_input["Lindi"] 
    X_scaled = scaler_X.transform(X)
    return X, y, X_scaled

# Streamlit UI
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'> Prediksi Volume Air Lindi Menggunakan Algoritma Artificial Neural Network</h1>",
    unsafe_allow_html=True
)

# ===============================================
# MODIFIKASI: MENAMBAHKAN LOGO DI ATAS SIDEBAR
# ===============================================
st.sidebar.image(
    "logo.png", 
    use_container_width=True # <--- GUNAKAN PARAMETER BARU INI
)
# ===============================================

# --- URUTAN SKENARIO ---
for _ in range(4):
    st.sidebar.markdown("   ")
    
menu = st.sidebar.radio(
    "📋 MENU ", 
    ["Skenario 1: Eksperimen", "Skenario 2: TESTING BATCH", "Skenario 3: TESTING SINGLE"]
)

# ===============================================
# MODIFIKASI: MENAMBAHKAN TEKS UNIVERSITAS DI BAWAH SIDEBAR
# ===============================================
for _ in range(30):
    st.sidebar.markdown("   ")

st.sidebar.markdown(
    "<p style='text-align: center; font-size: small; color: gray;'>Dibangun oleh:</p>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<h3 style='text-align: center; color: #007BFF;'>Universitas Muhammadiyah Malang</h3>",
    unsafe_allow_html=True
)
# ===============================================

# =================================================================
# SKENARIO 1: EKSPERIMEN WHITE BOX (FULL)
# =================================================================
if menu == "Skenario 1: Eksperimen":
    st.subheader("🔬 Alur Eksperimen End-to-End ")
   
    # Inisialisasi status data
    if 'df_processed' not in st.session_state:
        st.session_state['df_processed'] = None
        
    tab1, tab2 = st.tabs(["1. Input & Pre-Proses Data", "2. Pembentukan Model & Hasil Evaluasi"])

    # -----------------------------------------------------------------
    # Tab 1: Input & Pre-Proses Data
    # -----------------------------------------------------------------
    with tab1:
        st.markdown("### 1. 📥 Input Data Awal ")
        uploaded_excel = st.file_uploader("Upload File Excel ", type=["xlsx"])
        
        if uploaded_excel:
            try:
                # Membaca data mentah
                df_raw = pd.read_excel(io.BytesIO(uploaded_excel.getvalue()))
                
                # --- Tampilkan Jumlah Data Mentah ---
                st.info(f"Jumlah Data Mentah: **{len(df_raw)} baris**.")
                st.markdown("#### Pratinjau Data Mentah:")
                st.dataframe(df_raw.head(50))
                
                # --- Langkah Pre-Proses (sesuai Colab) ---
                st.markdown("### 2. ⚙️ Pre-Proses Data (One-Hot Encoding)")
                st.code("""
                            # One-hot encoding kolom 'DDD_CAR'
                            df_onehot = pd.get_dummies(df['DDD_CAR'], prefix='DDD_CAR', dtype=int)
                            # Gabungkan hasil one-hot dan hapus kolom asli
                            df = pd.concat([df, df_onehot], axis=1)
                            df = df.drop(columns=['DDD_CAR'])
                            df.to_csv("DATAPROSES.csv", index=False)
                        """)
                
                if 'DDD_CAR' in df_raw.columns:
                    # Lakukan OHE secara aktual
                    df_onehot = pd.get_dummies(df_raw['DDD_CAR'], prefix='DDD_CAR', dtype=int)
                    df_processed = pd.concat([df_raw, df_onehot], axis=1)
                    df_processed = df_processed.drop(columns=['DDD_CAR'])
                    
                    st.success("✅ Pre-Proses (OHE) Selesai!")
                    
                    # --- Tampilkan Jumlah Data Proses ---
                    st.info(f"Jumlah Data Setelah Proses (DATAPROSES.csv): **{len(df_processed)} baris**.")
                    st.markdown("#### Pratinjau Data Proses (`DATAPROSES.csv`):")
                    st.dataframe(df_processed.head(50))
                    
                    # --- Tombol Download ---
                    csv_data = df_processed.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Unduh DATAPROSES.csv",
                        data=csv_data,
                        file_name='DATAPROSES.csv',
                        mime='text/csv',
                        help="Klik untuk mengunduh data yang sudah diproses OHE."
                    )
                    
                    st.session_state['df_processed'] = df_processed
                else:
                    st.error("❌ File Excel harus memiliki kolom 'DDD_CAR' untuk One-Hot Encoding.")
                    st.session_state['df_processed'] = None

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses file: {str(e)}")
        else:
            st.warning("Mohon unggah file Excel data mentah Anda.")

    # -----------------------------------------------------------------
    # Tab 2: Pembentukan Model & Hasil Evaluasi
    # -----------------------------------------------------------------
    with tab2:
        st.markdown("### 3. 🔪 Split Data & Scaling")
        if st.session_state['df_processed'] is not None and MODEL_LOADED:
            df_processed = st.session_state['df_processed']

            if 'Lindi' not in df_processed.columns:
                st.error("Data proses tidak memiliki kolom 'Lindi'.")
                st.stop()
                
            X = df_processed.drop(columns=["Lindi", "TANGGAL"])
            y = df_processed["Lindi"]
            
            # Mencocokkan kolom dengan yang digunakan saat training (penting)
            missing_cols = set(SCALER_FEATURES) - set(X.columns)
            if missing_cols:
                for col in missing_cols:
                    X[col] = 0
            
            X = X[SCALER_FEATURES]

            # Simulasi train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Scaling Data (menggunakan scaler yang sudah dimuat)
            X_train_scaled = scaler_X.transform(X_train) 
            X_test_scaled = scaler_X.transform(X_test)
            
            # Simpan data split di session state
            st.session_state['X_train_scaled'] = X_train_scaled
            st.session_state['y_train'] = y_train
            st.session_state['X_test_scaled'] = X_test_scaled
            st.session_state['y_test'] = y_test
            
            st.code("""
                        # Pisahkan fitur & target
                        X = df.drop(columns=["Lindi", "TANGGAL"])
                        y = df["Lindi"]

                        # Split train-test (30% test, random_state=42)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, random_state=42
                        )

                        # Scaling fitur (StandardScaler)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    """)
            st.success(f"✅ Data berhasil dibagi: **Data Training** ({len(X_train)} baris) dan **Data Uji (Test)** ({len(X_test)} baris).")
            st.markdown("---")
            
            st.markdown("### 4. 🧠 Pembentukan Model & Pelatihan (ANN)")
            st.code("""
                        # Konfigurasi Model (ANN/MLPRegressor)
                        mlp = MLPRegressor(
                            hidden_layer_sizes=(64, 64),
                            activation='tanh', solver='adam',
                            alpha=0.001, learning_rate_init=0.005,
                            learning_rate="adaptive", batch_size=128,
                            max_iter=5000, random_state=42,
                            early_stopping=True, n_iter_no_change=60
                        )
                        # Pelatihan Model (Dijalankan di Colab)
                        mlp.fit(X_train_scaled, y_train)
                 """)
            st.info("⚠️ Proses pelatihan telah dilakukan sebelumnya. Kami menggunakan model yang sudah dimuat untuk simulasi evaluasi.")
            st.markdown("---")

            # --- Hasil Evaluasi ---
            st.markdown("### 5. 📊 Hasil Evaluasi (Simulasi)")
            
            X_train_scaled = st.session_state['X_train_scaled']
            y_train = st.session_state['y_train']
            X_test_scaled = st.session_state['X_test_scaled']
            y_test = st.session_state['y_test']
            
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Perhitungan Metrik
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            corr_train, _ = pearsonr(y_train, y_train_pred)
            corr_test, _ = pearsonr(y_test, y_test_pred)

            col_train_res, col_test_res = st.columns(2)
            
            with col_train_res:
                st.markdown("#### Hasil Evaluasi Data **Training**")
                st.metric("📈 R² Score", f"{train_r2:.4f}")
                st.metric("📊 RMSE", f"{train_rmse:.2f}")
                st.metric("✅ Pearson's r", f"{corr_train:.4f}")
            
            with col_test_res:
                st.markdown("#### Hasil Evaluasi Data **Testing**")
                st.metric("📈 R² Score", f"{test_r2:.4f}")
                st.metric("📊 RMSE", f"{test_rmse:.2f}")
                st.metric("✅ Pearson's r", f"{corr_test:.4f}")

            # TIME SERIES PLOT (Grafik Naik Turun)
            st.markdown("---")
            st.markdown("##### 📈 Time Series: Aktual vs Prediksi")
            fig_ts, axs_ts = plt.subplots(1, 2, figsize=(14, 6))

            # --- Plot Train Time Series ---
            axs_ts[0].plot(y_train.values, label="Aktual", marker='o', markersize=3, linestyle='-')
            axs_ts[0].plot(y_train_pred, label="Prediksi", marker='x', markersize=3, linestyle='--')
            axs_ts[0].set_title(f"Training Data Time Series (RMSE: {train_rmse:.2f})")
            axs_ts[0].set_xlabel("Indeks Data")
            axs_ts[0].set_ylabel("Lindi (m³/hari)")
            axs_ts[0].legend()
            axs_ts[0].grid(True)

            # --- Plot Test Time Series ---
            axs_ts[1].plot(y_test.values, label="Aktual", marker='o', markersize=3, linestyle='-')
            axs_ts[1].plot(y_test_pred, label="Prediksi", marker='x', markersize=3, linestyle='--')
            axs_ts[1].set_title(f"Testing Data Time Series (RMSE: {test_rmse:.2f})")
            axs_ts[1].set_xlabel("Indeks Data")
            axs_ts[1].set_ylabel("Lindi (m³/hari)")
            axs_ts[1].legend()
            axs_ts[1].grid(True)

            plt.tight_layout()
            st.pyplot(fig_ts)


            # Scatter Plot TRAIN vs TEST (Kode sebelumnya)
            st.markdown("##### 📌 Scatter Plot: Aktual vs Prediksi")
            fig_sc, axs_sc = plt.subplots(1, 2, figsize=(12, 6))

            # --- Plot Train ---
            axs_sc[0].scatter(y_train, y_train_pred, alpha=0.6, color="blue", edgecolor="k")
            axs_sc[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2) 
            axs_sc[0].set_title(f"Training Data (r = {corr_train:.4f})")
            axs_sc[0].set_xlabel("Aktual (m³/hari)")
            axs_sc[0].set_ylabel("Prediksi (m³/hari)")
            axs_sc[0].grid(True)
            
            # --- Plot Test ---
            axs_sc[1].scatter(y_test, y_test_pred, alpha=0.6, color="green", edgecolor="k")
            axs_sc[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
            axs_sc[1].set_title(f"Testing Data (r = {corr_test:.4f})")
            axs_sc[1].set_xlabel("Aktual (m³/hari)")
            axs_sc[1].set_ylabel("Prediksi (m³/hari)")
            axs_sc[1].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig_sc)


            st.markdown("---")
            st.markdown("### 6. 💾 Simpan Model")
            st.code("""
                        # Simpan model MLPRegressor & Scaler (Dijalankan di Colab)
                        # joblib.dump(mlp, "model_leachate_ann.pkl")
                        # joblib.dump(scaler, "scaler_X.pkl")
                    """)
            st.success("✅ Model dan Scaler telah disimpan sebagai `.pkl` file dan berhasil dimuat di aplikasi ini.")

        else:
            st.warning("Mohon selesaikan langkah **'1. Input & Pre-Proses Data'** di tab sebelumnya terlebih dahulu.")


# =================================================================
# SKENARIO 2: UPLOAD CSV (TIDAK BERUBAH)
# =================================================================
elif menu == "Skenario 2: TESTING BATCH":
    st.subheader("📂 Upload File CSV")
    st.info("Upload file CSV yang sudah diproses untuk diuji dengan model yang ada.")
    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"], key="upload_csv_s2")

    if uploaded_file and MODEL_LOADED:
        df = pd.read_csv(uploaded_file)
        
        if 'TANGGAL' in df.columns:
            try:
                df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
                df = df.sort_values(by='TANGGAL').reset_index(drop=True)
            except:
                st.warning("Kolom TANGGAL mungkin tidak dalam format tanggal yang benar. Diabaikan untuk visualisasi.")

        if 'Lindi' not in df.columns:
             st.error("❌ File CSV harus memiliki kolom 'Lindi' (nilai target) untuk evaluasi.")
        else:
            max_index = len(df) - 1
            
            st.markdown("### ⚙️ Atur Rentang Data yang Ditampilkan")
            start_index, end_index = st.slider(
                "Pilih Rentang Indeks Data:",
                0, max_index, (0, max_index)
            )
            
            df_filtered = df.iloc[start_index:end_index+1]

            if not df_filtered.empty:
                st.dataframe(df_filtered, use_container_width=True)

                try:
                    X, y_true, X_scaled = scale_features(df_filtered, scaler_X)
                except KeyError as e:
                    st.error(f"❌ Error: Kolom fitur {e} tidak ditemukan di file yang diunggah. Pastikan file sudah diproses OHE.")
                    st.stop()
                    
                y_pred = model.predict(X_scaled)
                
                # --- Perhitungan Metrik ---
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                pearson_r, p_value = pearsonr(y_true.values.flatten(), y_pred.flatten())

                # --- Tampilan Metrik ---
                st.markdown("### 📊 Hasil Evaluasi Model")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📉 MSE", f"{mse:.2f}")
                    st.metric("📊 RMSE", f"{rmse:.2f}")
                with col2:
                    st.metric("📏 MAE", f"{mae:.2f}")
                    st.metric("📈 R² Score", f"{r2:.4f}")
                with col3:
                    st.metric("✅ Pearson's r", f"{pearson_r:.4f}")
                    st.metric("🔬 P-value", f"{p_value:.2e}")
                
                st.markdown("""
                    <div style='margin-top: 10px; font-size: 14px;'>
                    <ul>
                        <li><strong>MSE</strong> (Mean Squared Error): Rata-rata kuadrat dari selisih antara prediksi dan aktual.</li>
                        <li><strong>RMSE</strong> (Root MSE): Akar dari MSE. Semakin kecil, semakin akurat prediksi.</li>
                        <li><strong>MAE</strong> (Mean Absolute Error): Rata-rata selisih absolut antara prediksi dan aktual.</li>
                        <li><strong>R2</strong> (R-squared): Proporsi varians dalam variabel terikat (dependen) yang dapat dijelaskan oleh model regresi linier, melalui variabel bebas (independen). Nilainya berkisar antara 0 hingga 1.</li>
                        <li><strong>Pearson's r</strong>: Mengukur korelasi linier antara aktual dan prediksi.</li>
                        <li><strong>P-value</strong>: Menunjukkan signifikansi statistik korelasi. P-value < 0.05 biasanya signifikan.</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")
                
                # Plot 1: Prediksi vs Aktual
                st.markdown("### 📈 Time Series: Prediksi vs Aktual")
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(df_filtered.index, y_true, label="Aktual", marker="o", linestyle='-', markersize=4)
                ax1.plot(df_filtered.index, y_pred, label="Prediksi", marker="x", linestyle='--', markersize=4)
                ax1.set_xlabel("Indeks Data")
                ax1.set_ylabel("Lindi (m³/hari)")
                ax1.set_title("Time Series Prediksi vs Aktual")
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
                
                if 'TANGGAL' in df_filtered.columns:
                    monthly_labels = df_filtered['TANGGAL'].dt.strftime('%b-%y')
                    month_indices = monthly_labels.drop_duplicates().index.tolist()
                    month_names = monthly_labels.loc[month_indices].tolist()

                    ax3.set_xlabel("Bulan")
                    ax3.set_xticks(month_indices)
                    ax3.set_xticklabels(month_names, rotation=90)
                    
                    for i in range(len(month_indices) - 1):
                        start_of_next_month = month_indices[i+1]
                        ax3.axvline(x=start_of_next_month - 0.5, color='red', linestyle='--', linewidth=1)
                else:
                    ax3.set_xlabel("Indeks Data")

                ax3.bar(df_filtered.index, np.abs(y_true.values.flatten() - y_pred.flatten()))
                ax3.set_ylabel("Error Absolut (m³/hari)")
                ax3.set_title("Error Absolut")
                ax3.grid(axis='y')
                
                st.pyplot(fig3)
            else:
                st.warning("Rentang data yang dipilih tidak valid. Silakan pilih rentang yang lain.")


# =================================================================
# SKENARIO 3: INPUT MANUAL (HARIAN) (TIDAK BERUBAH)
# =================================================================
elif menu == "Skenario 3: TESTING SINGLE":
    if not MODEL_LOADED:
        st.stop()
        
    st.markdown("### 📥 Input Fitur Prediksi Harian Secara Manual")
    
    feature_values = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_values['TN'] = st.number_input("TN (Suhu Min)", value=22.20, step=0.1)
        feature_values['TX'] = st.number_input("TX (Suhu Maks)", value=31.06, step=0.1)
        feature_values['TAVG'] = st.number_input("TAVG (Suhu Rata-rata)", value=25.21, step=0.1)
    
    with col2:
        feature_values['RH_AVG'] = st.number_input("RH_AVG (Kelembaban Rata-rata)", value=83.32, step=0.1)
        feature_values['RR'] = st.number_input("RR (Curah Hujan)", value=10.13, step=0.1)
        feature_values['SS'] = st.number_input("SS (Durasi Matahari)", value=5.26, step=0.1)
    
    with col3:
        feature_values['FF_X'] = st.number_input("FF_X (Kec. Angin Maks)", value=2.74, step=0.1)
        feature_values['DDD_X'] = st.number_input("DDD_X (Arah Angin Maks)", value=198.01, step=1.0)
        feature_values['FF_AVG'] = st.number_input("FF_AVG (Kec. Angin Rata-rata)", value=1.41, step=0.1)
    
    st.markdown("---")
    st.markdown("#### Input Arah Angin (Harian)")
    st.info("Pilih satu arah angin yang paling dominan untuk hari ini.")

    wind_directions = ['DDD_CAR_C', 'DDD_CAR_E', 'DDD_CAR_NW', 'DDD_CAR_S', 'DDD_CAR_SE', 'DDD_CAR_SW', 'DDD_CAR_W']
    dominant_direction = st.radio("Arah Angin Dominan", options=wind_directions, index=0, horizontal=True)
    
    for direction in wind_directions:
        feature_values[direction] = 0
    
    feature_values[dominant_direction] = 1
    
    col_empty1, col_button, col_empty2 = st.columns([1, 2, 1])
    with col_button:
        prediksi_clicked = st.button("🔍 Prediksi Lindi", key="prediksi_harian", use_container_width=True)
    
    if prediksi_clicked:
        try:
            expected_columns = SCALER_FEATURES 
            
            input_data = {key: feature_values[key] for key in expected_columns}

            X_manual_df = pd.DataFrame([input_data])
            
            X_manual_scaled = scaler_X.transform(X_manual_df)
            
            y_pred = model.predict(X_manual_scaled)
            
            st.markdown(
                f"""
                <div style='background-color:#007BFF;padding:20px;border-radius:10px;margin-top:20px; color: white;'>
                    <h3 style='text-align:center;'>✅ Prediksi Volume Air Lindi:</h3>
                    <h2 style='text-align:center;'>{y_pred[0]:.2f} m³/hari</h2>
                </div>
                """, unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"❌ Terjadi error saat prediksi. Pastikan semua input sudah diisi dengan benar. Error: {str(e)}")
