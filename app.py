import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import io
import base64

# =============================================================================
# 1. KONFIGURASI HALAMAN & CSS
# =============================================================================
st.set_page_config(
    page_title="Leachate Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="auto"
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_style(png_file):
    try:
        bin_str = get_base64(png_file)
        st.markdown(
            f"""
            <style>
            /* --- ANIMASI --- */
            @keyframes slideDownSmooth {{
                0% {{ opacity: 0; transform: translateY(-40px); filter: blur(5px); }}
                100% {{ opacity: 1; transform: translateY(0); filter: blur(0px); }}
            }}
            @keyframes popIn {{
                0% {{ opacity: 0; transform: scale(0.95); }}
                60% {{ opacity: 1; transform: scale(1.02); }}
                100% {{ transform: scale(1); }}
            }}

            /* --- TEKS GLOBAL --- */
            h1, h2, h3, h4, h5, p, label, span, div.stMarkdown, div.stMetricLabel, li, div[data-testid="stDialog"] {{
                color: #e0e0e0 !important;
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }}

            /* --- 1. KEYFRAMES ANIMASI NEON BERJALAN --- */
            @keyframes rgbFlow {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}

            @keyframes slideDownSmooth {{
                0% {{ opacity: 0; transform: translateY(-40px); filter: blur(5px); }}
                100% {{ opacity: 1; transform: translateY(0); filter: blur(0px); }}
            }}
            
            /* --- BACKGROUND UTAMA --- */
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

           /* --- KONTAINER UTAMA (RGB BORDER ONLY) --- */
            .block-container {{
                aposition: relative; /* Wajib ada agar border menempel */
                animation: slideDownSmooth 1.2s cubic-bezier(0.23, 1, 0.32, 1) both;
                
                /* Warna Kaca Gelap Transparan (Tengahnya) */
                background-color: rgba(28, 31, 38, 0.2); 
                
                /* --- PERUBAHAN DISINI (MEMBUAT LEBAR PENUH) --- */
                max-width: 100vw !important;  /* Paksa lebar maksimum seukuran layar */
                width: 100% !important;       /* Paksa lebar 100% */
                padding: 2rem 2rem !important; /* Sesuaikan padding agar tidak terlalu mepet */
                /* ----------------------------------------------- */
                border-radius: 20px;
                padding: 2.5rem 3rem;
                margin-top: 2rem;
                box-shadow: 0 20px 50px rgba(0,0,0,0.8);
                backdrop-filter: blur(10px);
                border: none; /* Border asli dimatikan */
            }}

            .block-container::before {{
                content: "";
                position: absolute;
                inset: 0;
                border-radius: 20px; 
                padding: 3px; /* KETEBALAN GARIS NEON */
                
                /* Warna Pelangi */
                background: linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000); 
                background-size: 400% 400%; /* Diperbesar agar animasi jalan terlihat */
                animation: rgbFlow 2s linear infinite; /* Kecepatan jalan */
                
                /* TEKNIK MASKING: Membuat tengahnya bolong sehingga transparansi kaca di layer bawah terlihat */
                -webkit-mask: 
                linear-gradient(#fff 0 0) content-box, 
                linear-gradient(#fff 0 0);
                -webkit-mask-composite: xor;
                mask-composite: exclude;
                
                pointer-events: none; /* Agar tetap bisa klik isi konten */
                z-index: 2; /* Di atas background kaca */
            }}

            /* --- SIDEBAR --- */
            section[data-testid="stSidebar"] {{
                background-color: rgba(20, 23, 28, 0.4); /* Kaca Gelap */
                backdrop-filter: blur(10px);
                border: none; /* Hapus border biasa */
                position: relative; /* Wajib agar neon menempel */
            }}
            section[data-testid="stSidebar"]::before {{
                content: "";
                position: absolute;
                top: 0; 
                right: 0; 
                bottom: 0;
                width: 3px; /* KETEBALAN GARIS NEON */
                
                /* Warna Pelangi Vertikal */
                background-size: 400% 400%;
                animation: rgbFlow 1s linear infinite; /* Animasi Jalan */
                
                z-index: 2;
            }}
            section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] h1 {{ color: #ffffff !important; }}

            /* --- TOMBOL --- */
            div.stButton > button {{
                background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
                color: white !important;
                border: none;
                border-radius: 12px;
                padding: 0.6rem 1.5rem;
                font-weight: 700;
                transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
                box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
            }}
            div.stButton > button:hover {{
                transform: translateY(-2px) scale(1.02);
                box-shadow: 0 8px 25px rgba(0, 114, 255, 0.6);
            }}
            
            /* --- WIDGET INPUT --- */
            div[data-baseweb="input"], div[data-baseweb="select"] > div {{
                background-color: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                color: white !important;
            }}

            /* --- MODAL DIALOG (FIXED DARK GLASS) --- */
            div[role="dialog"] {{
                position: relative; /* Penting agar border neon menempel */
                
                /* Warna Kaca Gelap Transparan */
                background-color: rgba(28, 31, 38, 0.2) !important; 
                backdrop-filter: blur(10px);
                
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.8);
                color: #e0e0e0 !important;
                
                border: none !important;
            }}
            
            div[role="dialog"]::before {{
                content: "";
                position: absolute;
                inset: 0; /* Menempel di ujung */
                border-radius: 20px; 
                padding: 3px; /* KETEBALAN NEON */
                
                /* Warna Pelangi */
                background: linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000); 
                background-size: 400% 400%;
                animation: rgbFlow 1s linear infinite;
                
                /* MASKING: Membuat tengah bolong */
                -webkit-mask: 
                   linear-gradient(#fff 0 0) content-box, 
                   linear-gradient(#fff 0 0);
                -webkit-mask-composite: xor;
                mask-composite: exclude;
                
                pointer-events: none;
                z-index: 2;
            }}

            /* --- MENGHILANGKAN TOMBOL 'X' (CLOSE) DI DIALOG --- */
            div[role="dialog"] button[aria-label="Close"] {{
                display: none !important;
            }}
            
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("⚠️ File 'TPA.jpg' tidak ditemukan.")

set_style('TPA.jpg')

# =============================================================================
# 2. LOGIKA BACKEND & MODEL
# =============================================================================
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
except Exception as e:
    st.error(f"Error System: {str(e)}")
    MODEL_LOADED = False

def scale_features(df_input, scaler_X):
    feature_names = SCALER_FEATURES
    X = df_input[feature_names]
    y = df_input["Lindi"] 
    X_scaled = scaler_X.transform(X)
    return X, y, X_scaled

# =============================================================================
# 3. SIDEBAR NAVIGATION (UPDATED LOGIC)
# =============================================================================
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.markdown("<h2 style='text-align:center; color:#00C6FF;'>AI LEACHATE</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CSS Radio Button
    st.markdown("""
        <style>
        /* Container Dasar Tombol */
        div[role="radiogroup"] label {
            background-color: rgba(255,255,255,0.05) !important;
            color: #ccc !important;
            padding: 12px 15px;
            border-radius: 10px;
            margin-bottom: 8px;
            border: 1px solid rgba(255,255,255,0.05);
            
            /* Transisi Halus untuk efek geser */
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
        }
        
        /* Efek Hover: GESER KANAN 6px */
        div[role="radiogroup"] label:hover {
            background-color: rgba(0, 198, 255, 0.15) !important;
            border-color: #00C6FF;
            
            /* INI BAGIAN ANIMASI GESERNYA */
            transform: translateX(6px); 
            
            color: white !important;
            box-shadow: -3px 0 10px rgba(0, 198, 255, 0.2);
        }
        
        /* Efek Aktif (Terpilih) */
        div[role="radiogroup"] label[data-checked="true"] {
             background: linear-gradient(90deg, rgba(0,198,255,0.2), transparent) !important;
             border-left: 4px solid #00C6FF !important;
             color: #00C6FF !important;
             font-weight: bold;
             
             /* Geser sedikit untuk menandakan aktif */
             transform: translateX(2px);
        }
        </style>
    """, unsafe_allow_html=True)

    # --- STATE MANAGEMENT ---
    # Perbarui opsi menu
    menu_options = ["🏠 Home", "📊 Testing Batch", "📝 Testing Single"]
    
    if "active_menu" not in st.session_state:
        st.session_state["active_menu"] = menu_options[0]

    # --- DIALOG POPUP ---
    @st.dialog("⚠️ Konfirmasi Perpindahan")
    def confirm_switch(new_selection):
        st.markdown(
            f"""
            <div style='text-align: center; color: #e0e0e0;'>
                <p style='font-size: 1.1em;'>
                    Anda akan berpindah ke <br>
                    <b style='color: #00C6FF;'>{new_selection}</b>
                </p>
                <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; font-size: 0.9em; margin: 20px 0; border: 1px solid rgba(255,255,255,0.1);'>
                    ❗ Tindakan ini akan <b>MENGHAPUS (RESET)</b> semua hasil analisis dan data yang sedang tampil saat ini.
                </div>
                <p>Apakah Anda yakin ingin melanjutkan?</p>
            </div>
            """, unsafe_allow_html=True
        )
        
        col_y, col_n = st.columns(2)
        with col_y:
            if st.button("✅ YA, Reset", type="primary", use_container_width=True):
                # 1. Hapus Data Session
                keys = ['df_processed', 'X_train_scaled', 'y_train', 'X_test_scaled', 'y_test']
                for k in keys: 
                    if k in st.session_state: del st.session_state[k]
                
                # 2. Update Halaman Aktif
                st.session_state["active_menu"] = new_selection
                st.rerun()
                
        with col_n:
            if st.button("❌ Batal", use_container_width=True):
                # Kembalikan Radio ke posisi semula
                st.session_state["nav_radio"] = st.session_state["active_menu"]
                st.rerun()

    # --- CALLBACK (MODIFIED LOGIC) ---
    def on_nav_change():
        target = st.session_state["nav_radio"]
        current = st.session_state["active_menu"]
        
        # JIKA sedang di HOME -> Langsung pindah (Tidak perlu peringatan)
        if current == menu_options[0]: 
            st.session_state["active_menu"] = target
            
        # JIKA sedang di Skenario 2 atau 3 -> Munculkan Peringatan
        elif target != current:
            confirm_switch(target)

    # --- RENDER RADIO BUTTON ---
    selection = st.radio(
        "NAVIGASI UTAMA",
        menu_options,
        key="nav_radio",
        index=menu_options.index(st.session_state["active_menu"]),
        on_change=on_nav_change
    )
    
    menu = st.session_state["active_menu"]

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 12px; color: #888; background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px;'>
            Developed by:<br>
            <strong style='color: #00C6FF; font-size: 13px; letter-spacing: 0.5px;'>Universitas Muhammadiyah Malang</strong>
        </div>
        """, unsafe_allow_html=True
    )

# =============================================================================
# 4. KONTEN UTAMA
# =============================================================================

# HEADER
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 40px;'>
        <h1 style='margin: 0; font-weight: 800; letter-spacing: 1.5px; text-shadow: 0 4px 8px rgba(0,0,0,0.5);'>PREDIKSI VOLUME AIR LINDI</h1>
        <p style='font-size: 16px; opacity: 0.7; color: #00C6FF !important; font-weight: 500; letter-spacing: 2px; text-transform: uppercase;'>Artificial Neural Network Implementation</p>
        <div style='height: 3px; width: 60px; background: #00C6FF; margin: 20px auto; border-radius: 2px; box-shadow: 0 0 10px #00C6FF;'></div>
    </div>
    """, unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# HALAMAN: HOME
# -----------------------------------------------------------------------------
if menu == "🏠 Home":
    # Konten Home (Placeholder)
    st.markdown(
        """
        <div style='text-align: center; padding: 20px;'>
            <p style='font-size: 18px; color: #ccc; margin-top: 10px;'>
                Sistem ini dirancang untuk membantu memprediksi volume air lindi (leachate) 
                di Tempat Pembuangan Akhir menggunakan algoritma Jaringan Syaraf Tiruan (ANN).
            </p>
            <br>
            <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); display: inline-block; text-align: left;'>
                <h4 style='color: #00C6FF; margin-bottom: 15px;'>🔍 Fitur Utama:</h4>
                <ul style='list-style-type: none; padding: 0; color: #ddd;'>
                    <li style='margin-bottom: 10px;'>📊 <b>Testing Batch:</b> Evaluasi model dengan dataset besar (CSV).</li>
                    <li style='margin-bottom: 10px;'>📝 <b>Testing Single:</b> Prediksi harian cepat dengan input manual.</li>
                    <li style='margin-bottom: 10px;'>📈 <b>Visualisasi:</b> Grafik interaktif Time Series dan Scatter Plot.</li>
                </ul>
            </div>
            <br><br>
            <p style='font-size: 14px; color: #888;'>Silakan pilih menu di sidebar kiri untuk memulai analisis.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# SKENARIO 2: TESTING BATCH
# -----------------------------------------------------------------------------
elif menu == "📊 Testing Batch":
    st.markdown("### 📂 Upload File CSV (Evaluasi Lengkap)")
    
    uploaded_file = st.file_uploader("Upload dataset CSV terproses", type=["csv"], key="upload_csv_s2")

    if uploaded_file and MODEL_LOADED:
        df = pd.read_csv(uploaded_file)
        
        if 'TANGGAL' in df.columns:
            try:
                df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
                df = df.sort_values(by='TANGGAL').reset_index(drop=True)
            except: pass

        if 'Lindi' not in df.columns:
             st.error("❌ File CSV harus memiliki kolom 'Lindi'.")
        else:
            max_index = len(df) - 1
            st.markdown("#### 🎚️ Atur Rentang Data")
            start_index, end_index = st.slider("", 0, max_index, (0, max_index))
            df_filtered = df.iloc[start_index:end_index+1]

            if not df_filtered.empty:
                with st.expander("📄 Lihat Dataframe Terfilter"):
                    st.dataframe(df_filtered, use_container_width=True)

                try:
                    X, y_true, X_scaled = scale_features(df_filtered, scaler_X)
                    y_pred = model.predict(X_scaled)
                except KeyError as e:
                    st.error(f"Kolom fitur {e} tidak ditemukan."); st.stop()
                
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                pearson_r, p_value = pearsonr(y_true.values.flatten(), y_pred.flatten())

                st.markdown("---")
                st.markdown("### 📊 Hasil Evaluasi Statistik")
                col1, col2, col3 = st.columns(3)
                col1.metric("📉 MSE", f"{mse:.2f}")
                col2.metric("📊 RMSE", f"{rmse:.2f}")
                col3.metric("📏 MAE", f"{mae:.2f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("📈 R² Score", f"{r2:.4f}")
                col5.metric("✅ Pearson's r", f"{pearson_r:.4f}")
                col6.metric("🔬 P-value", f"{p_value:.2e}")

                st.markdown("---")

                # Grafik 1
                st.markdown("#### 1. 📈 Time Series: Prediksi vs Aktual")
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                fig1.patch.set_alpha(0); ax1.patch.set_alpha(0)
                ax1.plot(df_filtered.index, y_true, label="Aktual", marker="o", linestyle='-', markersize=4, color='#00E676', alpha=0.7)
                ax1.plot(df_filtered.index, y_pred, label="Prediksi", marker="x", linestyle='--', markersize=4, color='#00B4DB')
                ax1.set_xlabel("Indeks Data", color='#ccc'); ax1.set_ylabel("Lindi (m³/hari)", color='#ccc')
                ax1.tick_params(colors='#ccc')
                ax1.spines['bottom'].set_color('#555'); ax1.spines['left'].set_color('#555')
                ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
                ax1.legend(facecolor='#1c1f26', labelcolor='white', edgecolor='#333')
                ax1.grid(True, linestyle=':', color='white', alpha=0.1)
                st.pyplot(fig1)

                # Grafik 2
                st.markdown("#### 2. 📌 Scatter Plot: Korelasi")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                fig2.patch.set_alpha(0); ax2.patch.set_alpha(0)
                ax2.scatter(y_true, y_pred, alpha=0.7, color="#FFC107", edgecolor="white", s=60)
                ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'w--', lw=2, label="Perfect Fit")
                ax2.set_xlabel("Aktual (m³/hari)", color='#ccc'); ax2.set_ylabel("Prediksi (m³/hari)", color='#ccc')
                ax2.set_title(f"Pearson r = {pearson_r:.2f}", color='#ccc')
                ax2.tick_params(colors='#ccc')
                ax2.spines['bottom'].set_color('#555'); ax2.spines['left'].set_color('#555')
                ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
                ax2.legend(facecolor='#1c1f26', labelcolor='white', edgecolor='#333')
                ax2.grid(True, linestyle=':', color='white', alpha=0.1)
                st.pyplot(fig2)
                
                # Grafik 3
                st.markdown("#### 3. 📊 Error Absolut per Data")
                fig3, ax3 = plt.subplots(figsize=(10, 5)) 
                fig3.patch.set_alpha(0); ax3.patch.set_alpha(0)
                if 'TANGGAL' in df_filtered.columns:
                    monthly_labels = df_filtered['TANGGAL'].dt.strftime('%b-%y')
                    month_indices = monthly_labels.drop_duplicates().index.tolist()
                    month_names = monthly_labels.loc[month_indices].tolist()
                    ax3.set_xlabel("Bulan", color='#ccc')
                    ax3.set_xticks(month_indices)
                    ax3.set_xticklabels(month_names, rotation=45, ha='right', color='#ccc')
                    for i in range(len(month_indices) - 1):
                        start_of_next_month = month_indices[i+1]
                        ax3.axvline(x=start_of_next_month - 0.5, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
                else:
                    ax3.set_xlabel("Indeks Data", color='#ccc')
                    ax3.tick_params(axis='x', colors='#ccc')

                errors = np.abs(y_true.values.flatten() - y_pred.flatten())
                ax3.bar(df_filtered.index, errors, color='#FF5252', alpha=0.8)
                ax3.set_ylabel("Error Absolut (m³/hari)", color='#ccc')
                ax3.tick_params(axis='y', colors='#ccc')
                ax3.spines['bottom'].set_color('#555'); ax3.spines['left'].set_color('#555')
                ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
                ax3.grid(axis='y', linestyle=':', color='white', alpha=0.1)
                st.pyplot(fig3)
            else:
                st.warning("⚠️ Rentang data kosong.")

# -----------------------------------------------------------------------------
# SKENARIO 3: TESTING SINGLE
# -----------------------------------------------------------------------------
elif menu == "📝 Testing Single":
    if not MODEL_LOADED: st.stop()
    
    st.markdown("### 📝 Input Parameter Harian")
    
    with st.container():
        st.markdown("<div style='background:rgba(255,255,255,0.02); padding:25px; border-radius:15px; border: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            tn = st.number_input("TN (Suhu Min)", 22.20)
            tx = st.number_input("TX (Suhu Maks)", 31.06)
            tavg = st.number_input("TAVG (Suhu Rata)", 25.21)
        with c2:
            rh = st.number_input("RH_AVG (Kelembaban)", 83.32)
            rr = st.number_input("RR (Curah Hujan)", 10.13)
            ss = st.number_input("SS (Matahari)", 5.26)
        with c3:
            ffx = st.number_input("FF_X (Angin Max)", 2.74)
            dddx = st.number_input("DDD_X (Arah Max)", 198.0)
            ffavg = st.number_input("FF_AVG (Angin Rata)", 1.41)
            
        st.markdown("---")
        st.markdown("#### Arah Angin Dominan")
        wind_opts = ['DDD_CAR_C', 'DDD_CAR_E', 'DDD_CAR_NW', 'DDD_CAR_S', 'DDD_CAR_SE', 'DDD_CAR_SW', 'DDD_CAR_W']
        dom_dir = st.radio("Pilih arah:", wind_opts, horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 HITUNG PREDIKSI LINDI", use_container_width=True):
        try:
            feat = {
                'TN': tn, 'TX': tx, 'TAVG': tavg, 'RH_AVG': rh, 'RR': rr, 'SS': ss,
                'FF_X': ffx, 'DDD_X': dddx, 'FF_AVG': ffavg
            }
            for w in wind_opts: feat[w] = 0
            feat[dom_dir] = 1
            
            df_in = pd.DataFrame([feat])
            df_in = df_in[SCALER_FEATURES]
            X_sc = scaler_X.transform(df_in)
            y_pred = model.predict(X_sc)[0]
            
            st.markdown("---")
            st.markdown(
                f"""
                <div style='
                    animation: popIn 0.8s cubic-bezier(0.68, -0.55, 0.27, 1.55);
                    background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
                    color: white;
                    padding: 40px;
                    border-radius: 20px;
                    text-align: center;
                    box-shadow: 0 10px 40px rgba(0, 114, 255, 0.4);
                    margin-top: 15px;
                    border: 1px solid rgba(255,255,255,0.2);
                '>
                    <h3 style='color: white !important; margin:0; opacity:0.9; letter-spacing: 1px;'>ESTIMASI VOLUME LINDI</h3>
                    <h1 style='font-size: 4.5em; margin: 15px 0; font-weight: 800; text-shadow: 0px 0px 20px rgba(255,255,255,0.5); color: white !important;'>
                        {y_pred:.2f}
                    </h1>
                    <p style='font-size: 1.3em; font-weight: 600; color: white !important;'>Meter Kubik (m³) / Hari</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Gagal memprediksi: {str(e)}")


