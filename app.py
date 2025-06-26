import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import os 

# --- 0. Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Prediksi Konsumsi Listrik")

st.title("💡 Aplikasi Prediksi Konsumsi Listrik")
st.write("Unggah file CSV Anda yang berisi data fitur untuk mendapatkan prediksi konsumsi listrik (kWh).")

# --- 1. Muat Aset yang Disimpan ---
# Pastikan file-file .pkl berada di direktori yang sama dengan app.py
MODEL_PATH = "electricity_consumption_linear_regression_model.pkl"
SCALER_PATH = "scaler.pkl"
CAPPING_BOUNDS_PATH = "capping_bounds.pkl"
FINAL_FEATURE_COLUMNS_PATH = "final_feature_columns.pkl"
OHE_COLS_PATH = "categorical_cols_for_ohe.pkl"
NUM_COLS_TO_SCALE_PATH = "numerical_cols_to_scale.pkl"

@st.cache_resource # Cache resource agar model/scaler tidak dimuat ulang setiap kali ada interaksi
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        capping_bounds = joblib.load(CAPPING_BOUNDS_PATH)
        final_feature_columns = joblib.load(FINAL_FEATURE_COLUMNS_PATH)
        ohe_categorical_cols = joblib.load(OHE_COLS_PATH)
        numerical_features_to_scale = joblib.load(NUM_COLS_TO_SCALE_PATH)
        return model, scaler, capping_bounds, final_feature_columns, ohe_categorical_cols, numerical_features_to_scale
    except FileNotFoundError as e:
        st.error(f"Error: File aset tidak ditemukan. Pastikan semua file .pkl berada di direktori yang sama. Detail: {e}")
        st.stop() 
    except Exception as e:
        st.error(f"Error saat memuat aset. Pastikan format file benar. Detail: {e}")
        st.stop()

model, scaler, capping_bounds, final_feature_columns, ohe_categorical_cols, numerical_features_to_scale = load_assets()

st.success("Model dan Preprocessor berhasil dimuat! Siap memprediksi. ✅")

# --- 2. Widget Unggah File ---
st.subheader("Unggah File Data Anda")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    df_predict_raw = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah (5 baris pertama):")
    st.dataframe(df_predict_raw.head())
    
    # Simpan ID jika ada (untuk submission)
    predict_ids = None
    if 'ID' in df_predict_raw.columns:
        predict_ids = df_predict_raw['ID']
        df_predict_raw = df_predict_raw.drop(columns=['ID'])
        st.info("Kolom 'ID' ditemukan dan disimpan untuk hasil akhir.")
    else:
        predict_ids = df_predict_raw.index
        st.warning("Kolom 'ID' tidak ditemukan di file yang diunggah. Menggunakan indeks sebagai ID.")

    st.subheader("Memulai Preprocessing Data...")

    # --- 3. Lakukan Preprocessing yang SAMA PERSIS pada data yang diunggah ---
    # Ini adalah langkah-langkah yang sama seperti di notebook pelatihan Anda

    try:
        df_processed = df_predict_raw.copy()

        # (a) Konversi 'date' dan Ekstraksi Fitur Waktu
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'])
            df_processed['year'] = df_processed['date'].dt.year
            df_processed['year'] = df_processed['year'].astype('object') # Tetap object karena di OHE
            df_processed['month'] = df_processed['date'].dt.month_name()
            df_processed['day_of_week'] = df_processed['date'].dt.day_name()
            df_processed = df_processed.drop(columns=['date'], errors='ignore') # Hapus kolom date asli
        else:
            st.error("Error: Kolom 'date' tidak ditemukan di file yang diunggah. Pastikan format data benar.")
            st.stop()

        # (b) Penanganan 'wind_direction_10m_dominant' (jika ada)
        if 'wind_direction_10m_dominant' in df_processed.columns:
            # Perhatikan: Ini diasumsikan numerik dan akan discale, bukan di OHE
            df_processed['wind_direction_10m_dominant'] = df_processed['wind_direction_10m_dominant'].round().astype('Int64')
        
        st.write("Preprocessing Awal (Ekstraksi Tanggal, Konversi Tipe) Selesai. ✅")

        # (c) Penanganan Outlier (Capping IQR) - Menggunakan batas dari data training
        # Iterasi melalui kolom yang dicapping saat training dan terapkan pada data upload
        for col in capping_bounds.keys(): # Kunci di capping_bounds adalah nama kolom yang dicapping
            if col in df_processed.columns and col != 'electricity_consumption': # Target tidak ada di data test
                upper_bound = capping_bounds[col]['upper_bound']
                df_processed[col] = df_processed[col].clip(upper=upper_bound, lower=df_processed[col].min())
        st.write("Penanganan Outlier (Capping IQR) Selesai. ✅")

        # (d) One-Hot Encoding
        # Penting: Pastikan kolom kategorikal di data yang diunggah sama persis dengan yang di training
        # 'ohe_categorical_cols' adalah daftar nama kolom kategorikal asli dari training
        
        # OHE data yang diunggah
        df_encoded = pd.get_dummies(df_processed, columns=ohe_categorical_cols, drop_first=False)
        
        # Konversi boolean ke int jika pd.get_dummies menghasilkan True/False
        for col in df_encoded.columns:
            if df_encoded[col].dtype == bool:
                df_encoded[col] = df_encoded[col].astype(int)
        
        st.write("One-Hot Encoding Selesai. ✅")

        # (e) Penyelarasan Kolom (CRITICAL)
        # Pastikan kolom df_encoded cocok dengan final_feature_columns (dari X_train_scaled.columns)
        # Ini akan menambah kolom yang hilang (diisi 0) dan menghapus kolom yang tidak ada di training
        missing_cols_in_uploaded = set(final_feature_columns) - set(df_encoded.columns)
        new_cols_in_uploaded = set(df_encoded.columns) - set(final_feature_columns)

        for col in missing_cols_in_uploaded:
            df_encoded[col] = 0 # Tambah kolom yang hilang, isi dengan 0
            
        df_encoded = df_encoded[final_feature_columns] # Reindex dan atur ulang urutan kolom
        
        if new_cols_in_uploaded:
            st.warning(f"Peringatan: Ada kolom baru di file yang diunggah yang tidak ada di data training: {new_cols_in_uploaded}. Kolom tersebut diabaikan.")
        
        st.write("Penyelarasan Kolom Selesai. ✅")


        # (f) Standarisasi
        # 'numerical_features_to_scale' adalah daftar kolom numerik yang perlu discale (dari training)
        df_final_features = df_encoded.copy() # Ambil DataFrame yang sudah di-encoded dan diselaraskan
        
        # Filter hanya kolom yang benar-benar ada dan perlu diskala
        actual_numerical_features_to_scale_in_upload = [col for col in numerical_features_to_scale if col in df_final_features.columns]

        if actual_numerical_features_to_scale_in_upload:
            df_final_features[actual_numerical_features_to_scale_in_upload] = scaler.transform(df_final_features[actual_numerical_features_to_scale_in_upload])
            st.write("Standarisasi Fitur Numerik Selesai. ✅")
        else:
            st.warning("Tidak ada fitur numerik yang diidentifikasi untuk standarisasi di file yang diunggah.")
            
        st.success("Semua tahap Preprocessing Selesai! ✨")

        # --- 4. Lakukan Prediksi ---
        st.subheader("Melakukan Prediksi...")
        predictions = model.predict(df_final_features)
        
        # Pastikan prediksi non-negatif
        predictions[predictions < 0] = 0

        st.success("Prediksi Berhasil! 🎉")

        # --- 5. Tampilkan Hasil dan Opsi Unduh ---
        st.subheader("Hasil Prediksi Konsumsi Listrik")
        
        # Buat DataFrame hasil untuk ditampilkan/diunduh
        result_df = pd.DataFrame({
            'ID': predict_ids, # Menggunakan ID yang disimpan
            'electricity_consumption': predictions
        })

        st.dataframe(result_df)

        # Opsi Unduh Hasil
        csv_file = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil Prediksi (CSV)",
            data=csv_file,
            file_name="predicted_electricity_consumption.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan selama preprocessing atau prediksi: {e}")
        st.write("Silakan periksa kembali format file dan nama kolom Anda.")
