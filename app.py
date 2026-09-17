from pathlib import Path

import streamlit as st

from portable_forest import load_portable_forest


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EconoMind AI",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name("model_ekonomi_1m.joblib")
    return load_portable_forest(model_path)


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("📊 EconoMind AI")

st.caption(
    "Sistem Prediksi Keputusan Harga Pasar — "
    "IT Incubation DSE Teknik Informatika UIN Malang"
)

st.divider()


# ============================================================
# INPUT
# ============================================================

biaya_produksi = st.slider(
    "Biaya Produksi per Unit (Rp)",
    min_value=20_000,
    max_value=400_000,
    value=75_000,
    step=5_000
)

inflasi = st.slider(
    "Persentase Inflasi Daerah (%)",
    min_value=-1.0,
    max_value=15.0,
    value=3.5,
    step=0.1
)

kompetitor = st.slider(
    "Jumlah Kompetitor Aktif",
    min_value=0,
    max_value=30,
    value=7,
    step=1
)

permintaan = st.selectbox(
    "Tren Permintaan Konsumen",
    [
        "Rendah",
        "Sedang",
        "Tinggi"
    ],
    index=1
)

kelangkaan = st.selectbox(
    "Kelangkaan Bahan Baku",
    [
        "Normal",
        "Langka"
    ]
)

sentimen = st.selectbox(
    "Sentimen Media Sosial",
    [
        "Negatif",
        "Netral",
        "Positif"
    ],
    index=1
)


# ============================================================
# MAPPING
# ============================================================

dict_permintaan = {
    "Rendah": -0.7,
    "Sedang": 0.0,
    "Tinggi": 0.7
}

dict_kelangkaan = {
    "Normal": 25.0,
    "Langka": 75.0
}

dict_sentimen = {
    "Negatif": -0.7,
    "Netral": 0.0,
    "Positif": 0.7
}


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Prediksi Keputusan Harga",
    type="primary",
    use_container_width=True
):

    input_data = [
        biaya_produksi,
        inflasi,
        kompetitor,
        dict_permintaan[permintaan],
        dict_kelangkaan[kelangkaan],
        dict_sentimen[sentimen],
    ]

    hasil = model.predict_one(input_data)

    st.divider()

    if hasil == "NAIK":

        st.success(
            "📈 Prediksi: NAIK"
        )

        st.write(
            "Harga jual direkomendasikan meningkat berdasarkan "
            "kondisi biaya produksi, pasokan, permintaan, dan "
            "indikator pasar."
        )

    elif hasil == "TURUN":

        st.error(
            "📉 Prediksi: TURUN"
        )

        st.write(
            "Harga jual direkomendasikan menurun berdasarkan "
            "tekanan persaingan dan kondisi pasar."
        )

    else:

        st.info(
            "➖ Prediksi: STABIL"
        )

        st.write(
            "Harga jual direkomendasikan tetap karena kondisi "
            "pasar relatif seimbang."
        )
