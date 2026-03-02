# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from datetime import datetime

# ─── TensorFlow / Keras ───────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Custom objects → um das 'groups':1 Problem zu umgehen
custom_objects = {
    'DepthwiseConv2D': lambda **kwargs: DepthwiseConv2D(**{k: v for k, v in kwargs.items() if k != 'groups'})
}

# ─── Supabase ─────────────────────────────────────────────────────
from st_supabase_connection import SupabaseConnection

# Connection (secrets.toml wird automatisch verwendet)
conn = st.connection(
    name="supabase",
    type=SupabaseConnection,
    ttl=None
)

# ─── Konfiguration ────────────────────────────────────────────────
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)

# ─── Labels laden ─────────────────────────────────────────────────
@st.cache_data
def lade_labels():
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        klassen = [line.split(" ", 1)[1].strip() if " " in line else line for line in lines]
        return klassen
    except Exception as e:
        st.error(f"labels.txt Fehler: {e}")
        return ["hose", "pullover", "jacke", "sonstiges"]

# ─── Modell laden ─────────────────────────────────────────────────
@st.cache_resource
def lade_modell():
    try:
        # Wichtig: custom_objects + compile=False
        model = load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False   # ← erlaubt das Laden trotz potenzieller Warnungen
        )
        return model
    except Exception as e:
        st.error(f"Modell-Ladefehler:\n{str(e)}")
        st.stop()

# ─── Bild vorbereiten (Teachable Machine kompatibel) ──────────────
def bild_vorbereiten(bild):
    bild = ImageOps.fit(bild, IMG_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(bild).astype("float32")
    normalisiert = (array / 127.5) - 1.0
    return np.expand_dims(normalisiert, axis=0)

# ─── Ergebnis in Supabase speichern ───────────────────────────────
def speichere_ergebnis(klasse, sicherheit, filename="unbekannt"):
    try:
        daten = {
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_class": klasse,
            "confidence": round(float(sicherheit), 2),
            "filename": filename
        }
        conn.table("predictions").insert(daten).execute()
    except Exception as e:
        st.warning(f"Supabase Speicherfehler: {str(e)}")

# ─── Streamlit App ────────────────────────────────────────────────
st.set_page_config(page_title="Kleidungs-Klassifizierer", layout="centered")

st.title("🧥 Kleidungs-Klassifizierer")
st.write("Pullover · Jacke · Hose · Sonstiges")

klassen = lade_labels()
modell  = lade_modell()

st.caption("Erkannte Klassen: " + ", ".join(klassen))

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("Klassifizieren"):
        with st.spinner("Analyse läuft …"):
            try:
                input_array = bild_vorbereiten(image)
                prediction = modell.predict(input_array, verbose=0)[0]

                idx = np.argmax(prediction)
                confidence = float(prediction[idx]) * 100
                result_class = klassen[idx]

                st.success(f"**{result_class}**")
                st.write(f"Sicherheit: **{confidence:.1f} %**")

                st.subheader("Wahrscheinlichkeiten")
                for name, prob in zip(klassen, prediction):
                    p = float(prob) * 100
                    st.progress(p / 100)
                    st.write(f"{name}: {p:.1f} %")

                speichere_ergebnis(result_class, confidence, uploaded_file.name)

            except Exception as e:
                st.error(f"Fehler bei der Vorhersage: {str(e)}")

# ─── Debug / Übersicht ────────────────────────────────────────────
if st.button("Letzte 5 Einträge aus Supabase"):
    try:
        data = conn.table("predictions") \
                   .select("*") \
                   .order("timestamp", desc=True) \
                   .limit(5) \
                   .execute().data

        if data:
            for row in data:
                st.write(f"{row.get('timestamp','–')} | **{row.get('predicted_class','–')}** | {row.get('confidence','–')}% | {row.get('filename','–')}")
        else:
            st.info("Noch keine Einträge")
    except Exception as e:
        st.error(f"Datenbankfehler: {str(e)}")

st.markdown("---")
st.caption("Modell: keras_model.h5 | Supabase: predictions | 224×224")
