# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from datetime import datetime

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# Supabase
from st_supabase_connection import SupabaseConnection

# Supabase Connection (secrets.toml wird automatisch genutzt)
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
            zeilen = f.readlines()
        klassen = []
        for zeile in zeilen:
            zeile = zeile.strip()
            if zeile and not zeile.startswith("#"):
                if " " in zeile:
                    klassen.append(zeile.split(" ", 1)[1].strip())
                else:
                    klassen.append(zeile)
        return klassen
    except Exception as e:
        st.error(f"labels.txt konnte nicht geladen werden: {e}")
        return ["hose", "pullover", "Jacken", "sonstiges"]

# ─── Modell laden ─────────────────────────────────────────────────
@st.cache_resource
def lade_modell():
    try:
        modell = load_model(MODEL_PATH, compile=False)
        st.info(f"Modell erfolgreich geladen (TensorFlow {tf.__version__})")
        return modell
    except Exception as e:
        st.error(
            f"Modell konnte nicht geladen werden:\n\n{str(e)}\n\n"
            "→ Lösung: tensorflow==2.15.0 (oder 2.12.0) in requirements.txt verwenden\n"
            "→ Python-Version: 3.9–3.11 empfohlen (nicht 3.12+)"
        )
        st.stop()

# ─── Bild vorbereiten ─────────────────────────────────────────────
def bild_vorbereiten(bild):
    bild = ImageOps.fit(bild, IMG_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(bild).astype("float32")
    normalisiert = (array / 127.5) - 1.0
    return np.expand_dims(normalisiert, axis=0)

# ─── In Supabase speichern ────────────────────────────────────────
def speichere_ergebnis(klasse, sicherheit, filename="unbekannt"):
    try:
        daten = {
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_class": klasse,
            "confidence": round(float(sicherheit), 2),
            "filename": filename,
        }
        conn.table("predictions").insert(daten).execute()
    except Exception as e:
        st.warning(f"Supabase-Speicherfehler: {e}")

# ─── Streamlit App ────────────────────────────────────────────────
st.set_page_config(page_title="Kleidungs-Klassifizierer", layout="centered")
st.title("🧥 Kleidungs-Klassifizierer")
st.write("Lade ein Bild hoch (Pullover, Jacke, Hose oder Sonstiges)")

klassen = lade_labels()
modell = lade_modell()

st.write("**Erwartete Klassen:**", ", ".join(klassen))

hochgeladenes_bild = st.file_uploader("Bild auswählen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if hochgeladenes_bild is not None:
    bild = Image.open(hochgeladenes_bild).convert("RGB")
    st.image(bild, caption="Dein hochgeladenes Bild", use_column_width=True)

    if st.button("Jetzt klassifizieren"):
        with st.spinner("Analysiere Bild..."):
            try:
                eingabe = bild_vorbereiten(bild)
                vorhersage = modell.predict(eingabe, verbose=0)[0]
                index = np.argmax(vorhersage)
                sicherheit = float(vorhersage[index]) * 100

                ergebnis = klassen[index]
                st.success(f"**Ergebnis:** {ergebnis}")
                st.write(f"Sicherheit: **{sicherheit:.1f} %**")

                st.subheader("Wahrscheinlichkeiten")
                for name, prob in zip(klassen, vorhersage):
                    prozent = float(prob) * 100
                    st.progress(prozent / 100)
                    st.write(f"{name}: {prozent:.1f} %")

                speichere_ergebnis(ergebnis, sicherheit, hochgeladenes_bild.name)

            except Exception as e:
                st.error(f"Fehler bei Vorhersage: {e}")

# ─── Letzte Vorhersagen ───────────────────────────────────────────
if st.button("Letzte 5 Vorhersagen aus Supabase"):
    try:
        response = conn.table("predictions") \
                       .select("*") \
                       .order("timestamp", desc=True) \
                       .limit(5) \
                       .execute()

        if response.data:
            st.subheader("Letzte Einträge")
            for row in response.data:
                st.write(f"{row.get('timestamp', '–')} | **{row.get('predicted_class', '–')}** | {row.get('confidence', '–')}% | {row.get('filename', '–')}")
        else:
            st.info("Noch keine Einträge")
    except Exception as e:
        st.error(f"Ladefehler: {e}")

st.markdown("---")
st.caption("Modell: keras_model.h5 | TensorFlow 2.15.0 empfohlen | 224×224")
