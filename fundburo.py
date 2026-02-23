# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import io

# ─── Konfiguration ────────────────────────────────────────────────
MODEL_PATH   = "keras_model.h5"
LABELS_PATH  = "labels.txt"
IMG_SIZE     = (224, 224)

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
                # "0 hose" → nur "hose"
                if " " in zeile:
                    klassen.append(zeile.split(" ", 1)[1].strip())
                else:
                    klassen.append(zeile)
        return klassen
    except Exception as e:
        st.error(f"labels.txt konnte nicht geladen werden: {e}")
        return ["hose", "pullover", "Jacken", "sonstiges"]  # Fallback


# ─── Modell laden (nur einmal) ────────────────────────────────────
@st.cache_resource
def lade_modell():
    try:
        modell = load_model(MODEL_PATH, compile=False)
        return modell
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden:\n{e}")
        st.stop()


# ─── Bild vorbereiten (genau wie Teachable Machine) ───────────────
def bild_vorbereiten(bild):
    # Auf 224×224 bringen (Center Crop + Resize)
    bild = ImageOps.fit(bild, IMG_SIZE, Image.Resampling.LANCZOS)
    
    # In Array umwandeln + Normalisieren [-1 .. +1]
    array = np.asarray(bild).astype("float32")
    normalisiert = (array / 127.5) - 1.0
    
    # Batch-Dimension hinzufügen
    return np.expand_dims(normalisiert, axis=0)


# ─── Streamlit App ────────────────────────────────────────────────
st.set_page_config(page_title="Kleidungs-Klassifizierer", layout="centered")

st.title("🧥 Kleidungs-Klassifizierer")
st.write("Lade ein Bild hoch (Pullover, Jacke, Hose oder Sonstiges)")

# Labels & Modell einmal laden
klassen = lade_labels()
modell  = lade_modell()

st.write("**Erwartete Klassen:**", ", ".join(klassen))

# Bild-Upload
hochgeladenes_bild = st.file_uploader("Bild auswählen (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

if hochgeladenes_bild is not None:
    # Bild anzeigen
    bild = Image.open(hochgeladenes_bild).convert("RGB")
    st.image(bild, caption="Dein hochgeladenes Bild", use_column_width=True)

    # Klassifizieren-Button
    if st.button("Jetzt klassifizieren"):
        with st.spinner("Analysiere Bild..."):
            try:
                eingabe = bild_vorbereiten(bild)
                vorhersage = modell.predict(eingabe, verbose=0)[0]

                index = np.argmax(vorhersage)
                sicherheit = float(vorhersage[index]) * 100

                st.success(f"**Ergebnis:** {klassen[index]}")
                st.write(f"Sicherheit: **{sicherheit:.1f} %**")

                # Alle Wahrscheinlichkeiten als Balken
                st.subheader("Wahrscheinlichkeiten")
                for i, (name, prob) in enumerate(zip(klassen, vorhersage)):
                    prozent = float(prob) * 100
                    st.progress(prozent / 100)
                    st.write(f"{name}: {prozent:.1f} %")

            except Exception as e:
                st.error(f"Fehler bei der Vorhersage: {e}")

# Hinweise unten
st.markdown("---")
st.caption("Modell: keras_Model.h5 | Labels: labels.txt | Auflösung: 224×224")
st.caption("Funktioniert am besten mit gut beleuchteten, zentrierten Kleidungsfotos")
