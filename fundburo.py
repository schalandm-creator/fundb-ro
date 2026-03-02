
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import io
from datetime import datetime

# ─── Supabase Verbindung ─────────────────────────────────────────────
conn = st.connection("supabase", type="supabase")   # Nutzt automatisch secrets.toml

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
        return load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden:\n{e}")
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
            # Optional: "user_id": st.session_state.user_id,  # falls du Login hast
        }
        response = conn.table("predictions").insert(daten).execute()
        if hasattr(response, "error") and response.error:
            st.warning("Konnte nicht in Supabase speichern: " + str(response.error))
    except Exception as e:
        st.warning(f"Supabase Fehler: {e}")

# ─── Streamlit App ────────────────────────────────────────────────
st.set_page_config(page_title="Kleidungs-Klassifizierer", layout="centered")
st.title("🧥 Kleidungs-Klassifizierer")
st.write("Lade ein Bild hoch → Klassifizieren → Ergebnis wird in Supabase gespeichert")

klassen = lade_labels()
modell = lade_modell()

st.write("**Erwartete Klassen:**", ", ".join(klassen))

hochgeladenes_bild = st.file_uploader("Bild auswählen (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

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

                # Alle Wahrscheinlichkeiten
                st.subheader("Wahrscheinlichkeiten")
                for i, (name, prob) in enumerate(zip(klassen, vorhersage)):
                    prozent = float(prob) * 100
                    st.progress(prozent / 100)
                    st.write(f"{name}: {prozent:.1f} %")

                # ─── Ergebnis in Supabase speichern ───
                speichere_ergebnis(ergebnis, sicherheit, hochgeladenes_bild.name)

            except Exception as e:
                st.error(f"Fehler: {e}")

# ─── Optional: Letzte Vorhersagen anzeigen ─────────────────────────
if st.button("Letzte 5 Vorhersagen aus Supabase laden"):
    try:
        response = conn.table("predictions") \
                       .select("*") \
                       .order("timestamp", desc=True) \
                       .limit(5) \
                       .execute()
        
        if response.data:
            st.subheader("Letzte Klassifikationen")
            for row in response.data:
                st.write(f"{row['timestamp']} | **{row['predicted_class']}** | {row['confidence']}% | Datei: {row['filename']}")
        else:
            st.info("Noch keine Einträge vorhanden.")
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")

st.markdown("---")
st.caption("Modell: keras_model.h5 | Supabase: predictions-Tabelle | 224×224")
