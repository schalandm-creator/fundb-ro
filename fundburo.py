
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from supabase import create_client
import os
import io
import uuid

# ── Konfiguration ────────────────────────────────────────────────
MODEL_PATH    = "keras_Model.h5"           # muss im gleichen Ordner liegen
LABELS_PATH   = "labels.txt"
IMG_SIZE      = (224, 224)
BUCKET_NAME   = "wardrobe"                 # dein Bucket-Name

# Supabase via Streamlit Secrets (oder .env)
supabase = create_client(
    st.secrets["connections"]["supabase"]["SUPABASE_URL"],
    st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
)

# ── Cache: Modell + Labels nur einmal laden ─────────────────────
@st.cache_resource
def load_everything():
    # Labels laden
    if not os.path.isfile(LABELS_PATH):
        st.error(f"labels.txt nicht gefunden!")
        st.stop()
    
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    class_names = [line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) == 2 else line for line in lines]
    
    # Modell laden
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Modell nicht gefunden: {MODEL_PATH}")
        st.stop()
    
    model = load_model(MODEL_PATH, compile=False)
    return class_names, model

class_names, model = load_everything()

# ── Bild vorbereiten ─────────────────────────────────────────────
def prepare_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32)
        normalized = (image_array / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)
    except Exception as e:
        st.error(f"Fehler beim Verarbeiten des Bildes: {e}")
        return None

# ── Upload zu Supabase Storage ──────────────────────────────────
def upload_to_supabase(image_bytes, filename):
    file_ext = os.path.splitext(filename)[1].lower() or ".jpg"
    unique_name = f"{uuid.uuid4()}{file_ext}"
    # Hier ohne User-Ordner – für echten User: f"user_{user_id}/{unique_name}"
    path = unique_name
    
    try:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=path,
            file=image_bytes,
            file_options={"content-type": f"image/{file_ext.lstrip('.') or 'jpeg'}"}
        )
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path)
        return path, public_url
    except Exception as e:
        st.warning(f"Upload nach Supabase fehlgeschlagen: {e}")
        return None, None

# ── Streamlit UI ─────────────────────────────────────────────────
st.title("Kleidungs-Klassifizierer 🧥👕")
st.write("Lade ein Bild hoch (Hose, Pullover, Jacke, Sonstiges) – KI sagt dir, was es ist!")

uploaded_file = st.file_uploader("Wähle ein Bild...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # zurücksetzen für weitere Nutzung
    st.image(image_bytes, caption="Dein hochgeladenes Bild", use_container_width=True)

    with st.spinner("Analysiere Bild..."):
        data = prepare_image(image_bytes)
        if data is not None:
            prediction = model.predict(data, verbose=0)[0]
            idx = np.argmax(prediction)
            confidence = float(prediction[idx])
            category = class_names[idx]

            # Ergebnis anzeigen
            st.subheader("Ergebnis")
            st.success(f"**{category.upper()}**  (Sicherheit: {confidence:.1%})")

            st.markdown("**Alle Wahrscheinlichkeiten:**")
            for name, prob in zip(class_names, prediction):
                st.write(f"{name:12}: {float(prob):6.1%}")

            # ── Optional: Speichern in Supabase ──────────────────────
            if st.button("Bild + Ergebnis in Supabase speichern"):
                with st.spinner("Speichere..."):
                    storage_path, public_url = upload_to_supabase(image_bytes, uploaded_file.name)
                    
                    if storage_path:
                        # User-ID: Für Demo hartcodiert – später via Auth
                        user_id = "deine-test-user-uuid-hier"   # ← ändern!

                        data_to_insert = {
                            "user_id": user_id,
                            "storage_path": storage_path,
                            "predicted_category": category,
                            "confidence": confidence
                        }
                        try:
                            res = supabase.table("wardrobe_items").insert(data_to_insert).execute()
                            st.success(f"Gespeichert! (ID: {res.data[0]['id']})")
                            if public_url:
                                st.markdown(f"**Öffentliche URL:** {public_url}")
                        except Exception as e:
                            st.error(f"DB-Fehler: {e}")
                    else:
                        st.warning("Speichern abgebrochen (Upload fehlgeschlagen)")
else:
    st.info("Lade ein Bild hoch, um zu starten.")

# Footer
st.markdown("---")
st.caption("Basierend auf deinem Teachable-Machine-Modell • Streamlit + Supabase • 2026")
