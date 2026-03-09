#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kleidungs-Klassifizierer (Hose / Pullover / Jacken / Sonstiges)
+ Speichert das Bild in Supabase Storage
"""

import os
import sys
import uuid
from datetime import datetime

# ── Supabase ────────────────────────────────────────────────
try:
    from supabase import create_client, Client
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("FEHLER: supabase oder python-dotenv nicht installiert.")
    print("Bitte ausführen:  pip install supabase python-dotenv")
    sys.exit(1)

# ── core libs ───────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("FEHLER: numpy nicht installiert.")
    sys.exit(1)

try:
    from PIL import Image, ImageOps
except ImportError:
    print("FEHLER: Pillow nicht installiert.")
    sys.exit(1)

try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("FEHLER: TensorFlow nicht installiert.")
    sys.exit(1)

# ────────────────────────────────────────────────
#  Konfiguration
# ────────────────────────────────────────────────

MODEL_PATH    = "keras_Model.h5"
LABELS_PATH   = "labels.txt"
IMG_SIZE      = (224, 224)

BUCKET_NAME   = "wardrobe"                  # ← deinen Bucket-Namen hier ändern
USER_ID       = "dein-test-user-uuid"       # ← später aus Auth holen, z. B. supabase.auth.get_user()

SUPABASE_URL  = https://tqkxrvbkdywhfuogsysq.supabase.co
SUPABASE_KEY  = eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRxa3hydmJrZHl3aGZ1b2dzeXNxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI0NTc4MTMsImV4cCI6MjA4ODAzMzgxM30.7C6QoI3yn_97HHTMBN2_DT8QZ8I-QzPbVkC3R23eW8U

if not SUPABASE_URL or not SUPABASE_KEY:
    print("FEHLER: SUPABASE_URL oder SUPABASE_ANON_KEY in .env fehlt!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ────────────────────────────────────────────────

def load_labels(path: str) -> list[str]:
    if not os.path.isfile(path):
        print(f"FEHLER: {path} nicht gefunden!")
        sys.exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    class_names = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            class_names.append(parts[1].strip())
        else:
            class_names.append(line)
    return class_names


def prepare_image(image_path: str) -> np.ndarray:
    if not os.path.isfile(image_path):
        print(f"FEHLER: Bild nicht gefunden → {image_path}")
        sys.exit(1)
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
            
        image = ImageOps.fit(image, IMG_SIZE, resample_filter)
        
        image_array = np.asarray(image, dtype=np.float32)
        normalized = (image_array / 127.5) - 1.0
        return np.expand_dims(normalized, axis=0)
    
    except Exception as e:
        print(f"FEHLER beim Verarbeiten des Bildes: {e}")
        sys.exit(1)


def upload_to_supabase(image_path: str) -> tuple[str, str | None]:
    """Lädt das Bild hoch und gibt (storage_path, public_url) zurück"""
    file_ext = os.path.splitext(image_path)[1].lower()  # .jpg, .png, ...
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Ordner pro User → verhindert Namenskonflikte & organisiert
    storage_path = f"{USER_ID}/{unique_filename}"
    
    try:
        with open(image_path, "rb") as f:
            res = supabase.storage.from_(BUCKET_NAME).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": f"image/{file_ext.lstrip('.') or 'jpeg'}"}
            )
        
        if hasattr(res, "error") and res.error:
            raise Exception(f"Upload-Fehler: {res.error.message}")
        
        # Öffentliche URL (nur wenn Bucket public ist!)
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)
        
        return storage_path, public_url
    
    except Exception as e:
        print(f"Supabase Upload fehlgeschlagen: {e}")
        return storage_path, None


def main():
    if len(sys.argv) != 2:
        prog = os.path.basename(sys.argv[0])
        print("\nVerwendung:")
        print(f"  python {prog} dein_bild.jpg")
        print(f"  python {prog} pfad/zum/bild.png\n")
        sys.exit(1)

    image_path = sys.argv[1]

    class_names = load_labels(LABELS_PATH)
    print(f"Klassen: {', '.join(class_names)}")

    if not os.path.isfile(MODEL_PATH):
        print(f"FEHLER: Modell-Datei nicht gefunden → {MODEL_PATH}")
        sys.exit(1)

    print("Lade Modell... ", end="", flush=True)
    model = load_model(MODEL_PATH, compile=False)
    print("fertig")

    print("Verarbeite Bild... ", end="", flush=True)
    data = prepare_image(image_path)
    print("fertig")

    print("Berechne Vorhersage... ", end="", flush=True)
    prediction = model.predict(data, verbose=0)[0]
    print("fertig")

    idx = np.argmax(prediction)
    confidence = float(prediction[idx])

    # ── Ergebnis ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ERGEBNIS")
    print("═" * 60)
    print(f"  Klasse       :  {class_names[idx]:<12}")
    print(f"  Sicherheit   :  {confidence:>7.2%}")
    print("═" * 60)

    print("\nAlle Wahrscheinlichkeiten:")
    for i, (name, prob) in enumerate(zip(class_names, prediction)):
        print(f"  {name:12} : {float(prob):6.2%}")

    # ── NEU: Bild hochladen ────────────────────────────────
    print("\nSpeichere Bild in Supabase Storage ... ", end="", flush=True)
    storage_path, public_url = upload_to_supabase(image_path)
    print("fertig")

    print(f"  → Storage-Pfad : {storage_path}")
    if public_url:
        print(f"  → Öffentliche URL: {public_url}")
    else:
        print("  → Bucket ist nicht public → signed URL nötig (später erweiterbar)")

    print()


if __name__ == "__main__":
    main()
