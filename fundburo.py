#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kleidungs-Klassifizierer (Hose / Pullover / Jacken / Sonstiges)
Benutzt dein keras_Model.h5 + labels.txt
"""

import os
import sys
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# ────────────────────────────────────────────────
#  Konfiguration – passe bei Bedarf an
# ────────────────────────────────────────────────

MODEL_PATH    = "keras_Model.h5"           # dein Modell
LABELS_PATH   = "labels.txt"               # deine Labels
IMG_SIZE      = (224, 224)                 # Teachable Machine Standard

# ────────────────────────────────────────────────

def load_labels(path: str) -> list[str]:
    """Liest labels.txt und gibt nur die Klassennamen zurück"""
    if not os.path.isfile(path):
        print(f"FEHLER: {path} nicht gefunden!")
        sys.exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # Teachable-Machine-Format: "0 hose" → "hose"
    class_names = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            class_names.append(parts[1].strip())
        else:
            class_names.append(line)
    return class_names


def prepare_image(image_path: str) -> np.ndarray:
    """Bild laden, auf 224×224 bringen, normalisieren (wie Teachable Machine)"""
    if not os.path.isfile(image_path):
        print(f"FEHLER: Bild nicht gefunden → {image_path}")
        sys.exit(1)
    
    try:
        # Bild öffnen und RGB erzwingen
        image = Image.open(image_path).convert("RGB")
        
        # Auf exakt 224×224 zuschneiden (Center-Crop + Resize)
        image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
        
        # in numpy Array umwandeln
        image_array = np.asarray(image, dtype=np.float32)
        
        # Normalisierung: [-1, +1] wie Teachable Machine
        normalized = (image_array / 127.5) - 1.0
        
        # Batch-Dimension hinzufügen → Shape (1, 224, 224, 3)
        return np.expand_dims(normalized, axis=0)
    
    except Exception as e:
        print(f"FEHLER beim Verarbeiten des Bildes: {e}")
        sys.exit(1)


def main():
    # ── Kommandozeilen-Argument prüfen ──────────────────────────────
    if len(sys.argv) != 2:
        print("\nVerwendung:")
        print("  python classify.py dein_bild.jpg")
        print("  python classify.py pfad/zum/bild.png\n")
        sys.exit(1)

    image_path = sys.argv[1]

    # ── Labels laden ────────────────────────────────────────────────
    class_names = load_labels(LABELS_PATH)
    print(f"Klassen: {', '.join(class_names)}")

    # ── Modell laden ────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        print(f"FEHLER: Modell-Datei nicht gefunden → {MODEL_PATH}")
        print("→ Lege dein trainiertes Modell als 'keras_Model.h5' ab")
        sys.exit(1)

    print("Lade Modell... ", end="", flush=True)
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("fertig")
    except Exception as e:
        print(f"FEHLER beim Laden des Modells:\n{e}")
        sys.exit(1)

    # ── Bild vorbereiten ────────────────────────────────────────────
    print("Verarbeite Bild... ", end="", flush=True)
    data = prepare_image(image_path)
    print("fertig")

    # ── Vorhersage ──────────────────────────────────────────────────
    print("Berechne Vorhersage... ", end="", flush=True)
    prediction = model.predict(data, verbose=0)[0]
    print("fertig")

    # Höchste Wahrscheinlichkeit
    idx = np.argmax(prediction)
    confidence = float(prediction[idx])

    # ── Ergebnis ausgeben ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ERGEBNIS")
    print("═" * 60)
    print(f"  Klasse       :  {class_names[idx]:<12}")
    print(f"  Sicherheit   :  {confidence:>7.2%}")
    print("═" * 60)

    # Optional: alle Klassen mit Prozent anzeigen
    print("\nAlle Wahrscheinlichkeiten:")
    for i, (name, prob) in enumerate(zip(class_names, prediction)):
        print(f"  {name:12} : {float(prob):6.2%}")
    print()


if __name__ == "__main__":
    main()
