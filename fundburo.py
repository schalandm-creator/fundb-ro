
# classify_simple.py
# Sehr einfacher Code – Kleidung vorhersagen (Hose, Pullover, Jacken, Sonstiges)

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import sys

# ──── Einstellungen ────────────────────────────────────────
MODEL_DATEI   = "keras_Model.h5"
LABELS_DATEI  = "labels.txt"
BILD_GROESSE  = (224, 224)


def lade_klassen():
    with open(LABELS_DATEI, "r", encoding="utf-8") as f:
        zeilen = f.readlines()
    klassen = []
    for zeile in zeilen:
        if zeile.strip():
            # "0 hose" → nur "hose" behalten
            teil = zeile.strip().split(" ", 1)
            if len(teil) == 2:
                klassen.append(teil[1])
            else:
                klassen.append(zeile.strip())
    return klassen


def bild_vorbereiten(pfad):
    bild = Image.open(pfad).convert("RGB")
    bild = ImageOps.fit(bild, BILD_GROESSE, Image.Resampling.LANCZOS)
    array = np.asarray(bild).astype("float32")
    normalisiert = (array / 127.5) - 1
    # Batch-Dimension → Form (1, 224, 224, 3)
    return np.expand_dims(normalisiert, axis=0)


def main():
    if len(sys.argv) != 2:
        print("\nVerwendung:")
        print("   python classify_simple.py dein_bild.jpg\n")
        sys.exit(1)

    bild_pfad = sys.argv[1]

    # Labels laden
    klassen = lade_klassen()
    print("Klassen:", ", ".join(klassen))

    # Modell laden
    try:
        modell = load_model(MODEL_DATEI, compile=False)
    except Exception as e:
        print("Fehler beim Laden des Modells:", e)
        sys.exit(1)

    # Bild vorbereiten
    try:
        eingabe = bild_vorbereiten(bild_pfad)
    except Exception as e:
        print("Fehler beim Bild:", e)
        sys.exit(1)

    # Vorhersage
    vorhersage = modell.predict(eingabe)[0]
    index = np.argmax(vorhersage)
    sicherheit = vorhersage[index]

    print("\n" + "-"*50)
    print(f"Klasse:     {klassen[index]}")
    print(f"Sicherheit: {sicherheit:.4f}  ({sicherheit*100:.1f} %)")
    print("-"*50)


if __name__ == "__main__":
    main()
