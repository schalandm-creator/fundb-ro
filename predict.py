# predict.py
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sys

MODEL_PATH = "models/clothing_model.h5"
LABEL_PATH = "labels.txt"

np.set_printoptions(suppress=True)

def load_class_names():
    with open(LABEL_PATH, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [line.split(" ", 1)[1].strip() for line in lines]


def prepare_image(img_path):
    size = (224, 224)
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img, dtype="float32")
    normalized = (img_array / 127.5) - 1
    return np.expand_dims(normalized, axis=0)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py bild.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print("Bild nicht gefunden.")
        sys.exit(1)

    if not os.path.isfile(MODEL_PATH):
        print("Kein trainiertes Modell gefunden.")
        print("→ zuerst trainieren (train_simple.py)")
        sys.exit(1)

    model = load_model(MODEL_PATH, compile=False)
    class_names = load_class_names()

    data = prepare_image(image_path)
    prediction = model.predict(data)[0]
    idx = np.argmax(prediction)
    confidence = float(prediction[idx])

    print("\n" + "="*50)
    print(f"  Klasse:       {class_names[idx]}")
    print(f"  Sicherheit:   {confidence:6.2%}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
