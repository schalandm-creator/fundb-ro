# collect_images.py
import os
import shutil
from pathlib import Path

DATA_DIR = Path("data/train")
CATEGORIES = ["hose", "pullover", "jacken", "sonstiges"]

def setup_directories():
    for cat in CATEGORIES:
        (DATA_DIR / cat).mkdir(parents=True, exist_ok=True)

def move_file(src_path: Path, category: str):
    if category not in CATEGORIES:
        print(f"Ungültige Kategorie: {category}")
        return False
    
    target_dir = DATA_DIR / category
    target_path = target_dir / src_path.name
    
    # Vermeiden von Überschreiben
    if target_path.exists():
        stem, ext = src_path.stem, src_path.suffix
        i = 1
        while target_path.exists():
            target_path = target_dir / f"{stem}_{i}{ext}"
            i += 1
    
    shutil.move(str(src_path), str(target_path))
    print(f"→ {src_path.name:40} → {category}")
    return True


def main():
    setup_directories()
    
    print("Kategorien:")
    for i, cat in enumerate(CATEGORIES, 1):
        print(f"  {i}) {cat}")
    print("  x) überspringen / löschen")
    print("-" * 50)
    
    while True:
        src = input("\nPfad zum Bild (oder Enter zum Beenden): ").strip()
        if not src:
            break
        
        src_path = Path(src).expanduser().resolve()
        if not src_path.is_file():
            print("Datei nicht gefunden.")
            continue
        
        print(f"Gefunden: {src_path.name}")
        
        choice = input("Kategorie (1–4 oder x): ").strip().lower()
        
        if choice in ("x", "exit", "q"):
            print("Überspringe.")
            continue
        
        try:
            idx = int(choice) - 1
            cat = CATEGORIES[idx]
        except:
            print("Ungültige Eingabe → überspringe")
            continue
        
        move_file(src_path, cat)


if __name__ == "__main__":
    main()
    print("\nFertig. Du kannst jetzt trainieren.")
