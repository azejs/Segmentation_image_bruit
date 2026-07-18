"""
Script de lancement de l'application GUI.
Vérifications des dépendances avant ouverture.
"""

import sys
import subprocess
from pathlib import Path

REQUIRED = ["cv2", "numpy", "PIL", "matplotlib", "sklearn", "scipy"]

def check_deps():
    missing = []
    for pkg in REQUIRED:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing

if __name__ == "__main__":
    missing = check_deps()
    if missing:
        print(f"[!] Dépendances manquantes : {missing}")
        print("    Lancez : pip install -r requirements.txt")
        sys.exit(1)

    # Ajouter le dossier racine au path
    sys.path.insert(0, str(Path(__file__).parent))
    from app import App
    app = App()
    app.mainloop()
