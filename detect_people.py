import cv2
import numpy as np
from ultralytics import YOLO
import json

import time
import yt_dlp
import os
import logging
from typing import Tuple, Optional, Union

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

class DemoPersonCounter:
    def __init__(self, model_path: str = "yolov8n.pt", max_width: int = 1280, skip_frames: int = 0):
        """
        Initialise le compteur de personnes pour demo.
        
        Args:
            model_path (str): Chemin vers le modèle YOLO
            max_width (int): Largeur max des frames pour optimisation
            skip_frames (int): Nombre de frames à ignorer entre deux détections
        """
        self.model = YOLO(model_path)
        self.person_count = 0
        self.frame_count = 0
        self.max_width = max_width
        self.skip_frames = skip_frames

        # Style overlay
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.color = (0, 255, 0)
        self.thickness = 2

    def get_youtube_stream_url(self, youtube_url: str) -> Optional[str]:
        """Extrait l'URL de streaming directe depuis YouTube"""
        try:
            ydl_opts = {
                "format": "best[height<=720]",
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info["url"]
        except Exception as e:
            logging.error(f"Erreur extraction YouTube: {e}")
            return None

    def connect_to_source(self, source: Union[str, int]) -> Optional[cv2.VideoCapture]:
        """
        Établit la connexion à une source vidéo.
        """
        logging.info(f"Tentative de connexion à la source: {source}")

        if "youtube.com" in str(source) or "youtu.be" in str(source):
            stream_url = self.get_youtube_stream_url(source)
            if not stream_url:
                return None
            source = stream_url

        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.isOpened():
            logging.info("✅ Connexion établie avec succès")
            return cap
        logging.error("❌ Impossible de lire depuis cette source")
        return None

    def detect_persons(self, frame: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Détecte les personnes dans une frame.
        """
        results = self.model(frame, verbose=False)
        person_count = 0
        annotated_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # 0 = personne (COCO)
                    person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.color, self.thickness)
                    cv2.putText(
                        annotated_frame,
                        f"Personne {conf:.2f}",
                        (x1, y1 - 10),
                        self.font,
                        self.font_scale,
                        self.color,
                        1,
                    )
        return person_count, annotated_frame

    def add_info_overlay(self, frame: np.ndarray, person_count: int, source_info: str = "Demo") -> None:
        """Ajoute un overlay d'informations sur la frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        info_text = [
            f"Source: {source_info}",
            f"Personnes detectees: {person_count}",
            f"Frame: {self.frame_count}",
            f"Timestamp: {time.strftime('%H:%M:%S')}",
            "Appuyez sur 'q' pour quitter",
        ]

        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (15, 30 + i * 22), self.font, 0.5, (255, 255, 255), 1)

    def save_results_to_json(self, person_count: int, source_name: str):
        """
        Sauvegarde les résultats de la frame actuelle dans un fichier JSON.
        Chaque frame sera ajoutée avec timestamp, source et nombre de personnes.
        """
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_name,
            "frame_number": self.frame_count,
            "person_count": person_count
        }

        # Crée un dossier 'results' si non existant
        os.makedirs("results", exist_ok=True)
        json_file = "results/detection_results.json"

        # Si le fichier existe, on charge et ajoute le nouveau résultat
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(result)

        # Écriture finale
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

    def run_demo(self, source: Union[str, int], source_name: str = "Demo") -> bool:
        """Exécute la détection en temps réel depuis une source vidéo."""
        cap = self.connect_to_source(source)
        if not cap:
            return False

        logging.info(f"🚀 Démarrage détection sur {source_name}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("⚠️ Fin de la source ou erreur de lecture")
                    break

                self.frame_count += 1

                # Redimensionnement
                h, w = frame.shape[:2]
                if w > self.max_width:
                    scale = self.max_width / w
                    frame = cv2.resize(frame, (self.max_width, int(h * scale)))

                # Skip frames si configuré
                if self.skip_frames and self.frame_count % (self.skip_frames + 1) != 0:
                    continue

                # Détection
                person_count, annotated_frame = self.detect_persons(frame)
                self.person_count = person_count
                self.save_results_to_json(person_count, source_name)


                # Overlay
                self.add_info_overlay(annotated_frame, person_count, source_name)
                cv2.imshow("Demo - Compteur de Personnes", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            logging.info("⏹️ Arrêt demandé par l'utilisateur")
        except Exception as e:
            logging.error(f"Erreur pendant la détection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("🧹 Ressources libérées")
        return True


def main():
    """Menu principal"""
    counter = DemoPersonCounter()

    demo_sources = {
        "1": {"source": 0, "name": "Webcam locale"},
        "2": {"source": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4", "name": "Big Buck Bunny"},
        "3": {"source": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4", "name": "Vidéo sample"},
    }

    while True:
        print("\n📋 Choisissez une source vidéo:")
        print("1. 📷 Webcam locale")
        print("2. 🎬 Big Buck Bunny")
        print("3. 📹 Vidéo sample")
        print("4. 📁 Fichier local")
        print("5. 🔗 YouTube")
        print("6. 🌐 URL personnalisée")
        print("0. ❌ Quitter")

        choice = input("\n👉 Votre choix (0-6): ").strip()
        if choice == "0":
            break
        elif choice in demo_sources:
            counter.run_demo(demo_sources[choice]["source"], demo_sources[choice]["name"])
        elif choice == "4":
            path = input("📁 Chemin vers le fichier: ").strip()
            if os.path.exists(path):
                counter.run_demo(path, os.path.basename(path))
            else:
                logging.error("Fichier introuvable")
        elif choice == "5":
            url = input("🔗 URL YouTube: ").strip()
            counter.run_demo(url, "YouTube")
        elif choice == "6":
            url = input("🔗 URL personnalisée: ").strip()
            counter.run_demo(url, "Custom URL")
        else:
            logging.warning("Choix invalide")


if __name__ == "__main__":
    main()
