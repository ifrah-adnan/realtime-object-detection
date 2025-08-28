import cv2
import numpy as np
from ultralytics import YOLO
import time
import yt_dlp
import os

class DemoPersonCounter:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialise le compteur de personnes pour demo
        
        Args:
            model_path (str): Chemin vers le modèle YOLO
        """
        self.model = YOLO(model_path)
        self.person_count = 0
        self.frame_count = 0
        
        # Configuration pour l'affichage
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.color = (0, 255, 0)  # Vert
        self.thickness = 2
        
    def get_youtube_stream_url(self, youtube_url):
        """Extrait l'URL de streaming directe depuis YouTube"""
        try:
            ydl_opts = {
                'format': 'best[height<=720]',  # Qualité raisonnable
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                stream_url = info['url']
                print(f"✅ URL de streaming extraite: {stream_url[:100]}...")
                return stream_url
                
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction YouTube: {e}")
            return None
    
    def connect_to_source(self, source):
        """
        Établit la connexion à différents types de sources
        
        Args:
            source: Peut être une URL, un fichier vidéo, ou un numéro de webcam
        """
        print(f"Connexion à la source: {source}")
        
        if 'youtube.com' in str(source) or 'youtu.be' in str(source):
            stream_url = self.get_youtube_stream_url(source)
            if stream_url:
                source = stream_url
            else:
                return None
        
        cap = cv2.VideoCapture(source)
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ret, frame = cap.read()
        if ret and frame is not None:
            print("✅ Connexion établie avec succès!")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        else:
            cap.release()
            print("❌ Impossible de lire depuis cette source")
            return None
        
    def detect_persons(self, frame):
        """
        Détecte les personnes dans une frame
        
        Args:
            frame: Image à analyser
            
        Returns:
            tuple: (nombre de personnes, frame annotée)
        """
        # Exécution de la détection YOLO
        results = self.model(frame, verbose=False)
        
        person_count = 0
        annotated_frame = frame.copy()
        
        # Parcourir les résultats de détection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Vérifier si c'est une personne (classe 0 dans COCO)
                    if int(box.cls[0]) == 0:  # 0 = personne
                        person_count += 1
                        
                        # Obtenir les coordonnées de la boîte
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Dessiner la boîte englobante
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.color, self.thickness)
                        
                        # Ajouter le texte de confiance
                        label = f'Personne {confidence:.2f}'
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                  self.font, 0.5, self.color, 1)
        
        return person_count, annotated_frame
    
    def add_info_overlay(self, frame, person_count, source_info="Demo"):
        """Ajoute les informations sur l'image"""
        height, width = frame.shape[:2]
        
        # Fond semi-transparent pour le texte
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Informations à afficher
        info_text = [
            f"Source: {source_info}",
            f"Personnes detectees: {person_count}",
            f"Frame: {self.frame_count}",
            f"Timestamp: {time.strftime('%H:%M:%S')}",
            "Appuyez sur 'q' pour quitter"
        ]
        
        # Afficher chaque ligne d'information
        for i, text in enumerate(info_text):
            y_position = 30 + (i * 22)
            cv2.putText(frame, text, (15, y_position), 
                       self.font, 0.5, (255, 255, 255), 2)
    
    def run_demo(self, source, source_name="Demo"):
        """Lance la détection en temps réel"""
        cap = self.connect_to_source(source)
        if cap is None:
            return False
            
        print(f"🚀 Démarrage de la détection sur {source_name}...")
        print("Appuyez sur 'q' pour quitter")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Fin de la source ou erreur de lecture")
                    break
                
                self.frame_count += 1
                
                # Redimensionner pour de meilleures performances
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = 1280
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Détecter les personnes
                person_count, annotated_frame = self.detect_persons(frame)
                self.person_count = person_count
                
                # Ajouter les informations sur l'image
                self.add_info_overlay(annotated_frame, person_count, source_name)
                
                # Afficher le résultat
                cv2.imshow('Demo - Compteur de Personnes', annotated_frame)
                
                # Contrôles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  
                    
        except KeyboardInterrupt:
            print("\n⏹️ Arrêt demandé par l'utilisateur")
        except Exception as e:
            print(f"❌ Erreur:  {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🧹 Ressources libérées")
            return True

def main():
    """Menu principal pour choisir la source"""
    counter = DemoPersonCounter()
    
    print("="*60)
    print("🎥 DEMO - COMPTEUR DE PERSONNES AVEC YOLO")
    print("="*60)
    
    # Sources de démonstration prêtes à utiliser
    demo_sources = {
        "1": {
            "source": 0,  # Webcam par défaut
            "name": "Webcam localee"
        },
        "2": {
            "source": "https://sample-videos.com/zip/10/mp4/720/SampleVideo_720x480_1mb.mp4",
            "name": "Vidéo test en ligne"
        },
        "3": {
            "source": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
            "name": "Big Buck Bunny (demo)"
        },
        "4": {
            "source": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
            "name": "Vidéo sample MP4"
        }
    }
    
    while True:
        print("\n📋 Choisissez une source vidéo:")
        print("1. 📷 Webcam (caméra locale)")
        print("2. 🌐 Vidéo test en ligne")
        print("3. 🎬 Big Buck Bunny (demo)")
        print("4. 📹 Sample MP4")
        print("5. 🎞️ Fichier vidéo local")
        print("6. 📺 YouTube (nécessite yt-dlp)")
        print("7. 🔗 URL personnalisée")
        print("8. 📡 Votre flux caméra IP")
        print("0. ❌ Quitter")
        
        choice = input("\n👉 Votre choix (0-8): ").strip()
        
        if choice == "0":
            print("👋 Au revoir!")
            break
            
        elif choice in demo_sources:
            source_info = demo_sources[choice]
            success = counter.run_demo(source_info["source"], source_info["name"])
            
        elif choice == "5":
            file_path = input("📁 Chemin vers le fichier vidéo: ").strip()
            if os.path.exists(file_path):
                counter.run_demo(file_path, f"Fichier: {os.path.basename(file_path)}")
            else:
                print("❌ Fichier introuvable!")
                
        elif choice == "6":
            youtube_url = input("🔗 URL YouTube: ").strip()
            print("⚠️ Installation de yt-dlp requise: pip install yt-dlp")
            counter.run_demo(youtube_url, "YouTube")
            
        elif choice == "7":
            custom_url = input("🔗 URL personnalisée: ").strip()
            counter.run_demo(custom_url, "URL personnalisée")
            
        elif choice == "8":
            ip_url = input("📡 URL de votre caméra IP (ex: http://192.168.1.100:8080/video): ").strip()
            counter.run_demo(ip_url, "Caméra IP")
            
        else:
            print("❌ Choix invalide!")
        
        counter.frame_count = 0

if __name__ == "__main__":
    main()