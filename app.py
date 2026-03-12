import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Konfiguration der Seite
st.set_page_config(page_title="Fundkiste Klassifizierung", layout="centered")

# 1. Modell und Labels laden
@st.cache_resource
def load_keras_model():
    # Exakter Dateiname nach deiner Vorgabe
    model_path = "keras_model(1).h5"
    label_path = "labels.txt"
    
    # Überprüfung, ob die Datei im Hauptverzeichnis existiert
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Die Datei '{model_path}' wurde nicht gefunden. Bitte prüfe den Namen im GitHub-Repo.")
    
    # Laden des Keras-Modells
    model = load_model(model_path, compile=False)
    
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Die Datei '{label_path}' fehlt im Repository.")
        
    with open(label_path, "r") as f:
        class_names = f.readlines()
        
    return model, class_names

# App-Logik: Ressourcen laden
try:
    model, class_names = load_keras_model()
except Exception as e:
    st.error(f"❌ Fehler: {e}")
    st.info("Hinweis: Achte darauf, dass 'keras_model(1).h5' direkt neben dieser app.py Datei liegt.")
    st.stop()

# 2. Benutzeroberfläche
st.title("🔍 Fundkiste Bild-Erkennung")
st.write("Lade ein Bild hoch oder nutze die Kamera, um ein Objekt zu identifizieren.")

option = st.radio("Quelle wählen:", ("Kamera nutzen", "Bild hochladen"))

img_file = None
if option == "Kamera nutzen":
    img_file = st.camera_input("Foto aufnehmen")
else:
    img_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

# 3. Vorhersage-Berechnung
if img_file is not None:
    # Bild öffnen und vorbere
