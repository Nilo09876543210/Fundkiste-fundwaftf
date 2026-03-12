import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Konfiguration der Seite
st.set_page_config(page_title="Fundkiste Klassifizierung", layout="centered")

# 1. Modell und Labels laden
@st.cache_resource
def load_keras_model():
    # Der Dateiname wurde auf deinen Wunsch angepasst
    # Sicherstellen, dass die Datei im GitHub-Repo existiert
    model = load_model("keras_model(1).h5", compile=False) 
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

# Laden der Ressourcen
try:
    model, class_names = load_keras_model()
except Exception as e:
    st.error(f"Fehler beim Laden des Modells: {e}")
    st.stop()

# 2. Benutzeroberfläche
st.title("🔍 Fundkiste Bild-Erkennung")
st.write("Wähle eine Quelle, um ein Objekt zu klassifizieren.")

option = st.radio("Quelle wählen:", ("Kamera nutzen", "Bild hochladen"))

img_file = None
if option == "Kamera nutzen":
    img_file = st.camera_input("Foto aufnehmen")
else:
    img_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

# 3. Vorhersage-Logik
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption
