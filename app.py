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
    # Bild öffnen und vorbereiten
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Gewähltes Bild", use_container_width=True)

    # Bildgröße für das Modell anpassen (Teachable Machine Standard: 224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalisierung der Daten
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    
    # Batch-Dimension hinzufügen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage durchführen
    with st.spinner('Objekt wird analysiert...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

    # Ergebnis-Anzeige
    st.divider()
    # Entferne die ersten Zeichen (z.B. "0 ") vom Label-Namen
    display_name = class_name[2:] if class_name[0].isdigit() else class_name
    st.subheader(f"Ergebnis: {display_name}")
    st.progress(float(confidence_score))
    st.write(f"Wahrscheinlichkeit: {100 * confidence_score:.2f}%")
