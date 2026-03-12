import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# 1. Konfiguration der Seite
st.set_page_config(page_title="Fundkiste Bild-Erkennung", layout="centered")

# 2. Modell und Labels laden
@st.cache_resource
def load_keras_model():
    # Dateinamen exakt nach deiner Angabe
    model_path = "keras_model(1).h5"
    label_path = "labels(1).txt"
    
    # Überprüfen, ob die Model-Datei existiert
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Datei '{model_path}' nicht gefunden. Bitte prüfen!")
    
    # Laden des Modells
    model = load_model(model_path, compile=False)
    
    # Überprüfen, ob die Labels existieren
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Datei '{label_path}' nicht gefunden. Bitte prüfen!")
        
    with open(label_path, "r") as f:
        class_names = f.readlines()
        
    return model, class_names

# App-Logik starten
try:
    model, class_names = load_keras_model()
except Exception as e:
    st.error(f"❌ Fehler: {e}")
    st.info("Hinweis: Beide Dateien müssen im Hauptverzeichnis deines Repos liegen.")
    st.stop()

# 3. Benutzeroberfläche (UI)
st.title("🔍 Fundkiste Bild-Erkennung")
st.write("Identifiziere Objekte aus der Fundkiste.")

option = st.radio("Bildquelle wählen:", ("Kamera nutzen", "Bild hochladen"))

img_file = None
if option == "Kamera nutzen":
    img_file = st.camera_input("Foto aufnehmen")
else:
    img_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

# 4. Bildverarbeitung und Vorhersage
if img_file is not None:
    # Bild laden
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Eingabebild", use_container_width=True)

    # Vorbereitung für das Modell (Größe 224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalisierung der Daten
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage (Inferenz)
    with st.spinner('Objekt wird erkannt...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

    # Ergebnis-Anzeige
    st.divider()
    # Entfernt Index-Nummern am Anfang des Labels (z.B. "0 Schlüssel" -> "Schlüssel")
    display_name = class_name[2:] if class_name[0].isdigit() else class_name
    st.subheader(f"Ergebnis: {display_name}")
    st.progress(float(confidence_score))
    st.write(f"Sicherheit: {100 * confidence_score:.2f}%")
