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
    # Dateinamen exakt ohne Leerzeichen vor der Klammer
    model_path = "keras_model(1).h5"
    label_path = "labels(1).txt"
    
    # Check ob Modell da ist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Datei '{model_path}' nicht gefunden.")
    
    # Modell laden
    model = load_model(model_path, compile=False)
    
    # Check ob Labels da sind
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Datei '{label_path}' nicht gefunden.")
        
    with open(label_path, "r") as f:
        class_names = f.readlines()
        
    return model, class_names

# App starten
try:
    model, class_names = load_keras_model()
except Exception as e:
    st.error(f"❌ Fehler: {e}")
    st.stop()

# 3. Benutzeroberfläche
st.title("🔍 Fundkiste Bild-Erkennung")

option = st.radio("Quelle:", ("Kamera", "Upload"))

img_file = None
if option == "Kamera":
    img_file = st.camera_input("Foto")
else:
    img_file = st.file_uploader("Bild wählen", type=["jpg", "jpeg", "png"])

# 4. Vorhersage
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Vorbereiten
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Analyse
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Ergebnis
    st.divider()
    display_name = class_name[2:] if class_name[0].isdigit() else class_name
    st.subheader(f"Objekt: {display_name}")
    st.write(f"Sicherheit: {100 * confidence_score:.2f}%")
