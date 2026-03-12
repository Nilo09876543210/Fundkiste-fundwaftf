import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# 1. Konfiguration
st.set_page_config(page_title="Fundkiste Bild-Erkennung", layout="centered")

# 2. Ressourcen laden mit Kompatibilitäts-Fix
@st.cache_resource
def load_resources():
    model_path = "keras_model(1).h5"
    label_path = "labels(1).txt"
    
    if not os.path.exists(model_path):
        st.error(f"Datei '{model_path}' nicht gefunden!")
        st.stop()
        
    # Fix für 'DepthwiseConv2D' Fehler bei älteren Modellen
    try:
        # Wir definieren die Layer-Klasse explizit als 'custom_objects'
        custom_objects = {
            "DepthwiseConv2D": tf.keras.layers.DepthwiseConv2D
        }
        model = load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        st.stop()
    
    if not os.path.exists(label_path):
        st.error(f"Datei '{label_path}' nicht gefunden!")
        st.stop()
        
    with open(label_path, "r") as f:
        class_names = f.readlines()
        
    return model, class_names

# Laden ausführen
model, class_names = load_resources()

# 3. Benutzeroberfläche (UI)
st.title("🔍 Fundkiste Bild-Erkennung")
st.write("Identifiziere Objekte aus der Fundkiste.")

option = st.radio("Quelle:", ("Kamera", "Upload"))

img_file = None
if option == "Kamera":
    img_file = st.camera_input("Foto aufnehmen")
else:
    img_file = st.file_uploader("Bild wählen", type=["jpg", "jpeg", "png"])

# 4. Vorhersage
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Bild für das Modell vorbereiten (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalisierung (Standard für Teachable Machine)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage treffen
    with st.spinner('Analysiere Bild...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence = prediction[0][index]

    # Ergebnis-Anzeige
    st.divider()
    # Entfernt Index-Nummer (z.B. "0 Schlüssel" -> "Schlüssel")
    display_name = class_name[2:] if class_name[0].isdigit() else class_name
    st.subheader(f"Ergebnis: {display_name}")
    st.write(f"Sicherheit: {100 * confidence:.2f}%")
