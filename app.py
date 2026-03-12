import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Seite konfigurieren
st.set_page_config(page_title="Fundkiste Bild-Erkennung", layout="centered")

@st.cache_resource
def load_resources():
    model_path = "keras_model(1).h5"
    label_path = "labels(1).txt"
    
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        st.error("Model- oder Label-Datei fehlt im Repository!")
        st.stop()
        
    # In TF 2.15 sollte compile=False ausreichen
    model = load_model(model_path, compile=False)
    
    with open(label_path, "r") as f:
        class_names = f.readlines()
        
    return model, class_names

# Ressourcen laden
model, class_names = load_resources()

st.title("🔍 Fundkiste Bild-Erkennung")

img_file = st.camera_input("Foto aufnehmen") or st.file_uploader("Oder Bild hochladen", type=["jpg", "png"])

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Vorbereitung
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    with st.spinner('Analysiere...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence = prediction[0][index]

    st.divider()
    # Entfernt Index (z.B. "0 Schlüssel" -> "Schlüssel")
    display_name = class_name[2:] if class_name[0].isdigit() else class_name
    st.subheader(f"Ergebnis: {display_name}")
    st.write(f"Sicherheit: {100 * confidence:.2f}%")
