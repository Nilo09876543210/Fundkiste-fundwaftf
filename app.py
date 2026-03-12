import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# 1. Seiteneinstellungen
st.set_page_config(page_title="Fundkiste Bild-Erkennung", layout="centered")

# 2. Ressourcen laden (Modell und Labels)
@st.cache_resource
def load_resources():
    model_path = "keras_model(1).h5"
    label_path = "labels(1).txt"
    
    # Prüfen, ob die Model-Datei existiert
    if not os.path.exists(model_path):
        st.error(f"Datei '{model_path}' wurde im Repository nicht gefunden!")
        st.stop()
        
    # Modell laden
    model = load_model(model_path, compile=False)
    
    # Prüfen, ob die Label-Datei existiert
    if not os.path.exists(label_path):
        st.error(f"Datei '{label_path}' wurde im Repository nicht gefunden!")
        st.stop()
        
    with open(label_path, "r") as f:
        class_names = f.readlines()
        
    return model, class_names

# Ressourcen abrufen
model, class_names = load_resources()

# 3. Benutzeroberfläche
st.title("🔍 Fundkiste Bild-Erkennung")
st.write("Mache ein Foto oder lade ein Bild hoch.")

option = st.radio("Quelle wählen:", ("Kamera nutzen", "Bild hochladen"))

img_file = None
if option == "Kamera nutzen":
    img_file = st.camera_input("Foto aufnehmen")
else:
    img_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

# 4. Bildverarbeitung und Vorhersage
if img_file is not None:
    # Bild öffnen
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Eingabe", use_container_width=True)

    # Vorbereitung (224x224 Pixel)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalisierung
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    with st.spinner('Objekt wird erkannt...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.divider()
    # Entfernt Index-Nummern (z.B. "0 Schlüssel" -> "Schlüssel")
    display_name = class_name[2:] if class_name[0].isdigit() else class_name
    st.subheader(f"Ergebnis: {display_name}")
    st.progress(float(confidence_score))
    st.write(f"Sicherheit: {100 * confidence_score:.2f}%")
