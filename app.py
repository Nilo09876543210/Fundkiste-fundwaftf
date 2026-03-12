import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from datetime import datetime

# 1. Konfiguration & Session State für die "Datenbank"
st.set_page_config(page_title="Fundkiste Sharing", layout="wide")

if 'history' not in st.session_state:
    st.session_state['history'] = []

# 2. Ressourcen laden
@st.cache_resource
def load_resources():
    model_path = "keras_model(1).h5"
    label_path = "labels(1).txt"
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        st.error("Dateien fehlen!")
        st.stop()
    model = load_model(model_path, compile=False)
    with open(label_path, "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_resources()

# 3. Layout: Zwei Spalten (Links: Scan, Rechts: Übersicht)
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Neues Objekt scannen")
    img_file = st.camera_input("Foto machen") or st.file_uploader("Bild wählen")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        
        # Vorhersage-Logik
        size = (224, 224)
        img_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized).astype(np.float32)
        normalized_img = (img_array / 127.5) - 1
        data = np.expand_dims(normalized_img, axis=0)
        
        prediction = model.predict(data)
        index = np.argmax(prediction)
        label = class_names[index].strip()[2:] if class_names[index].strip()[0].isdigit() else class_names[index].strip()
        confidence = prediction[0][index]

        if st.button(f"Als '{label}' speichern"):
            # In die "historische" Liste speichern
            new_entry = {
                "label": label,
                "time": datetime.now().strftime("%H:%M:%S"),
                "image": image,
                "confidence": f"{100 * confidence:.1f}%"
            }
            st.session_state['history'].insert(0, new_entry)
            st.success(f"{label} wurde zur Liste hinzugefügt!")

with col2:
    st.header("📂 Alle Funde (heute)")
    if not st.session_state['history']:
        st.info("Noch keine Objekte hochgeladen.")
    else:
        for item in st.session_state['history']:
            with st.expander(f"{item['time']} - {item['label']} ({item['confidence']})"):
                st.image(item['image'], use_container_width=True)
