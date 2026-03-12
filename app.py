import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Konfiguration der Seite
st.set_page_config(page_title="Schul-Fundkiste KI", page_icon="🏫")

st.title("🔍 KI-Fundkiste")
st.write("Mache ein Foto von einem Fundstück, und die KI hilft beim Einordnen!")

# 1. Modell und Labels laden
@st.cache_resource
def load_keras_model():
    # Stelle sicher, dass die Dateien in deinem GitHub Repo liegen
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_keras_model()

# 2. Kamera- oder Upload-Funktion
option = st.radio("Quelle wählen:", ("Kamera nutzen", "Bild hochladen"))

if option == "Kamera nutzen":
    img_file = st.camera_input("Nimm das Fundstück auf")
else:
    img_file = st.file_uploader("Wähle ein Bild aus", type=["jpg", "jpeg", "png"])

if img_file:
    # Bild anzeigen und vorbereiten
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Dein Foto", use_container_width=True)

    # Bild für Teachable Machine Modell anpassen (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage treffen
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.subheader(f"Ergebnis: {class_name[2:]}") # Schneidet die Nummerierung (z.B. "0 ") ab
    st.info(f"Sicherheit der KI: {round(confidence_score * 100, 2)}%")

    # Zusatz-Logik für die Fundkiste
    if confidence_score > 0.7:
        st.success(f"Das Objekt wurde als **{class_name[2:]}** erkannt und im System markiert.")
        # Hier könnte man später eine Datenbank-Anbindung (z.B. Google Sheets) einbauen
    else:
        st.warning("Die KI ist sich unsicher. Bitte versuche es bei besserem Licht nochmal.")
