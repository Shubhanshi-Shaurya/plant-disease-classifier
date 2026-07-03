import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os


st.set_page_config(
    page_title="AI Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_disease_model():
    model_path = "disease_classifier.h5"
    return tf.keras.models.load_model(model_path)

try:
    model = load_disease_model()
except Exception as e:
    st.error(f"Error loading model file. Make sure 'disease_classifier.h5' is in the same folder. Details: {e}")

@st.cache_data
def load_class_labels():
    labels_file = "classes.txt"
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        st.error(" Critical Error: 'classes.txt' not found in the project folder!")
        return []

CLASS_NAMES = load_class_labels()


def preprocess_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    img_array = np.asarray(image)
    
    if img_array.shape[-1] == 4:
        image = image.convert("RGB")
        img_array = np.asarray(image)
        
    img_tensor = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_tensor)
    highest_index = np.argmax(predictions[0])
    confidence = predictions[0][highest_index]
    
    return CLASS_NAMES[highest_index], confidence


st.title("🌿 AI Plant Disease Diagnostics")
st.write("Upload a clear photo of an infected plant leaf to get an instant diagnosis and confidence rating.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Leaf Image", width=350)
    
    st.write("")
    if st.button("Analyze Leaf", type="primary"):
        with st.spinner("Running AI Analysis..."):
            try:
                label, confidence_score = preprocess_and_predict(image, model)
                
                clean_label = label.replace("___", " - ").replace("_", " ")
                
                st.success("### Analysis Complete!")
                st.metric(label="Predicted Diagnosis", value=clean_label)
                st.metric(label="Confidence Level", value=f"{confidence_score * 100:.2f}%")
                
                if confidence_score < 0.60:
                    st.warning(" Low confidence warning. Make sure the leaf is centered, well-lit, and occupying most of the camera frame.")
                    
            except Exception as ex:
                st.error(f"Prediction failed. Ensure your CLASS_NAMES count matches the model outputs. Error: {ex}")

