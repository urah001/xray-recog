import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('xray_model.keras')

# Class labels (should match your training classes)
class_labels = ['Normal', 'Brachymetatarsia', 'FemoralFracture', 'HipFracture','Cardiomegalymild','NormalChest']  
# Change as per your dataset

# Function to preprocess and predict
def predict_xray(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence


# Streamlit UI
st.title("X-ray Disease Detection App")
st.write("Upload an X-ray image to predict possible diseases.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray.', use_container_width=True)
    
    if st.button('Predict'):
        label, confidence = predict_xray(img)
        st.success(f"Prediction: {label} ({confidence*100:.2f}% confidence)")
