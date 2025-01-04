import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Define the model path
model_path = r"C:\Users\imran\OneDrive\Desktop\PlantVillage\app\plant_disease_prediction_model (1).keras"

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Load the class indices (for labels)
working_dir = os.path.dirname(os.path.abspath(__file__))
class_indices_file = os.path.join(working_dir, "class_indices.json")

@st.cache_data
def load_class_indices():
    with open(class_indices_file, "r") as f:
        class_indices = json.load(f)
    return {int(k): v for k, v in class_indices.items()}  # Ensure keys are integers

class_indices = load_class_indices()

# Function to load and preprocess the image
def load_and_preprocess_image(image, target_size=(256, 256)):
    """
    Resize the image to the target size and preprocess it.
    """
    image = image.convert('RGB')  # Ensure RGB format
    img = image.resize(target_size)  # Resize to target size
    img_array = np.array(img, dtype="float32") / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    """
    Preprocess the image and make predictions using the model.
    """
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = predictions[0][predicted_class_index]
    predicted_class_name = class_indices.get(predicted_class_index, "Unknown")
    return predicted_class_name, confidence_score

# Streamlit app UI
st.title('Plant Disease Classifier')
st.write("Upload an image of a plant leaf to predict the disease class.")

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)

    # Resize the uploaded image for display (reduce size)
    display_image = image.resize((300, 300))  # Resized for display purposes
    st.image(display_image, caption="Uploaded Image (Resized)", use_column_width=False)

    if st.button('Classify'):
        try:
            # Predict the class of the uploaded image
            prediction, confidence = predict_image_class(model, image, class_indices)
            st.success(f'Prediction: {prediction}')
            st.info(f'Confidence: {confidence:.2%}')
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
else:
    st.info("Please upload an image to begin.")
