import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('model_dogbreed.h5')

class_indices = {
    0 : 'Bernese Mountain Dog',
    1 : 'Border Collie',
    2 : 'Golden Retriever',
    3 : 'Jack Russel', 
    4 : 'Siberian Husky' }

# Define function for prediction
def predict_breed(img):
    img = image.load_img(img, target_size=(256, 256))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    breed_index = np.argmax(prediction, axis=1)[0]
    return class_indices[breed_index]

# Streamlit interface
st.title('Dog Breed Classifier')
st.text("This app can only recognize Burnese Mountain Dog, Border Collie, Golden Retriever,")
st.text("Jack Russel, and Siberian Husky.")
st.text("Other dogs such as Beagle will only result to random prediction")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    breed = predict_breed(uploaded_file)
    st.write(f"Predicted Breed: {breed}")
