import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load your trained model (update with the correct path or method)
model = load_model('my_model.h5')  # or use 'my_model_directory' for TensorFlow format

# Set up Streamlit
st.title("Disease Image Generator")
st.write("Upload an image of the disease:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Generate predictions
    predictions = model.predict(img_array)
    st.write("Predictions:", predictions)

    # Ask for number of similar images to generate
    num_images = st.number_input("How many similar images would you like to generate?", min_value=1, max_value=100, value=1)

    # Data augmentation
    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generate and display images
    for i in range(num_images):
        img_iterator = data_gen.flow(img_array)
        augmented_img = next(img_iterator)[0].astype(np.float32)  # Get the augmented image
        plt.subplot(1, num_images, i + 1)
        plt.imshow(augmented_img)
        plt.axis('off')

    st.pyplot(plt)
