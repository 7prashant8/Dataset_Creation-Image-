import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load pre-trained model or load a saved model
@st.cache_resource
def create_model():
    # Initialize the MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()  # Initialize model

# Streamlit web app
st.title("Data Doppler - Disease Image Generator")
st.write("Upload an image of a disease to generate similar images.")

# File uploader for the user's input image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(224, 224))  # Resize image to fit model input size
    img_array = img_to_array(img) / 255.0  # Normalize image array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Show original uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict disease class
    predictions = model.predict(img_array)
    st.write("Model Predictions:", predictions)

    # Generate similar images using data augmentation
    num_images = st.number_input("How many similar images would you like to generate?", min_value=1, max_value=10, value=3)

    # Set up data augmentation for generating similar images
    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Display the generated images
    st.write("Generated Similar Images:")
    fig, axs = plt.subplots(1, num_images, figsize=(15, 15))
    img_iterator = data_gen.flow(img_array)

    for i in range(num_images):
        generated_img = next(img_iterator)[0]
        axs[i].imshow(generated_img)
        axs[i].axis('off')
    st.pyplot(fig)

st.write("**Note:** This tool is for illustrative purposes and may not accurately represent disease images.")
