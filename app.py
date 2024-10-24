from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# Other necessary imports and your app setup...
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your model here (if needed)
# from model import your_model_function  # Replace with your model import

# Function to load and preprocess an image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to generate augmented images
def generate_images(num_images):
    sample_img_path = 'path/to/sample/image.jpg'  # Replace with a valid sample image path
    sample_img = load_image(sample_img_path)

    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    generated_images = []
    for i in range(num_images):
        img_iterator = data_gen.flow(sample_img)
        augmented_img = next(img_iterator)[0].astype(np.float32)  # Get the augmented image
        generated_images.append(augmented_img)

    return generated_images

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_images = int(request.form['num_images'])
        generated_images = generate_images(num_images)

        # Save generated images
        img_paths = []
        for i, img in enumerate(generated_images):
            img_path = f'static/generated_images/image_{i}.png'
            plt.imsave(img_path, img)
            img_paths.append(img_path)

        return render_template('index.html', img_paths=img_paths)

    return render_template('index.html', img_paths=None)

if __name__ == '__main__':
    app.run(debug=True)
