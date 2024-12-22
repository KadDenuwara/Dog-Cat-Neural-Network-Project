import os
import sys
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Load the trained model
model = load_model('cat_dog_model.h5')


# Function to predict
def predict_image(image_path):
    # Load and preprocess the image
    test_image = image.load_img(str(image_path), target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Predict
    prediction = model.predict(test_image)
    class_label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
    return class_label


# Main function
if __name__ == '__main__':
    # Check if an image path is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
    else:
        image_path = sys.argv[1]
        try:
            result = predict_image(image_path)
            print(f'The image is a {result}.')  # Only print the result
        except Exception as e:
            print(f"Error: {str(e)}")
