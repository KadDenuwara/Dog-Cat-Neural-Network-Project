import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('cat_dog_model.h5')


# Function to predict
def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 150))  # Resize to match input size
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    class_label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
    return class_label


# Test the function
if __name__ == '__main__':
    image_path = input("Enter the image path: ")
    result = predict_image(image_path)
    print(f'The image is a {result}.')
