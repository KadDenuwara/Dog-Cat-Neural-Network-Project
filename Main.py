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


# Test the function
if __name__ == '__main__':

    while True:
        run = input("Do you want to run this model? y/n: ")
        if run == 'y':
            image_path = input("Enter the image path: ")
            result = predict_image(image_path)
            print(f'The image is a {result}.')

        else:
            break


