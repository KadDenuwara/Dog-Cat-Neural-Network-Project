import os
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Import database setup
from database import session, Prediction

# Create folder if it doesn't exist
UPLOAD_FOLDER = "uploaded_images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model('cat_dog_model.h5')


# Function to predict
def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    class_label = 'Dog' if prediction[0][0] == 1 else 'Cat'
    return class_label


# Function to delete a prediction record and image file
def delete_prediction(record_id, file_path):
    try:
        # Delete the record from the database
        record = session.query(Prediction).filter_by(id=record_id).first()
        if record:
            session.delete(record)
            session.commit()

        # Delete the image file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        st.success("Prediction and image deleted successfully!")
    except Exception as e:
        st.error(f"Error while deleting: {str(e)}")


# App title
st.title("Cat and Dog Classifier with Image Storage and Delete Option")

# File uploader
uploaded_file = st.file_uploader("Upload an image file (JPEG or PNG):", type=["jpg", "jpeg", "png"])

if st.button("Classify"):
    if uploaded_file is not None:
        try:
            # Save the uploaded image to the 'uploaded_images' folder
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())

            # Predict the class of the image
            result = predict_image(image_path)

            # Display result
            st.write(f"The image is classified as: **{result}**")

            # Save prediction to database
            new_prediction = Prediction(image_name=uploaded_file.name, file_path=image_path, prediction=result)
            session.add(new_prediction)
            session.commit()

            st.write("Prediction saved to database.")
        except Exception as e:
            st.write(f"Error occurred: {str(e)}")
    else:
        st.write("Please upload an image before classifying.")

# View and manage database content
st.write("### Previous Predictions")
results = session.query(Prediction).all()
for record in results:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Image Name:** {record.image_name}")
        st.write(f"**Prediction:** {record.prediction}")
        st.write(f"**Timestamp:** {record.timestamp}")
    with col2:
        if st.button(f"Delete", key=f"delete_{record.id}"):
            delete_prediction(record.id, record.file_path)
            #st.experimental_rerun()  # Refresh the page to update the view
