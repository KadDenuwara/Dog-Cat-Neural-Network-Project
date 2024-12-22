import streamlit as st
import subprocess

# App title
st.title("Cat and Dog Classifier")

# Input for the image path
image_path = st.text_input("Enter the path to your image file:")

# Button to classify the image
if st.button("Classify"):
    if image_path:
        try:
            # Run the main.py script with the image path
            result = subprocess.run(
                ["python", "main.py", image_path],
                capture_output=True,
                text=True
            )

            # Check for errors in the subprocess
            if result.returncode != 0:
                st.write("Error occurred:")
                st.write(result.stderr)  # Display any error messages
            else:
                st.write(f"Result: {result.stdout.strip()}")  # Display the result
        except Exception as e:
            st.write(f"Exception: {str(e)}")
    else:
        st.write("Please enter a valid image path.")
