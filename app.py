import streamlit as st
import subprocess
import re

# Function to clean terminal output (remove ANSI escape sequences)
def clean_output(output):
    ansi_escape = re.compile(r'(?:\x1b[@-_][0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', output)

# Function to extract the last sentence from the output
def extract_last_sentence(output):
    lines = output.strip().splitlines()
    # Get the last line and check for "The image is ..."
    for line in reversed(lines):
        if "The image is" in line:
            return line.strip()
    return "Result not found."

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
                st.write(clean_output(result.stderr))  # Display any error messages
            else:
                # Clean and display the extracted result
                cleaned_result = clean_output(result.stdout)
                last_sentence = extract_last_sentence(cleaned_result)
                st.write(f"Result: {last_sentence}")
        except Exception as e:
            st.write(f"Exception: {str(e)}")
    else:
        st.write("Please enter a valid image path.")
