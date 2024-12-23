import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from database import session, Prediction
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

UPLOAD_FOLDER = "uploaded_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "supersecretkey"

# Load the trained model
model = load_model("cat_dog_model.h5")

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

# Route: Home
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Predict the image
            result = predict_image(filepath)

            # Save prediction to database
            new_prediction = Prediction(image_name=filename, file_path=filepath, prediction=result)
            session.add(new_prediction)
            session.commit()

            flash(f"The image is classified as: {result}")
            return redirect(url_for("home"))

    # Fetch all predictions for display
    predictions = session.query(Prediction).all()
    return render_template("index.html", predictions=predictions)

# Route: Delete
@app.route("/delete/<int:id>")
def delete_prediction(id):
    record = session.query(Prediction).filter_by(id=id).first()
    if record:
        # Delete image file
        if os.path.exists(record.file_path):
            os.remove(record.file_path)
        # Delete record
        session.delete(record)
        session.commit()
        flash("Prediction deleted successfully!")
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
    import os
    from flask import Flask, request, render_template, redirect, url_for, flash
    from werkzeug.utils import secure_filename
    from database import session, Prediction
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from keras.preprocessing import image
    import numpy as np

    UPLOAD_FOLDER = "uploaded_images"
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.secret_key = "supersecretkey"

    # Load the trained model
    model = load_model("cat_dog_model.h5")

    # Ensure upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)


    # Helper function to check file extension
    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


    # Prediction function
    def predict_image(image_path):
        test_image = image.load_img(image_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image)
        return "Dog" if prediction[0][0] > 0.5 else "Cat"


    # Route: Home
    @app.route("/", methods=["GET", "POST"])
    def home():
        if request.method == "POST":
            # Handle file upload
            if "file" not in request.files:
                flash("No file part")
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                flash("No selected file")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # Predict the image
                result = predict_image(filepath)

                # Save prediction to database
                new_prediction = Prediction(image_name=filename, file_path=filepath, prediction=result)
                session.add(new_prediction)
                session.commit()

                flash(f"The image is classified as: {result}")
                return redirect(url_for("home"))

        # Fetch all predictions for display
        predictions = session.query(Prediction).all()
        return render_template("index.html", predictions=predictions)


    # Route: Delete
    @app.route("/delete/<int:id>")
    def delete_prediction(id):
        record = session.query(Prediction).filter_by(id=id).first()
        if record:
            # Delete image file
            if os.path.exists(record.file_path):
                os.remove(record.file_path)
            # Delete record
            session.delete(record)
            session.commit()
            flash("Prediction deleted successfully!")
        return redirect(url_for("home"))


    if __name__ == "__main__":
        app.run(debug=True)
