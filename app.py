from flask import Flask, request, render_template
import joblib
import numpy as np
from feature_extraction import extract_features

app = Flask(__name__)

# Load trained model
model = joblib.load("phishing_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form["url"]
        features = np.array(extract_features(url)).reshape(1, -1)
        prediction = model.predict(features)[0]  # Make prediction

        result = "Safe" if prediction == 1 else "Scam"
        return render_template("index.html", url=url, result=result)

    return render_template("index.html", url="", result="")

if __name__ == "__main__":
    app.run(debug=True)
