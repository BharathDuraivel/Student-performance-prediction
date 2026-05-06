from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Age": float(request.form["Age"]),
        "Gender": int(request.form["Gender"]),
        "Ethnicity": int(request.form["Ethnicity"]),
        "ParentalEducation": int(request.form["ParentalEducation"]),
        "StudyTimeWeekly": float(request.form["StudyTimeWeekly"]),
        "Absences": int(request.form["Absences"]),
        "Tutoring": int(request.form["Tutoring"]),
        "ParentalSupport": int(request.form["ParentalSupport"]),
        "Extracurricular": int(request.form["Extracurricular"]),
        "Sports": int(request.form["Sports"]),
        "Music": int(request.form["Music"]),
        "Volunteering": int(request.form["Volunteering"]),
        "GPA": float(request.form["GPA"])
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    grades = {
        0.0: "A",
        1.0: "B",
        2.0: "C",
        3.0: "D",
        4.0: "F"
    }

    result = grades[prediction]

    return render_template(
        "result.html",
        prediction=result
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)