import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

classifier = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_model():
    preg = int(input("No. of pregnancies: "))
    glucose = float(input("Glucose Level: "))
    bp = float(input("Blood Pressure: "))
    skin = float(input("Skin Thickness: "))
    insulin = float(input("Insulin Level: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))

    new_data = pd.DataFrame([[
    preg, glucose, bp, skin, insulin, bmi, dpf, age]], columns=[
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"])

    new_data = scaler.transform(new_data)

    result = classifier.predict(new_data)
    if result[0] == 1:
        print("Patient is Diabetic")
    else:
        print("Patient is Not Diabetic")

predict_model()