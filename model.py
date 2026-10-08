import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import joblib

df_train = pd.read_csv("Datasets/Training.csv")
df_test = pd.read_csv("Datasets/Testing.csv")
df = pd.concat([df_train, df_test], ignore_index=True)

x = df.drop(columns='Outcome')
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state = 2)
scaler = StandardScaler()
scaler.fit_transform(x_train)
standard_data = scaler.transform(x_test)

classifier = svm.SVC(kernel = 'rbf', C= 1, gamma = 1)
classifier.fit(x_train, y_train)

joblib.dump(classifier, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

'''
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
    preg, glucose, bp, skin, insulin, bmi, dpf, age]], columns=x.columns)

    new_data = scaler.transform(new_data)

    result = classifier.predict(new_data)
    if result[0] == 1:
        print("Patient is Diabetic")
    else:
        print("Patient is Not Diabetic") 
'''