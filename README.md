# Diabetes Prediction Using SVM

A machine learning project that predicts whether a person is **Diabetic or Non-Diabetic** based on diagnostic measurements using a **Support Vector Machine (SVM)** classifier.

The project includes the complete machine learning workflow, from data preprocessing and model training to saving the trained model and making predictions using a separate Python script.

> **Disclaimer:** This is a simple educational machine learning project. The model can make incorrect predictions and **must not be used for real medical diagnosis or clinical decisions.**

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Libraries Used](#libraries-used)
* [Data Preprocessing](#data-preprocessing)
* [Machine Learning Model](#machine-learning-model)
* [Project Structure](#project-structure)
* [Process](#process)
* [Model Evaluation](#model-evaluation)
* [Making Predictions](#making-predictions)
* [Visualizations](#visualizations)
* [What I Learned](#what-i-learned-through-this-project)
* [Disclaimer](#disclaimer)

---

## Overview

This project uses the **Diabetes Dataset from Kaggle** to build a binary classification model using a **Support Vector Machine (SVM)**.

The project covers:

* Data analysis
* Data preprocessing
* Feature scaling
* Train/test splitting
* SVM model training
* Model evaluation
* Data visualization
* Saving the trained model
* Making predictions using a Python script

The model predicts whether a person is:

* **Diabetic**
* **Non-Diabetic**

The prediction is based on the following diagnostic measurements:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

---

## Dataset

The dataset contains diagnostic measurements of patients.

### Features

| Feature                    | Description                    |
| -------------------------- | ------------------------------ |
| Pregnancies                | Number of pregnancies          |
| Glucose                    | Plasma glucose concentration   |
| Blood Pressure             | Diastolic blood pressure       |
| Skin Thickness             | Triceps skin fold thickness    |
| Insulin                    | 2-Hour serum insulin           |
| BMI                        | Body Mass Index                |
| Diabetes Pedigree Function | Diabetes hereditary risk score |
| Age                        | Age of the patient             |

### Target Variable

* `Outcome`

Where:

* `0` = Non-Diabetic
* `1` = Diabetic

---

## Libraries Used

The project was developed using Python and the following libraries:

* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Data preprocessing, SVM model, and evaluation
* **Joblib** – Saving and loading the trained model

---

## Data Preprocessing

The dataset was loaded and analyzed using Pandas.

The input features were separated from the target variable and the data was divided into training and testing sets using an **80:20 ratio** with random_state = 2.

Since SVM models are sensitive to differences in feature scales, **StandardScaler** from Scikit-learn was used to standardize the input features.

The scaler was fitted only on the training data and then used to transform both the training and testing data.

This helps prevent information from the testing data from being used during model training.

---

## Machine Learning Model

A **Support Vector Machine (SVM)** classifier with an **RBF (Radial Basis Function) kernel** was used.

The model was configured with:

```text
Kernel: RBF
C: 1
Gamma: 1
```

The model was trained using the training dataset and evaluated using the unseen testing dataset.

After training, the trained model and scaler are saved locally using `joblib`. These files are **not included in this repository** and can be generated on the user's machine by running the notebook.

---

## Project Structure

The project is organized approximately as follows:

```text
Diabetes-Prediction-SVM/
│
├── Dataset/
│   ├── Testing.csv
|   └── Training.csv
│
├── Visualization/
│   ├── Diabetes vs Non-Diabetes patients.png
│   ├── Correlation Heatmap.png
│   ├── Correlation with Outcome.png
│   └── Accuracy.jpg
│
├── diabetes_prediction.ipynb
├── predict.py
├── model.py
└── README.md
```

### Important Files

**`diabetes_prediction.ipynb`**

Contains the complete process of:

* Data loading
* Data analysis
* Preprocessing
* Train/test splitting
* Model training
* Model evaluation
* Visualization
* Saving the trained model and scaler

**`predict.py`**

Loads the locally generated scaler and trained SVM model and uses them to make predictions on new input data.

### Generated Files

Running the notebook generates the following files locally:

```text
classifier.joblib
scaler.joblib
```

These files contain:

* **`classifier.joblib`** – The trained SVM classifier.
* **`scaler.joblib`** – The fitted `StandardScaler` used to transform new input data in the same way as the training data.

These generated model files are **not uploaded to the repository**. Users can generate them on their own machine by running the model-training sections of the notebook.

---

## Process

The project follows these steps:

### 1. Load the Dataset

The diabetes dataset is loaded using Pandas.

### 2. Analyze the Dataset

The dataset is inspected to understand its features, target variable, and distribution of diabetic and non-diabetic patients.

### 3. Prepare the Data

The input features are separated from the `Outcome` column.

### 4. Split the Dataset

The dataset is divided into training and testing data using an **80:20 ratio**.

```text
80% → Training data
20% → Testing data
```

A `random_state` of `2` is used to make the split reproducible.

### 5. Standardize the Features

`StandardScaler` is used to bring the features to a similar scale.

This is particularly important for SVM because SVM is affected by the scale of the input features.

### 6. Train the SVM Model

An SVM classifier using an RBF kernel is created and trained using the standardized training data.

### 7. Evaluate the Model

The trained model is tested using the testing dataset to determine how well it performs on unseen data.

### 8. Save the Model

The trained SVM classifier and scaler are saved locally using `joblib`.

The generated files are:

```text
classifier.joblib
scaler.joblib
```

These files are generated on the user's machine and are not included in the repository.

### 9. Make Predictions

Once the model and scaler have been generated, `predict.py` can load them and make predictions on new input data.

---

## Model Evaluation

The model achieved the following results during the project:

| Dataset       | Accuracy |
| ------------- | -------: |
| Training Data |     100% |
| Testing Data  |   97.11% |

The training accuracy is higher than the testing accuracy, which indicates that the model performs better on the data it was trained on than on unseen data.

> **Note:** Accuracy alone does not provide a complete picture of a medical classification model's performance. Metrics such as precision, recall, F1-score, sensitivity, specificity, and the confusion matrix can provide additional information.

---

## Making Predictions

After the model has been trained and the required files have been generated, predictions can be made without retraining the SVM.

The `predict.py` script loads:

```text
classifier.joblib
scaler.joblib
```

It then takes the required diagnostic measurements, scales the input using the saved scaler, and passes the processed data to the trained SVM classifier.

The model returns a prediction indicating whether the input is classified as:

```text
Diabetic
```

or

```text
Non-Diabetic
```

### Generating the Model Files

Before running `predict.py`, run the model-training notebook:

```text
diabetes_prediction.ipynb
```

This will generate:

```text
classifier.joblib
scaler.joblib
```

in the project directory.

### Running the Prediction Script

Once the files have been generated, run:

```bash
python predict.py
```

The prediction script can then be used to provide new patient measurements and obtain a prediction from the trained model.

---

## Visualizations

### Diabetes vs Non-Diabetes Patients

![Diabetes vs Non-Diabetes Patients](Visualization/Diabetes%20vs%20Non-Diabetes%20patients.png)

### Correlation Heatmap

![Correlation Heatmap](Visualization/Correlation%20Heatmap.png)

### Correlation with Outcome

![Correlation with Outcome](Visualization/Correlation%20with%20Outcome.png)

### Accuracy

![Accuracy](Visualization/Accuracy.jpg)

---

## What I Learned Through This Project

Through this project, I learned:

* How to load and analyze a dataset using Pandas
* How to separate features and target variables
* How train/test splitting works
* Why feature scaling is important for SVM
* How `StandardScaler` works
* How an SVM classifier works
* How the RBF kernel is used
* How to train a machine learning model
* How to evaluate a classification model
* How to visualize data and model results
* How to save a trained machine learning model using `joblib`
* How to load a saved model for future predictions
* How to create a separate Python prediction script

---

## Disclaimer

This project is intended **for educational purposes only**.

The predictions generated by this model should **not** be considered medical advice or a medical diagnosis. The model may produce incorrect predictions and has not been validated for clinical use.

It is intended only to demonstrate the basic workflow of building, saving, and using a machine learning classification model.
