import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np

# Load your models (make sure to update the path to where your models are saved)
models = {
    'Random Forest': pickle.load(open('random_forest_model.pkl', 'rb')),
    'SVM - Linear Kernel': pickle.load(open('svm_linear_model.pkl', 'rb')),
    'SVM - RBF Kernel': pickle.load(open('svm_rbf_model.pkl', 'rb')),
    'SVM - Poly Kernel': pickle.load(open('svm_poly_model.pkl', 'rb')),
    'Logistic Regression': pickle.load(open('logistic_regression_model.pkl', 'rb')),
    'KNN': pickle.load(open('knn_model.pkl', 'rb')),
    'Decision Tree': pickle.load(open('decision_tree_model.pkl', 'rb')),
    'Naive Bayes': pickle.load(open('naive_bayes_model.pkl', 'rb')),
    'XGBoost': pickle.load(open('xgboost_model.pkl', 'rb'))
}

def predict_diabetes():
    # Collect the input values
    inputs = np.array([[
        float(entry_pregnancies.get()),
        float(entry_glucose.get()),
        float(entry_blood_pressure.get()),
        float(entry_skin_thickness.get()),
        float(entry_insulin.get()),
        float(entry_bmi.get()),
        float(entry_diabetes_pedigree_function.get()),
        float(entry_age.get())
    ]])

    # Get selected model from dropdown
    selected_model = models[model_choice.get()]
    
    # Make prediction
    prediction = selected_model.predict(inputs.reshape(1, -1))[0]
    
    # Display prediction
    result_label.config(text=f'Prediction: {"Diabetic" if prediction == 1 else "Not Diabetic"}')

# Setup the main window
root = tk.Tk()
root.title('Diabetes Prediction')

# Create entry fields for inputs
tk.Label(root, text="Pregnancies:").grid(row=0, column=0)
entry_pregnancies = tk.Entry(root)
entry_pregnancies.grid(row=0, column=1)

tk.Label(root, text="Glucose Level:").grid(row=1, column=0)
entry_glucose = tk.Entry(root)
entry_glucose.grid(row=1, column=1)

tk.Label(root, text="Blood Pressure:").grid(row=2, column=0)
entry_blood_pressure = tk.Entry(root)
entry_blood_pressure.grid(row=2, column=1)

tk.Label(root, text="Skin Thickness:").grid(row=3, column=0)
entry_skin_thickness = tk.Entry(root)
entry_skin_thickness.grid(row=3, column=1)

tk.Label(root, text="Insulin Level:").grid(row=4, column=0)
entry_insulin = tk.Entry(root)
entry_insulin.grid(row=4, column=1)

tk.Label(root, text="BMI:").grid(row=5, column=0)
entry_bmi = tk.Entry(root)
entry_bmi.grid(row=5, column=1)

tk.Label(root, text="Diabetes Pedigree Function:").grid(row=6, column=0)
entry_diabetes_pedigree_function = tk.Entry(root)
entry_diabetes_pedigree_function.grid(row=6, column=1)

tk.Label(root, text="Age:").grid(row=7, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=7, column=1)

# Dropdown for model selection
tk.Label(root, text="Select Model:").grid(row=8, column=0)
model_choice = ttk.Combobox(root, values=list(models.keys()))
model_choice.grid(row=8, column=1)
model_choice.current(0)  # set default value

# Button to predict
predict_button = tk.Button(root, text="Predict", command=predict_diabetes)
predict_button.grid(row=9, columnspan=2)

# Label to display result
result_label = tk.Label(root, text="")
result_label.grid(row=10, columnspan=2)

# Start the GUI
root.mainloop()
