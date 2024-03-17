#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import gradio as gr  # Import Gradio library

# Load the dataset
data = pd.read_csv("C:/Nirmala/GUVI/breast cancer/breast_cancer_dataset.csv")

# Data Preprocessing
# Encode target variable
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Split data into features and target
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_scaled, y)

# Prediction function
def predict_diagnosis(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                      compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                      fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                      smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                      fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
                      smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
                      symmetry_worst, fractal_dimension_worst):
    data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                      compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                      fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                      smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                      fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
                      smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
                      symmetry_worst, fractal_dimension_worst]])
    data_scaled = scaler.transform(data)
    prediction = rf_classifier.predict(data_scaled)
    if prediction[0] == 1:
        return "Malignant"
    else:
        return "Benign"

# Create Gradio interface
interface = gr.Interface(predict_diagnosis, 
                          [
                              "number", "number", "number", "number", "number",
                              "number", "number", "number", "number", "number",
                              "number", "number", "number", "number", "number",
                              "number", "number", "number", "number", "number",
                              "number", "number", "number", "number", "number",
                              "number", "number", "number", "number", "number"
                          ], 
                          "label")
interface.launch(share=True)  # Set share=True to generate a shareable link

