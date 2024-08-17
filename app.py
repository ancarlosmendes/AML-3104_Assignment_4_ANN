import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
ann_mdl = tf.keras.models.load_model('ann_mdl.h5')

# Load the encoders and scaler
scaler_variables = pickle.load(open('scaler_variables.pkl','rb'))

# Streamlit app
st.title('Breast Cancer Prediction')

# User input
radius_mean = st.number_input('Radius mean:')
texture_mean = st.number_input('Texture mean:')
perimeter_mean = st.number_input('Perimeter mean:')
area_mean = st.number_input('Area mean:')
smoothness_mean = st.number_input('Smoothness mean:')
compactness_mean = st.number_input('Compactness mean:')
concavity_mean = st.number_input('Concavity mean:')
concave_points_mean = st.number_input('Concave points mean:')
symmetry_mean = st.number_input('Symmetry mean:')
fractal_dimension_mean = st.number_input('Fractal dimension mean:')
radius_se = st.number_input('Radius se:')
texture_se = st.number_input('Texture se:')
perimeter_se = st.number_input('Perimeter se:')
area_se = st.number_input('Area se:')
smoothness_se = st.number_input('Smoothness se:')
compactness_se = st.number_input('Compactness se:')
concavity_se = st.number_input('Concavity se:')
concave_points_se = st.number_input('Concave points se:')
symmetry_se = st.number_input('Symmetry se:')
fractal_dimension_se = st.number_input('Fractal dimension se:')
radius_worst = st.number_input('Radius worst:')
texture_worst = st.number_input('Texture worst:')
perimeter_worst = st.number_input('Perimeter worst:')
area_worst = st.number_input('Area worst:')
smoothness_worst = st.number_input('Smoothness worst:')
compactness_worst = st.number_input('Compactness worst:')
concavity_worst = st.number_input('Concavity worst:')
concave_points_worst = st.number_input('Concave points worst:')
symmetry_worst = st.number_input('Symmetry worst:')
fractal_dimension_worst = st.number_input('Fractal dimension worst:')


# Prepare the input data
data = pd.DataFrame({
    'radius_mean': [radius_mean],
    'texture_mean': [texture_mean],
    'perimeter_mean': [perimeter_mean],
    'area_mean': [area_mean],
    'smoothness_mean': [smoothness_mean],
    'compactness_mean': [compactness_mean],
    'concavity_mean' : [concavity_mean],
    'concave_points_mean': [concave_points_mean],
    'symmetry_mean': [symmetry_mean],
    'fractal_dimension_mean': [fractal_dimension_mean],
    'radius_se': [radius_se],
    'texture_se': [texture_se],
    'perimeter_se': [perimeter_se],
    'area_se': [area_se],
    'smoothness_se': [smoothness_se],
    'compactness_se': [compactness_se],
    'concavity_se': [concavity_se],
    'concave_points_se': [concave_points_se],
    'symmetry_se': [symmetry_se],
    'fractal_dimension_se': [fractal_dimension_se],
    'radius_worst': [radius_worst],
    'texture_worst': [texture_worst],
    'perimeter_worst': [perimeter_worst],
    'area_worst': [area_worst],
    'smoothness_worst': [smoothness_worst],
    'compactness_worst': [compactness_worst],
    'concavity_worst': [concavity_worst],
    'concave_points_worst': [concave_points_worst],
    'symmetry_worst': [symmetry_worst],
    'fractal_dimension_worst': [fractal_dimension_worst]
})


# Scale the input data
input_data_scaled = scaler_variables.transform(data)

# Predict churn
predict_bc = ann_mdl.predict(input_data_scaled)
predict_proba = predict_bc[0][0]

st.write(f'Breast Cancer Probability: {predict_proba:.2f}')

if predict_proba > 0.5:
    st.write('This woman is likely to have breast cancer')
else:
    st.write('This woman is not likely to have breast cancer')
