import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib # Import joblib for loading the model

st.title('Iris Flower Species Prediction')
st.write('This app predicts the Iris flower species (setosa, versicolor, or virginica) based on sepal and petal measurements.')

# --- Load the pre-trained model ---
# Assuming the trained model is saved as 'svm_model.pkl' in the same directory
@st.cache_resource # Use st.cache_resource for models
def load_model():
    try:
        model = joblib.load('svm_iris_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'svm_model.pkl' not found. Please ensure it's in the same directory as app.py")
        st.stop()

model = load_model()

st.sidebar.header('Input Features')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input Features')
st.write(df_input)

# Make prediction
prediction = model.predict(df_input)

st.subheader('Prediction')
st.write(f'The predicted Iris species is: **{prediction[0]}**')
