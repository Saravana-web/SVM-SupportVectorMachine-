import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

st.title('Iris Flower Species Prediction')
st.write('This app predicts the Iris flower species (setosa, versicolor, or virginica) based on sepal and petal measurements.')

# Load the dataset (assuming the CSV is in the same directory as app.py)
@st.cache_data
def load_data():
    data = pd.read_csv('Iris_dataset(SVM).csv')
    return data

ir = load_data()

# Prepare the data
x = ir[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = ir['flower_name']

# Train the model
# Using a small test_size for deployment to ensure enough training data,
# but in a real scenario, you might train on the full dataset or a robust split.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42) # Adjusted test_size for full training in app

model = SVC(kernel='linear')
model.fit(x_train, y_train)

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
