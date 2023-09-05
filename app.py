import streamlit as st
import numpy as np
import pandas as pd
#import tensorflow as tf
#from tensorflow import keras
#from sklearn.preprocessing import StandardScaler

# Load your pre-trained model
model = keras.Sequential()
model.add(keras.layers.Dense(6, input_dim=6, activation='relu', name='dense_1'))
model.add(keras.layers.Dense(400, activation='relu', name='dense_2'))
model.add(keras.layers.Dense(800, activation='relu', name='dense_3'))
model.add(keras.layers.Dense(1200, activation='relu', name='dense_4'))
model.add(keras.layers.Dense(4000, activation='relu', name='dense_5'))
model.add(keras.layers.Dense(1, activation='sigmoid', name='output_layer'))


# Load pre-trained weights (modify the path accordingly)
model.load_weights('model.h5')

# Streamlit UI
st.title("Kidney Stone Prediction")

st.sidebar.header("User Input")

# Add input fields for user to enter feature values
gravity = st.sidebar.number_input("Gravity", min_value=0.0, max_value=100.0, value=0.0)
ph = st.sidebar.number_input("pH", min_value=0.0, max_value=14.0, value=0.0)
osmo = st.sidebar.number_input("Osmo", min_value=0.0, max_value=100.0, value=0.0)
cond = st.sidebar.number_input("Cond", min_value=0.0, max_value=100.0, value=0.0)
urea = st.sidebar.number_input("Urea", min_value=0.0, max_value=100.0, value=0.0)
calc = st.sidebar.number_input("Calc", min_value=0.0, max_value=100.0, value=0.0)

# Make predictions when the user clicks the "Predict" button
if st.sidebar.button("Predict"):
    # Create a new data point from user inputs
    user_data = np.array([[gravity, ph, osmo, cond, urea, calc]])
    # Create a StandardScaler object

    
    # Perform data normalization
    scaler = StandardScaler()
    # Fit the scaler to your training data (modify with your actual training data)
    # scaler.fit(training_data)
    # Transform user input data
    user_data_scaled = scaler.fit_transform(user_data)

    # Make predictions using the loaded model
    prediction = model.predict(user_data_scaled)

    # Display the prediction
    if prediction[0][0] > 0.5:
        st.write("Prediction: Kidney Stone Detected")
    else:
        st.write("Prediction: No Kidney Stone Detected")

# Optionally, you can add some information or instructions
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("1. Enter the values for the features.")
st.sidebar.markdown("2. Click the 'Predict' button to see the prediction.")

# You can also add some additional information or description about your model or dataset
st.write("This is a Streamlit app for kidney stone prediction.")
