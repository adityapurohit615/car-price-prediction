import pickle

import numpy as np
import pandas as pd
import streamlit as st

car_df = pickle.load(open('car.pkl', 'rb'))
pipeline = pickle.load((open('pipeline.pkl', 'rb')))


def predict(li):
    predictions = pipeline.predict(li)
    predictions = np.round(predictions, 2)
    return predictions


st.title('Car Prediction Model')
st.text('This app predicts the car you want to sell.Try filling the information below')

company = st.selectbox(
    'Select the company',
    (car_df['Company'].unique()))

company_cars = car_df[car_df['Company'] == company]

name = st.selectbox(
    'Select the model',
    (company_cars['Name'].unique()))

year = st.text_input('Year', 'Number of years')

fuel_type = st.selectbox(
    'Select the Fuel type',
    (company_cars['Fuel_type'].unique()))

kms_driven = st.text_input('Kilometers driven', 'Enter the number of kilometers')

if st.button('Predict'):
    query_df = pd.DataFrame({'Name': [name],
                             'Kms_driven': [kms_driven],
                             'Fuel_type': [fuel_type],
                             'Year': [year],
                             'Company': [company]})

    # Use the pipeline to make predictions
    prediction = predict(query_df)

    # Display the prediction
    st.title(prediction[0])
