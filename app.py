import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model("ml_gradient_airbnb")
st.title("Sistema de predicción de precios Upgrade-Hub para la ciudad de Roma")

neighbourhood = st.selectbox('Barrio', options=[
    'VIII Appia Antica', 'I Centro Storico', 'II Parioli/Nomentano',
       'V Prenestino/Centocelle', 'XIII Aurelia',
       'VII San Giovanni/Cinecittà', 'XII Monte Verde', 'IX Eur',
       'IV Tiburtina', 'XV Cassia/Flaminia', 'XIV Monte Mario',
       'VI Roma delle Torri', 'III Monte Sacro', 'X Ostia/Acilia',
       'XI Arvalia/Portuense'
])



accommodates = st.slider('Número de Personas', min_value=1, max_value=17, value=1)
room_type = st.selectbox('Tipo de Habitación', options=['Private room', 'Entire home/apt', 'Shared room'])


input_data = pd.DataFrame([[
    neighbourhood, accommodates, room_type,
]], columns=['neighbourhood', 'accommodates', 'room_type_x'])


if st.button('¡Descubre el precio!'):
    prediction = predict_model(model, data=input_data)
    st.write(str(prediction["prediction_label"].values[0]) + ' euros')