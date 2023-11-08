import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
from streamlit_lottie import st_lottie
from PIL import Image
import requests

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="My first predict AIRBNB prices app", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()




# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
#img_contact_form = Image.open("images/yt_contact_form.png")
#img_lottie_animation = Image.open("images/yt_lottie_animation.png")


# ---- HEADER SECTION ----

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Hi, I am Ignacio :wave:")
        st.write("##")
        st.title("A newbie Data Analyst From Barcelona")
        st.write("I specialize in data analysis, data visualization, machine learning, and possess in-depth knowledge of cloud technologies and SQL. My passion lies in transforming data into actionable insights to drive strategic decision-making and organizational growth.")
        st.write( 'Welcome to my first project in Streamlit, an app to predict the price with AIRBNB service in Roma')
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")





# Model
model = load_model("ml_gradient_airbnb.pkl")
st.header("Price Prediction System for Roma")

neighbourhood = st.selectbox('Neighbourhood', options=[
    'VIII Appia Antica', 'I Centro Storico', 'II Parioli/Nomentano',
       'V Prenestino/Centocelle', 'XIII Aurelia',
       'VII San Giovanni/Cinecittà', 'XII Monte Verde', 'IX Eur',
       'IV Tiburtina', 'XV Cassia/Flaminia', 'XIV Monte Mario',
       'VI Roma delle Torri', 'III Monte Sacro', 'X Ostia/Acilia',
       'XI Arvalia/Portuense'
])



accommodates = st.slider('Nº accommodates', min_value=1, max_value=17, value=1)
room_type = st.selectbox('Room Type', options=['Private room', 'Entire home/apt', 'Shared room'])


input_data = pd.DataFrame([[
    neighbourhood, accommodates, room_type,
]], columns=['neighbourhood', 'accommodates', 'room_type_x'])


if st.button('¡Descubre el precio!'):
    prediction = predict_model(model, data=input_data)
    st.write(str(prediction["prediction_label"].values[0]) + ' euros')