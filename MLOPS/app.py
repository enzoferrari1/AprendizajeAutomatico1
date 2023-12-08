from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from clases_y_funciones import CustomWinsoring, CustomStandardScaler, CustomDummiesEncoder, Load_Keras_Model, pipe_regression, pipe_classification, X_train

import streamlit as st
import joblib
import pandas as pd
import os


input_features = ['Location', 'MinTemp', 'MaxTemp', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'Month']

categorical_features = ['Location', 'WindGustDir','WindDir9am','WindDir3pm','Month']

def get_user_input():
    """
    esta función genera los inputs del frontend de streamlit para que el usuario pueda cargar los valores.
    Además, contiene el botón para hacer el submit y obtener la predicción.
    No hace falta hacerlo así, las posibilidades son infinitas.
    """
    input_dict = {}

    with st.form(key='my_form'):
        for feat in input_features:
            if feat in categorical_features:
                input_value = st.selectbox(f'Introduzca valor para {feat}', X_train[feat].unique())
                input_dict[feat] = input_value
            else:
                input_value = st.number_input(f"Enter value for {feat}", value=0.0, step=0.01)
                input_dict[feat] = input_value


        submit_button = st.form_submit_button(label='Submit')
    input_df = pd.DataFrame([input_dict])
    input_df.to_csv('input_df.csv', index=False)
    return input_df, submit_button


user_input, submit_button = get_user_input()


# When the 'Submit' button is pressed, perform the prediction
if submit_button:
    # Predict wine quality
    prediction = pipe_classification.predict(user_input)
    prediction_classification_value = prediction[0]
    llueve_bool = prediction_classification_value > 0.5

    # Display the prediction
    st.header("Predicción de lluvia ¿mañana llueve?")
    st.write(prediction_classification_value)
    st.write(llueve_bool)
    if llueve_bool:
        st.header("Cantidad de lluvia")
        prediction = pipe_regression.predict(user_input)
        prediction_regression_value = prediction[0]
        st.write(prediction_regression_value)
    


st.markdown(
    """
    Link del repositorio:<br>
    [Github](https://github.com/enzoferrari1/AprendizajeAutomatico1)
    """, unsafe_allow_html=True
)