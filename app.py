import streamlit as st
import pandas as pd
import joblib 


model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Attack Prediction")
st.markdown("Provide the following Details")

age = st.slider("Age",15,80,25)
sex  = st.selectbox('Sex',['M','F'])
pain = st.selectbox("Type of Chest Pain",['Atypica ANgina','Non Typical Angina','Asymtomatic','Typical ANgina'])
restingBp = st.number_input("Resting Blood Pressure ",80,200,120)
cholesterol = st.number_input("Cholesterol",40,400,65)
fasting_bp = st.selectbox('Fasting blood pressure above 120 mg/dl',['Yes','NO'])
restingEcg = st.selectbox("Resting ECG",['Normal','ST','LVH'])
max_hr  =st.slider("Maximum Heart Rate ",60,220,150)
exercise_agina = st.selectbox('Exercise Induced Agina ',['Yes','No'])
oldpeak = st.slider('Oldpeak Depression' ,0.0,6.0,1.0)
st_slope = st.selectbox('ST SLOPE ',['UP',"Flat",'Down'])


if (st.button("Predict")):
    raw_input={

        'Age':age,
        'restingBp':restingBp,
        'Cholesterol':cholesterol,
        'Fasting BP':fasting_bp,
        'max_hr':max_hr,
        'Oldpeak':oldpeak,
        'Sex'+sex:1,
        'Chestpain'+pain:1,
        'Resting ECG'+ restingEcg:1,
        "Exercise_agina"+ exercise_agina:1,
        'St_slope'+ st_slope:1

    }

    input_df =pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df = input_df[expected_columns]       
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    if prediction==1:
        st.success('Hight risk of heart disease')

    else:
        st.success("Low risk of heart disease")