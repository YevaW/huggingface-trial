import streamlit as st
import pandas as pd
import joblib

st.header('FTDS Model Deployment')
st.write("""
Created by FTDS Curriculum Team

Use the sidebar to select input features.
""")

@st.cache
def fetch_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ardhiraka/PFDS_sources/master/campus.csv')
    return df

df = fetch_data()
st.write(df)

st.sidebar.header('User Input Features')

def user_input():
    gender = st.sidebar.selectbox('Gender', df['gender'].unique())
    ssc = st.sidebar.number_input('Secondary School Points', value=67.00)
    hsc = st.sidebar.number_input('High School Points', 0.0, value=91.0)
    hsc_s = st.sidebar.selectbox('High School Spec', df['hsc_s'].unique())
    degree_p = st.sidebar.number_input('Degree Points', 0.0, value=58.0)
    degree_t = st.sidebar.selectbox('Degree Spec', df['degree_t'].unique())
    workex = st.sidebar.selectbox('Work Experience?', df['workex'].unique())
    etest_p = st.sidebar.number_input('Etest Points', 0.0, value=78.00)
    spec = st.sidebar.selectbox('Specialization', df['specialisation'].unique())
    mba_p = st.sidebar.number_input('MBA Points', 0.0, value=54.55)

    data = {
        'gender': gender,
        'ssc_p': ssc,
        'hsc_p': hsc,
        'hsc_s': hsc_s,
        'degree_p': degree_p,
        'degree_t': degree_t,
        'workex': workex,
        'etest_p': etest_p,
        'specialisation':spec,
        'mba_p': mba_p
    }
    features = pd.DataFrame(data, index=[0])
    return features

input = user_input()

st.subheader('User Input')
st.write(input)

load_model = joblib.load("my_model.pkl")

prediction = load_model.predict(input)

if prediction == 1:
    prediction = 'Placed'
else:
    prediction = 'Not Placed'

st.write('Based on user input, the placement model predicted: ')
st.write(prediction)
  