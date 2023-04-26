import streamlit as st
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# importing model
pipe = pickle.load(open('F:/Data Science/Data Sets/Laptop Price Prediction/pipe.pkl','rb'))
df = pickle.load(open('F:/Data Science/Data Sets/Laptop Price Prediction/df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,32])

# Gpu
gpu = st.selectbox('Gpu_Brand',df['Gpu'].unique())

# OpSys
opsys = st.selectbox('OpSys',df['OpSys'].unique())
                     
# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS Display
ips = st.selectbox('IPS_Panel',['No','Yes'])

# Screen Size 
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','230x1440'])

# Cpu
cpu = st.selectbox('Cpu_Brand',df['Cpu_Brand'].unique())

# Hard Drive (HDD)
hdd = st.selectbox('HDD',[0,128,256,512,1024,2048])

# SSd
ssd = st.selectbox('SSD',[0,128,256,512,1024,2048])

if st.button('Predict Price'):
    # Query
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == "Yes":
        ips = 1
    else:
        ips = 0
    
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = int(((x_res**2 + y_res**2)**0.5)/screen_size)

    query = np.array([company,type,ram,gpu,opsys,weight,touchscreen,ips,ppi,cpu,hdd,ssd])
    query = query.reshape(1,12)

    st.title("Predicted Laptop Price is:" + str(int(np.exp(pipe.predict(query)[0]))))
