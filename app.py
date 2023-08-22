import streamlit as st
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model

col_logo, col_name = st.columns([1, 3])

# Company logo (replace 'logo.png' with the actual path to your logo image)
logo_path = 'png-clipart-medical-symbol-who-logo-thumbnail.png'
col_logo.image(logo_path, use_column_width=True)

# Company name
col_name.title("**AI In Medical Domain**")

st.title("EEG Signals Segmentation and Classification for Schizophrenia")
st.write("**Our Model Gives an Accuracy of 99.95%**")

model = load_model('schiz2.h5')
uploaded_file = st.file_uploader("Choose a .edf file to upload", type=["edf","fif","set","bdf"], key="file")

if uploaded_file is not None:
    file_name = os.path.basename(uploaded_file.name)
    st.write(f"File Name: {file_name}")
    st.write(f"File Size: {uploaded_file.size} bytes")
    
    
    
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension in [".edf"]:
        raw = mne.io.read_raw_edf(file_name,preload=True)
    elif file_extension in [".fif"]:
        raw = mne.io.read_raw_fif(file_name,preload=True)
    elif file_extension in [".bdf"]:
        raw = mne.io.read_raw_bdf(file_name,preload=True)
    elif file_extension in [".set"]:
        raw = mne.io.read_raw_eeglab(file_name,preload=True)    
    
    raw.resample(sfreq=250)
    data = raw.get_data()
    df = pd.DataFrame(data)
    df = df.transpose()
    info = df.values
    info = (info - np.mean(info))/np.std(info)
    pca = PCA(n_components=10)
    pca.fit(info)
    info = pca.transform(info)

    X = info[:15000]
    X = X.reshape([1,15000,10])

    Y = model.predict(X)
    ans = Y[0][0] * 100

    st.write("There is " + str(ans) + "% probability that this person suffers from " + "**Schizophrenia**" + ".")

    if(ans > 50):
        st.write("**We Recommend seeing a Doctor Immediately.**")

    os.remove(file_name)
