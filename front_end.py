import streamlit as st
import pandas as pd
import pika
import requests
import json
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


host = "localhost"
port = 5672
user = "guest"
password = "guest"
url = "http://localhost:8000"
upload = st.file_uploader("Upload excel.")


run_id = st.text_input("Run ID")
if st.button("Load model") and (run_id is not None or run_id != ""):
    resp = requests.get(url + "/model/" + run_id) 
    st.write(resp.content)
else:
    st.write("Click the button to load model.")

resp = requests.get(url + "/model/current") 
run_id = resp.content.decode("utf-8")
st.write(f"Current model: {run_id}")

if upload is not None:
    data = pd.read_excel("./data/data_after_EDA.xlsx")
    data = data.iloc[:, 1:]
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, credentials=pika.PlainCredentials(user, password)))
    channel = connection.channel()
    channel.queue_declare(queue="Pe", durable=True)
    channel.basic_publish(exchange='', routing_key="Pe", body=data.to_json().encode('utf-8'))
    connection.close()
    resp = requests.get(url + "/predict/Pe").json()

    data = pd.DataFrame.from_dict(json.loads(resp))
    st.write("<h2 style='font-size: 28px;'>Sample data:</h2>", unsafe_allow_html=True)
    st.table(data.head(3))

# metrics
    st.write("<h2 style='font-size: 28px;'>Metrics:</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.write("MSE")
            st.write(mean_squared_error(data["PE"], data["y_pred"]))

    with col2:
        with st.container(border=True):
            st.write("RMSE")
            st.write(root_mean_squared_error(data["PE"], data["y_pred"]))

    with col3:
        with st.container(border=True):
            st.write("R2")
            st.write(r2_score(data["PE"], data["y_pred"]))


# actual vs pred
    st.write("<h2 style='font-size: 28px;'>Actual vs Predicted Pe using kNN Regression (100 Instances)</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(15, 6))
    plt.ylim(400, 510)
    plt.grid(axis='y')
    plt.plot(np.arange(0, 100, 1), data["PE"][0:100], label="Actual")
    plt.plot(np.arange(0, 100, 1), data["y_pred"][0:100], label="Predicted")
    plt.xlabel('Instances')
    plt.ylabel('PE')
    plt.legend()
    st.pyplot(plt)

