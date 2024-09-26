import streamlit as st
import pandas as pd
import pika
import requests
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score, recall_score, RocCurveDisplay


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
    st.table(data.head(3))

   #%% SCores
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.write("Precision")
            st.write(precision_score(data["PE"], data["y_pred"], average='micro'))

    with col2:
        with st.container(border=True):
            st.write("Accuracy")
            st.write(accuracy_score(data["PE"], data["y_pred"]))

    with col3:
        with st.container(border=True):
            st.write("F1 Score")
            st.write(f1_score(data["PE"], data["y_pred"], average='micro'))

    with col4:  
        with st.container(border=True):
            st.write("Recall")
            st.write(recall_score(data["PE"], data["y_pred"],average='micro'))

     #%% figures - heatmap
    st.pyplot(ConfusionMatrixDisplay.from_predictions(data["PE"], data["y_pred"]).figure_)

