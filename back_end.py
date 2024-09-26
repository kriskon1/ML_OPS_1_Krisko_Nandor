from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd
import pika
import time
import logging
import uvicorn

model = None
client = None
signature = None
rabbit_connection = None
channel = None
current_run_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, rabbit_connection, channel
    client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
    credentials = pika.PlainCredentials(username="guest", password="guest")
    while rabbit_connection is None:
        try:
            rabbit_connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost", port=5672, credentials=credentials, heartbeat=0))
        except pika.exceptions.AMQPConnectionError:
            logging.error(f"Connection to RabbitMQ failed at localhost:5672. Retrying...")
            time.sleep(0.3)
    channel = rabbit_connection.channel()
    channel.basic_qos(prefetch_count=1)
                
    yield
    channel.close()
    rabbit_connection.close()
    return


app = FastAPI(lifespan=lifespan)


@app.get("/") 
async def read_root():
    return "Hello, this is the starting page"


@app.get("/model/current")
def get_model_state():
    global current_run_id

    if current_run_id is None:
        return "No model is loaded"
    else:
        return current_run_id 


@app.get("/model/{run_id}")
def get_mlflow_model(run_id: str):
    global model, client, signature, current_run_id
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}//model")
    run_data_dict = client.get_run(run_id).data.to_dictionary()
    signature = eval(run_data_dict["params"]["input"])
    current_run_id = run_id
    return f"Successfully loaded model {run_id}."


@app.get("/predict/{queue}")
async def predict(queue):
    global channel
    method_frame, header_frame, body = channel.basic_get(queue)
    data = body.decode("utf-8")
    channel.basic_ack(method_frame.delivery_tag)
    data = pd.read_json(data)
    y = model.predict(data.loc[:, signature])
    data["y_pred"] = y
    # data.to_excel("output.xlsx", index=False)
    return data.to_json()



if __name__ == "__main__":
    uvicorn.run("back_end:app", host="localhost", port=8000, reload=True)
    
