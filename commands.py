import requests
import pandas as pd
import pika

def post_data(data, queue_name, host = "localhost", port = 5672, user = "guest", password = "guest"):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, credentials=pika.PlainCredentials(user, password)))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_publish(exchange='', routing_key=queue_name, body=data.encode('utf-8'))
    connection.close()

if __name__ == "__main__":
    url = "http://localhost:8000"
    run_id = "a6752b569d284b429608c3632a7cb509"
    resp = requests.get(url + "/model/" + run_id)    
    print(resp)

    data = pd.read_excel("./data/data_after_EDA.xlsx")
    data = data.iloc[:, 1:]
    post_data(data.to_json(), "Pe")
    resp = requests.get(url + "/predict/Pe")
    print(resp.content)

