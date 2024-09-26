import mlflow
import pandas as pd


mlflow.set_tracking_uri("http://127.0.0.1:5000")
run_id = "a6752b569d284b429608c3632a7cb509"
model = mlflow.sklearn.load_model(f"runs:/{run_id}//model")

data = pd.read_excel("./data/data_after_EDA.xlsx")
data = data.iloc[:, 1:]
client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
run_data_dict = client.get_run(run_id).data.to_dictionary()
print(model.predict(data.loc[:, eval(run_data_dict["params"]["input"])]))

