import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn


# load and prep data
data = pd.read_excel("./data/data_after_EDA.xlsx")
data = data.iloc[:, 1:]
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# train and pred
knn_params = [{"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10],
               "p": [1, 2]}]

knn = GridSearchCV(estimator=KNeighborsRegressor(),
                   param_grid=knn_params,
                   scoring="r2", cv=5,
                   return_train_score=True,
                   verbose=1)

knn.fit(X, y)
model = knn.best_estimator_
score = knn.best_score_
y_pred = knn.predict(X)


# Mlflow server setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient("http://127.0.0.1:5000")
name = "Pe"
try:
        client.create_experiment(name)
except Exception as e:
            pass   
experiment_id = client.get_experiment_by_name(name).experiment_id

with mlflow.start_run(experiment_id=experiment_id):
    run_id = mlflow.active_run().info.run_id
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("input", X.columns.to_list())

print("Done!")

