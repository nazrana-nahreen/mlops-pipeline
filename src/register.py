"""Pick best run, register it, export to models/champion for serving."""
import shutil, yaml, mlflow, mlflow.sklearn
from mlflow import MlflowClient

p = yaml.safe_load(open("params.yaml"))
mlflow.set_tracking_uri(p["tracking_uri"])
client = MlflowClient()

exp = client.get_experiment_by_name("house-price")
best = client.search_runs([exp.experiment_id], order_by=["metrics.rmse ASC"], max_results=1)[0]
print("Best:", best.data.params["model"], "rmse", round(best.data.metrics["rmse"]))

mv = mlflow.register_model(f"runs:/{best.info.run_id}/model", "house-price-model")
client.set_registered_model_alias("house-price-model", "champion", mv.version)

shutil.rmtree("models/champion", ignore_errors=True)
mlflow.sklearn.save_model(
    mlflow.sklearn.load_model("models:/house-price-model@champion"),
    "models/champion", serialization_format="cloudpickle")
print(f"Registered v{mv.version} as @champion -> models/champion")