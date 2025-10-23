import mlflow.pytorch
import torch

def load_latest_model():
    client = mlflow.tracking.MlflowClient(tracking_uri="http://mlflow:5000")
    experiment = client.get_experiment_by_name("mnist_cnn_experiment")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("No MLflow runs found. Train a model first.")

    run_id = runs[0].info.run_id
    print(f"Loading latest model from run_id: {run_id}")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    model.eval()
    return model
