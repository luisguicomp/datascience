# Command to execute script locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Command to run Docker image: docker run -d -p 8000:8000 <fastapi-app-name>:latest

import pandas as pd
import io

from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from typing import List

from utils.data_processing import save_log_prediction

# Create FastAPI instance
app = FastAPI()

# Initiate MLflow client
client = MlflowClient()

# Load best model (based on logloss) amongst all experiment runs
all_exps = [exp.experiment_id for exp in client.list_experiments()]
runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
run_id, exp_id = runs.loc[runs['metrics.Acuracia'].idxmin()]['run_id'], runs.loc[runs['metrics.Acuracia'].idxmin()]['experiment_id']
print(f'Loading best model: Run {run_id} of Experiment {exp_id}')
best_model = mlflow.pyfunc.load_pyfunc(f"mlruns/{exp_id}/{run_id}/artifacts/model/")

# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    X_h2o = test_df
    # Generate predictions with best model (output is H2O frame)
    preds = best_model.predict(X_h2o)
    # Convert predictions into JSON format
    json_compatible_item_data = jsonable_encoder(preds.tolist())
    return JSONResponse(content=json_compatible_item_data)

@app.post("/evaluation")
async def evaluation(vector: List[List[float]]):
    print('[+] Initiate Evaluation')
    preds = best_model.predict(vector)
    print(type(best_model))
    data = {
        "run_id": run_id,
        "pred": preds.tolist()
    }
    json_compatible_item_data = jsonable_encoder(data)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/predict_array")
async def predict_array(vector: List[List[float]]):
    print('[+] Initiate Array Prediction')
    preds = best_model.predict(vector)
    save_log_prediction(vector, preds)
    json_compatible_item_data = jsonable_encoder(preds.tolist())
    return JSONResponse(content=json_compatible_item_data)

@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the API from the model</h2>
    <p> The MLflow models and FastAPI instances have been set up successfully </p>
    <p> You can view the docs by heading to localhost:8000/docs </p>
    <p> Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests </p>
    </body>
    """
    return HTMLResponse(content=content)