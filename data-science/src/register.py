
import os
import argparse
import logging
import mlflow
import pandas as pd
from pathlib import Path

mlflow.start_run()  # Starting the MLflow experiment run

def main():
    # Argument parser setup for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the trained model")  # Path to the trained model artifact
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    args = parser.parse_args()

    # Load the trained model from the provided path
    model = mlflow.sklearn.load_model(args.model_path)

    print("Registering the best trained machine failure prediction model")
    
    # Register the model in the MLflow Model Registry under the name "machine_failure_prediction_model"
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="train_used_cars_model",  # Descriptive model name for registration
        artifact_path="decision_tree_cars_data_classifier"  # Path to store model artifacts
    )

    # End the MLflow run
    mlflow.end_run()

if __name__ == "__main__":
    main()
