
# Required imports for training
import mlflow
import argparse
import numpy as np
import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

mlflow.start_run()  # Start the MLflow experiment run

os.makedirs("./outputs", exist_ok=True)  # Create the "outputs" directory if it doesn't exist

def select_first_file(path):
    """Selects the first file in a folder, assuming there's only one file.
    Args:
        path (str): Path to the directory or file to choose.
    Returns:
        str: Full path of the selected file.
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument('--n_estimators', type=int, default='10', 
                        help='Number of Trees in the Forest. More trees generally improve accuracy but increase computation time.')
    parser.add_argument('--max_depth', type=int, default=None, 
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')
    parser.add_argument("--model_output", type=str, help="Path of output model")
    args = parser.parse_args()

    # Load datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Split the data into features (X) and target (y) 
    y_train = train_df['price']  # Specify the target column
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # Initialize and train a RandomForest Regressor
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)  # Provide the arguments for RandomForestRegressor
    model.fit(X_train, y_train)  # Train the model

    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")  # Provide the model name
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict using the RandomForest Regressor on test data
    yhat_test = model.predict(X_test)  # Predict the test data

    # Compute and log mean squared error for test data
    mse = mean_squared_error(y_test, yhat_test)
    print('Mean Squared Error of RandomForest Regressor on test set: {:.2f}'.format(mse))
    mlflow.log_metric("MSE", float(mse))  # Log the MSE

    # Save the Model
    model_path = "best_model.pkl"
    joblib.dump(model, model_path)

    # Save the model
    mlflow.sklearn.log_model(sk_model=model, artifact_path=args.model_output)  # Save the model

    # Output the trained model
    mlflow.sklearn.save_model(model, args.model_output)

    mlflow.end_run()  # Ending the MLflow experiment run

if __name__ == "__main__":
    main()
