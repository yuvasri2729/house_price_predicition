import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.data_preprocessing import load_data, preprocess_data

def train_and_save_model(data_path, model_path):
    """Train model with location and save"""
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained successfully. MSE: {mse:.2f}")

    # Save model + preprocessor
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "preprocessor": preprocessor}, f)

if __name__ == "__main__":
    data_path = "dataset/house_prices_500_with_location.csv"  # Updated dataset
    model_path = "models/model.pkl"
    train_and_save_model(data_path, model_path)
