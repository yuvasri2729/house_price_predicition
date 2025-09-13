import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(path):
    """Load dataset from CSV file"""
    return pd.read_csv(path)

def preprocess_data(df):
    """Preprocessing with location encoding"""
    # Fill missing numeric values
    df = df.fillna(df.mean(numeric_only=True))

    # Features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Identify categorical columns
    cat_cols = ["location"]
    num_cols = [col for col in X.columns if col not in cat_cols]

    # One-Hot encode location and scale numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(), cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor
