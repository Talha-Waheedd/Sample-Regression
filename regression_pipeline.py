import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

def load_data():
    """Generates synthetic regression data for the sample template."""
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 6)])
    df['target'] = y
    return df

def main():
    # 1. Load Data
    print("Loading data...")
    df = load_data()
    
    # 2. Split Data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Evaluate Model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Model Performance ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

if __name__ == "__main__":
    main()
