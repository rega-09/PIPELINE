import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class CleanNumericFromString(BaseEstimator, TransformerMixin):
    def __init__(self, column, pattern=r"[\d\.]+"):
        self.column = column
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].astype(str).apply(
            lambda x: float(re.findall(self.pattern, x)[0]) if re.findall(self.pattern, x) else np.nan
        )
        return X

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("✅ Loaded dataset:", df.shape)
    return df

def clean_data(df):
    df = df.copy()
    df.drop(columns=["S.No.", "New_Price"], errors='ignore', inplace=True)
    for col in ["Mileage", "Engine", "Power"]:
        df = CleanNumericFromString(col).transform(df)
    return df

def build_pipeline(df, target_column="Price"):
    X = df.drop(columns=[target_column])
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    print("\n🔢 Numeric features:", numeric_features)
    print("🔠 Categorical features:", categorical_features)
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    return preprocessor

def preprocess_and_split(df, target_column="Price"):
    df_clean = clean_data(df)
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    preprocessor = build_pipeline(df_clean, target_column)
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42)
    car_data_input, car_price_output = X_train, y_train
    print("\n✅ Training data ready:")
    print("🧪 car_data_input:", car_data_input.shape)
    print("🎯 car_price_output:", car_price_output.shape)
    return car_data_input, car_price_output

if __name__ == "__main__":
    df = load_data("cars.csv")
    car_data_input, car_price_output = preprocess_and_split(df, target_column="Price")
    print("\n🚗 Full preprocessing pipeline executed successfully!")
