"""Model training and prediction utilities."""
from typing import Mapping
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_average",
    "previous_exam_score",
    "sleep_hours_per_day",
]
TARGET_COLUMN = "final_exam_score"

def train_model(data: pd.DataFrame):
    """Train a reproducible Random Forest regression model and return its test split."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test

def predict_score(model, features: Mapping[str, float]) -> float:
    """Predict a bounded exam score from a mapping of feature names to values."""
    row = pd.DataFrame([[float(features[name]) for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(row)[0])
    return max(0.0, min(100.0, prediction))
