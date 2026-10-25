"""Basic tests for data cleaning and model prediction."""
from pathlib import Path
import pandas as pd
from src.data_cleaning import clean_student_data
from src.model import train_model, predict_score, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_and_cleaning():
    raw = pd.read_csv(ROOT / "data" / "student_performance.csv")
    cleaned = clean_student_data(raw)
    assert len(cleaned) >= 10
    assert cleaned["student_id"].is_unique
    assert cleaned["final_exam_score"].between(0, 100).all()

def test_model_prediction_is_bounded():
    data = clean_student_data(pd.read_csv(ROOT / "data" / "student_performance.csv"))
    model, _, _ = train_model(data)
    features = {
        "study_hours_per_day": 4,
        "attendance_percentage": 80,
        "assignment_average": 75,
        "previous_exam_score": 70,
        "sleep_hours_per_day": 7,
    }
    score = predict_score(model, features)
    assert 0 <= score <= 100
    assert len(FEATURE_COLUMNS) == 5
