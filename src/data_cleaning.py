"""Data validation and cleaning helpers."""
import pandas as pd

REQUIRED_COLUMNS = [
    "student_id",
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_average",
    "previous_exam_score",
    "sleep_hours_per_day",
    "final_exam_score",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]

def clean_student_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, coerce numeric fields, handle missing values, and remove invalid rows."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    cleaned = data[REQUIRED_COLUMNS].copy()
    cleaned["student_id"] = cleaned["student_id"].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["student_id", TARGET_COLUMN_NAME])
    # Use column medians to fill missing numeric input values.
    for column in NUMERIC_COLUMNS:
        if column != TARGET_COLUMN_NAME:
            median = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(0 if pd.isna(median) else median)

    # Keep plausible values and prevent impossible marks/percentages.
    cleaned = cleaned[cleaned["student_id"].ne("")]
    cleaned = cleaned[cleaned[TARGET_COLUMN_NAME].between(0, 100)]
    for column in ["attendance_percentage", "assignment_average", "previous_exam_score"]:
        cleaned[column] = cleaned[column].clip(0, 100)
    cleaned["study_hours_per_day"] = cleaned["study_hours_per_day"].clip(0, 24)
    cleaned["sleep_hours_per_day"] = cleaned["sleep_hours_per_day"].clip(0, 24)
    cleaned = cleaned.drop_duplicates(subset=["student_id"]).reset_index(drop=True)

    if len(cleaned) < 10:
        raise ValueError("At least 10 valid student rows are required to train the demonstration model.")
    return cleaned

TARGET_COLUMN_NAME = "final_exam_score"
