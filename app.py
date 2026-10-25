"""Streamlit dashboard for the Student Performance Tracker internship project."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.model import FEATURE_COLUMNS, TARGET_COLUMN, train_model, predict_score
from src.data_cleaning import clean_student_data

st.set_page_config(page_title="Student Performance Tracker", page_icon="🎓", layout="wide")
sns.set_theme(style="whitegrid")

st.title("🎓 Student Performance Tracker")
st.caption("An educational Machine Learning dashboard using a synthetic dataset. Predictions are estimates, not official grades.")

DATA_PATH = ROOT / "data" / "student_performance.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def get_trained_model(data: pd.DataFrame):
    return train_model(data)

try:
    raw_data = load_data(str(DATA_PATH))
    data = clean_student_data(raw_data)
except Exception as exc:
    st.error(f"Could not load or clean the dataset: {exc}")
    st.stop()

model, X_test, y_test = get_trained_model(data)
test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
rmse = mean_squared_error(y_test, test_predictions) ** 0.5
r2 = r2_score(y_test, test_predictions)

with st.sidebar:
    st.header("Dashboard controls")
    search_id = st.text_input("Find student ID", placeholder="e.g. STU025").strip().upper()
    score_filter = st.selectbox("Performance group", ["All", "Needs support (<50)", "Developing (50–69.9)", "Strong (70+)"])
    st.divider()
    st.caption("Dataset: 300 synthetic student records")
    st.caption("Model: Random Forest Regressor")

filtered = data.copy()
if search_id:
    filtered = filtered[filtered["student_id"].str.upper().str.contains(search_id, na=False)]
if score_filter == "Needs support (<50)":
    filtered = filtered[filtered[TARGET_COLUMN] < 50]
elif score_filter == "Developing (50–69.9)":
    filtered = filtered[(filtered[TARGET_COLUMN] >= 50) & (filtered[TARGET_COLUMN] < 70)]
elif score_filter == "Strong (70+)":
    filtered = filtered[filtered[TARGET_COLUMN] >= 70]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students shown", f"{len(filtered):,}")
c2.metric("Average exam score", f"{data[TARGET_COLUMN].mean():.1f}/100")
c3.metric("Average attendance", f"{data['attendance_percentage'].mean():.1f}%")
c4.metric("Model MAE", f"{mae:.2f} points", help="Mean absolute error on the held-out test set; lower is better.")

tab_overview, tab_students, tab_predict, tab_model = st.tabs(["Overview", "Student records", "Predict a score", "Model details"])

with tab_overview:
    st.subheader("Performance overview")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=data, x=TARGET_COLUMN, bins=15, kde=True, ax=ax)
        ax.set_title("Distribution of final exam scores")
        ax.set_xlabel("Final exam score (0–100)")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=data, x="study_hours_per_day", y=TARGET_COLUMN, hue="attendance_percentage", palette="viridis", ax=ax)
        ax.set_title("Study time and final exam score")
        ax.set_xlabel("Study hours per day")
        ax.set_ylabel("Final exam score")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Average score by performance group")
    group_labels = pd.cut(data[TARGET_COLUMN], bins=[-0.1, 50, 70, 100.1], labels=["Needs support", "Developing", "Strong"], right=False)
    counts = group_labels.value_counts().reindex(["Needs support", "Developing", "Strong"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of students")
    ax.set_xlabel("")
    ax.set_title("Student count by score band")
    st.pyplot(fig)
    plt.close(fig)

with tab_students:
    st.subheader("Student records")
    st.write("Use the sidebar to search by ID or filter by score group.")
    display = filtered.copy()
    display["performance_group"] = pd.cut(
        display[TARGET_COLUMN], bins=[-0.1, 50, 70, 100.1],
        labels=["Needs support", "Developing", "Strong"], right=False
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download displayed records as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_filtered.csv",
        mime="text/csv"
    )
    if filtered.empty:
        st.info("No students match these filters. Try another student ID or performance group.")

with tab_predict:
    st.subheader("Predict a student's exam score")
    st.write("Enter academic and study-related details. The model estimates a score from 0 to 100.")
    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            study_hours = st.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
            attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
            assignment = st.slider("Assignment average (%)", 0.0, 100.0, 75.0, 1.0)
        with b:
            previous = st.slider("Previous exam score (%)", 0.0, 100.0, 70.0, 1.0)
            sleep = st.slider("Sleep hours per day", 0.0, 12.0, 7.0, 0.5)
        submitted = st.form_submit_button("Predict score", type="primary")
    if submitted:
        features = {
            "study_hours_per_day": study_hours,
            "attendance_percentage": attendance,
            "assignment_average": assignment,
            "previous_exam_score": previous,
            "sleep_hours_per_day": sleep
        }
        prediction = predict_score(model, features)
        st.success(f"Estimated final exam score: {prediction:.1f} / 100")
        if prediction < 50:
            st.warning("Suggested interpretation: consider additional study support and regular progress reviews.")
        elif prediction < 70:
            st.info("Suggested interpretation: consistent practice may help improve performance.")
        else:
            st.info("Suggested interpretation: maintain effective study habits and monitor progress.")
        st.caption("This is a demonstration model trained on synthetic data. Do not use it to make high-stakes academic decisions.")

with tab_model:
    st.subheader("Model evaluation")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")
    m3.metric("R² score", f"{r2:.3f}")
    st.write("Metrics are calculated on a held-out test set (20% of the dataset).")
    st.write("**Input features used:**", ", ".join(FEATURE_COLUMNS))
    st.write("**Target predicted:**", TARGET_COLUMN)
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature importance (Random Forest)")
    ax.set_xlabel("Relative importance")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Feature importance describes how the fitted model used inputs; it does not prove causation.")
