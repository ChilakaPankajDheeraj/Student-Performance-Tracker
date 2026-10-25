# Student Performance Tracker

**Internship project:** Machine Learning  
**Intern ID:** `[Enter your intern ID]`  
**Full name:** `[Enter your full name]`  
**Duration (weeks):** `[Enter the number of weeks]`  
**Project scope:** Clean and analyze student performance data, visualize trends, and estimate final exam scores with a supervised Machine Learning regression model.

## Overview

Student Performance Tracker is an educational dashboard built with Python, Streamlit, Pandas, Matplotlib, Seaborn, and scikit-learn. It uses a **synthetic dataset** of 300 fictional student records. The dashboard summarizes results, filters student records, shows visualizations, evaluates a Random Forest regression model, and estimates a score from user-provided inputs.

> **Important:** All included records are synthetic, not real student data. Predictions are demonstrations and must not be used to make high-stakes academic decisions.

## Features

- Data validation, missing-value handling, duplicate removal, and range checks.
- Summary metrics for student count, average score, attendance, and model error.
- Score distribution, study-hours scatter plot, and performance-group chart.
- Search by student ID and filter by score band.
- Download filtered records as CSV.
- Random Forest regression model to estimate final exam score.
- Evaluation using MAE, RMSE, and R².
- Feature-importance chart.
- Basic automated tests.

## Folder structure

```text
Student-Performance-Tracker/
├── app.py
├── data/
│   └── student_performance.csv
├── notebooks/
├── screenshots/
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   └── model.py
├── tests/
│   └── test_project.py
├── .gitignore
├── README.md
└── requirements.txt
```

The `notebooks/` and `screenshots/` folders are reserved for optional analysis notebooks and your own application screenshots.

## Dataset description

The dataset contains 300 fictional students and these columns:

| Column | Meaning |
|---|---|
| `student_id` | Fictional unique student identifier |
| `study_hours_per_day` | Average study hours per day |
| `attendance_percentage` | Attendance percentage |
| `assignment_average` | Average assignment score |
| `previous_exam_score` | Score in a previous exam |
| `sleep_hours_per_day` | Average sleep hours per day |
| `final_exam_score` | Final exam score, the model target |

The data is generated with a reproducible random seed and a little random variation to make it suitable for demonstration. Relationships in synthetic data do not represent verified educational research.

## Machine Learning workflow

1. Load the CSV file with Pandas.
2. Validate required columns and convert numeric fields.
3. Fill missing input values with medians, remove invalid target rows and duplicate IDs, and constrain values to plausible ranges.
4. Split the dataset into training and test sets (80% / 20%).
5. Train a `RandomForestRegressor`.
6. Evaluate held-out predictions using MAE, RMSE, and R².
7. Accept new inputs in the dashboard and produce an estimated score.

## Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- VS Code (recommended) or another Python editor
- Internet access for the initial package installation

## Run in VS Code on Windows

Open the project folder in VS Code. In the VS Code terminal, run these commands from the project root.

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

If the `py -3.11` launcher is unavailable, try:

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in your browser. Stop the server with `Ctrl+C` in the terminal.

## Test the project

With the virtual environment active, run:

```powershell
python -m pytest -q
```

Both tests should pass. If `pytest` is not found, confirm that the virtual environment is activated and run `python -m pip install -r requirements.txt` again.

## How to use the dashboard

1. Open the **Overview** tab to see the dataset's score distribution and relationships.
2. Use the sidebar to search for a student ID such as `STU025` or filter performance groups.
3. Open **Student records** to review or download filtered records.
4. Open **Predict a score**, enter the five inputs, and click **Predict score**.
5. Open **Model details** to view evaluation metrics and feature importance.

## Screenshots to capture for submission

After running the application, save genuine screenshots of your own running project in the `screenshots/` folder:

1. `dashboard.png` — Overview tab showing summary metrics and charts.
2. `student_records.png` — Student records table and sidebar filters.
3. `prediction.png` — Prediction form and a displayed prediction result.
4. `model_evaluation.png` — MAE, RMSE, R², and feature importance.
5. `terminal_run.png` — VS Code terminal showing the Streamlit command and successful startup (hide personal information if visible).
6. `tests_passed.png` — terminal showing `python -m pytest -q` passing.

Do not create fake screenshots or claim tests passed until you run them.

## GitHub submission

Create a new **public** repository named `Student-Performance-Tracker` (or use a private repository if your internship permits it). From the project root, run:

```powershell
git init
git add .
git commit -m "Build Student Performance Tracker ML dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Student-Performance-Tracker.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If the repository already has a remote, check it with `git remote -v` before adding another. Never upload passwords, tokens, API keys, or private student data.

After pushing, open the repository on GitHub and confirm that `app.py`, `src/`, `data/`, `requirements.txt`, and `README.md` are present. Submit the repository URL through your internship's official submission process.

## Submission checklist

- [ ] Complete source code uploaded to GitHub.
- [ ] Unique repository created for this project.
- [ ] README includes intern ID, full name, duration in weeks, project name, and project scope.
- [ ] Dataset file uploaded.
- [ ] Requirements file uploaded.
- [ ] Code comments and clear folder structure included.
- [ ] Dashboard tested locally.
- [ ] Automated tests run and results checked.
- [ ] Genuine output screenshots saved in `screenshots/`.
- [ ] Documentation and screenshots uploaded.
- [ ] GitHub repository link submitted to the internship provider.

## Limitations and future improvements

This is a demonstration project trained on synthetic data. It cannot establish real-world student outcomes or causal relationships. Potential improvements include validating against an approved real dataset, adding privacy controls and user authentication, comparing multiple algorithms, and tracking performance over time.

## Academic integrity note

Explain the data-generation process, cleaning steps, model features, evaluation metrics, and limitations in your own words when presenting the project. Fill in the internship-specific details before submitting.
