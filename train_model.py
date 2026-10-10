from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mental_health_dataset_1000.csv"
MODEL_PATH = BASE_DIR / "mental_health_pipeline.pkl"

# Features deliberately kept for a smaller, easier-to-use assessment form.
FEATURES = [
    "Age", "Sleep_Hours", "Sleep_Quality", "Interest_in_Work",
    "Academic_Pressure", "Work_Pressure", "Financial_Stress",
    "Mood_Swings", "Stress_Level", "Anxiety_Level", "Depression_Score",
    "Concentration", "Confusion", "Aggression", "Relationship_Conflicts",
    "Physical_Activity", "Low_Energy", "Social_Isolation", "Family_History",
    "Panic_Attack", "Screen_Time", "Self_Confidence", "Happiness"
]
TARGET = "Mental_Health_Status"
CATEGORICAL = [
    "Sleep_Quality", "Mood_Swings", "Confusion", "Relationship_Conflicts",
    "Low_Energy", "Social_Isolation", "Family_History", "Panic_Attack"
]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

def main():
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[FEATURES + [TARGET]].drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    X = df[FEATURES]
    y = df[TARGET].astype(str)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    # Retrain on all rows for the app after reporting holdout metrics.
    pipeline.fit(X, y)
    joblib.dump({"pipeline": pipeline, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
