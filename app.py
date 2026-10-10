from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mental_health_pipeline.pkl"
app = Flask(__name__)

FEATURES = [
    "Age", "Sleep_Hours", "Sleep_Quality", "Interest_in_Work",
    "Academic_Pressure", "Work_Pressure", "Financial_Stress",
    "Mood_Swings", "Stress_Level", "Anxiety_Level", "Depression_Score",
    "Concentration", "Confusion", "Aggression", "Relationship_Conflicts",
    "Physical_Activity", "Low_Energy", "Social_Isolation", "Family_History",
    "Panic_Attack", "Screen_Time", "Self_Confidence", "Happiness"
]
CATEGORICAL = {
    "Sleep_Quality", "Mood_Swings", "Confusion", "Relationship_Conflicts",
    "Low_Energy", "Social_Isolation", "Family_History", "Panic_Attack"
}
CHOICES = {
    "Sleep_Quality": ["Poor", "Fair", "Good"],
    "Mood_Swings": ["Yes", "No"],
    "Confusion": ["Yes", "No"],
    "Relationship_Conflicts": ["Yes", "No"],
    "Low_Energy": ["Yes", "No"],
    "Social_Isolation": ["Yes", "No"],
    "Family_History": ["Yes", "No"],
    "Panic_Attack": ["Yes", "No"],
}
NUMERIC_RANGES = {
    "Age": (18, 100, 1, "Age"),
    "Sleep_Hours": (0, 24, 0.1, "Sleep hours per night"),
    "Interest_in_Work": (1, 5, 1, "Interest in work (1–5)"),
    "Academic_Pressure": (1, 5, 1, "Academic pressure (1–5)"),
    "Work_Pressure": (1, 5, 1, "Work pressure (1–5)"),
    "Financial_Stress": (1, 5, 1, "Financial stress (1–5)"),
    "Stress_Level": (1, 5, 1, "Stress level (1–5)"),
    "Anxiety_Level": (1, 5, 1, "Anxiety level (1–5)"),
    "Depression_Score": (1, 5, 1, "Depression score (1–5)"),
    "Concentration": (1, 5, 1, "Concentration (1–5)"),
    "Aggression": (1, 5, 1, "Aggression (1–5)"),
    "Physical_Activity": (0, 10, 1, "Physical activity (0–10)"),
    "Screen_Time": (0, 24, 0.1, "Screen time in hours"),
    "Self_Confidence": (1, 5, 1, "Self-confidence (1–5)"),
    "Happiness": (1, 10, 1, "Happiness (1–10)"),
}

def load_artifact():
    if not MODEL_PATH.exists():
        return None
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"]

@app.route("/")
def home():
    return render_template("index.html", choices=CHOICES, numeric_ranges=NUMERIC_RANGES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = {}
        for feature in FEATURES:
            raw = request.form.get(feature, "").strip()
            if not raw:
                return render_template("result.html", error=f"Please provide: {feature}")
            if feature in CATEGORICAL:
                if raw not in CHOICES[feature]:
                    return render_template("result.html", error=f"Invalid value for {feature}.")
                row[feature] = raw
            else:
                low, high, _step, label = NUMERIC_RANGES[feature]
                value = float(raw)
                if not low <= value <= high:
                    return render_template("result.html", error=f"{label} must be between {low} and {high}.")
                row[feature] = value
        pipeline = load_artifact()
        if pipeline is None:
            return render_template("result.html", error="Model not found. Run python train_model.py first.")
        input_df = pd.DataFrame([row], columns=FEATURES)
        prediction = str(pipeline.predict(input_df)[0])
        return render_template("result.html", prediction=prediction)
    except (ValueError, TypeError) as exc:
        return render_template("result.html", error=f"Please check your entries. {exc}")

if __name__ == "__main__":
    app.run(debug=True)
