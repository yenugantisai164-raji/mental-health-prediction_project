# Mental Health Prediction — Flask + Machine Learning

## Files
- `app.py`: Flask backend and `/predict` endpoint.
- `templates/index.html`: frontend questionnaire.
- `templates/result.html`: prediction result page.
- `train_model.py`: trains a fresh model from the CSV and saves `mental_health_pipeline.pkl`.
- `mental_health_dataset_1000.csv`: supplied dataset.
- `requirements.txt`: Python dependencies.

## Run in VS Code (Windows)
1. Extract this ZIP to a folder and open that folder in VS Code.
2. Open Terminal > New Terminal.
3. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Train a model from the included CSV:
   ```bash
   python train_model.py
   ```
   This creates `mental_health_pipeline.pkl`.
6. Start the website:
   ```bash
   python app.py
   ```
7. Open `http://127.0.0.1:5000` in your browser. Keep the terminal running.

## Dataset columns removed from model
The original dataset has 31 input features plus `Mental_Health_Status` as the target. This project uses 23 input features and the target. It excludes:
- `Gender`
- `Occupation`
- `Education`
- `Marital_Status`
- `Smoking`
- `Alcohol`
- `Therapy_History`
- `Medication`

These were omitted to make the form smaller and avoid using demographic or treatment-history fields in this starter version. This is a project-design choice, not proof that they are irrelevant. If your research goal requires them, add them consistently to `FEATURES`, preprocessing, and the form.

The original target is synthetically generated in the provided notebook using rules based on some input columns. The resulting predictions are only as meaningful as that synthetic labeling process and should not be presented as clinically validated.

## Important
- Train a new model with `train_model.py`. Do not use the old `.pkl` files: their preprocessing may not match this reduced feature list.
- The app intentionally does not require MySQL; this keeps the project easy to run locally. Do not hard-code database passwords in source files.
- This is an educational screening demo, not a diagnostic tool. Do not make treatment decisions from its output.
