# Health Prediction System

A beginner-friendly AIML project that demonstrates multi-class disease prediction from symptoms using Python, pandas, scikit-learn, and Streamlit.

> **Important:** This is an educational demonstration, not a medical device or diagnostic tool. The included dataset is synthetic and illustrative. Do not use its predictions to make health decisions. If you are worried about symptoms, contact a qualified healthcare professional; seek urgent help for severe or rapidly worsening symptoms.

## What the project does

- Presents a searchable symptom selector in a Streamlit web app.
- Trains a Random Forest classifier from the included CSV dataset.
- Shows the model's top condition class and other model matches.
- Reports a held-out split score for classroom demonstration.
- Runs locally or can be deployed to Streamlit Community Cloud.

## Dataset

`data/symptom_disease.csv` contains 2,400 generated rows across 15 illustrative labels. Each row has 25 binary symptom columns and a `disease` label. The examples are generated from hand-authored symptom profiles with random variation in `generate_dataset.py`; they are not patient records and do not represent medically validated symptom frequencies. The data is intentionally small and simplified for demonstrating an ML workflow. It cannot establish real-world accuracy.

Labels: common cold, flu, COVID-19, allergy, migraine, food poisoning, gastroenteritis, asthma, bronchitis, pneumonia, urinary tract infection, diabetes, arthritis, dermatitis, and GERD.

## Machine learning approach

The app loads the CSV, uses an 80/20 stratified train/test split, and trains a scikit-learn `RandomForestClassifier` with 250 trees. The app calculates accuracy on the held-out split for a simple classroom metric. Since the data is synthetic and generated from recognizable profiles, this score only measures performance on this toy dataset. The displayed per-class model score is not a medical probability or diagnosis.

## Run on your computer

You need Python 3.10 or newer and an internet connection for the initial package installation.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, use the `streamlit run app.py` command from a terminal opened after installing the requirements, or activate with the command appropriate for your shell.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in your browser.

## Regenerate the dataset (optional)

The CSV is included, so this step is not required. To reproduce it after installing the dependencies:

```bash
python generate_dataset.py
```

The generator uses a fixed random seed, so repeated runs produce the same CSV.


## Project structure

```text
Health Prediction System/
├── app.py                     # Streamlit user interface and model workflow
├── generate_dataset.py        # Reproducible synthetic dataset generator
├── requirements.txt           # Python package dependencies
├── README.md                  # Project, setup, and deployment guide
└── data/
    └── symptom_disease.csv    # Included training dataset
```

