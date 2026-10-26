"""Educational symptom-based disease prediction demo."""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "symptom_disease.csv"
SYMPTOMS = [
    "fever", "cough", "sore_throat", "runny_nose", "sneezing", "headache",
    "body_ache", "fatigue", "nausea", "vomiting", "diarrhea", "abdominal_pain",
    "chest_pain", "shortness_of_breath", "wheezing", "rash", "itching",
    "joint_pain", "chills", "loss_of_taste_or_smell", "dizziness", "frequent_urination",
    "excessive_thirst", "back_pain", "burning_urination",
]


@st.cache_resource
def load_model():
    data = pd.read_csv(DATA_PATH)
    X, y = data[SYMPTOMS], data["disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test)), len(data)


st.set_page_config(page_title="Health Prediction System", page_icon="🩺", layout="wide")
st.title("🩺 Health Prediction System")
st.write("An AIML project that uses a Random Forest model to estimate a likely condition from selected symptoms.")

with st.sidebar:
    st.header("About this project")
    st.write("The model is trained on a small, illustrative synthetic dataset included with this project.")
    st.caption("For learning and demonstration only. This is not a medical diagnosis.")

st.warning("This demo cannot diagnose illness or replace a qualified clinician. For severe or urgent symptoms, seek medical care.")
st.subheader("Select the symptoms you are experiencing")
st.caption("Choose only symptoms that apply. Predictions can be unreliable, especially for symptoms or conditions not represented in this teaching dataset.")

labels = {s: s.replace("_", " ").title() for s in SYMPTOMS}
selected = st.multiselect("Symptoms", SYMPTOMS, format_func=lambda s: labels[s], placeholder="Type to search symptoms")

if st.button("Predict condition", type="primary", use_container_width=True):
    if not selected:
        st.info("Select at least one symptom to get a prediction.")
    else:
        model, accuracy, n_rows = load_model()
        sample = pd.DataFrame([{s: int(s in selected) for s in SYMPTOMS}])
        probabilities = model.predict_proba(sample)[0]
        ranked = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        name, confidence = ranked[0]
        st.subheader("Model estimate")
        st.write(f"**Most likely class in this dataset:** {name.replace('_', ' ').title()}")
        st.caption("The percentage below reflects the model's relative score among its known classes. It is not the probability that you have this disease.")
        st.progress(float(confidence), text=f"Model score: {confidence:.0%}")
        with st.expander("Other model matches"):
            for disease, score in ranked[1:4]:
                st.write(f"{disease.replace('_', ' ').title()} — {score:.0%}")
        st.caption(f"Model validation accuracy on a held-out split: {accuracy:.0%} ({n_rows:,} illustrative rows). This score is not clinical evidence.")

with st.expander("Dataset and model details"):
    st.write("Each row is a generated teaching example with binary symptom columns and one condition label. The app trains a Random Forest classifier when needed; no personal information is saved.")
    st.write("Conditions represented: common cold, flu, COVID-19, allergy, migraine, food poisoning, gastroenteritis, asthma, bronchitis, pneumonia, urinary tract infection, diabetes, arthritis, dermatitis, and GERD.")
