"""Create the project's reproducible illustrative symptom dataset."""

from pathlib import Path
import random

import pandas as pd

RNG = random.Random(2026)
SYMPTOMS = [
    "fever", "cough", "sore_throat", "runny_nose", "sneezing", "headache",
    "body_ache", "fatigue", "nausea", "vomiting", "diarrhea", "abdominal_pain",
    "chest_pain", "shortness_of_breath", "wheezing", "rash", "itching",
    "joint_pain", "chills", "loss_of_taste_or_smell", "dizziness", "frequent_urination",
    "excessive_thirst", "back_pain", "burning_urination",
]

# Illustrative symptom likelihoods, not clinical data or medical advice.
PROFILES = {
    "common_cold": ["cough", "sore_throat", "runny_nose", "sneezing", "headache", "fatigue"],
    "flu": ["fever", "cough", "headache", "body_ache", "fatigue", "chills", "sore_throat"],
    "covid_19": ["fever", "cough", "fatigue", "headache", "body_ache", "loss_of_taste_or_smell", "shortness_of_breath"],
    "allergy": ["runny_nose", "sneezing", "itching", "rash", "cough"],
    "migraine": ["headache", "nausea", "vomiting", "dizziness", "fatigue"],
    "food_poisoning": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever", "chills"],
    "gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever", "fatigue"],
    "asthma": ["cough", "shortness_of_breath", "wheezing", "chest_pain"],
    "bronchitis": ["cough", "fatigue", "chest_pain", "shortness_of_breath", "fever"],
    "pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue", "chills"],
    "urinary_tract_infection": ["frequent_urination", "burning_urination", "back_pain", "fever", "abdominal_pain"],
    "diabetes": ["frequent_urination", "excessive_thirst", "fatigue", "dizziness"],
    "arthritis": ["joint_pain", "fatigue", "back_pain"],
    "dermatitis": ["rash", "itching"],
    "gerd": ["chest_pain", "nausea", "abdominal_pain", "cough"],
}

rows = []
for disease, core in PROFILES.items():
    for _ in range(160):
        row = {symptom: 0 for symptom in SYMPTOMS}
        for symptom in SYMPTOMS:
            # Core symptoms are common in the class; occasional absent/present
            # variation makes examples less perfectly deterministic.
            probability = 0.78 if symptom in core else 0.045
            row[symptom] = int(RNG.random() < probability)
        if not any(row.values()):
            row[RNG.choice(core)] = 1
        row["disease"] = disease
        rows.append(row)

RNG.shuffle(rows)
destination = Path(__file__).parent / "data" / "symptom_disease.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=SYMPTOMS + ["disease"]).to_csv(destination, index=False)
print(f"Wrote {len(rows)} illustrative records to {destination}")
