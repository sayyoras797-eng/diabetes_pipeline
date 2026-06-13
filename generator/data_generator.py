import random
import numpy as np
from datetime import datetime, timezone, timedelta

TASHKENT_TZ = timezone(timedelta(hours=5))

PHYSICAL_ACTIVITY_LEVELS = ["Low", "Medium", "High"]
SMOKING_STATUSES = ["Never", "Former", "Current"]
ALCOHOL_LEVELS = ["None", "Moderate", "High"]

# Lifestyle factors and their contribution to the logistic risk score.
# Family history (genetics) and smoking are well-established diabetes
# risk factors (ADA); regular physical activity is protective.
ACTIVITY_RISK = {"Low": 0.25, "Medium": 0.0, "High": -0.20}
SMOKING_RISK = {"Never": 0.0, "Former": 0.05, "Current": 0.20}
ALCOHOL_RISK = {"None": 0.0, "Moderate": 0.05, "High": 0.15}
FAMILY_HISTORY_RISK = 0.45


def generate_patient_record():
    age = random.randint(20, 75)
    gender = random.choice(["Male", "Female"])

    # Lifestyle factors
    physical_activity = random.choices(PHYSICAL_ACTIVITY_LEVELS, weights=[0.35, 0.40, 0.25])[0]
    smoking_status = random.choices(SMOKING_STATUSES, weights=[0.55, 0.25, 0.20])[0]
    alcohol_consumption = random.choices(ALCOHOL_LEVELS, weights=[0.45, 0.40, 0.15])[0]
    family_history = 1 if random.random() < 0.30 else 0

    # Low activity tends to raise BMI; high activity lowers it
    activity_bmi_shift = {"Low": 1.5, "Medium": 0.0, "High": -1.5}[physical_activity]
    bmi = round(random.gauss(28.5, 6.0) + activity_bmi_shift, 1)
    bmi = max(15.0, min(55.0, bmi))

    # Glucose rises with age and BMI, like in real clinical data
    glucose = 90 + 0.6 * (bmi - 25) + 0.3 * (age - 40) + random.gauss(0, 20)
    glucose = round(max(60.0, min(280.0, glucose)), 1)

    # Blood pressure rises with age and BMI
    blood_pressure = 70 + 0.5 * (bmi - 25) + 0.2 * (age - 40) + random.gauss(0, 8)
    blood_pressure = round(max(40.0, min(130.0, blood_pressure)), 1)

    # HbA1c reflects long-term glucose control, so it tracks glucose
    hba1c = 4.8 + 0.018 * (glucose - 90) + random.gauss(0, 0.6)
    hba1c = round(max(4.0, min(14.0, hba1c)), 2)

    # Insulin resistance increases with BMI, glucose and low activity
    activity_insulin_shift = {"Low": 5.0, "Medium": 0.0, "High": -5.0}[physical_activity]
    insulin = 50 + 1.2 * (bmi - 25) + 0.8 * (glucose - 100) + activity_insulin_shift + random.gauss(0, 30)
    insulin = round(max(0.0, min(500.0, insulin)), 1)

    # Skinfold thickness tracks body fat (BMI)
    skin_thickness = 18 + 0.7 * (bmi - 25) + random.gauss(0, 5)
    skin_thickness = round(max(5.0, min(80.0, skin_thickness)), 1)

    pregnancies = random.randint(0, 10) if gender == "Female" else 0

    # Intercept shifts the baseline so an average patient is low-risk,
    # matching real-world diabetes prevalence (~10-15%). The gaussian
    # term represents unmeasured factors (other genetics, diet details)
    # not captured by the measured features, so the relationship isn't
    # perfectly deterministic.
    risk_score = (
        -1.45 +
        0.025 * (glucose - 100) +
        0.04 * (bmi - 25) +
        0.35 * (hba1c - 5.0) +
        0.02 * (age - 40) +
        0.01 * (blood_pressure - 80) +
        0.003 * (insulin - 100) +
        0.01 * (skin_thickness - 25) +
        0.05 * pregnancies +
        ACTIVITY_RISK[physical_activity] +
        SMOKING_RISK[smoking_status] +
        ALCOHOL_RISK[alcohol_consumption] +
        FAMILY_HISTORY_RISK * family_history +
        random.gauss(0, 0.4)
    )
    diabetes_risk = round(float(1 / (1 + np.exp(-risk_score))), 4)
    label = 1 if diabetes_risk > 0.5 else 0

    return {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "hba1c": hba1c,
        "insulin": insulin,
        "skin_thickness": skin_thickness,
        "pregnancies": pregnancies,
        "physical_activity": physical_activity,
        "smoking_status": smoking_status,
        "alcohol_consumption": alcohol_consumption,
        "family_history": family_history,
        "diabetes_risk": diabetes_risk,
        "label": label,
        "recorded_at": datetime.now(TASHKENT_TZ)
    }


def generate_batch(n: int = 50):
    return [generate_patient_record() for _ in range(n)]
