"""Small Telco Churn sample from the deployed synthetic generator.

The drift rules match src/generate_dataset.py in telco-churn-synthetic.
Row count comes from params.yaml so DVC can version the dataset.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params():
    with open(ROOT / "params.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate(n_samples: int, seed: int, output_file: Path) -> None:
    random.seed(seed)
    np.random.seed(seed)
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)
    total_days = (end - start).days
    rows = []
    for _ in range(n_samples):
        record_date = start + timedelta(days=random.randint(0, total_days))
        progress = (record_date - start).days / total_days
        fiber_prob = 0.40 + 0.25 * progress
        dsl_prob = 0.40 - 0.20 * progress
        no_inet_prob = 0.20 - 0.05 * progress
        echeck_prob = max(0.15, 0.40 - 0.25 * progress)
        m2m_prob = max(0.30, 0.55 - 0.25 * progress)
        streaming_boost = 0.3 * progress
        senior_prob = max(0.08, 0.18 - 0.12 * progress)

        gender = random.choice(["Male", "Female"])
        senior_citizen = 1 if random.random() < senior_prob else 0
        has_partner = random.choices(["Yes", "No"], weights=[52 + 10 * progress, 48 - 10 * progress])[0]
        has_dependents = "Yes" if random.random() < (0.3 - 0.1 * progress) else "No"
        tenure = int(np.random.beta(2 + progress, 3 - 0.5 * progress) * 72)
        tenure = max(0, min(tenure, 72))
        phone_service = "Yes" if random.random() < 0.92 else "No"
        internet_service = random.choices(
            ["DSL", "Fiber optic", "No"],
            weights=[dsl_prob, fiber_prob, no_inet_prob],
        )[0]
        if internet_service == "No":
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
        else:
            base_yes = 0.5 + streaming_boost
            online_security = "Yes" if random.random() < (base_yes * 0.7) else "No"
            online_backup = "Yes" if random.random() < (base_yes * 0.8) else "No"
            device_protection = "Yes" if random.random() < (base_yes * 0.75) else "No"
            tech_support = "Yes" if random.random() < (base_yes * 0.6) else "No"
            streaming_tv = "Yes" if random.random() < (base_yes + 0.1) else "No"
            streaming_movies = "Yes" if random.random() < (base_yes + 0.1) else "No"
        multiple_lines = "No phone service" if phone_service == "No" else (
            "Yes" if random.random() < 0.45 + 0.1 * progress else "No"
        )
        contract = random.choices(
            ["Month-to-month", "One year", "Two year"],
            weights=[m2m_prob, (1 - m2m_prob) * 0.6, (1 - m2m_prob) * 0.4],
        )[0]
        paperless_billing = "Yes" if random.random() < 0.59 + 0.15 * progress else "No"
        payment_method = random.choices(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            weights=[echeck_prob, 0.25, 0.25 + 0.1 * progress, 0.25 + 0.15 * progress],
        )[0]
        base = 20.0
        if phone_service == "Yes":
            base += 25
            if multiple_lines == "Yes":
                base += 18
        if internet_service == "DSL":
            base += 50
        elif internet_service == "Fiber optic":
            base += 82 + 10 * progress
        extra_count = sum(
            flag == "Yes"
            for flag in (
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
            )
        )
        base += extra_count * (8 + 3 * progress)
        if contract == "One year":
            base *= 0.94
        elif contract == "Two year":
            base *= 0.88 - 0.03 * progress
        monthly_charges = round(max(18.5, base + np.random.normal(0, 6)), 2)
        total_charges = round(monthly_charges * tenure * random.uniform(0.97, 1.03), 2)
        churn_base = 0.45
        if contract == "Month-to-month":
            churn_base += 0.35
        if payment_method == "Electronic check":
            churn_base += 0.18
        if internet_service == "Fiber optic":
            churn_base += 0.08
        if tenure < 12:
            churn_base += 0.25 - tenure * 0.02
        churn_base -= 0.20 * progress
        churn = "Yes" if random.random() < churn_base else "No"
        customer_id = f"{random.randint(1000, 9999)}-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"
        rows.append(
            [
                customer_id,
                gender,
                senior_citizen,
                has_partner,
                has_dependents,
                tenure,
                phone_service,
                multiple_lines,
                internet_service,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                contract,
                paperless_billing,
                payment_method,
                monthly_charges,
                total_charges,
                churn,
                record_date.strftime("%Y-%m-%d"),
            ]
        )
    columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
        "RecordDate",
    ]
    frame = pd.DataFrame(rows, columns=columns).sort_values("RecordDate")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_file, index=False)
    print(f"wrote {len(frame)} rows to {output_file}")


if __name__ == "__main__":
    params = load_params()
    generate(
        n_samples=int(params["generate"]["samples"]),
        seed=int(params["generate"]["seed"]),
        output_file=ROOT / "data" / "raw" / "telco_churn.csv",
    )
