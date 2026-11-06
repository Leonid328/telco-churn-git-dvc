"""Train the churn model from params.yaml. Used by the DVC pipeline."""

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def fit_and_score(features: Path, C: float, max_iter: int, seed: int) -> dict:
    frame = pd.read_csv(features)
    target = frame.pop("Churn")
    x_train, x_test, y_train, y_test = train_test_split(
        frame, target, test_size=0.2, random_state=seed, stratify=target
    )
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    metrics = {
        "model": "logreg",
        "C": float(C),
        "accuracy": round(float(accuracy_score(y_test, model.predict(x_test))), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "rows": int(len(frame)),
    }
    model_path = ROOT / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


def main() -> None:
    with open(ROOT / "params.yaml", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    train = params["train"]
    metrics = fit_and_score(
        ROOT / "data" / "prepared" / "features.csv",
        C=float(train["C"]),
        max_iter=int(train["max_iter"]),
        seed=int(train["seed"]),
    )
    metrics_path = ROOT / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
