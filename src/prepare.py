"""Turn the raw Telco CSV into a numeric table for training."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "raw" / "telco_churn.csv")
    raw["Churn"] = (raw["Churn"] == "Yes").astype(int)
    features = raw.drop(columns=["customerID", "RecordDate"])
    features = pd.get_dummies(features, drop_first=True)
    out = ROOT / "data" / "prepared" / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    print(f"wrote {features.shape[0]} x {features.shape[1]} to {out}")


if __name__ == "__main__":
    main()
