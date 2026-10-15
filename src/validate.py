import sys
import pandas as pd
import yaml
from clean import clean, parse_price

REQUIRED = ["Bedrooms", "Bathrooms", "Floor_no", "Floor_area",
            "City", "Price_in_taka", "Location"]


def validate(p: dict) -> list[str]:
    errors = []
    raw = pd.read_csv(p["data_path"])
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        return [f"Missing columns: {missing}"]
    if parse_price(raw["Price_in_taka"]).isna().mean() > 0.01:
        errors.append("Price_in_taka: >1% unparseable")
    df = clean(raw, p)
    if len(df) < p["min_rows"]:
        errors.append(f"Only {len(df)} rows after cleaning (<{p['min_rows']})")
    if df[p["target"]].isna().any():
        errors.append("Target has nulls after cleaning")
    if df["city"].nunique() < 2:
        errors.append("City has <2 unique values")
    print(f"raw={len(raw)} clean={len(df)} dropped={len(raw) - len(df)}")
    return errors


if __name__ == "__main__":
    p = yaml.safe_load(open("params.yaml"))
    errs = validate(p)
    if errs:
        print("VALIDATION FAILED:", *errs, sep="\n- ")
        sys.exit(1)
    print("Data validation passed")