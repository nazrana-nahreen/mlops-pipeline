import re
import pandas as pd

FEATURES_NUM = ["bedrooms", "bathrooms", "floor_no", "floor_area"]
FEATURES_CAT = ["city", "area"]
FEATURES = FEATURES_NUM + FEATURES_CAT


def parse_price(s: pd.Series) -> pd.Series:
    # "Tk 39,000,000" -> 39000000
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9]", "", regex=True),
                         errors="coerce")


def parse_floor(v):
    # "3" -> 3, "8th" -> 8, "G+7" / "Merin City" -> NaN
    m = re.fullmatch(r"\s*(\d+)(st|nd|rd|th)?\s*", str(v))
    return float(m.group(1)) if m else float("nan")


def clean(raw: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()
    out = pd.DataFrame({
        "bedrooms": df["Bedrooms"],
        "bathrooms": df["Bathrooms"],
        "floor_no": df["Floor_no"].map(parse_floor),
        "floor_area": df["Floor_area"],
        "city": df["City"].str.lower().str.strip(),
        # "Block L, Bashundhara R-A" -> "bashundhara r-a"
        "area": df["Location"].fillna("unknown").str.split(",").str[-1]
                  .str.lower().str.strip(),
        p["target"]: parse_price(df["Price_in_taka"]),
    })
    lo, hi = p["price_range"]
    alo, ahi = p["area_range"]
    keep = (out[p["target"]].between(lo, hi)
            & out["floor_area"].between(alo, ahi)
            & (out["bedrooms"].isna() | (out["bedrooms"] <= p["max_bedrooms"]))
            & (out["floor_no"].isna() | (out["floor_no"] <= p["max_floor"])))
    return out[keep].reset_index(drop=True)