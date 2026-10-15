import sys; sys.path.insert(0, "src")
import pandas as pd
from clean import parse_price, parse_floor


def test_parse_price():
    assert parse_price(pd.Series(["Tk 39,000,000"])).iloc[0] == 39000000


def test_parse_floor():
    assert parse_floor("8th") == 8
    assert pd.isna(parse_floor("Merin City - Purbach"))