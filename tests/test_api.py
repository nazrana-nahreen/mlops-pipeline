from fastapi.testclient import TestClient
from app.main import app


def test_health_and_predict():
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is True
        r = c.post("/predict", json={"bedrooms": 3, "bathrooms": 3, "floor_no": 4,
                                     "floor_area": 1500, "city": "dhaka", "area": "gulshan"})
        assert r.status_code == 200
        assert r.json()["predicted_price_taka"] > 0


def test_bad_input_rejected():
    with TestClient(app) as c:
        assert c.post("/predict", json={"floor_area": -5}).status_code == 422