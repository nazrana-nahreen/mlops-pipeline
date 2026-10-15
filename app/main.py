import mlflow.sklearn, pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = mlflow.sklearn.load_model("models/champion")
    yield


app = FastAPI(title="House Price API", version="1.0", lifespan=lifespan)


class House(BaseModel):
    bedrooms: float = Field(3, ge=1, le=10)
    bathrooms: float = Field(3, ge=1, le=10)
    floor_no: float = Field(4, ge=0, le=30)
    floor_area: float = Field(1500, ge=200, le=10000, description="sq ft")
    city: str = Field("dhaka", examples=["dhaka", "chattogram"])
    area: str = Field("gulshan", examples=["gulshan", "bashundhara r-a"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(h: House):
    price = float(state["model"].predict(pd.DataFrame([h.model_dump()]))[0])
    return {"predicted_price_taka": round(price), "predicted_price_lakh": round(price / 1e5, 1)}