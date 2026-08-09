from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests

app = FastAPI(
    title="Iris Classifier API",
    description="FastAPI server connected to KServe Iris Classifier",
    version="1.0.0"
)

# KServe endpoint exposed using kubectl port-forward
KSERVE_URL = (
    "http://localhost:8083"
    "/v1/models/iris-classifier:predict"
)


class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


@app.get("/")
def root():
    return {
        "message": "Iris Classifier API is running",
        "model": "iris-classifier"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: IrisRequest):

    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    payload = {
        "instances": [features]
    }

    try:
        response = requests.post(
            KSERVE_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        predictions = result.get("predictions")

        if not predictions:
            raise HTTPException(
                status_code=502,
                detail="KServe returned no predictions"
            )

        probabilities = predictions[0]

        predicted_class = probabilities.index(
            max(probabilities)
        )

        class_names = [
            "Iris Setosa",
            "Iris Versicolor",
            "Iris Virginica"
        ]

        return {
            "input": {
                "sepal_length": data.sepal_length,
                "sepal_width": data.sepal_width,
                "petal_length": data.petal_length,
                "petal_width": data.petal_width
            },
            "prediction": {
                "class": predicted_class,
                "label": class_names[predicted_class],
                "probabilities": probabilities
            }
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"KServe request failed: {str(e)}"
        )