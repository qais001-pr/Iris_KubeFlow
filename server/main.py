from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import requests
import io
from PIL import Image


app = FastAPI(
    title="Iris Classifier API",
    description="FastAPI server connected to KServe Iris Classifier",
    version="1.0.0"
)


# KServe endpoint exposed using kubectl port-forward
KSERVE_URL = "http://localhost:8083/v1/models/iris-classifier:predict"


# ============================================================
# Pydantic Request Model
# ============================================================

class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


# ============================================================
# Iris Class Names
# ============================================================

CLASS_NAMES = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]


# ============================================================
# Helper Function: Call KServe
# ============================================================

def call_kserve(features: list[float]):
    """
    Send feature values to KServe and return prediction.
    """

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

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"KServe request failed: {str(e)}"
        )

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

    return {
        "class": predicted_class,
        "label": CLASS_NAMES[predicted_class],
        "probabilities": probabilities
    }


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Iris Classifier API is running",
        "model": "iris-classifier"
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# Prediction Using Numerical Features
# ============================================================

@app.post("/predict")
def predict_through_data(data: IrisRequest):

    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    prediction = call_kserve(features)

    return {
        "input": {
            "sepal_length": data.sepal_length,
            "sepal_width": data.sepal_width,
            "petal_length": data.petal_length,
            "petal_width": data.petal_width
        },
        "prediction": prediction
    }


# ============================================================
# Convert Image Into Iris Features
# ============================================================

def image_to_features(image: Image.Image):
    """
    Convert an uploaded image into four numerical features.

    NOTE:
    This is only a demonstration/example mapping.
    A real Iris image classifier would normally use
    a trained computer-vision model to extract features.
    """

    width, height = image.size

    aspect_ratio = width / height if height else 1

    sepal_length = round(
        4.5 + min(width / 500, 1.5),
        2
    )

    sepal_width = round(
        2.5 + min(height / 500, 1.5),
        2
    )

    petal_length = round(
        1.0 + min(aspect_ratio * 2.0, 4.0),
        2
    )

    petal_width = round(
        0.2 + min(aspect_ratio * 0.5, 1.0),
        2
    )

    return [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]


# ============================================================
# Prediction Using Image
# ============================================================

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    # Validate file type
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file"
        )

    # Read image
    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Convert image to Iris features
    features = image_to_features(image)

    # Call KServe
    prediction = call_kserve(features)

    return {
        "filename": file.filename,

        "features": {
            "sepal_length": features[0],
            "sepal_width": features[1],
            "petal_length": features[2],
            "petal_width": features[3]
        },

        "prediction": prediction
    }
