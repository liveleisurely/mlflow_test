from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import io
import torch
from model_load import load_latest_model

app = FastAPI(title="MNIST CNN Inference API")

model = load_latest_model()

@app.get("/")
def root():
    return {"msg": "MNIST inference API is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()
    return {"prediction": int(pred)}
