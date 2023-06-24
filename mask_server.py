# Server
from fastapi import FastAPI, Depends, HTTPException
import uvicorn

# Request Models
from pydantic import BaseModel

# TensorFlow
from tensorflow.keras.models import load_model

# Image 
import cv2
import numpy as np
import base64

# Detector
from mask_detector import detect_and_predict_mask

class MaskModel(BaseModel):
    image: str
    
# load our serialized face detector model from disk
prototxtPath = "./models/deploy.prototxt"
weightsPath = "models/res10_300x300_ssd_iter_140000 (1).caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("./models/mask_detector.model")


app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/")
async def getAllPlates(reqBody: MaskModel):
    # Getting Images
    np_arr = np.frombuffer(base64.b64decode(reqBody.image), np.uint8)  # frombuffer is more stable than fromstring
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Convert bits to a numpy array
    
    
    (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
    print(type(list(preds)))
     
    label = []
    for pred in preds:
        (mask, withoutMask) = pred
        label.append("Mask" if mask > withoutMask else "No Mask")
    
    return {"Response": label}
   
if __name__ == "__main__":
    uvicorn.run("mask_server:app", host="127.0.0.1", port=8002, reload=True)