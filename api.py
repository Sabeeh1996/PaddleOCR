from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import PaddleOCRFunc


app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running!"}


@app.post("/process-image/")
async def process_image_endpoint(OCR_input_image: UploadFile = File(None)):
    try:

              
        contents = await OCR_input_image.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        returnimg = PaddleOCRFunc.run_detection(image)
        barcodes = PaddleOCRFunc.ocr_paddleocr(returnimg)

        return barcodes

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")