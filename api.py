from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import PaddleOCRFunc
from typing import List
import gc

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running!"}


@app.post("/process-image1/")
async def process_image_endpoint(OCR_input_image: UploadFile = File(None)):
    try:

              
        contents = await OCR_input_image.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        returnimg,box = PaddleOCRFunc.run_detection(image)
       # print(returnimg)

       # if(returnimg == 1):
        #    barcodes = PaddleOCRFunc.ocr_paddleocr(image)
        #else:
        if(box == 1):
            barcodes = PaddleOCRFunc.ocr_paddleocr(image)
        else:    
            barcodes = PaddleOCRFunc.ocr_paddleocr(returnimg)
            
        print(barcodes)
        return barcodes

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        
        
@app.post("/process-image/")
async def process_images_endpoint(OCR_input_image: List[UploadFile] = File(...)):
    try:
        results = []
        
        for OCR_input_images in OCR_input_image:
            contents = await OCR_input_images.read()
            np_image = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {OCR_input_images.filename}")
            
            returnimg, box = PaddleOCRFunc.run_detection(image)
            
            if box == 1:
                barcodes = PaddleOCRFunc.ocr_paddleocr(image)
            else:
                barcodes = PaddleOCRFunc.ocr_paddleocr(returnimg)

            results.append({
               # "filename": OCR_input_image.filename,
                "barcodes": barcodes
            })

        return {"results": results}
        del results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")