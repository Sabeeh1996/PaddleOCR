from fastapi import FastAPI,Form, UploadFile, File, HTTPException
import numpy as np
import cv2
import PaddleOCRFunc
import OptimizePaddleOCRFunc
from typing import List
import gc
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins (like ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict this to ["POST", "GET"] if needed
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API is running!"}


@app.post("/process-image1/")
async def process_image_endpoint1(OCR_input_image: UploadFile = File(None)):
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
        


def process_single_image(image_path):
    """This is the wrapper for parallel processing, it handles one image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"image_path": image_path, "error": f"Invalid image file or path"}

        returnimg, box = OptimizePaddleOCRFunc.run_detection(image)

        if box == 1:
            barcodes = OptimizePaddleOCRFunc.ocr_paddleocr(image)
        else:
            barcodes = OptimizePaddleOCRFunc.ocr_paddleocr(returnimg)

        return {"image_path": image_path, "barcodes": barcodes}

    except Exception as e:
        return {"image_path": image_path, "error": str(e)}     
                   
 


        
@app.post("/process-images-from-db")
def process_images_from_db(image_paths: List[str] = Form(...)):
    try:
        results = []
        
        # Safe number of workers (limit to available cores, minimum of 1)
        total_cores = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() returns None
        max_workers = max(1, min(total_cores - 1, 5))  # Use up to 5 workers, adjust as needed
        print(max_workers)
        with ProcessPoolExecutor(max_workers = max_workers) as executor:
            futures = {executor.submit(process_single_image, path): path for path in image_paths}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # Optional: timeout per image processing
                    results.append(result)
                except Exception as processing_error:
                    results.append(f"Error processing {futures[future]}: {str(processing_error)}")

        return {"results": results}

    except asyncio.CancelledError:
        # This happens if the request is cancelled (e.g., client disconnects)
        print("Request cancelled, cleaning up...")
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")



        
        
