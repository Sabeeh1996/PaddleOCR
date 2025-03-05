from paddleocr import PaddleOCR
import json
import numpy as np
from PIL import Image
from difflib import SequenceMatcher
import re
import os
import cv2
import matplotlib.pyplot as plt
from functools import lru_cache
import gc
from pyzbar.pyzbar import decode as pyzbar_decode
import onnxruntime

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, enable_mkldnn=True)
QR_DETECTOR = cv2.QRCodeDetector()  # OpenCV QR detector

CONFIG = {
    'min_text_length': 3,
    'min_confidence': 0.70,
    'similarity_threshold': 0.7,
    'angles': [90,180, 270],
    'show_preview': True,
    'preview_size': (400, 400),
    'qr_scan_attempts': 2  # Number of different preprocessing attempts for QR
}
def text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def merge_similar_texts(text_data):
    @lru_cache(maxsize=None)
    def cached_text_similarity(text1, text2):
        return text_similarity(text1, text2)

    merged = []
    sorted_texts = sorted(text_data, key=lambda x: (-len(x['original']), -x['confidence']))
    
    for entry in sorted_texts:
        original_text = entry['original']
        matched = False
        for i, existing_entry in enumerate(merged):
            existing_original_text = existing_entry['original']
            if cached_text_similarity(original_text, existing_original_text) > CONFIG['similarity_threshold']:
                if len(original_text) > len(existing_original_text) or (len(original_text) == len(existing_original_text) and entry['confidence'] > existing_entry['confidence']):
                    merged[i] = entry  # Replace the existing entry
                matched = True
                break
        if not matched:
            merged.append(entry)
    
    return {entry['original']: entry for entry in merged}

def enhance_for_qr(image_cv):
    """Enhanced preprocessing for QR detection with multiple techniques"""
    gray = image_cv
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def rotate_images_pil(image, angles):
    """
    Rotates a given PIL image to multiple angles and returns a list of rotated images.
    
    :param image: PIL Image object.
    :param angles: List of angles (in degrees) to rotate the image.
    :return: List of rotated PIL images.
    """
    return [image.rotate(angle, expand=True) for angle in angles]

def detect_qr_opencv(image_cv):
    """Detect QR code using OpenCV's QRCodeDetector"""
    data, bbox, _ = QR_DETECTOR.detectAndDecode(image_cv)
    return data.strip() if data and data.strip() else None

def detect_qr_pyzbar(image_cv):
    """Detect QR code using PyZbar"""
    decoded_objects = pyzbar_decode(image_cv)
    for obj in decoded_objects:
        if obj.type == 'QRCODE':
            return obj.data.decode('utf-8').strip()
    return None

def detect_qr_code(image_cv):
    """Multi-method QR detection with fallback"""
    if len(image_cv.shape) == 2:
        grayscale_image = image_cv
    else:
        grayscale_image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

    qr_data = detect_qr_opencv(grayscale_image)
    if qr_data:
        return qr_data

    preprocessed = [
        grayscale_image,
        enhance_for_qr(grayscale_image),
        cv2.medianBlur(grayscale_image, 3)
    ]

    for img in preprocessed[:CONFIG['qr_scan_attempts']]:
        qr_data = detect_qr_opencv(img) or detect_qr_pyzbar(img)
        if qr_data:
            return qr_data

    return None

def run_detection(image_path):
    conf_threshold=0.3
    nms_threshold=0.3
    onnx_model_path = r"best.onnx"
    # Helper function: letterbox resize with padding
    def letterbox(image, target_size):
        h, w = image.shape[:2]
        th, tw = target_size
        ratio = min(th / h, tw / w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        resized = cv2.resize(image, (new_w, new_h))
        new_image = np.full((th, tw, 3), 114, dtype=np.uint8)
        pad_h = (th - new_h) // 2
        pad_w = (tw - new_w) // 2
        new_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return new_image, ratio, (pad_w, pad_h)
    
    # Load the ONNX model and retrieve input shape information
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_height, input_width = input_shape[2], input_shape[3]
    
    # Load and preprocess the image
    #image = cv2.imread(image_path)
    image = image_path
    if image is None:
        print("Error: Could not load image.")
        return
    original_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_img, ratio, (pad_w, pad_h) = letterbox(image_rgb, (input_height, input_width))
    resized_img = resized_img.astype(np.float32) / 255.0  # Normalize
    input_tensor = resized_img.transpose(2, 0, 1)[np.newaxis, ...]  # Add batch dimension

    # Run inference on the preprocessed image
    outputs = session.run(None, {input_name: input_tensor})
    output = outputs[0]

    # Adjust output shape if needed (assuming model output shape is [1, 84, num_detections])
    output = output.transpose(0, 2, 1)[0]  # Now shape: (num_detections, 84)
    boxes = output[:, :4]  # [x_center, y_center, w, h]
    scores = output[:, 4:]  # Class probabilities

    # Compute detection confidence and class IDs
    confidences = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    # Filter out detections with low confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    cropped_rgb = None
    if boxes.shape[0] == 0:
        print("No detections found.")
        return cropped_rgb,1

    # Convert from center (x, y, w, h) to corner coordinates (x1, y1, x2, y2)
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    # Adjust coordinates to remove letterbox padding and scale back to original image dimensions
    x1 = (x1 - pad_w) / ratio
    y1 = (y1 - pad_h) / ratio
    x2 = (x2 - pad_w) / ratio
    y2 = (y2 - pad_h) / ratio

    # Clip coordinates so they remain within image boundaries
    x1 = np.clip(x1, 0, original_image.shape[1])
    y1 = np.clip(y1, 0, original_image.shape[0])
    x2 = np.clip(x2, 0, original_image.shape[1])
    y2 = np.clip(y2, 0, original_image.shape[0])
    
    # Prepare boxes in [x, y, w, h] format for NMS
    boxes_xywh = np.column_stack([x1, y1, x2 - x1, y2 - y1])
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), confidences.tolist(), conf_threshold, nms_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        final_boxes = np.column_stack([x1, y1, x2, y2])[indices]
        final_confidences = confidences[indices]
        final_class_ids = class_ids[indices]

        # Loop through final detections and display each result
        for box, confidence, class_id in zip(final_boxes, final_confidences, final_class_ids):
            x1_disp, y1_disp, x2_disp, y2_disp = map(int, box)
            print(f'Class: {class_id}, Confidence: {confidence:.2f}, Box: ({x1_disp}, {y1_disp}, {x2_disp}, {y2_disp})')
            
            # Crop and display detection on the original image
            cropped = original_image[y1_disp:y2_disp, x1_disp:x2_disp]
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
           # plt.figure(figsize=(4, 4))
           # plt.imshow(cropped_rgb)
            return cropped_rgb,2
           # plt.title(f'Class: {int(class_id)}, Conf: {confidence:.2f}')
            #plt.axis('off')
            #plt.show()
    else:
        print("No detections after NMS")
        
def ocr_paddleocr(img_file):
    
    rotated_list = []
    try:
        image_path = Image.fromarray(img_file)
        #print(image_path)
        #pil_img = Image.open(image_path)
        pil_img = image_path.convert('L')
        img_cv_original = np.array(pil_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    qr_data = detect_qr_code(img_cv_original)
    
    blurred = cv2.GaussianBlur(img_cv_original, (0, 0), 1.5)
    unsharp = cv2.addWeighted(img_cv_original, 1.5, blurred, -0.5, 0)
    enhanced = cv2.convertScaleAbs(unsharp)
    
    pil_img = Image.fromarray(enhanced)
    rotated_list = rotate_images_pil(pil_img, CONFIG['angles'])
    
    all_texts = []
    for i, rotated_img in enumerate(rotated_list):
        try:
            rotated_cv = np.array(rotated_img)
            
            if not qr_data:
                qr_rotated = detect_qr_code(rotated_cv)
                if qr_rotated:
                    qr_data = qr_rotated
                    print(f"QR Code Detected at ° Rotation: {qr_data}")
                    break
            
            try:
                result = ocr.ocr(rotated_cv, cls=True)
            except Exception as e:
                print(f"Rotation error °: {str(e)}")
                continue
            
            if not result:
                continue
            
            valid_words = (word_info[1] for line in result if line 
                           for word_info in line if word_info and len(word_info) >= 2)
            
            for text_data in valid_words:
                if len(text_data) < 2:
                    continue
                text, confidence = text_data[0], text_data[1]
                if len(text) >= CONFIG['min_text_length'] and confidence >= CONFIG['min_confidence']:
                    all_texts.append({
                        'original': text,
                        'confidence': confidence,
                    })
                    
        except Exception as e:
            print(f"Rotation error {angle}°: {str(e)}")         

    if not all_texts:
        print("No text found in any rotation")
       # return
        
    del rotated_list, pil_img
    
    merged_results = merge_similar_texts(all_texts)
    final_output = sorted(merged_results.values(), key=lambda x: (-x['confidence'], -len(x['original'])))
    del all_texts
    
    print("\nFinal Merged OCR Results:")
    barcode_regex = re.compile(r"(\d{8,}(/?\d+)*)")
    extracted_barcode = None
    extracted_igp_no = None
    extracted_roll_number = None
    extracted_meter = None
    extracted_dimension = None
    extracted_weave = None
    extracted_cotton = None
    extracted_spandex = None
    extracted_text = None
    for result in final_output:
        text = result['original']
        text_no_space = "".join(text.split())
        textslash = text_no_space.replace("/", "")
        if barcode_regex.fullmatch(textslash):
            extracted_barcode = text_no_space
            if qr_data is not None:
                extracted_barcode = qr_data
                
    del final_output
    if extracted_barcode is None and qr_data is not None:
        extracted_barcode = qr_data
    gc.collect()
   # print("response",response)
    response = {
        "Plain_text": extracted_text,
        "Barcode_Number": extracted_barcode,
        "IGP_Number": extracted_igp_no,
        "Roll_Number": extracted_roll_number,
        "Meters": extracted_meter,
        "Composition": {
            "Cotton": extracted_cotton,
            "Spandex": extracted_spandex
        },
        "Dimensions": extracted_dimension,
        "Weave": extracted_weave
    }
   # print({
    #    "Barcode_Number": extracted_barcode,
    #    "QR_Code": qr_data
    #})

   # print(response)
 
    return response  # Return as JSON object, not a string








           






