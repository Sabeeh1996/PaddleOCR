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

session = onnxruntime.InferenceSession(r"best.onnx")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_height, input_width = input_shape[2], input_shape[3]

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

def letterbox(image, target_size):
    """Resize image with padding to fit model input size."""
    h, w = image.shape[:2]
    th, tw = target_size
    ratio = min(th / h, tw / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    padded_image = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_h, pad_w = (th - new_h) // 2, (tw - new_w) // 2
    padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return padded_image, ratio, (pad_w, pad_h)

def convert_boxes(boxes, ratio, pad_w, pad_h, original_shape):
    """Convert boxes from center format to original image coordinates."""
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1 = (x_center - w / 2 - pad_w) / ratio, (y_center - h / 2 - pad_h) / ratio
    x2, y2 = (x_center + w / 2 - pad_w) / ratio, (y_center + h / 2 - pad_h) / ratio

    # Clip boxes to image size
    x1, y1 = np.clip(x1, 0, original_shape[1]), np.clip(y1, 0, original_shape[0])
    x2, y2 = np.clip(x2, 0, original_shape[1]), np.clip(y2, 0, original_shape[0])

    return np.column_stack([x1, y1, x2, y2])

def run_detection(image):
    CONF_THRESHOLD=0.2
    NMS_THRESHOLD=0.2

    """Runs YOLO detection on an input image."""
    original_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_img, ratio, (pad_w, pad_h) = letterbox(image_rgb, (input_height, input_width))

    input_tensor = resized_img.astype(np.float32) / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)[np.newaxis, ...]

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    output = outputs[0].transpose(0, 2, 1)[0]  # Shape: (num_detections, 84)

    boxes, scores = output[:, :4], output[:, 4:]
    confidences, class_ids = np.max(scores, axis=1), np.argmax(scores, axis=1)

    mask = confidences > CONF_THRESHOLD
    if not np.any(mask):
        print("No detections found.")
        return None, 1

    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    # Convert boxes to image coordinates
    boxes_xyxy = convert_boxes(boxes, ratio, pad_w, pad_h, original_image.shape)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) == 0:
        print("No detections after NMS.")
        return None, 1

    indices = indices.flatten()

    for idx in indices:
        x1, y1, x2, y2 = map(int, boxes_xyxy[idx])
        class_id, confidence = class_ids[idx], confidences[idx]
        print(f"Class: {class_id}, Confidence: {confidence:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

        cropped = original_image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return cropped_rgb, 2  # Return the first detected object

    return None, 1  # Fallback, but should rarely hit
        
def ocr_paddleocr(img_file):
    rotated_list = []
    try:
        image_path = Image.fromarray(img_file)
        pil_img = image_path.convert('L')
        img_cv_original = np.array(pil_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # ✅ FIRST QR CHECK - Before rotation (Original image)
    qr_data = detect_qr_code(img_cv_original)
    if qr_data:
        print(f"QR Code Detected in Original Image: {qr_data}")

    # Preprocess for better OCR and QR
    blurred = cv2.GaussianBlur(img_cv_original, (0, 0), 1.5)
    unsharp = cv2.addWeighted(img_cv_original, 1.5, blurred, -0.5, 0)
    enhanced = cv2.convertScaleAbs(unsharp)

    pil_img = Image.fromarray(enhanced)
    rotated_list = rotate_images_pil(pil_img, CONFIG['angles'])

    all_texts = []
    for i, rotated_img in enumerate(rotated_list):
        try:
            rotated_cv = np.array(rotated_img)

            # ✅ SECONDARY QR CHECK - Only if QR was not found earlier
            if not qr_data:
                qr_rotated = detect_qr_code(rotated_cv)
                if qr_rotated:
                    qr_data = qr_rotated
                    print(f"QR Code Detected at Rotation {CONFIG['angles'][i]}°: {qr_data}")
                    break  # No need to check further rotations for QR if found

            # OCR Process (existing)
            try:
                result = ocr.ocr(rotated_cv, cls=True)
            except Exception as e:
                print(f"OCR error: {str(e)}")
                continue

            if not result:
                continue

            valid_words = (
                word_info[1] for line in result if line 
                for word_info in line if word_info and len(word_info) >= 2
            )

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
            print(f"Rotation error at {CONFIG['angles'][i]}°: {str(e)}")

    if not all_texts:
        print("No text found in any rotation")

    del rotated_list, pil_img

    merged_results = merge_similar_texts(all_texts)
    final_output = sorted(merged_results.values(), key=lambda x: (-x['confidence'], -len(x['original'])))
    del all_texts

    # Barcode and Text Extraction Logic (unchanged from your code)
    barcode_regex = re.compile(r"(\d{8,}(/?\d+)*)")
    extracted_barcode = None

    for result in final_output:
        text = result['original']
        text_no_space = "".join(text.split())
        textslash = text_no_space.replace("/", "")
        if barcode_regex.fullmatch(textslash):
            extracted_barcode = text_no_space
            if qr_data:  # Prefer QR data if available
                extracted_barcode = qr_data

    del final_output
    if not extracted_barcode and qr_data:
        extracted_barcode = qr_data

    gc.collect()

    response = {
        "Plain_text": None,
        "Barcode_Number": extracted_barcode,
        #"IGP_Number": None,
        #"Roll_Number": None,
        #"Meters": None,
        #"Composition": {
         #   "Cotton": None,
         #   "Spandex": None
        #},
        #"Dimensions": None,
        #"Weave": None
    }

    return response

"""
def ocr_paddleocr1(img_file):
    rotated_list = []
    extracted_barcode = None  # Store detected barcode early

    try:
        image_path = Image.fromarray(img_file)
        pil_img = image_path.convert('L')
        img_cv_original = np.array(pil_img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # ✅ FIRST QR CHECK - Before rotation (Original image)
    qr_data = detect_qr_code(img_cv_original)
    if qr_data:
        print(f"QR Code Detected in Original Image: {qr_data}")
        extracted_barcode = qr_data  # Prefer QR if found

    # Preprocess for better OCR and QR
    blurred = cv2.GaussianBlur(img_cv_original, (0, 0), 1.5)
    unsharp = cv2.addWeighted(img_cv_original, 1.5, blurred, -0.5, 0)
    enhanced = cv2.convertScaleAbs(unsharp)

    pil_img = Image.fromarray(enhanced)
    rotated_list = rotate_images_pil(pil_img, CONFIG['angles'])

    all_texts = []
    for i, rotated_img in enumerate(rotated_list):
        try:
            rotated_cv = np.array(rotated_img)

            # ✅ SECONDARY QR CHECK - If no QR was found earlier
            if not qr_data:
                qr_rotated = detect_qr_code(rotated_cv)
                if qr_rotated:
                    qr_data = qr_rotated
                    print(f"QR Code Detected at Rotation {CONFIG['angles'][i]}°: {qr_data}")
                    extracted_barcode = qr_data  # Prefer QR
                    break  # ✅ STOP further rotation

            # OCR Process
            try:
                result = ocr.ocr(rotated_cv, cls=True)
            except Exception as e:
                print(f"OCR error: {str(e)}")
                continue

            if not result:
                continue

            valid_words = (
                word_info[1] for line in result if line 
                for word_info in line if word_info and len(word_info) >= 2
            )

            for text_data in valid_words:
                if len(text_data) < 2:
                    continue
                text, confidence = text_data[0], text_data[1]
                if len(text) >= CONFIG['min_text_length'] and confidence >= CONFIG['min_confidence']:
                    all_texts.append({
                        'original': text,
                        'confidence': confidence,
                    })

                    # ✅ STOP further rotation if barcode text is enough
                    text_no_space = "".join(text.split()).replace("/", "")
                    if re.fullmatch(r"(\d{8,}(/?\d+)*)", text_no_space):
                        extracted_barcode = text_no_space
                        print(f"✅ Barcode Found at Rotation {CONFIG['angles'][i]}°: {extracted_barcode}")
                        break  # ✅ STOP further rotation

        except Exception as e:
            print(f"Rotation error at {CONFIG['angles'][i]}°: {str(e)}")

        # ✅ Exit early if barcode is found
        if extracted_barcode:
            break

    if not all_texts:
        print("No text found in any rotation")

    del rotated_list, pil_img

    merged_results = merge_similar_texts(all_texts)
    final_output = sorted(merged_results.values(), key=lambda x: (-x['confidence'], -len(x['original'])))
    del all_texts

    # Barcode Extraction Logic
    if not extracted_barcode:
        for result in final_output:
            text = result['original']
            text_no_space = "".join(text.split()).replace("/", "")
            if re.fullmatch(r"(\d{8,}(/?\d+)*)", text_no_space):
                extracted_barcode = text_no_space
                if qr_data:  # Prefer QR data if available
                    extracted_barcode = qr_data
                break  # ✅ STOP searching if barcode is found

    del final_output
    if not extracted_barcode and qr_data:
        extracted_barcode = qr_data

    gc.collect()

    response = {
        "Plain_text": None,
        "Barcode_Number": extracted_barcode,
    }

    return response

"""


           






