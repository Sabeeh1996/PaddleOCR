
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode, ZBarSymbol
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import json

def get_barcode_area(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Remove salt-and-pepper noise
    gray = cv2.medianBlur(gray, 3)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Threshold to extract extreme white pixels
    _, thresh = cv2.threshold(enhanced, 160, 255, cv2.THRESH_BINARY)

    # Find contours of the bright regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Get the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Ensure the region is large enough to be a sticker
    if w * h > 1000:
        return image[y:y+h, x:x+w], (x, y, w, h)

    return None, None





def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1,6, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# In[6]:


def remove_salt_and_pepper_noise(image, kernel_size=3):
    """Apply median filtering to remove salt and pepper noise."""
    return cv2.medianBlur(image, kernel_size)


# In[7]:




def remove_salt_and_pepper_noise(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def get_barcode_corners(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = remove_salt_and_pepper_noise(gray, kernel_size=3)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Apply adaptive thresholding to enhance edges
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            # Approximate contour to polygon
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            # Ensure it's a quadrilateral
            if len(approx) == 4:
                corners = np.squeeze(approx)
                return corners  # Returns four corner points

    return None


# In[8]:


def preprocess_image(image):
    if isinstance(image, str):  # If input is a file path, read the image
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif len(image.shape) == 3:  # Convert RGB/BGR to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")
    
    # Ensure the image is in uint8 format
    image = image.astype(np.uint8)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image, thresh

def find_largest_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None

def fix_stretched_barcode(image):
    image, thresh = preprocess_image(image)
    largest_contour = find_largest_contour(thresh)
    
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        barcode_region = image[y:y+h, x:x+w]
        
        fixed_width = 300
        fixed_height = 150
        fixed_barcode = cv2.resize(barcode_region, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
        
        return fixed_barcode  # Return processed barcode instead of saving it
    else:
        raise ValueError("No barcode found in the image.")


# In[9]:




def fix_distorted_text(image_path):
    # Load image
    image = image_path
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to improve text structure
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Deskew the image
    coords = np.column_stack(np.where(morph > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = morph.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(morph, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed


# In[10]:


def gaussian_threshold(image):
    #gray = image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# In[49]:



def correct_image_rotation(image):
    """Corrects image rotation using PaddleOCR."""
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image, cls=True)
    angles = [-word[-1]["angle"] for line in result if line for word in line if isinstance(word[-1], dict) and "angle" in word[-1]]
    
    rotation_angle = np.median(angles) if angles else 0
    print(f"Detected rotation angle: {rotation_angle:.2f} degrees")
    
    if rotation_angle != 0:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return image


# In[50]:


def mean_threshold(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)


# In[51]:




def preprocess_image(image, use_otsu=False, apply_threshold=True):
    """Preprocess image with CLAHE, denoising, morphological filtering, and optional thresholding"""
    
    # Convert image to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Light denoising to preserve text edges
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # Morphological Opening to remove small noise
    kernel = np.ones((2,2), np.uint8)  # Small kernel to preserve thin text
    morph = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    if apply_threshold:
        if use_otsu:
            # Otsu's Thresholding
            _, processed = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Adaptive Thresholding (Better for non-uniform lighting)
            processed = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 15, 5)
    else:
        processed = morph  # Skip thresholding for better OCR accuracy

    return processed



# In[53]:




def distort_correction(image):
    height, width = image.shape[:2]
    # Define source and destination points for correction
    src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_pts = np.float32([[0, 0], [width, 0], [int(0.1 * width), height], [int(0.9 * width), height]])  # Adjust based on distortion pattern

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected_image = cv2.warpPerspective(image, M, (width, height))
    return corrected_image


# In[54]:



def fix_sticker_distortion(image, expand_ratio=1.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sticker_corners = None

    for contour in contours:
        # Approximate contour with a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Sticker should have 4 corners
            sticker_corners = approx.reshape(4, 2)
            break

    if sticker_corners is None:
        print("Sticker corners not found!")
        return image  # Return original image if sticker is not detected

    # Compute the center of the sticker
    center_x, center_y = np.mean(sticker_corners, axis=0)

    # Expand each corner outward to grab the entire sticker
    expanded_corners = []
    for corner in sticker_corners:
        dx = corner[0] - center_x
        dy = corner[1] - center_y
        new_x = center_x + dx * expand_ratio
        new_y = center_y + dy * expand_ratio
        expanded_corners.append([new_x, new_y])

    expanded_corners = np.array(expanded_corners, dtype="float32")

    # Sort points (top-left, top-right, bottom-left, bottom-right)
    rect = np.zeros((4, 2), dtype="float32")
    s = expanded_corners.sum(axis=1)
    rect[0] = expanded_corners[np.argmin(s)]  # Top-left
    rect[2] = expanded_corners[np.argmax(s)]  # Bottom-right

    diff = np.diff(expanded_corners, axis=1)
    rect[1] = expanded_corners[np.argmin(diff)]  # Top-right
    rect[3] = expanded_corners[np.argmax(diff)]  # Bottom-left

    # Compute width and height dynamically for the sticker
    width_A = np.linalg.norm(rect[1] - rect[2])  # Top-right to Bottom-right
    width_B = np.linalg.norm(rect[0] - rect[3])  # Top-left to Bottom-left
    height_A = np.linalg.norm(rect[0] - rect[1])  # Top-left to Top-right
    height_B = np.linalg.norm(rect[3] - rect[2])  # Bottom-left to Bottom-right

    max_width = int(max(width_A, width_B))
    max_height = int(max(height_A, height_B))

    dst = np.array([[0, 0], [max_width - 1, 0], 
                    [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    # Compute transformation matrix and warp the entire sticker
    M = cv2.getPerspectiveTransform(rect, dst)
    fixed_sticker = cv2.warpPerspective(image, M, (max_width, max_height))

    return fixed_sticker


# In[55]:


def fix_sticker_perspective_v5(image, output_size=(600, 300)):
    """
    Fixes perspective distortion by finding the largest rectangular region that resembles a sticker.
    Ensures the full sticker is detected, avoiding QR code misdetections.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance contrast
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and filter out small ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    sticker_contour = None
    image_area = image.shape[0] * image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Ignore small regions (likely noise or QR code)
        if area < 0.02 * image_area:  # Sticker should be at least 2% of the image size
            continue

        # Approximate contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Ensure it has four corners (likely a rectangular sticker)
        if len(approx) == 4:
            sticker_contour = approx
            break

    if sticker_contour is None:
        print("Sticker not detected properly!")
        return image

    # Order points for perspective transformation
    rect = np.zeros((4, 2), dtype="float32")
    s = sticker_contour.sum(axis=1)
    rect[0] = sticker_contour[np.argmin(s)]  # Top-left
    rect[2] = sticker_contour[np.argmax(s)]  # Bottom-right

    diff = np.diff(sticker_contour, axis=1)
    rect[1] = sticker_contour[np.argmin(diff)]  # Top-right
    rect[3] = sticker_contour[np.argmax(diff)]  # Bottom-left

    # Define the destination points (fixed rectangular shape)
    width, height = output_size
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # Compute perspective transform and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    fixed_sticker = cv2.warpPerspective(image, M, (width, height))

    # Display the corrected sticker
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(fixed_sticker, cv2.COLOR_BGR2RGB))
    plt.title("Warped Symmetrical Sticker")
    plt.axis("off")
    plt.show()

    return fixed_sticker

# This version now correctly ignores small QR code regions and detects the full sticker. 🚀


# In[56]:



def fix_text_dimensions(image_path, output_size=(500, 200)):
    # Load image
    #image = cv2.imread(image_path)
    image = image_path
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to extract text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of text regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest bounding box (assumed to be text)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Crop the text region
    text_region = image[y:y+h, x:x+w]

    # Deskew the text using Hough Line Transform
    edges = cv2.Canny(text_region, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

        if angles:
            median_angle = np.median(angles)
            (h, w) = text_region.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1)
            text_region = cv2.warpAffine(text_region, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Resize to uniform output dimensions
    fixed_text = cv2.resize(text_region, output_size, interpolation=cv2.INTER_CUBIC)

    return fixed_text


# In[57]:



def fix_text_dimensions1(image, output_size=(500, 200)):
    # Convert to grayscale and enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Dilate to strengthen faint text
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None  # No text detected

    # Get the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    text_region = image[y:y+h, x:x+w]

    # Edge detection and deskew
    edges = cv2.Canny(text_region, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=5)
    
    if lines is not None:
        angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for line in lines for x1, y1, x2, y2 in line]
        if angles:
            median_angle = np.median(angles)
            (h, w) = text_region.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1)
            text_region = cv2.warpAffine(text_region, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Resize and return
    fixed_text = cv2.resize(text_region, output_size, interpolation=cv2.INTER_CUBIC)
    return fixed_text


# In[60]:



def correct_perspective(image_path, output_size=(400, 200)):
    """
    Corrects the perspective of a label-like object in an image.
    
    Parameters:
    - image_path: str, path to the image file.
    - output_size: tuple, desired width and height of the corrected image.
    
    Returns:
    - The warped image with corrected perspective.
    """
    # Load the image
    image = image_path
   # image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to enhance contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to avoid picking small inner regions
    contours = [c for c in contours if cv2.contourArea(c) > 5000]  # Adjust threshold as needed
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Approximate the largest contour to a quadrilateral
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            corners = approx
            break
    else:
        raise ValueError("No quadrilateral detected!")

    # Extract points
    pts1 = np.float32([corners[i][0] for i in range(4)])
    
    # Sort points in order: [top-left, top-right, bottom-left, bottom-right]
    sum_pts = pts1.sum(axis=1)
    diff_pts = np.diff(pts1, axis=1)

    top_left = pts1[np.argmin(sum_pts)]
    bottom_right = pts1[np.argmax(sum_pts)]
    top_right = pts1[np.argmin(diff_pts)]
    bottom_left = pts1[np.argmax(diff_pts)]

    ordered_pts = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

    # Destination points for a symmetrical rectangle
    width, height = output_size
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Compute the transformation matrix
    matrix = cv2.getPerspectiveTransform(ordered_pts, pts2)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped






def detect_qr_barcode(image):
    """Detects QR codes and barcodes in an image, drawing bounding boxes without requiring 4 points."""
    decoded_objects = decode(image, symbols=[ZBarSymbol.QRCODE, ZBarSymbol.EAN13, ZBarSymbol.EAN8, ZBarSymbol.CODE128, ZBarSymbol.CODE39])
    detected_codes = []

    for obj in decoded_objects:
        try:
            # Decode barcode data
            data = obj.data.decode('utf-8')
            detected_codes.append(data)
            
            # Get bounding box
            points = obj.polygon
            if points:  # Ensure there are points detected
                pts = np.array(points, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Put decoded text on the image
                cv2.putText(image, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Error decoding barcode: {e}")

    return detected_codes, image





from fuzzywuzzy import process  # For fuzzy text matching

def fuzzy_extract(text, keywords, threshold=80):
    """
    Fuzzy match a list of keywords in the given text.

    Args:
        text (str): The OCR extracted text.
        keywords (list): Possible variations of the keyword.
        threshold (int): Matching accuracy (0-100), higher is stricter.

    Returns:
        str or None: The best match found, or None if no match.
    """
    best_match, score = process.extractOne(text, keywords)
    return best_match if score >= threshold else None

def ocr_paddleocr(image):
    """
    Perform OCR on the given image using PaddleOCR and structure the extracted information into a JSON response.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        dict: JSON response containing structured fabric label information.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    result = ocr.ocr(gray)

    # Extract text from OCR result
    extracted_text = []
    if result:
        for line in result:
            for word_info in line:
                if len(word_info) > 1:
                    extracted_text.append(word_info[1][0])

    # Join extracted text into a single string
    extracted_text = " ".join(extracted_text).strip()

    # Debugging: Print extracted text
    print("Extracted Text:", extracted_text)


from fuzzywuzzy import process

def fuzzy_match(extracted_text, pattern_list, threshold=80):
    """
    Try regex first; if it fails, use fuzzy matching to find the best match.
    
    Args:
        extracted_text (str): The full OCR-extracted text.
        pattern_list (list): Possible keyword variations to search for.
        threshold (int): Minimum match confidence (0-100).

    Returns:
        str or None: Best-matched text or None if not found.
    """
    # Try regex patterns first
    for pattern in pattern_list:
        match = re.search(pattern, extracted_text, re.IGNORECASE)
        if match:
            return match.group(1)  # Return the captured group

    # If regex fails, apply fuzzy matching
    best_match, score = process.extractOne(extracted_text, pattern_list)
    return best_match if score >= threshold else None

def ocr_paddleocr(image):
    """
    Perform OCR on the given image using PaddleOCR and extract structured fabric label information.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        str: JSON response containing structured fabric label information.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    result = ocr.ocr(gray)

    # Extract text from OCR result
    extracted_text = []
    if result:
        for line in result:
            for word_info in line:
                if len(word_info) > 1:
                    extracted_text.append(word_info[1][0])

    # Convert list to single text string
    extracted_text = " ".join(extracted_text).strip()

    # Debugging: Print extracted text
    print("Extracted Text:", extracted_text)

    weave_patterns = [
    "DO8BY LENO", "TWILL", "SATIN", "PLAIN WEAVE", "HERRINGBONE", "BASKET WEAVE"
    ]

    # **Regex + Fuzzy Matching for Each Field**
    barcode_number = fuzzy_match(extracted_text, [r"^(\d+/\d+)"])  # First number
    igp_number = fuzzy_match(extracted_text, [r"IGP[#:\s]*([\d-]+)", r"IGP\s*(\d+-\d+-\d+-\d+)",r"[1I]G[PR][#:\s]*([\d-]+)"])
    roll_number = fuzzy_match(extracted_text, [r"Roll[#:\s]*(\d+)",r"R[o0]l[#:\s]*(\d+)"])
    meters = fuzzy_match(extracted_text, [r"METER[S]?:?\s*(\d+)"])
    cotton = fuzzy_match(extracted_text, [r"(\d+)%\s*Cotton"])
    spandex = fuzzy_match(extracted_text, [r"(\d+)%\s*SPANDEX"])
    dimensions = fuzzy_match(extracted_text, [r"(\d+x\d+\s*\d+”?)"])
    #weave = fuzzy_match(extracted_text, [r"(DO8BY\s*LENO)"])
    weave_match = re.search(r'"\s*([\w\s-]+)$', extracted_text)
    weave = weave_match.group(1).strip() if weave_match else None
    # Fabric details extraction
    fabric_details_match = re.search(r"\((.*?)\)", extracted_text)
    fabric_details = fabric_details_match.group(1).split() if fabric_details_match else []

    # Construct JSON response
    response = {
        "Barcode_Number": barcode_number,
        "IGP_Number": igp_number,
        "Roll_Number": int(roll_number) if roll_number and roll_number.isdigit() else None,
        "Meters": int(meters) if meters and meters.isdigit() else None,
        "Composition": {
            "Cotton": f"{cotton}%" if cotton else None,
            "Spandex": f"{spandex}%" if spandex else None
        },
       # "Fabric_Details": {
        #    "CD": fabric_details[0] if len(fabric_details) > 0 else None,
        #    "CM": fabric_details[1] if len(fabric_details) > 1 else None,
        #    "40D": "40D" in extracted_text
        #},
        "Dimensions": dimensions,
        "Weave": weave
        #"QR_Code": "Detected"  # Placeholder for QR detection
    }

    # Remove keys with None values
    response = {k: v for k, v in response.items() if v is not None}
    print(extracted_text)

    return response  # Return as JSON object, not a string


def ocr_paddleocr1(image):
    
    
    #barcode_image, bbox = get_barcode_area(image)
    # Check if image is already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale, no need to convert

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    result = ocr.ocr(gray)

    # Extract text from OCR result
    extracted_text = " ".join([word_info[1][0] for line in result for word_info in line if len(word_info) > 1])
    
    return extracted_text.strip()

def Image_Preview(Imgtype, image_path):
    image = image_path
    if image is None:
        raise FileNotFoundError("Image not found")
    
    detected_codes, image_with_boxes = detect_qr_barcode(image)
   # print(image)
    extracted_text = ocr_paddleocr(image)
    
    
    
    if detected_codes:
        print(f"QR/Barcode Data: {detected_codes}")
    if extracted_text:
        print(f"OCR Extracted Text: {extracted_text}")


###################### Importing Image ########################
#directory_path = 'images 2'  # Replace with your actual directory path
#for image_name in os.listdir(directory_path):
 #   image_path = os.path.join(directory_path, image_name)
  #  if os.path.isfile(image_path):
   #     image = cv2.imread(image_path)
    #    Image_Preview("Processed Image", image)
     #   barcode_image, bbox = get_barcode_area(image)
      #  if barcode_image is not None:
       #     Image_Preview("Barcode Area", barcode_image)
           
           






