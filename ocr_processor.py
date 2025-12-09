import os
import re
import json
import argparse
import numpy as np
import cv2
import pytesseract
import pandas as pd
from textblob import TextBlob
from fuzzywuzzy import process
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import math

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define a list of expected floor plan terms
EXPECTED_TERMS = [
    "bedroom", "bathroom", "kitchen", "toilet", "closet", "garage", "porch", "foyer",
    "dining", "living", "pantry", "laundry", "utility", "master", "storeroom", "hall",
    "great room", "study", "office", "stairs", "veranda", "deck", "balcony", "entry",
    "guest", "bath", "clos", "storage", "two car garage", "mstr", "rear porch", "front porch"
]

def clean_text(text):
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return str(TextBlob(text).correct()).strip()

def fuzzy_correct(text, threshold=80):
    """Correct text using fuzzy matching with expected terms."""
    match, score = process.extractOne(text, EXPECTED_TERMS)
    return match if score >= threshold else text

def classify_text(text):
    """Classify text into categories like Room, Dimension, or Other."""
    text = text.lower().strip()
    room_keywords = [
        "bedroom", "bathroom", "kitchen", "toilet", "porch", "garage",
        "hall", "closet", "foyer", "pantry", "utility", "storeroom",
        "study", "living", "dining", "great room", "entry"
    ]
    if any(word in text for word in room_keywords):
        return "Room"
    elif re.search(r"(\d+\s*[xX′']\s*\d+)|(\d+\s*-\s*\d+)", text):
        return "Dimension"
    else:
        return "Other"

def preprocess_image(image_path, save_path="temp_processed.png"):
    """Preprocess image to improve OCR accuracy."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    thresh = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(save_path, thresh)
    return save_path

def perform_ocr(image_path):
    """Perform OCR on the given image and return structured results."""
    # Read the image with OpenCV
    img_cv = cv2.imread(image_path)
    
    # Convert to RGB (Tesseract expects RGB)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Get OCR data with bounding boxes and confidence
    data = pytesseract.image_to_data(
        img_rgb, 
        output_type=pytesseract.Output.DICT,
        config='--psm 6'  # Assume a single uniform block of text
    )
    
    structured_output = []
    n_boxes = len(data['level'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if not text:  # Skip empty text
            continue
            
        confidence = int(data['conf'][i]) / 100.0  # Convert to 0-1 range
        if confidence < 0:  # Skip low confidence detections
            continue
            
        # Get bounding box coordinates
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        bbox = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]
        
        cleaned = clean_text(text)
        corrected = fuzzy_correct(cleaned)
        label_type = classify_text(corrected)
        
        structured_output.append({
            "original_text": text,
            "cleaned_text": corrected,
            "confidence": confidence,
            "bbox": bbox,
            "type": label_type
        })
    
    return structured_output

def draw_boxes_on_image(image_path, ocr_results, output_path="output/boxed_image.png"):
    """Draw bounding boxes and labels on the image."""
    # Read the image with OpenCV
    img = cv2.imread(image_path)
    
    # Define colors for different text types
    color_map = {
        "Room": (0, 255, 0),      # Green
        "Dimension": (255, 0, 0),  # Blue
        "Other": (0, 0, 255)       # Red
    }
    
    for item in ocr_results:
        bbox = item["bbox"]
        text = item["cleaned_text"]
        label = item["type"]
        
        # Get top-left and bottom-right points
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        
        # Get color based on label
        color = color_map.get(label, (0, 0, 0))
        
        # Draw rectangle and text
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add text background for better visibility
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(
            img, 
            text, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255) if label == "Other" else (255, 255, 255),  # White text
            1, 
            cv2.LINE_AA
        )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path

def detect_scale(text: str) -> Optional[Tuple[float, str]]:
    """
    Detect scale information from text.
    Returns a tuple of (scale_factor, unit) if detected, else None.
    """
    # Common scale patterns
    patterns = [
        # 1:100 format
        (r'(?i)scale\s*[\:\-]?\s*(\d+)\s*[\:\=]\s*(\d+)', 1, 0, 1, ""),
        # 1" = 10' format
        (r'(?i)(\d+)\s*["\']?\s*\=\s*(\d+)\s*(["\']|ft|foot|feet)', 1, 0, 1, "ft"),
        # 1cm = 1m format
        (r'(?i)(\d+)\s*cm\s*\=\s*(\d+)\s*(m|meter|metre)', 1, 0, 1, "m"),
        # 1:100m format
        (r'(?i)scale\s*[\:\-]?\s*(\d+)\s*\:\s*(\d+)\s*(m|meter|metre|cm|mm|ft|foot|feet|\"|\')', 1, 0, 1, ""),
    ]
    
    for pattern, num1_idx, num2_idx, unit_idx, unit_override in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                num1 = float(match.group(num1_idx + 1))
                num2 = float(match.group(num2_idx + 1))
                unit = unit_override if unit_override else match.group(unit_idx + 1).lower() if unit_idx < len(match.groups()) else ""
                
                # Convert to scale factor (pixels per meter)
                if 'cm' in unit:
                    scale_factor = num2 / (num1 * 0.01)  # cm to m
                elif 'mm' in unit:
                    scale_factor = num2 / (num1 * 0.001)  # mm to m
                elif any(x in unit for x in ['ft', 'foot', 'feet', "'", '"']):
                    scale_factor = num2 / (num1 * 0.3048)  # ft to m
                else:
                    scale_factor = num2 / num1
                
                return scale_factor, "m"
            except (IndexError, ValueError, ZeroDivisionError):
                continue
    
    return None

def convert_dimension(text: str, scale_factor: float) -> Dict[str, Union[str, float]]:
    """
    Convert a dimension text to real-world units.
    Returns a dictionary with original text, cleaned text, and converted values.
    """
    # Common dimension patterns
    patterns = [
        # 12' x 15' format
        (r'(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)\s*["\']', 1, 0.3048),  # feet to meters
        # 12 x 15 format (assume meters if no unit)
        (r'(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)', 1, 1.0),  # assume meters
        # 12m x 15m format
        (r'(\d+\.?\d*)\s*m\s*[xX×]\s*(\d+\.?\d*)\s*m', 1, 1.0),  # meters
        # 12cm x 15cm format
        (r'(\d+\.?\d*)\s*cm\s*[xX×]\s*(\d+\.?\d*)\s*cm', 0.01, 0.01),  # cm to m
    ]
    
    cleaned_text = text.strip()
    
    for pattern, unit1_scale, unit2_scale in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                dim1 = float(match.group(1)) * unit1_scale
                dim2 = float(match.group(2)) * unit2_scale
                
                # Calculate area in square meters
                area = dim1 * dim2
                
                return {
                    "original_text": text,
                    "cleaned_text": cleaned_text,
                    "converted": True,
                    "dimensions_meters": [dim1, dim2],
                    "area_sqm": area,
                    "original_units": "m" if unit1_scale == 1.0 else "ft"
                }
            except (IndexError, ValueError):
                continue
    
    # If no pattern matched, return the original text
    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "converted": False,
        "dimensions_meters": None,
        "area_sqm": None,
        "original_units": None
    }

def process_image(input_path: str, output_dir: str = "output", use_preprocessing: bool = True, 
                 conf_threshold: float = 0.4, hide_other: bool = True,
                 manual_scale: Optional[float] = None, manual_scale_unit: str = "m") -> List[Dict]:
    """
    Process an image file with OCR and save the results.
    
    Args:
        input_path (str): Path to the input image file
        output_dir (str): Directory to save output files
        use_preprocessing (bool): Whether to use image preprocessing
        conf_threshold (float): Minimum confidence threshold (0-1)
        hide_other (bool): Whether to hide 'Other' category in results
        manual_scale (float, optional): Manual scale factor (pixels per meter)
        manual_scale_unit (str): Unit for manual scale ('m' or 'ft')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing image: {input_path}")
    
    # Preprocess image if needed
    if use_preprocessing:
        print("Applying image preprocessing...")
        processed_path = preprocess_image(input_path)
    else:
        processed_path = input_path
    
    # Perform OCR
    print("Running OCR...")
    results = perform_ocr(processed_path)
    
    # Detect scale from OCR results if not provided manually
    scale_factor = None
    scale_unit = manual_scale_unit
    
    if manual_scale is not None:
        scale_factor = manual_scale
        if scale_unit == 'ft':
            scale_factor *= 0.3048  # Convert feet to meters
    else:
        # Try to detect scale from OCR results
        for item in results:
            if item["type"] == "Dimension":
                scale_info = detect_scale(item["original_text"])
                if scale_info:
                    scale_factor, scale_unit = scale_info
                    break
    
    # Process dimensions if scale is available
    if scale_factor is not None:
        for item in results:
            if item["type"] == "Dimension":
                dim_info = convert_dimension(item["original_text"], scale_factor)
                item.update(dim_info)
    # If manual scale is provided, ensure all dimensions are processed
    elif manual_scale is not None:
        scale_factor = manual_scale
        if manual_scale_unit == 'ft':
            scale_factor *= 0.3048  # Convert feet to meters
        for item in results:
            if item["type"] == "Dimension":
                dim_info = convert_dimension(item["original_text"], scale_factor)
                item.update(dim_info)
    
    # Filter results
    filtered = [
        r for r in results
        if r["confidence"] >= conf_threshold and (not hide_other or r["type"] != "Other")
    ]
    
    print(f"Found {len(filtered)} text elements matching the criteria")
    
    # Draw bounding boxes on image
    boxed_image_path = os.path.join(output_dir, "boxed_image.png")
    draw_boxes_on_image(processed_path, filtered, output_path=boxed_image_path)
    print(f"Saved annotated image to: {boxed_image_path}")
    
    # Save results to JSON
    results_metadata = {
        "scale_factor": scale_factor,
        "scale_unit": scale_unit,
        "items": filtered
    }
    
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_metadata, f, indent=2)
    print(f"Saved results to: {json_path}")
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "results.csv")
    df = pd.DataFrame(filtered)
    df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")
    
    return results_metadata

def main():
    parser = argparse.ArgumentParser(description='Process floor plan images with OCR')
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--no-preprocess', action='store_false', dest='preprocess', 
                       help='Disable image preprocessing')
    parser.add_argument('--conf-threshold', type=float, default=0.4,
                       help='Minimum confidence threshold (0-1)')
    parser.add_argument('--show-other', action='store_false', dest='hide_other',
                       help='Show "Other" category in results')
    
    args = parser.parse_args()
    
    process_image(
        input_path=args.image_path,
        output_dir=args.output_dir,
        use_preprocessing=args.preprocess,
        conf_threshold=args.conf_threshold,
        hide_other=args.hide_other
    )

if __name__ == "__main__":
    main()
