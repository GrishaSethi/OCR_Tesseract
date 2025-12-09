# 🏠 Floor Plan OCR Analyzer

A clean, modern Streamlit app and CLI to extract text from floor plan images using Tesseract OCR. Automatically identifies room names and dimension strings, draws color-coded boxes, and exports JSON/CSV with optional real-world unit conversion.

## 🌐 Live app
- Deployed: https://ocr-tesseract.streamlit.app/

## ✨ Features
- **OCR with context**: Uses Tesseract to extract text, bounding boxes, and confidence.
- **Smart cleaning**: Text normalization, spell correction, and fuzzy matching against common floor-plan terms.
- **Labeling**: Classifies text into `Room`, `Dimension`, or `Other`.
- **Scale handling**: Auto-detect basic scale formats or enter manual scale (meters or feet) to convert dimensions and compute areas.
- **Visual output**: Saves an annotated image with color-coded boxes (Room=green, Dimension=blue, Other=red).
- **Exports**: Download structured results as JSON and CSV.
- **Streamlit UI + CLI**: Use the web app or run from the terminal.


## 🖼️ Demo
- Sample annotated output is saved to: `output/boxed_image.png`


## 📦 Tech stack
- Python, Streamlit
- OpenCV, Pillow, NumPy, pandas
- Tesseract OCR via pytesseract
- TextBlob and fuzzywuzzy for text cleanup


## 📁 Project structure
```
ocr_tessera/
├─ app.py                 # Streamlit app
├─ ocr_processor.py       # OCR pipeline and utilities
├─ requirements.txt       # Python dependencies
├─ output/                # Generated results (annotated image, CSV, JSON)
├─ new.jpg                # Example image (optional)
└─ README.md              # This file
```


## ✅ Prerequisites
- Python 3.9+ recommended
- Tesseract OCR installed on your system
  - Windows (default path used in code): `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - macOS (Homebrew): `brew install tesseract`
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`

If Tesseract is installed in a different location, update the line in `ocr_processor.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\\Path\\To\\Tesseract-OCR\\tesseract.exe'
```


## 🚀 Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
```


## 🧭 Usage
### Streamlit app
Run the UI:
```bash
streamlit run app.py
```

Or open the deployed app: https://ocr-tesseract.streamlit.app/

Then open the local URL shown in the terminal. In the app:
- Upload a floor plan image (PNG/JPG/JPEG).
- Optionally enable preprocessing, adjust confidence threshold, and hide `Other`.
- Choose scale handling:
  - Auto-detect from image, or
  - Enter manual pixels-per-unit and unit (`m` or `ft`).
- Click “Analyze Floor Plan”.
- Download JSON/CSV and the annotated image.

### CLI
Process an image from the terminal:
```bash
python ocr_processor.py path/to/image.png \
  --output-dir output \
  --conf-threshold 0.4 \
  --show-other      # include 'Other' in results
  # add --no-preprocess to disable preprocessing
```
Outputs go to the specified `--output-dir` (default `output`).


## ⚙️ Configuration reference
- `use_preprocessing` (UI checkbox / `--no-preprocess` in CLI): Improves OCR by denoising + adaptive thresholding.
- `conf_threshold` (UI slider / `--conf-threshold`): Minimum OCR confidence to keep a detection.
- `hide_other` (UI checkbox / `--show-other`): Filter out items labeled `Other`.
- `manual_scale` and `manual_scale_unit` (UI only): Override auto-scale with pixels-per-unit in meters or feet.

Scale detection tries to recognize formats like:
- `Scale 1:100`
- `1" = 10'`
- `1 cm = 1 m`


## 📤 Outputs
- `output/boxed_image.png` — Annotated image with labeled boxes.
- `output/results.json` — Structured results including `scale_factor`, `scale_unit`, and `items`.
- `output/results.csv` — Tabular form of items.

Each item typically includes:
```json
{
  "original_text": "BEDROOM",
  "cleaned_text": "bedroom",
  "confidence": 0.92,
  "bbox": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
  "type": "Room",
  "converted": true,
  "dimensions_meters": [3.5, 4.2],
  "area_sqm": 14.7,
  "original_units": "m"
}
```


## 🧩 Tips & troubleshooting
- If OCR returns low-quality text, try enabling preprocessing in the UI, or supply a higher-resolution image.
- If dimensions don’t convert, provide the manual scale in the sidebar.
- On Windows, ensure Tesseract is installed at the path configured in `ocr_processor.py`.
- If Streamlit can’t display the annotated image, confirm `output/boxed_image.png` exists after running analysis.


## 🔒 Notes on privacy & usage
- Images you upload are processed locally. Temporary files are cleaned up where possible.


## 🙌 Acknowledgements
- Tesseract OCR
- Streamlit
- OpenCV, Pillow, NumPy, pandas
- TextBlob, fuzzywuzzy
