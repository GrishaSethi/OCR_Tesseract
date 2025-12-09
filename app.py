import streamlit as st
import os
import tempfile
from PIL import Image
import json
import pandas as pd
import numpy as np
from ocr_processor import process_image

# Set page config
st.set_page_config(
    page_title="Floor Plan OCR",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🏠 Floor Plan OCR Analyzer")
st.markdown("Upload a floor plan image to extract and analyze text elements like room names and dimensions.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    use_preprocessing = st.checkbox("Enable Image Preprocessing", value=True,
                                  help="Improves OCR accuracy by enhancing image quality")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05,
                              help="Minimum confidence score to include results")
    hide_other = st.checkbox("Hide 'Other' Category", value=True,
                           help="Hide results classified as 'Other'")
    
    st.markdown("---")
    st.subheader("Scale Settings")
    scale_option = st.radio("Scale Detection", 
                          ["Auto-detect from image", "Enter manually"],
                          help="Choose how to handle scale conversion")
    
    manual_scale = None
    manual_scale_unit = "m"
    
    if scale_option == "Enter manually":
        col1, col2 = st.columns(2)
        with col1:
            manual_scale = st.number_input(
                "Pixels per unit",
                min_value=0.1,
                max_value=10000.0,
                value=100.0,
                step=1.0,
                help="Number of pixels that represent one unit of measurement"
            )
        with col2:
            manual_scale_unit = st.selectbox(
                "Unit",
                ["m", "ft"],
                help="Measurement unit"
            )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses Tesseract OCR to extract text from floor plan images.")
    st.markdown("It can identify rooms, dimensions, and other text elements.")

# File uploader
uploaded_file = st.file_uploader("Upload a floor plan image", 
                               type=["png", "jpg", "jpeg"],
                               help="Supported formats: PNG, JPG, JPEG")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        # Process the image when the button is clicked
        if st.button("🔍 Analyze Floor Plan", use_container_width=True):
            with st.spinner("Processing image..."):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    image.save(tmp_file.name, format='PNG')
                    tmp_file_path = tmp_file.name
                
                try:
                    # Process the image with scale settings
                    results = process_image(
                        input_path=tmp_file_path,
                        use_preprocessing=use_preprocessing,
                        conf_threshold=conf_threshold,
                        hide_other=hide_other,
                        manual_scale=manual_scale if scale_option == "Enter manually" else None,
                        manual_scale_unit=manual_scale_unit
                    )
                finally:
                    # Clean up the temporary file
                    try:
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                    except Exception as e:
                        st.warning(f"Warning: Could not delete temporary file: {e}")
                
                # Display results
                st.success(f"✅ Found {len(results)} text elements")
                
                # Show results in an expandable section
                with st.expander("📋 View Extracted Text", expanded=True):
                    # Handle both dictionary and list return types
                    if isinstance(results, dict):
                        items = results.get('items', [])
                        scale_factor = results.get('scale_factor')
                        scale_unit = results.get('scale_unit', 'm')
                    else:  # Handle old list format
                        items = results
                        scale_factor = None
                        scale_unit = 'm'
                    
                    # If we have a manual scale, ensure we use it
                    if scale_option == "Enter manually" and manual_scale is not None:
                        scale_factor = manual_scale
                        if manual_scale_unit == 'ft':
                            scale_factor *= 0.3048  # Convert feet to meters
                    
                    # Create DataFrame from items
                    df = pd.DataFrame(items)
                    
                    # Display scale information if available
                    if scale_factor is not None:
                        st.info(f"Using scale: 1 pixel = {1/scale_factor:.4f} m (1 m = {scale_factor:.2f} pixels)")
                    else:
                        st.warning("No scale information detected. Dimensions will not be converted to real-world units.")
                    
                    # Add dimension columns if available
                    if 'dimensions_meters' in df.columns:
                        formatted_dims = []
                        
                        for _, row in df.iterrows():
                            try:
                                dims = row.get('dimensions_meters')
                                
                                # Skip if dims is not a list/tuple or doesn't have enough elements
                                if not isinstance(dims, (list, tuple, np.ndarray)) or len(dims) < 2:
                                    formatted_dims.append(row.get('cleaned_text', row.get('original_text', '')))
                                    continue
                                    
                                # Convert numpy arrays to list for easier handling
                                if hasattr(dims, 'tolist'):
                                    dims = dims.tolist()
                                
                                # Handle already converted dimensions
                                if row.get('converted', False):
                                    formatted_dims.append(f"{float(dims[0]):.2f} × {float(dims[1]):.2f} m")
                                    continue
                                
                                # Try to convert using scale factor if available
                                if scale_factor and scale_factor > 0:
                                    try:
                                        dim1 = float(dims[0]) / scale_factor
                                        dim2 = float(dims[1]) / scale_factor
                                        formatted_dims.append(f"{dim1:.2f} × {dim2:.2f} m (converted)")
                                        continue
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Fallback to raw values
                                formatted_dims.append(f"{dims[0]} × {dims[1]} (pixels)")
                                
                            except Exception:
                                formatted_dims.append(row.get('cleaned_text', row.get('original_text', '')))
                        
                        df['dimensions'] = formatted_dims
                    
                    # Add color coding for different types
                    def color_type(val):
                        color = 'lightgreen' if val == 'Room' else 'lightblue' if val == 'Dimension' else 'lightgray'
                        return f'background-color: {color}'
                    
                    # Display styled DataFrame
                    # Prepare columns to display
                    display_columns = ['type', 'cleaned_text', 'dimensions' if 'dimensions' in df.columns else 'original_text', 'confidence']
                    
                    # Configure column display names
                    column_config = {
                        "type": "Type",
                        "cleaned_text": "Cleaned Text",
                        "dimensions": "Dimensions (m)",
                        "original_text": "Original Text",
                        "confidence": st.column_config.NumberColumn(
                            "Confidence",
                            format="%.2f"
                        )
                    }
                    
                    # Remove any columns that don't exist in the DataFrame
                    display_columns = [col for col in display_columns if col in df.columns]
                    
                    # Display the DataFrame
                    st.dataframe(
                        df[display_columns].style.applymap(color_type, subset=['type']),
                        use_container_width=True,
                        column_config=column_config
                    )
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download JSON",
                            data=json.dumps({"items": items, "scale_factor": scale_factor, "scale_unit": scale_unit}, indent=2),
                            file_name="floor_plan_analysis.json",
                            mime="application/json"
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download CSV",
                            data=pd.DataFrame(items).to_csv(index=False).encode('utf-8'),
                            file_name="floor_plan_analysis.csv",
                            mime="text/csv"
                        )
                
                # Show annotated image
                st.subheader("Annotated Floor Plan")
                st.image("output/boxed_image.png", use_column_width=True)
                
                # Add download button for annotated image
                with open("output/boxed_image.png", "rb") as file:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=file,
                        file_name="annotated_floor_plan.png",
                        mime="image/png"
                    )

# Add some spacing at the bottom
st.markdown("")
st.markdown("---")
st.markdown("*Upload a floor plan image to get started*")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload** a floor plan image (PNG, JPG, or JPEG)
    2. Adjust the settings in the sidebar if needed:
       - Enable/disable image preprocessing
       - Set confidence threshold
       - Show/hide 'Other' category
    3. Click **'Analyze Floor Plan'** to process the image
    4. View and download the results
    """)
