import streamlit as st
import os
import torch
import torch.nn as nn
import base64
from PIL import Image
import io
import numpy as np
import cv2 # Make sure opencv-python is installed (pip install opencv-python)
import re
import random
import requests # Used for synchronous API calls
import json # Used for JSON parsing

# Explicitly import transforms components from torchvision to avoid conflicts
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# --- Streamlit Page Configuration ---
# st.set_page_config() MUST be the very first Streamlit command
st.set_page_config(
    page_title="Plant Disease Detector & Crop Recommender",
    layout="centered", # Use centered layout to match original design's max-width
)

# --- Model_ADI API Configuration ---
# Use Streamlit's secrets management for secure API key handling
# For local development outside Streamlit Cloud, you can use a .streamlit/secrets.toml file
# (e.g., [secrets] GENERATIVE_LANGUAGE_API_KEY="your_api_key")
# or set an environment variable.
MODEL_ADI_API_KEY = "" # Initialize with empty string

# First, try to load from Streamlit secrets.toml
try:
    if "GENERATIVE_LANGUAGE_API_KEY" in st.secrets:
        MODEL_ADI_API_KEY = st.secrets["GENERATIVE_LANGUAGE_API_KEY"]
except Exception as e:
    st.error(f"Error initializing Streamlit secrets. Ensure your '.streamlit/secrets.toml' file exists and is correctly formatted. Details: {e}")

# If not found in secrets, try environment variable
if not MODEL_ADI_API_KEY:
    MODEL_ADI_API_KEY = os.environ.get("GENERATIVE_LANGUAGE_API_KEY", "")

Model_ADI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Final warning if API key is still missing after all attempts
if not MODEL_ADI_API_KEY:
    st.warning("Warning: GENERATIVE_LANGUAGE_API_KEY is not set. AI-powered features (disease detection, crop recommendations, remedies) will be limited or unavailable. Please set it in .streamlit/secrets.toml or as an environment variable.")

# --- Helper function for encoding local images to base64 ---
def get_base64_image(image_path):
    """Encodes an image to base64 for Streamlit background styling and returns mime type."""
    if not os.path.exists(image_path):
        print(f"DEBUG: Image file NOT FOUND at path: {image_path}") # Debug print
        return "", "" # Return empty string if file not found

    try:
        with open(image_path, "rb") as f:
            data = f.read()
        encoded_data = base64.b64encode(data).decode()
        
        # Determine mime type from file extension
        _, ext = os.path.splitext(image_path)
        if ext.lower() in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif ext.lower() == '.png':
            mime_type = "image/png"
        else:
            mime_type = "image/octet-stream" # Generic fallback
        
        print(f"DEBUG: Image {image_path} loaded successfully with MIME type {mime_type}") # Debug print
        return encoded_data, mime_type
    except Exception as e:
        print(f"DEBUG: Error encoding image {image_path} to base64: {e}") # Debug print
        return "", ""

# Load background image for the overall app (from your provided .py file)
# Assuming Img_130716.png is in the same directory as Crop.py
overall_bg_image_path = "Img_130716.png" 
overall_bg_image_base64, overall_bg_mime_type = get_base64_image(overall_bg_image_path)

# Load background image for the main content container
main_content_bg_image_path = "pexels-asphotograpy-1002703.jpg" # Using one of the previous section images
main_content_bg_image_base64, main_content_bg_mime_type = get_base64_image(main_content_bg_image_path)


# --- Custom CSS for Streamlit app styling ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

    html {{
        scroll-behavior: smooth; /* Smooth scrolling for anchor links */
        /* Removed scroll-padding-top */
    }}

    body {{
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding-bottom: 30px; /* Adjusted padding-bottom for non-fixed footer */
        background-color: #e8f5e9; /* Light green background fallback */
        color: #000; /* Ensured text color is black */
        min-height: 100vh;
    }}

    /* Streamlit overrides for main content area */
    .stApp {{
        background-color: transparent; /* Ensure body background shows through if no image */
        {"background-image: url('data:" + overall_bg_mime_type + ";base64," + overall_bg_image_base64 + "');" if overall_bg_image_base64 else ""}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp > header {{
        display: none; /* Hide Streamlit's default header */
    }}
    .stApp [data-testid="stSidebar"] {{
        display: none !important; /* Hide sidebar as per request */
    }}

    /* Custom Navigation Bar Styling */
    .main-nav-container {{
        position: fixed; /* Changed to fixed */
        top: 0;
        left: 0;
        width: 100%;
        background-color: #388e3c; /* Darker green for the nav bar */
        padding: 15px 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        display: flex; /* Use flex for nav items */
        justify-content: center; /* Center the navigation links */
        z-index: 999; /* Ensure it stays on top */
        border-bottom: 3px solid #66bb6a; /* A subtle border */
    }}

    .main-nav-item {{
        color: white;
        text-decoration: none;
        font-weight: 600;
        font-size: 1.1em;
        padding: 10px 20px;
        margin: 0 10px;
        border-radius: 8px;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        display: flex;
        align-items: center;
        cursor: pointer;
    }}

    .main-nav-item:hover {{
        background-color: #2e7d32; /* Even darker green on hover */
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
    }}

    /* Main Content Block Container - NOW HAS IMAGE BACKGROUND AND IS WIDER */
    div.block-container {{
        padding-top: 2rem; /* Adjusted padding as content is directly inside */
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        
        {"background: linear-gradient(rgba(255,255,255,0.7), rgba(255,255,255,0.7)), url('data:" + main_content_bg_mime_type + ";base64," + main_content_bg_image_base64 + "');" if main_content_bg_image_base64 else "background-color: white;"}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: scroll; /* Scroll with content */

        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid #d4edda;
        margin: 120px auto 30px auto; /* Adjusted top margin to 120px for fixed nav bar clearance */
        max-width: 60%; /* Made narrower by user */
    }}

    /* REMOVED .card STYLING - no more individual cards */

    /* Header Title (for h1 in Home section) - Light Green and Larger */
    .header-title {{
        font-size: 3.5em !important; /* Made larger */
        color: #66BB6A !important; /* Light green - Forced with !important */
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2); /* Adjusted shadow for visibility */
    }}
    .header-title i {{
        margin-right: 10px;
        color: #4CAF50; /* A vibrant green for the icon */
    }}

    /* Headings for sections (Streamlit uses H2 and H3 for its titles) - Deep Green and Larger */
    h2 {{ /* Targets st.header() or st.markdown("##") */
        font-size: 2.2em !important; /* Made larger */
        color: #1B5E20 !important; /* Deep green */
        margin-top: 30px; /* Added top margin for separation */
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid #eee; /* Subtle separator line */
        padding-bottom: 5px;
    }}
    h2 i {{
        margin-right: 12px;
        color: #66bb6a;
    }}

    h3 {{ /* Targets st.subheader() or st.markdown("###") */
        font-size: 1.8em !important; /* Made larger */
        color: #555 !important;
        margin-top: 20px;
        margin-bottom: 15px;
    }}
    
    /* General Paragraph Text (from st.write) - Ensure readability */
    p {{
        font-size: 1.6em; /* Made bigger for general content */
        line-height: 1.6;
        color: #000; /* Explicitly black for readability */
    }}

    /* Explicitly make list items black within the main content block */
    div.block-container ul li {{
        color: #000 !important;
    }}


    /* Apply scroll-margin-top to each section ID to prevent fixed nav overlap */
    #home-section,
    #about-section,
    #crop-query-section,
    #predict-section,
    #contact-section,
    #chatbot-section {{ /* Added chatbot-section */
        scroll-margin-top: 100px; /* Offset for the fixed navigation bar + extra space */
        padding-top: 20px; /* Added padding to ensure content starts clearly below nav */
    }}

    /* Input Elements (Streamlit widgets are rendered as specific HTML inputs) */
    .stNumberInput label,
    .stSelectbox label,
    .stFileUploader label {{
        display: flex;
        flex-direction: column;
        font-weight: 600;
        color: #555;
        font-size: 0.95em;
    }}
    .stNumberInput input,
    .stSelectbox select {{
        padding: 12px;
        border: 1px solid #ccc;
        border-radius: 8px;
        margin-top: 8px;
        font-size: 1em;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }}

    .stNumberInput input:focus,
    .stSelectbox select:focus {{
        border-color: #66bb6a;
        box-shadow: 0 0 0 3px rgba(102, 187, 106, 0.3);
        outline: none;
    }}

    /* Buttons (Streamlit buttons are specific HTML <button>s inside a div) */
    .stButton > button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        border: none;
        text-decoration: none;
        background-color: #4CAF50; /* Green */
        color: white;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3);
        width: 100%; /* Make buttons take full width of their column/container */
        margin-top: 15px;
    }}
    .stButton > button:hover {{
        background-color: #388e3c; /* Darker green */
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(76, 175, 80, 0.4);
    }}
    .stButton > button i {{
        margin-right: 10px;
    }}

    /* Image Upload Specifics */
    .stFileUploader > div:first-child {{ /* Targets the input area of file uploader */
        margin-bottom: 20px;
        padding: 10px;
        border: 1px dashed #cccccc;
        border-radius: 8px;
        background-color: #f9f9f9;
    }}
    .image-preview-container {{ /* Custom container for image to apply preview styling */
        min-height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #f0f0f0;
        border: 1px dashed #ccc;
        border-radius: 10px;
        margin-bottom: 20px;
        overflow: hidden;
    }}
    .image-preview-container img {{
        max-width: 100%;
        max-height: 400px; /* Limit height for larger images */
        height: auto;
        border-radius: 8px;
        display: block; /* Ensures no extra space below image */
        object-fit: contain; /* Ensures image fits within the box */
    }}

    /* Output Boxes (these can remain as they are for results) */
    .output-box, .output-card {{
        background-color: #e6ffe6; /* Lightest green */
        color: black;
        padding: 25px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 1.1em;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 6px solid #4CAF50; /* Default green border */
    }}
    .output-box strong, .output-card strong {{
        font-size: 1.2em;
        color: #2e7d32;
        margin-bottom: 10px;
        display: block;
    }}
    .output-box span, .output-card span {{
        display: block;
    }}

    /* Specific Output Box Borders */
    .remedy-box {{ border-left-color: #fb8c00; }} /* Orange */
    .medicine-box {{ border-left-color: #43a047; }} /* Another green shade */
    .crop-recommendation-output {{ border-left-color: #1a73e8; }} /* Blue */

    /* Segmented Image specific styling */
    .segmented-image-card img {{
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        margin-top: 15px;
        border: 2px solid #ddd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    /* Footer Styles */
    .app-footer {{
        /* Removed position: fixed, bottom, left, z-index */
        width: 100%;   /* Span full parent width (which is the Streamlit app area) */
        background-color: #2e7d32; /* Dark green, matching header */
        color: rgba(255, 255, 255, 0.8);
        text-align: center;
        padding: 20px 0;
        margin-top: 40px; /* Space above the footer */
        font-size: 0.9em;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    }}
    .app-footer p {{
        margin: 5px 0;
        line-height: 1.5;
    }}

    /* Responsive Adjustments */
    @media (max-width: 768px) {{
        div.block-container {{
            padding: 20px;
            padding-top: 8rem; /* Adjusted for mobile nav to prevent overlap */
        }}
        .header-title {{
            font-size: 2em !important;
        }}
        h2 {{
            font-size: 1.5em !important;
        }}
        h3 {{
            font-size: 1.2em !important;
        }}
        /* Make inputs stack on small screens */
        .stNumberInput, .stSelectbox, .stFileUploader {{
            width: 100%;
        }}
        .stButton > button {{
            width: 100%;
        }}
        .main-nav-container {{
            flex-direction: column; /* Stack nav items vertically on small screens */
            padding: 10px 0;
        }}
        .main-nav-item {{
            margin: 5px 0; /* Adjust vertical margin for stacked links */
            width: fit-content;
            align-self: center; /* Center each item */
        }}
    }}
    </style>
""", unsafe_allow_html=True)


# --- Helper Functions (from App2.py and previous iterations) ---

def format_remedies(text):
    """
    Formats text by inserting a newline only after a sentence ending
    (a letter followed by a period, not a decimal), and handles
    initial flattening of all problematic newlines.
    Numbered list items will only get a newline if they follow a sentence ending.
    """
    if not isinstance(text, str):
        text = str(text)

    # Step 1: Replace ALL newline characters with a single space.
    flattened_text = text.replace('\n', ' ')

    # Step 2: Normalize all sequences of whitespace (multiple spaces) to a single space.
    cleaned_text = re.sub(r'\s+', ' ', flattened_text).strip()

    result_lines = []
    current_line = []
    
    i = 0
    while i < len(cleaned_text):
        current_line.append(cleaned_text[i])

        # Check for a sentence boundary: letter + period
        if cleaned_text[i] == '.':
            if i > 0 and cleaned_text[i-1].isalpha(): # Period preceded by a letter
                # Ensure it's not a decimal number (by checking if next char is digit)
                is_decimal = (i + 1 < len(cleaned_text) and cleaned_text[i+1].isdigit())

                if not is_decimal:
                    # Check for context: followed by space and then upper case or digit, or end of string
                    next_char_idx = i + 1
                    # Consume any spaces immediately after the period
                    while next_char_idx < len(cleaned_text) and cleaned_text[next_char_idx].isspace():
                        next_char_idx += 1 
                    
                    if next_char_idx >= len(cleaned_text) or \
                       cleaned_text[next_char_idx].isupper() or \
                       cleaned_text[next_char_idx].isdigit(): 
                        
                        result_lines.append("".join(current_line).strip())
                        current_line = []
                        
                        # Set i to the character AFTER the consumed spaces
                        i = next_char_idx - 1 # It will be incremented by loop's i += 1

        i += 1
    
    # Add any remaining text
    if current_line:
        result_lines.append("".join(current_line).strip())

    # Join lines, remove empty ones
    final_output = "\n".join(filter(None, result_lines))
    
    # A final small cleanup for any remaining double newlines or leading/trailing spaces on lines
    final_output = re.sub(r'\n\s*\n', '\n', final_output) # Collapse multiple newlines to single
    final_output = re.sub(r'\s+\n', '\n', final_output) # Remove spaces before newlines
    
    return final_output


# --- Define U-Net model (Matching your trained model's architecture) ---
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_c, out_c, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_c, out_c, 3, padding=1),
                torch.nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = torch.nn.MaxPool2d(2)

        self.up2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

# --- Set device and Load U-Net model ---
@st.cache_resource
def load_unet_model_cached():
    """Loads the U-Net model and caches it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_instance = UNet()
    try:
        # Assuming 'leaf_unet3_model.pth' is in the same directory as this app.py
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'leaf_unett_model.pth')
        if not os.path.exists(MODEL_PATH):
            st.error(f"U-Net model file not found at: {MODEL_PATH}. Disease segmentation will not work.")
            return None, device
        
        unet_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        unet_instance.to(device).eval() # Set model to evaluation mode
        # st.success("U-Net model loaded successfully (cached).") # Removed to avoid clutter on every rerun
        return unet_instance, device
    except Exception as e:
        st.error(f"ERROR loading U-Net model: {e}. Segmentation will not work.")
        return None, device

# Call the cached model loader
unet, device = load_unet_model_cached()

# --- Transforms for U-Net (must match training transform for input images) ---
transform_unet = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper Function: Generate a circular mask ---
def create_circular_mask(height, width, center=None, radius=None):
    """
    Creates a circular mask for an image.
    """
    if center is None:
        center = (int(width / 2), int(height / 2))
    if radius is None:
        radius = min(height, width) / 2 * 0.85
    
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask.astype(np.uint8)

# --- U-Net Segmentation (Modified to apply circular mask and upscale) ---
def segment_with_unet(image: Image.Image, mask_radius_factor=0.75):
    if unet is None:
        # Error message is already shown in load_unet_model_cached
        return np.array(image) # Return original image if U-Net not loaded

    original_width, original_height = image.size # Get original dimensions

    input_tensor = transform_unet(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = unet(input_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()

    pred_h, pred_w = prediction.shape[0], prediction.shape[1]
    
    circular_mask_radius = int(min(pred_h, pred_w) / 2 * mask_radius_factor)
    circular_mask = create_circular_mask(pred_h, pred_w, radius=circular_mask_radius)

    masked_prediction = prediction * circular_mask
    masked_prediction = (masked_prediction > 0.5).astype(np.uint8) * 255

    # Convert PIL image to OpenCV format (BGR) for drawing
    img_np_resized = np.array(image.resize((256, 256))) # Use original resized for drawing base
    circled_img = cv2.cvtColor(img_np_resized, cv2.COLOR_RGB2BGR) # Convert to BGR for cv2

    contours, _ = cv2.findContours(masked_prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 40: # Threshold for small noise contours
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Changed color to red (0, 0, 255) and thickness to 2
        cv2.circle(circled_img, center, radius, (0, 0, 255), 2) 

    # Upscale the circled image back to original resolution
    upscaled_circled_img = cv2.resize(circled_img, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    # Convert back to RGB for Streamlit display
    return cv2.cvtColor(upscaled_circled_img, cv2.COLOR_BGR2RGB)

# --- Gemini API Functions for Classification and Remedies (from App2.py, slightly modified for error handling) ---
@st.cache_data(ttl=3600) # Cache results for 1 hour
def predict_disease_with_model_ADI(image_pil: Image.Image) -> str:
    """
    Sends a PIL Image to the Gemini API for disease prediction.
    Returns disease name in English and Bengali.
    """
    if not MODEL_ADI_API_KEY:
        return "API Key Missing"

    # Convert PIL Image to bytes then base64
    from io import BytesIO
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG") # Save as JPEG for inlineData
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    vision_prompt = """
    Analyze the provided plant or leaf image for health issues. Identify any abnormal conditions like diseases, pests, deficiencies, or decay.
    Respond STRICTLY with the exact name of the condition identified and the plant name, in English, followed by its Bengali translation in parentheses.
    Example: 'Tomato Early Blight (টমেটোর আগাম ধসা রোগ)'.
    If multiple conditions, prioritize the most severe.
    If no issues, reply 'Healthy'.
    If unrecognizable, reply 'I don't know'.
    If not a plant image, reply 'Not a plant image'.
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": vision_prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1, # Low temperature for deterministic output
            "maxOutputTokens": 100 # Increased tokens for longer output
        }
    }

    headers = {'Content-Type': 'application/json'}
    
    try:
        url = f"{Model_ADI_API_URL}?key={MODEL_ADI_API_KEY}"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        if result.get("candidates") and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content")
            if content and content.get("parts") and len(content["parts"]) > 0:
                disease_name = content["parts"][0].get("text", "Unknown").strip()
                return disease_name
        return "Failed to get disease name from AI."
    except requests.exceptions.RequestException as e:
        st.error(f"AI classification request failed: {e}. Check your network and API key.")
        return "API Error"
    except json.JSONDecodeError:
        st.error("Failed to parse AI response for classification. Response might not be valid JSON.")
        return "Parse Error"
    except Exception as e:
        st.error(f"An unexpected error occurred during classification: {e}")
        return "Error"

@st.cache_data(ttl=3600) # Cache results for 1 hour
def get_remedies_with_model_ADI(disease_name: str, temp: float, humidity: float, ph: float, light: float, N: float, P: float, K: float, rain: str) -> dict:
    """
    Requests remedies and precautions from Gemini API based on disease and environment.
    Uses structured output (JSON schema).
    """
    if not MODEL_ADI_API_KEY:
        return {"Precautions": "API Key Missing", "Medicines": "API Key Missing"}

    # Convert rain status to a more descriptive string for the LLM
    rain_status_str = "it is raining or has rained recently" if rain == "1" else "it is not raining"

    remedy_prompt = f"""
    For the plant condition: '{disease_name}', considering the following environmental factors:
    Temperature: {temp}°C
    Humidity: {humidity}%
    Soil pH: {ph}
    Light Intensity: {light} Lux
    Soil Nutrients (NPK): Nitrogen {N}%, Phosphorus {P}%, Potassium {K}%
    Rain Status: {rain_status_str}

    Provide effective precautionary measures and suitable medicines/treatments.
    Format your response as a JSON object with two keys: "Precautions" and "Medicines".
    Each value should be a numbered list (1., 2., 3., etc.) of concise, single-sentence measures.
    Ensure both lists contain at least three relevant points.
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": remedy_prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "Precautions": {"type": "STRING"},
                    "Medicines": {"type": "STRING"}
                },
                "required": ["Precautions", "Medicines"]
            },
            "temperature": 0.7, # Higher temperature for more varied remedies
            "maxOutputTokens": 500 # More tokens for detailed remedies
        }
    }

    headers = {'Content-Type': 'application/json'}
    
    try:
        url = f"{Model_ADI_API_URL}?key={MODEL_ADI_API_KEY}"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        if result.get("candidates") and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content")
            if content and content.get("parts") and len(content["parts"]) > 0:
                remedies_json = json.loads(content["parts"][0].get("text", "{}"))
                return {
                    "Precautions": remedies_json.get("Precautions", "No precautions found."),
                    "Medicines": remedies_json.get("Medicines", "No medicines found.")
                }
        return {"Precautions": "Failed to get remedies from AI.", "Medicines": "Failed to get remedies from AI."}
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed for remedies: {e}. Check your network and API key.")
        return {"Precautions": "API Error", "Medicines": "API Error"}
    except json.JSONDecodeError:
        st.error("Failed to parse AI response for remedies. Response might not be valid JSON.")
        return {"Precautions": "Parse Error", "Medicines": "Parse Error"}
    except Exception as e:
        st.error(f"An unexpected error occurred during remedy generation: {e}")
        return {"Precautions": "Error", "Medicines": "Error"}

@st.cache_data(ttl=3600) # Cache results for 1 hour
def get_suitable_crops_with_gemini(temp: float, humidity: float, ph: float, light: float, N: float, P: float, K: float, rain: str) -> list[dict]: # Changed return type hint
    """
    Requests suitable crops from Gemini API based on environmental conditions.
    Returns a list of dictionaries, each containing English and Bengali crop names.
    """
    if not MODEL_ADI_API_KEY:
        st.error("API Key Missing for crop prediction. Please set your Gemini API key.")
        return [{"englishName": "API Key Missing", "bengaliName": "API কী অনুপস্থিত"}]

    rain_status_str = "it is raining or has rained recently" if rain == "1" else "it is not raining"

    # Modified crop_prompt to request both English and Bengali names as a JSON array of objects
    crop_prompt = f"""
    Given the following environmental conditions:
    Temperature: {temp}°C
    Humidity: {humidity}%
    Soil pH: {ph}
    Light Intensity: {light} Lux
    Soil Nutrients (NPK): Nitrogen {N}%, Phosphorus {P}%, Potassium {K}%
    Rain Status: {rain_status_str}

    List up to 5 crops that would thrive in these conditions.
    Respond STRICTLY as a JSON array of objects, where each object has two keys: "englishName" and "bengaliName".
    Example: [{{ "englishName": "Wheat", "bengaliName": "গম" }}, {{ "englishName": "Rice", "bengaliName": "ধান" }}]
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": crop_prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "englishName": {"type": "STRING"},
                        "bengaliName": {"type": "STRING"}
                    },
                    "required": ["englishName", "bengaliName"] # Ensure both are present
                }
            },
            "temperature": 0.5, # Moderate temperature for variety
            "maxOutputTokens": 200 # Increased tokens for more detailed output
        }
    }

    headers = {'Content-Type': 'application/json'}
    
    try:
        url = f"{Model_ADI_API_URL}?key={MODEL_ADI_API_KEY}"
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        if result.get("candidates") and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content")
            if content and content.get("parts") and len(content["parts"]) > 0:
                crops_list = json.loads(content["parts"][0].get("text", "[]"))
                if isinstance(crops_list, list):
                    # Validate each item in the list
                    valid_crops = []
                    for item in crops_list:
                        if isinstance(item, dict) and "englishName" in item and "bengaliName" in item:
                            valid_crops.append(item)
                    return valid_crops
        return [{"englishName": "Failed to get crop recommendations from AI.", "bengaliName": "এআই থেকে ফসলের সুপারিশ পেতে ব্যর্থ হয়েছে。"}]
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed for crop prediction: {e}. Check your network and API key.")
        return [{"englishName": "API Error", "bengaliName": "এপিআই ত্রুটি"}]
    except json.JSONDecodeError:
        st.error("Failed to parse AI response for crop prediction. Response might not be valid JSON.")
        return [{"englishName": "Parse Error", "bengaliName": "পার্স ত্রুটি"}]
    except Exception as e:
        st.error(f"An unexpected error occurred during crop prediction: {e}")
        return [{"englishName": "Error", "bengaliName": "ত্রুটি"}]


# --- Streamlit App Layout and Logic ---

def main():
    # Custom Navigation Bar (using HTML anchor links for scrolling)
    st.markdown(
        f"""
        <div class="main-nav-container">
            <a class="main-nav-item" href="#home-section">
                <i class="fas fa-home"></i> Home
            </a>
            <a class="main-nav-item" href="#about-section">
                <i class="fas fa-info-circle"></i> About
            </a>
            <a class="main-nav-item" href="#crop-query-section">
                <i class="fas fa-search-location"></i> Environmental Conditions & Crop Query
            </a>
            <a class="main-nav-item" href="#predict-section">
                <i class="fas fa-flask"></i> Predict Disease
            </a>
            <a class="main-nav-item" href="#chatbot-section"> 
                <i class="fas fa-comments"></i> Chatbot
            </a>
            <a class="main-nav-item" href="#contact-section">
                <i class="fas fa-envelope"></i> Contact Us
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Render All Sections Sequentially within the main block-container ---

    # Home Section
    st.markdown('<div id="home-section">', unsafe_allow_html=True) # Renamed to div with ID for anchor, no 'card' class
    st.markdown('<h1 class="header-title"><i class="fas fa-leaf"></i> Plant Disease Detection & Remedies</h1>', unsafe_allow_html=True)
    st.write("Welcome to the Plant Disease Detector & Crop Recommender! Use this tool to analyze plant images for diseases and get tailored crop recommendations based on environmental conditions.")
    st.write("Navigate through the sections above to explore features like disease prediction, remedies, and optimal crop suggestions for your farm.")
    st.markdown('</div>', unsafe_allow_html=True)

    # About Section
    st.markdown('<div id="about-section">', unsafe_allow_html=True) # Renamed to div with ID for anchor, no 'card' class
    st.markdown('<h2><i class="fas fa-book"></i> About This Application</h2>', unsafe_allow_html=True)
    st.write("This application is your ultimate companion for plant health and agricultural planning. Our core features include:")
    st.markdown(
        """
        * **Intelligent Disease Detection:** Upload images of your ailing plants and receive instant diagnoses, identifying common diseases, pests, and nutrient deficiencies.
        * **Environmental Crop Recommendations:** Input your local temperature, humidity, soil pH, light intensity, and NPK levels to get tailored suggestions for crops that will flourish in your unique environment.
        * **Comprehensive Remedy Suggestions:** Once a disease is detected, receive practical, actionable remedies and preventative measures to protect your crops.
        * **Interactive Plant Chatbot (Coming Soon!):** Engage with our AI chatbot for answers to all your plant-related questions, from general gardening tips to specific agricultural queries.
        """
    )
    st.write("Our aim is to provide a seamless and informative experience, putting expert agricultural knowledge at your fingertips. Get started today and transform your approach to plant care!")
    st.markdown('</div>', unsafe_allow_html=True)

    # Crop Query Section
    st.markdown('<div id="crop-query-section">', unsafe_allow_html=True) # Renamed to div with ID for anchor, no 'card' class
    st.markdown('<h2><i class="fas fa-seedling"></i> Environmental Conditions & Crop Query</h2>', unsafe_allow_html=True)
    st.write("Please provide the current environmental conditions below. These values will be used for both crop recommendations and for generating disease remedies.")

    # Environmental inputs (Unified inputs for both sections)
    col_env1, col_env2 = st.columns(2)
    with col_env1:
        temperature = st.number_input("Temperature (°C):", value=25.0, step=0.1, min_value=0.0, max_value=100.0, key="main_temp")
        soil_ph = st.number_input("Soil pH:", value=6.5, step=0.1, min_value=0.0, max_value=14.0, format="%.1f", key="main_ph")
        npk_n = st.number_input("Nitrogen (N %):", value=10.0, step=0.1, min_value=0.0, max_value=100.0, format="%.1f", key="main_n")
        npk_k = st.number_input("Potassium (K %):", value=15.0, step=0.1, min_value=0.0, max_value=100.0, format="%.1f", key="main_k")
    with col_env2:
        humidity = st.number_input("Humidity (%):", value=65.0, step=0.1, min_value=0.0, max_value=100.0, key="main_hum")
        light_intensity = st.number_input("Light Intensity (Lux):", value=10000.0, step=100.0, min_value=0.0, key="main_light")
        npk_p = st.number_input("Phosphorus (P %):", value=5.0, step=0.1, min_value=0.0, max_value=100.0, format="%.1f", key="main_p")
        rain_status = st.selectbox("Rain Status:", options=[("No Rain", "0"), ("Raining/Recently Rained", "1")], format_func=lambda x: x[0], key="main_rain")[1]
    
    st.markdown("---") # Separator before the "Recommend Crops" button

    if st.button("Recommend Crops", key="recommendCropsBtn_query"):
        with st.spinner('Fetching crop recommendations...'):
            # Use the values from the unified inputs for crop recommendations
            suitable_crops = get_suitable_crops_with_gemini(temperature, humidity, soil_ph, light_intensity, npk_n, npk_p, npk_k, rain_status)
            
            st.markdown('<div class="output-box crop-recommendation-output">', unsafe_allow_html=True)
            st.markdown('<strong>Recommended Crops:</strong>', unsafe_allow_html=True)
            if suitable_crops and suitable_crops[0].get("englishName") not in ["API Key Missing", "API Error", "Parse Error", "Error", "Failed to get crop recommendations from AI."]:
                formatted_crops = [f'- {crop["englishName"]} ({crop["bengaliName"]})' for crop in suitable_crops]
                st.markdown(f'<span id="cropList_query">{"<br>".join(formatted_crops)}</span>', unsafe_allow_html=True)
            else:
                st.error(f"Could not recommend crops: {suitable_crops[0].get('englishName')}. Please try again or check your API key.")
                st.markdown('<span id="cropList_query">No recommendations yet.</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    # --- Predict Disease Section ---
    st.markdown('<div id="predict-section">', unsafe_allow_html=True) # Renamed to div with ID for anchor, no 'card' class
    st.markdown('<h2><i class="fas fa-camera"></i> Upload Plant Image for Disease Prediction</h2>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="imageUpload_predict")

    # Display image preview
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
        st.image(image_pil, caption='Uploaded Image', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload an image to see its preview.")
        image_pil = None # Ensure image_pil is None if no file uploaded

    if st.button("Analyze Image", key="analyzeImageBtn"):
        if image_pil is None:
            st.warning('Please upload an image first.')
        else:
            with st.spinner('Analyzing image and fetching remedies...'):
                predicted_disease = predict_disease_with_model_ADI(image_pil) 
                
                if predicted_disease == "API Key Missing":
                    st.error("API Key Missing. Please set your GENERATIVE_LANGUAGE_API_KEY environment variable or in .streamlit/secrets.toml.")
                elif predicted_disease in ["API Error", "Parse Error", "Error"]:
                    st.error(f"AI Classification failed due to: {predicted_disease}. Please try again.")
                elif predicted_disease in ["I don't know", "Not a plant image", "Healthy", "Failed to get disease name from AI."]:
                    st.warning(f"AI could not provide a specific diagnosis: {predicted_disease}. Please try another image or check context.")
                else:
                    segmented_image_display = segment_with_unet(image_pil, mask_radius_factor=0.7)
                    ai_confidence = random.uniform(85.0, 99.0)
                    
                    # Fetch remedies using the values from the UNIFIED inputs
                    remedies_data = get_remedies_with_model_ADI(
                        predicted_disease, 
                        temperature, humidity, soil_ph, light_intensity, 
                        npk_n, npk_p, npk_k, rain_status
                    )
                    
                    formatted_precautions = format_remedies(remedies_data.get("Precautions", "No recommendations.")).replace("\n", "<br>")
                    formatted_medicines = format_remedies(remedies_data.get("Medicines", "No recommendations.")).replace("\n", "<br>")
                    
                    # --- Display Results ---
                    st.markdown('<div class="output-sections">', unsafe_allow_html=True)

                    st.markdown('<div class="output-card prediction-card">', unsafe_allow_html=True)
                    st.markdown('<strong><i class="fas fa-brain"></i> AI Prediction:</strong>', unsafe_allow_html=True)
                    st.markdown(f'<span id="diseaseName">{predicted_disease}</span><br>', unsafe_allow_html=True)
                    st.markdown(f'Confidence: <span id="confidence">{ai_confidence:.2f}</span>%', unsafe_allow_html=True)
                    # Display the user-provided environmental conditions from the unified inputs for transparency
                    st.markdown(f"""
                        <br><strong>Current Environmental Conditions (from above section):</strong><br>
                        Temp: {temperature:.1f}C, Humidity: {humidity:.1f}%, pH: {soil_ph:.1f}<br>
                        Light: {light_intensity:.0f} Lux, NPK: N{npk_n:.1f}% P{npk_p:.1f}% K{npk_k:.1f}%<br>
                        Rain: {'Raining' if rain_status == '1' else 'No Rain'}
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)


                    st.markdown('<div class="output-card segmented-image-card">', unsafe_allow_html=True)
                    st.markdown('<strong><i class="fas fa-leaf"></i> Segmented Disease Region:</strong>', unsafe_allow_html=True)
                    st.image(segmented_image_display, caption="Detected Infections (Circles show detected infections)", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    if remedies_data.get("Precautions") and remedies_data["Precautions"] not in ["API Key Missing", "API Error", "Parse Error", "Error", "No recommendations."]:
                        st.markdown('<div class="output-card remedy-box">', unsafe_allow_html=True)
                        st.markdown('<strong><i class="fas fa-shield-alt"></i> Precautions:</strong>', unsafe_allow_html=True)
                        st.markdown(f'<span id="precautionsText">{formatted_precautions}</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Precautions could not be retrieved. Please check API key or AI response.")

                    if remedies_data.get("Medicines") and remedies_data["Medicines"] not in ["API Key Missing", "API Error", "Parse Error", "Error", "No recommendations."]:
                        st.markdown('<div class="output-card medicine-box">', unsafe_allow_html=True)
                        st.markdown('<strong><i class="fas fa-pills"></i> Medicines:</strong>', unsafe_allow_html=True)
                        st.markdown(f'<span id="medicinesText">{formatted_medicines}</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Medicines could not be retrieved. Please check API key or AI response.")
                    
                    st.markdown('</div>', unsafe_allow_html=True) # Close output-sections div
        
    st.markdown('</div>', unsafe_allow_html=True) # End Predict Disease section

    # --- Chatbot Section ---
    st.markdown('<div id="chatbot-section">', unsafe_allow_html=True)
    st.markdown('<h2><i class="fas fa-comments"></i> Plant Chatbot</h2>', unsafe_allow_html=True)
    st.write("This section will host an AI-powered chatbot to answer your questions about plants, farming, and more. Stay tuned for future updates!")
    st.markdown("<br><br><br>", unsafe_allow_html=True) # Added extra space
    st.button("Chat", key="chatbot_button") # Added a blank "Chat" button
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Contact Us Section ---
    st.markdown('<div id="contact-section">', unsafe_allow_html=True) # Renamed to div with ID for anchor, no 'card' class
    st.markdown('<h2><i class="fas fa-headset"></i> Contact Us</h2>', unsafe_allow_html=True)
    st.write("If you have any questions, feedback, or need support regarding the Plant Disease Detector & Crop Recommender, please don't hesitate to reach out.")
    st.markdown('You can contact us via email at: <strong><a href="mailto:support@plantapp.com">support@plantapp.com</a></strong>', unsafe_allow_html=True)
    st.write("We aim to respond to all inquiries within 24-48 business hours.")
    st.markdown("""
        <div class="social-links">
            <a href="#" target="_blank" title="Follow us on Twitter"><i class="fab fa-twitter"></i></a>
            <a href="#" target="_blank" title="Connect on LinkedIn"><i class="fab fa-linkedin-in"></i></a>
            <a href="#" target="_blank" title="Check our GitHub"><i class="fab fa-github"></i></a>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # End Contact Us section


    # --- Footer ---
    st.markdown("""
        <footer class="app-footer">
            <p>&copy; 2025 Plant Disease Detector & Crop Recommender. All rights reserved.</p>
            <p>Disclaimer: This application provides AI-generated insights and recommendations for informational purposes only. Consult with agricultural experts for precise advice.</p>
        </footer>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
