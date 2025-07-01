# Crop-disease-detection-model-using-AI

![Alt text for the image](Output_images/web_layout.png)
🌿 Plant Disease Detector & Crop Recommender
This project offers an integrated solution for agriculture: AI-driven plant disease detection, environmental crop recommendations, and practical remedies. It combines a hardware system for data collection with a Streamlit web application powered by a custom U-Net model and Google's Gemini AI.

✨ Key Features at a Glance
Disease Detection: AI analyzes plant images for health issues.

Crop Recommendations: Suggestions based on environmental data.

Remedy Solutions: Practical advice for detected diseases.

Visual Segmentation: Highlights affected areas on images.

Hardware Integration: Real-world data input from custom sensors.

⚙️ Hardware Module
The project incorporates a custom-built hardware component that collects essential real-time environmental and soil data. This data directly informs the application's crop recommendations and remedy suggestions.


![](Output_images/hardware_module.jpeg)
Our custom hardware module for environmental and soil data collection.

📸 Application Layout & UI
Explore the main interfaces of the web application.

![](Output_images/web_layout2.png)
Overall Web Layouts
Get a feel for the application's design and navigation.


The intuitive navigation and clean design.


Another view of the application's structure.

📈 Environmental Conditions & Crop Recommendations
Input your environmental data (temperature, humidity, pH, NPK, light, rain status) to receive tailored crop suggestions. This data can come from our integrated hardware module.

![](Output_images/recommend_crop1.png)
Provide environmental details to get precise crop recommendations.
![](Output_images/recommend_crop2.png)

🌱 Disease Detection & Remedies Flow
See how the application diagnoses plant health and offers solutions.

# Sample1:
1. Upload & Analyze 
Upload your plant image. The AI will process it for disease detection.

![](Output_images/upload_image1.png)

2. Segmented Disease & Prediction with Medicine , Precautions
The U-Net model segments the disease regions, and the AI provides a diagnosis along with confidence.
Receive detailed recommendations for remedies, including both preventative measures and suitable medicines.
(This detail will be visible in the "Output Image with Segmented Disease" if that image captures the entire output section with medicine/precautions cards. If you have a separate image for just these, add another placeholder below.)

![](Output_images/predicted_image1.png)

Visualizing detected disease areas and AI's prediction.

# Sample2:
1. Upload & Analyze 
Upload your plant image. The AI will process it for disease detection.

![](Output_images/upload_image2.png)

2. Segmented Disease & Prediction with Medicine , Precautions
The U-Net model segments the disease regions, and the AI provides a diagnosis along with confidence.
Receive detailed recommendations for remedies, including both preventative measures and suitable medicines.
(This detail will be visible in the "Output Image with Segmented Disease" if that image captures the entire output section with medicine/precautions cards. If you have a separate image for just these, add another placeholder below.)

![](Output_images/predicted_image2.png)

Visualizing detected disease areas and AI's prediction.



🚀 Get Started
Live Demo (Deployed Version)
Experience the application directly in your browser:

👉 https://crop-disease-detection-model-using-ai.streamlit.app/

Local Setup
To run this project locally:

Clone:
```
git clone https://github.com/Creating-Content/Crop-disease-detection-model-using-AI
cd Crop-disease-detection-model-using-AI
```
Dependencies:
```
pip install -r requirements.txt
```
(Create requirements.txt with: streamlit, torch, torchvision, Pillow, numpy, opencv-python, requests)

API Key: Obtain a Google Gemini API Key and save it in .streamlit/secrets.toml:

# .streamlit/secrets.toml
```
GENERATIVE_LANGUAGE_API_KEY = "YOUR_API_KEY_HERE"
```
U-Net Model: Place your leaf_unett_model.pth file in the project's root directory.

Run:

streamlit run crop.py

🤝 Contributing
We welcome contributions! Feel free to open issues or pull requests.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

