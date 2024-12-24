import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
from io import BytesIO

# Set up Streamlit page
st.set_page_config(
    page_title="AI-Driven Pneumonia Diagnosis Tool",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Header image
header_image_url = "images/header.png"
st.image(header_image_url, use_column_width=True)

# Main title and description
st.title("AI-Driven Pneumonia Diagnosis Tool")
st.write(
    """
    Welcome to the AI-Driven Pneumonia Diagnosis Tool. This application utilizes state-of-the-art deep learning models 
    (ResNet50 and EfficientNetB0) to analyze chest X-ray images for pneumonia detection.
    """
)
st.write("---")

# Sidebar setup
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose an option:",
    ["Home", "Resize Image for ResNet50", "Resize Image for EfficientNetB0", "Make Predictions"]
)

# Load pre-trained models
@st.cache_resource
def load_models():
    """
    Load pre-trained models for pneumonia detection.
    """
    resnet50_model = load_model("resnet50_chest_xray_model.h5")
    efficientnetb0_model = load_model("efficientnetb0_chest_xray_model.h5")
    return resnet50_model, efficientnetb0_model

resnet50_model, efficientnetb0_model = load_models()

# Helper function to preprocess images
def preprocess_image(file_buffer, target_size):
    """
    Preprocess the uploaded image for resizing.

    Parameters:
    file_buffer: The uploaded file buffer or file-like object.
    target_size: A tuple specifying the target size for resizing the image.

    Returns:
    processed_image: A NumPy array with the resized image.
    """
    # Open the image file
    image = Image.open(file_buffer).convert("RGB")
    
    # Resize the image to the specified target size
    image = image.resize(target_size)
    
    # Save the resized image to a buffer
    output_buffer = BytesIO()
    image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)
    
    return image, output_buffer

if app_mode == "Home":
    st.write(
        """
        ### Instructions:
        1. Use the **Resize Image for ResNet50** or **Resize Image for EfficientNetB0** options in the sidebar 
           to convert your X-ray image to the expected dimensions.
        2. Once resized, download the image and upload it back under **Make Predictions** to get results.
        """
    )
    st.write("Upload a chest X-ray image to explore AI-driven pneumonia detection!")

elif app_mode == "Resize Image for ResNet50":
    st.subheader("Resize Image for ResNet50")
    uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
        resized_image, buffer = preprocess_image(uploaded_file, target_size=(224, 224))
        st.image(resized_image, caption="Resized Image (224x224)", use_column_width=True)
        st.download_button(
            label="Download Resized Image",
            data=buffer,
            file_name="resnet50_resized_image.jpg",
            mime="image/jpeg"
        )

elif app_mode == "Resize Image for EfficientNetB0":
    st.subheader("Resize Image for EfficientNetB0")
    uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
        resized_image, buffer = preprocess_image(uploaded_file, target_size=(150, 150))
        st.image(resized_image, caption="Resized Image (150x150)", use_column_width=True)
        st.download_button(
            label="Download Resized Image",
            data=buffer,
            file_name="efficientnetb0_resized_image.jpg",
            mime="image/jpeg"
        )

elif app_mode == "Make Predictions":
    st.subheader("Make Predictions")
    uploaded_file = st.file_uploader("Upload a Resized Chest X-Ray Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Resized Chest X-Ray", use_column_width=True)
        
        # Determine dimensions to preprocess image
        if image.size == (224, 224):
            model = resnet50_model
            model_name = "ResNet50"
        elif image.size == (150, 150):
            model = efficientnetb0_model
            model_name = "EfficientNetB0"
        else:
            st.error("Invalid image dimensions! Please resize the image using the tools provided in the sidebar.")
            st.stop()
        
        # Preprocess for prediction
        image_array = np.array(image) / 255.0
        processed_image = np.expand_dims(image_array, axis=0)

        # Perform prediction
        with st.spinner("Predicting..."):
            prediction = model.predict(processed_image)

        # Display results
        classes = ["Normal", "Pneumonia"]
        predicted_class = classes[int(prediction[0][0] > 0.5)]
        st.success(f"**{model_name} Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {prediction[0][0]:.2f}")




# Footer
st.write("---")
st.markdown(
    """
    **Disclaimer**: This tool is part of the research project *AI-Driven Diagnosis of Pneumonia Using Chest X-Ray Imaging*. 
    It is designed to showcase the potential of deep learning for medical diagnostics and is not intended for clinical use. 
    Always consult a healthcare professional for medical advice.
    """
)
st.write("Developed by Musiliu Bello.")
