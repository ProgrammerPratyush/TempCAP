import streamlit as st
import torch.nn as nn
import os
import shutil
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from typing import List

# my imports -
from torchvision import transforms
from PIL import Image
from io import BytesIO
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
# from tensorflow.keras.applications import ResNet50

# Load environment variables

# Load your pre-trained models
# plant_disease_model_weights_path = 'models for predictions/Crops/plant-disease-model.pth'
plant_disease_model_path = 'models for predictions/Crops/plant-disease-model-complete.pth'
soil_composition_model_path = 'models for predictions/Soil/composition/composition/two.h5'
soil_type_model_weights_path = 'models for predictions/Soil/soil type/soiltype/tycoon.weights.h5'

# Crop Disease Prediction Function
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Increase the input size
        transforms.ToTensor(),
    ])
    img = transform(image).unsqueeze(0)  # Add batch dimension
    return img


# Prediction function using the fully saved PyTorch model
def predict_crop_disease(image, model_path):
    try:
        # Load the complete model directly
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        # Preprocess the image
        img = preprocess_image(image)

        # Make predictions
        with torch.no_grad():
            predictions = model(img)
        predicted_class = torch.argmax(predictions, dim=1).item()

        # Map the prediction to class name
        return class_names[predicted_class]
    except Exception as e:
        print(f"Error in predicting crop disease: {e}")
        return None


# List of disease names corresponding to class indices
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Soil Composition Prediction Function------------------------------------------------------------------------
def predict_soil_composition(img):
    img = Image.open(img)
    img = img.resize((224, 224))  # Resize to the expected input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    model = load_model(soil_composition_model_path)
    predictions = model.predict(img)
    return predictions[0]


# Define image dimensions
img_width, img_height = 224, 224

# Manually specify the class labels
class_labels = {0: 'Black Soil', 1: 'Cinder Soil', 2: 'Red Soil', 3: 'Yellow Soil'}

def load_and_preprocess_image(img):
    img = img.resize((img_width, img_height))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Recreate the model architecture exactly as in training
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))

# Fine-tuning: Unfreeze the last 10 layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Add custom classifier layers with dropout and batch normalization for regularization
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model with the same configuration as training
initial_learning_rate = 1e-4
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the weights
model.load_weights('models for predictions/Soil/soil type/soiltype/tycoon.weights.h5')

# Weather API Function
def get_weather_info(api_key, lat, lon):
    # Implement the weather API call here using the provided API key
    pass

# Streamlit UI Layout
st.title("AgriVision: Your Agricultural Assistant")

st.header("Predict Crop Disease")
crop_image = st.file_uploader("Upload a Crop Image", type=["jpg", "png", "jpeg","JPG"])

if crop_image:
    st.image(crop_image, caption='Uploaded Crop Image', use_column_width=True)

    if st.button("Predict Disease"):
        # Predict crop disease using the complete model
        image = Image.open(crop_image).convert('RGB')
        crop_result = predict_crop_disease(image, plant_disease_model_path)

        if crop_result is not None:
            st.success(f"Predicted Crop Disease: {crop_result}")
        else:
            st.error("Error in predicting crop disease. Please check the logs.")

# Soil Composition Prediction Section
st.header("Predict Soil Composition")
soil_image_composition = st.file_uploader("Upload a Soil Image for Composition", type=["jpg", "png", "jpeg"], key="composition")
if soil_image_composition is not None:
    st.image(soil_image_composition, caption='Uploaded Soil Image', use_column_width=True)
    if st.button("Predict Composition"):
        composition_prediction = predict_soil_composition(soil_image_composition)
        st.write(f'The predicted soil composition is: {composition_prediction}')

# Soil Type Prediction
st.header("Predict Soil Type")
soil_image_type = st.file_uploader("Upload a Soil Image for Type", type=["jpg", "png", "jpeg"], key="type")
if soil_image_type is not None:
    st.image(soil_image_type, caption='Uploaded Soil Image', use_column_width=True)
    if st.button("Predict Soil Type"):
        # Convert the uploaded file to a PIL Image
        image = Image.open(BytesIO(soil_image_type.read()))

        # Preprocess the image
        input_image = load_and_preprocess_image(image)

        # Predict the class
        predictions = model.predict(input_image)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_labels[predicted_class[0]]

        # Display the prediction result
        st.write(f'The predicted soil type is: {predicted_label}')


# Weather Prediction Section
st.header("Get My Location Weather")
if st.button("Predict Weather"):
    # Code to handle location permission and weather prediction
    pass
