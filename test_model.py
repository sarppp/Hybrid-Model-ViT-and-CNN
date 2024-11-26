import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from model import HybridViT
from config import Config
import torch.nn.functional as F
import os
from torchvision.datasets import OxfordIIITPet
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_model(model_path, num_classes=37):
    # Load configuration
    config = Config()
    params = config.load_params()
    
    # Initialize model
    model = HybridViT(num_classes=num_classes, params=params)
    
    # Load trained weights with weights_only=True
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def preprocess_image(image_path):
    # Load the original image for display
    original_image = Image.open(image_path).convert('RGB')
    
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transform the image for model input
    image_tensor = transform(original_image)
    return image_tensor.unsqueeze(0), original_image  # Return both tensor and original image

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        return predicted_class.item(), confidence.item()

def get_class_names():
    # Create a temporary dataset instance to get class names
    dataset = OxfordIIITPet(root='data', split='trainval', download=False)
    return dataset.classes

def display_prediction(image, class_name, confidence):
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Resize image for better display while maintaining aspect ratio
    height, width = image_cv.shape[:2]
    max_dim = 800
    scale = max_dim / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image_cv = cv2.resize(image_cv, (new_width, new_height))
    
    # Prepare text
    text = f"{class_name} ({confidence:.2%})"
    
    # Get text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Position text at bottom center
    text_x = (new_width - text_size[0]) // 2
    text_y = new_height - 20
    
    # Draw white background rectangle for text
    padding = 10
    cv2.rectangle(image_cv, 
                 (text_x - padding, text_y - text_size[1] - padding),
                 (text_x + text_size[0] + padding, text_y + padding),
                 (255, 255, 255),
                 -1)
    
    # Draw text
    cv2.putText(image_cv, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    # Convert back to RGB for matplotlib
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Display image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def main():
    # Path to your model and test image
    model_path = 'best_model.pth'
    
    # Get class names
    print("Loading class names...")
    class_names = get_class_names()
    
    # Get image path from user
    image_path = input("Enter the path to your test image: ")
    
    try:
        # Load the model
        print("Loading model...")
        model = load_model(model_path)
        
        # Preprocess the image
        print("Preprocessing image...")
        image_tensor, original_image = preprocess_image(image_path)
        
        # Make prediction
        print("Making prediction...")
        predicted_class, confidence = predict(model, image_tensor)
        
        print(f"\nPrediction Results:")
        print(f"Predicted Class: {predicted_class} ({class_names[predicted_class]})")
        print(f"Confidence: {confidence:.2%}")
        
        # Display the image with prediction
        display_prediction(original_image, class_names[predicted_class], confidence)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
