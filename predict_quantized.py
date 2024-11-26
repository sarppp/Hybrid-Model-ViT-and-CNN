import os
import torch
import time
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from model import HybridViT
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_class_names():
    """Get class names from the dataset."""
    dataset = OxfordIIITPet(root='data', split='trainval', download=False)
    return dataset.classes

def load_models():
    """Load both original and quantized models."""
    print("\nLoading models...")
    
    # Load original model
    original_model = HybridViT()
    original_model.load_state_dict(torch.load("best_model.pth", map_location='cpu', weights_only=True))
    original_model.eval()
    
    # Load quantized model
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        quantized_model = torch.load("dynamic_quantized_model_x86.pth", map_location='cpu')
    quantized_model.eval()
    
    return original_model, quantized_model

def preprocess_image(image_path):
    """Preprocess the input image."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image

def get_predictions(model, image, class_names):
    """Get model predictions."""
    start_time = time.time()
    
    with torch.no_grad():
        output = model(image)
    
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_idx = torch.topk(probabilities, 1)
    
    class_name = class_names[top_idx.item()]
    probability = top_prob.item()
    
    return class_name, probability, inference_time

def print_prediction(model_name, class_name, probability):
    """Print prediction in a formatted way."""
    print(f"{model_name}: {class_name} ({probability*100:.2f}%)")

def display_predictions(image, orig_class, orig_prob, quant_class, quant_prob):
    """Display original and quantized predictions side by side."""
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Resize image for better display while maintaining aspect ratio
    height, width = image_cv.shape[:2]
    max_dim = 400
    scale = max_dim / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Create two copies of the resized image
    image_orig = cv2.resize(image_cv.copy(), (new_width, new_height))
    image_quant = cv2.resize(image_cv.copy(), (new_width, new_height))
    
    # Prepare texts
    orig_text = f"{orig_class} ({orig_prob*100:.2f}%)"
    quant_text = f"{quant_class} ({quant_prob*100:.2f}%)"
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Add text to both images
    for img, text in [(image_orig, orig_text), (image_quant, quant_text)]:
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (new_width - text_size[0]) // 2
        text_y = new_height - 20
        
        # Draw white background rectangle
        padding = 10
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (255, 255, 255),
                     -1)
        
        # Draw text
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    # Convert back to RGB for matplotlib
    image_orig_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    image_quant_rgb = cv2.cvtColor(image_quant, cv2.COLOR_BGR2RGB)
    
    # Display images side by side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_orig_rgb)
    plt.axis('off')
    plt.title("Original Model Prediction", pad=10)
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_quant_rgb)
    plt.axis('off')
    plt.title("Quantized Model Prediction", pad=10)
    
    # Adjust the layout to prevent text cutoff
    plt.tight_layout()
    
    # Keep the window open and bring it to front
    plt.show()

def main():
    # Get image path from user
    image_path = input("Enter the path to your test image: ")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    try:
        # Load class names and models
        print("\nLoading class names and models...")
        class_names = get_class_names()
        original_model, quantized_model = load_models()
        
        # Preprocess image and get predictions
        print("Processing image and making predictions...")
        image, image_pil = preprocess_image(image_path)
        orig_class, orig_prob, original_time = get_predictions(original_model, image, class_names)
        quant_class, quant_prob, quantized_time = get_predictions(quantized_model, image, class_names)
        
        # Print results
        print(f"\nPredictions for: {os.path.basename(image_path)}")
        print_prediction("Original Model", orig_class, orig_prob)
        print_prediction("Quantized Model", quant_class, quant_prob)
        speedup = original_time / quantized_time
        print(f"\nSpeedup: {speedup:.2f}x")
        
        # Display predictions
        display_predictions(image_pil, orig_class, orig_prob, quant_class, quant_prob)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
