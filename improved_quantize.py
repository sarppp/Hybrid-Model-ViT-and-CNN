import os
import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import psutil
from model import HybridViT

def print_memory_usage(prefix="Memory Usage:"):
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"\n{prefix}")
    print(f"CPU Memory: {cpu_memory:.2f} MB")
    print(f"GPU Memory: {gpu_memory:.2f} MB")

def create_calibration_dataloader(data_dir, batch_size=32):
    """Create a dataloader for calibration data."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def dynamic_quantization(model_path, save_path, backend='fbgemm'):
    """Perform dynamic quantization."""
    print(f"\n=== Starting Dynamic Quantization Process (backend: {backend}) ===")
    print_memory_usage()
    
    # Load original model
    original_model = HybridViT()
    original_model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    original_model.eval()
    
    # Configure dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        original_model,
        {nn.Linear, nn.Conv2d},  # Quantize linear and conv layers
        dtype=torch.qint8,
        inplace=False
    )
    
    # Save the full model
    torch.save(quantized_model, save_path)
    
    # Print model size comparison
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(save_path) / (1024 * 1024)
    reduction = ((original_size - quantized_size) / original_size) * 100
    
    print(f"\nModel size comparison:")
    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Size reduction: {reduction:.2f}%")
    
    return quantized_model

if __name__ == "__main__":
    model_path = "best_model.pth"
    
    print("Starting quantization process...")
    
    try:
        # Try x86 backend first
        print("\nAttempting quantization with fbgemm backend...")
        dynamic_model_x86 = dynamic_quantization(
            model_path=model_path,
            save_path="dynamic_quantized_model_x86.pth",
            backend='fbgemm'
        )
        
        # Try ARM backend
        print("\nAttempting quantization with qnnpack backend...")
        dynamic_model_arm = dynamic_quantization(
            model_path=model_path,
            save_path="dynamic_quantized_model_arm.pth",
            backend='qnnpack'
        )
        
        print("\nQuantization complete! Generated models:")
        print("- dynamic_quantized_model_x86.pth (for x86 processors)")
        print("- dynamic_quantized_model_arm.pth (for ARM processors)")
        
    except Exception as e:
        print(f"\nError during quantization: {str(e)}")
