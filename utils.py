import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

class PreprocessingVIT:
    @staticmethod
    def get_transforms():
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),  # Small rotations
            # Careful color augmentation to preserve breed-specific features
            transforms.ColorJitter(
                brightness=0.1,  # Subtle brightness changes
                contrast=0.1,    # Subtle contrast changes
                saturation=0.1,  # Subtle saturation changes
                hue=0.02        # Very subtle hue changes to preserve natural colors
            ),
            # Random cropping with padding to help focus on different parts of the animal
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),  # Small translations
                    scale=(0.95, 1.05),      # Subtle scaling
                    fill=255
                )
            ], p=0.5),
            # Random perspective to help with different viewing angles
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            # Gaussian blur to help with texture recognition
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return train_transform, val_transform

    def __init__(self, batch_size=32, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform, self.val_transform = self.get_transforms()
        
        # Load datasets
        self.train_dataset = OxfordIIITPet(
            root='./data', 
            split='trainval',
            transform=self.train_transform, 
            download=True
        )
        
        self.test_dataset = OxfordIIITPet(
            root='./data', 
            split='test',
            transform=self.val_transform, 
            download=True
        )
        
        # Store class names
        self.class_names = self.train_dataset.classes
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )
