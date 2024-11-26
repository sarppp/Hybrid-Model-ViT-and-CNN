import torch
import torch.nn as nn
import timm

class HybridViT(nn.Module):
    def __init__(self, num_classes=37, params=None):
        super().__init__()
        if params is None:
            from config import Config
            params = Config().load_params()
            
        # Load pre-trained ResNet features
        self.cnn = timm.create_model('resnet34', pretrained=True, features_only=True)
        
        # Load pre-trained ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Apply freezing based on parameters
        for i, (name, param) in enumerate(self.cnn.named_parameters()):
            if i < params['cnn_freeze_layers']:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        for i, (name, param) in enumerate(self.vit.named_parameters()):
            if 'blocks' in name and int(name.split('.')[1]) < params['vit_freeze_blocks']:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Classifier with parameter-based dropout
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.ReLU(),
            nn.Dropout(params['dropout_rate']),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # CNN forward pass
        cnn_features = self.cnn(x)[-1]  # Get the last layer features
        cnn_features = torch.mean(cnn_features, dim=[2, 3])  # Global average pooling
        
        # ViT forward pass
        vit_features = self.vit(x)
        
        # Combine features
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        
        # Classification
        return self.classifier(combined_features)
