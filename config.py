import json
import os

class Config:
    def __init__(self):
        self.config_path = 'best_params.json'
        self.default_params = {
            'dropout_rate': 0.3,  # Increased from 0.2
            'weight_decay': 0.01,
            'lr_cnn': 1e-5,  # Reduced from 3e-5
            'lr_vit': 1e-5,  # Reduced from 3e-5
            'lr_classifier': 3e-5,  # Reduced from 8e-5
            'cnn_freeze_layers': 20,
            'vit_freeze_blocks': 2,
            'label_smoothing': 0.15,
            'gradient_clip_val': 1.0,
            'accumulation_steps': 4,
            'scheduler': {
                'name': 'cosine_warm_restarts',
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 1e-6
            },
            'early_stopping': {
                'patience': 25,
                'min_delta': 0.0001,
                'monitor': 'accuracy'
            }
        }
    
    def save_params(self, params):
        with open(self.config_path, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Parameters saved to {self.config_path}")
    
    def load_params(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        print(f"No saved parameters found at {self.config_path}, using defaults")
        return self.default_params
