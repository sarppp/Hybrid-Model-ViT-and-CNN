import argparse
import torch
from torch.utils.data import DataLoader
from model import HybridViT
from train import train_model
from optimize import optimize_hyperparameters
from config import Config
from utils import PreprocessingVIT
from training_monitor import TrainingMonitor

def main():
    parser = argparse.ArgumentParser(description='Train or optimize HybridViT model')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    
    # Training Parameters
    # ------------------
    # Default: 3 (Quick training)
    # Recommended for full training: 10-30 epochs
    # Production: 50-100 epochs with early stopping
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for final training')
    
    # Optuna Parameters
    # ----------------
    # Default: 2 (Quick optimization)
    # Recommended: 5-10 epochs per trial
    # Production: 15-20 epochs per trial with early stopping
    parser.add_argument('--optuna-epochs', type=int, default=2, help='Number of epochs for each optimization trial')
    
    # Default: 16 (Quick iteration, less memory)
    # Recommended: 32-64 (Better convergence)
    # Maximum: Depends on GPU memory, typically 128-256
    # Research papers typically use: 64-128
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    # Default: 5 (Quick search, 20 minutes)
    # Recommended: 20-50 trials (few hours)
    # Production: 100+ trials (overnight)
    # Research papers typically use: 200-500 trials
    parser.add_argument('--trials', type=int, default=5, help='Number of optimization trials')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize monitor first
    transform = PreprocessingVIT()
    train_loader = DataLoader(transform.train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(transform.test_dataset, batch_size=args.batch_size)
    monitor = TrainingMonitor(transform=transform)
    monitor.log_message(f"Using device: {device}")
    
    if args.optimize:
        monitor.log_message("Running hyperparameter optimization...")
        monitor.log_message(f"Number of trials: {args.trials}")
        monitor.log_message(f"Epochs per trial: {args.optuna_epochs}")
        best_params = optimize_hyperparameters(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            n_trials=args.trials,
            num_epochs=args.optuna_epochs
        )
        monitor.log_message("\nBest parameters have been saved to best_params.json")
        
        monitor.log_message(f"\nTraining final model with best parameters for {args.epochs} epochs...")
        config = Config()
        params = config.load_params()
    else:
        monitor.log_message(f"Training model with default/saved parameters for {args.epochs} epochs...")
        config = Config()
        params = config.load_params()
    
    # Train model
    model = HybridViT(num_classes=37, params=params).to(device)
    model, monitor = train_model(
        model, 
        train_loader, 
        test_loader, 
        device,
        params=params,
        num_epochs=args.epochs,
        monitor=monitor  
    )
    
    # Save the trained model
    torch.save(model.state_dict(), 'best_model.pth')
    monitor.log_message("\nModel saved to best_model.pth")
    
    # Plot training metrics
    monitor.plot_metrics()

if __name__ == '__main__':
    main()
