import optuna
from optuna.trial import TrialState
import torch
from model import HybridViT
from train import train_model
from config import Config

def objective(trial, train_loader, test_loader, device, num_epochs=10):  
    print(f"\nTrial {trial.number} started...")
    
    # Hyperparameter Search Space
    # --------------------------
    # Each parameter has 3 ranges:
    # 1. Current: Quick optimization (20 minutes)
    # 2. Recommended: Thorough search (few hours)
    # 3. Research: From published papers
    
    params = {
        # Dropout Rate
        # Current: 0.2-0.4 (Focused range for quick results)
        # Recommended: 0.1-0.5 (Full exploration)
        # Research papers: 0.0-0.7
        'dropout_rate': trial.suggest_float("dropout_rate", 0.2, 0.4),
        
        # Weight Decay
        # Current: 0.005-0.05 (Moderate regularization)
        # Recommended: 0.001-0.1 (Full range)
        # Research papers: 1e-6-1e-1
        'weight_decay': trial.suggest_float("weight_decay", 0.005, 0.05, log=True),
        
        # Learning Rates
        # Current: 3e-5 to 3e-4 (Conservative range)
        # Recommended: 1e-5 to 1e-3 (Full fine-tuning range)
        # Research papers: 1e-6 to 1e-2
        'lr_cnn': trial.suggest_float("lr_cnn", 3e-5, 3e-4, log=True),
        'lr_vit': trial.suggest_float("lr_vit", 3e-5, 3e-4, log=True),
        'lr_classifier': trial.suggest_float("lr_classifier", 3e-5, 3e-4, log=True),
        
        # Layer Freezing
        # CNN Layers (ResNet34 has 33 layers)
        # Current: 10-25 (Middle layers)
        # Recommended: 0-30 (Full range)
        # Research: Varies by model
        'cnn_freeze_layers': trial.suggest_int("cnn_freeze_layers", 10, 25),
        
        # ViT Blocks (Base ViT has 12 blocks)
        # Current: 4-10 (Allow some adaptation)
        # Recommended: 0-12 (Full range)
        # Research: Usually freeze first 50-75%
        'vit_freeze_blocks': trial.suggest_int("vit_freeze_blocks", 4, 10)
    }
    
    print("Trial parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create and train model
    model = HybridViT(num_classes=37, params=params).to(device)
    model, monitor = train_model(model, train_loader, test_loader, device, params=params, num_epochs=num_epochs)
    
    # Calculate final metrics with more emphasis on recent performance
    last_2_train_acc = sum(monitor.train_accs[-2:]) / 2  # Changed from 3 to 2 epochs
    last_2_val_acc = sum(monitor.val_accs[-2:]) / 2
    stability_penalty = abs(last_2_train_acc - last_2_val_acc) * 0.1
    
    # More aggressive early stopping
    if trial.number > 2:  # Changed from 5 to 2
        median_score = trial.study.trials_dataframe()['value'].median()
        if last_2_val_acc < median_score - 5:  # Stop if significantly worse
            raise optuna.TrialPruned()
    
    final_score = last_2_val_acc - stability_penalty
    return final_score

def optimize_hyperparameters(train_loader, test_loader, device, n_trials=10, num_epochs=10):  
    print("\n" + "="*50)
    print("Starting Optuna hyperparameter optimization")
    print("="*50)
    print(f"Number of trials: {n_trials}")
    
    # Calculate estimated time
    batches_per_epoch = len(train_loader)
    estimated_seconds_per_batch = 2  # Adjusted estimate based on GPU acceleration
    estimated_minutes_per_epoch = (batches_per_epoch * estimated_seconds_per_batch) / 60
    estimated_total_minutes = estimated_minutes_per_epoch * num_epochs * n_trials
    
    print(f"Estimated time:")
    print(f"  - Batches per epoch: {batches_per_epoch}")
    print(f"  - Estimated minutes per epoch: {estimated_minutes_per_epoch:.1f}")
    print(f"  - Estimated total minutes: {estimated_total_minutes:.1f}")
    
    # Set timeout based on optimization mode
    if num_epochs <= 2 and n_trials <= 5:  # Quick optimization mode
        timeout = int(estimated_total_minutes * 60 * 1.2)  # Quick mode: estimated + 20% buffer
    else:  # Full optimization mode
        timeout = int(estimated_total_minutes * 60 * 1.5)  # Full mode: estimated + 50% buffer
    
    print(f"Timeout set to: {timeout//60} minutes")
    if timeout//60 > 60:
        print(f"Warning: This will take {timeout//3600} hours and {(timeout%3600)//60} minutes.")
        print("Consider reducing epochs or trials for faster results.")
    print("="*50 + "\n")
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),  # More aggressive pruning
        study_name="hybrid_vit_optimization"
    )
    
    # Wrap objective to include num_epochs
    objective_with_epochs = lambda trial: objective(trial, train_loader, test_loader, device, num_epochs=num_epochs)
    
    try:
        study.optimize(
            objective_with_epochs,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted! Using best parameters so far...")
    
    print("\nOptimization finished!")
    print("\nBest trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    config = Config()
    config.save_params(trial.params)
    
    return trial.params
