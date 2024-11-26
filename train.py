import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import HybridViT
from config import Config
from training_monitor import TrainingMonitor

class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.0001, monitor='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.monitor = monitor
        self.should_stop = False
    
    def __call__(self, val_loss, val_acc=None):
        if self.monitor == 'loss':
            metric = val_loss
            best_metric = self.best_loss
            is_better = lambda new, old: new < old - self.min_delta
        else:  # monitor accuracy
            metric = val_acc
            best_metric = self.best_acc
            is_better = lambda new, old: new > old + self.min_delta

        if best_metric is None:
            if self.monitor == 'loss':
                self.best_loss = val_loss
                self.best_acc = val_acc
            else:
                self.best_acc = val_acc
                self.best_loss = val_loss
        elif not is_better(metric, best_metric):
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            if self.monitor == 'loss':
                self.best_loss = val_loss
                self.best_acc = val_acc
            else:
                self.best_acc = val_acc
                self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, test_loader, device, params=None, num_epochs=100, monitor=None):
    if params is None:
        params = Config().load_params()
    
    monitor.log_message(f"Starting training with parameters: {params}")
    monitor.log_message(f"Number of epochs: {num_epochs}")
    monitor.log_message(f"Training device: {device}")
    monitor.log_message("=" * 50)
    
    # Calculate class weights if needed
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights().to(device)
    else:
        class_weights = None
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15, weight=class_weights)
    
    param_groups = [
        {'params': model.cnn.parameters(), 'lr': params['lr_cnn']},
        {'params': model.vit.parameters(), 'lr': params['lr_vit']},
        {'params': model.classifier.parameters(), 'lr': params['lr_classifier']}
    ]
    
    optimizer = AdamW(param_groups, weight_decay=params['weight_decay'])
    
    # Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Initial restart interval
        T_mult=2,  # Multiply T_i by this number after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    if monitor is None:
        monitor = TrainingMonitor()
    
    if monitor:
        monitor.train_loader = train_loader
    
    early_stopping = EarlyStopping(patience=25, min_delta=0.0001, monitor='accuracy')
    
    # Gradient accumulation steps
    accumulation_steps = 4
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        model.train()
        monitor.start_epoch()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            monitor.update_batch(loss.item() * accumulation_steps, outputs, labels, optimizer.param_groups[0]['lr'])
        
        # Step the scheduler based on epoch
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        example_images = []
        example_outputs = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if batch_idx == 0:
                    example_images = images[:5]
                    example_outputs = outputs[:5].cpu()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        monitor.end_epoch(val_loss, val_acc, all_preds, all_labels, example_images, example_outputs)
        
        # Check early stopping with both metrics
        early_stopping(val_loss, val_acc)
        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            monitor.log_message(f"Best validation loss: {early_stopping.best_loss:.4f}")
            monitor.log_message(f"Best validation accuracy: {early_stopping.best_acc:.2f}%")
            break
        
        if epoch == num_epochs - 1:
            monitor.plot_metrics()
    
    return model, monitor
