import torch
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from logger_config import setup_logger
import sys
import json


class TrainingMonitor:
    def __init__(self, log_dir='./logs', transform=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create plots directory at initialization
        self.plots_dir = os.path.join(log_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize logger
        self.logger, self.log_filename = setup_logger()
        
        # Store class names if transform is provided
        self.class_names = transform.class_names if transform is not None else None
        self.total_batches = None  # Will be set in start_epoch
        
        # Initialize metrics lists
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.current_train_loss = 0
        self.current_train_correct = 0
        self.current_train_total = 0
        self.batch_count = 0
        
        # Initialize log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        
        # Open log file in append mode and redirect stdout
        self.log_handle = open(self.log_file, 'a', buffering=1)  # Line buffering
        sys.stdout = self.log_handle
        
        self.logger.info("Training Log - Started at %s", datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.logger.info("=" * 50)
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("=" * 50)
        
    def log_message(self, message):
        self.logger.info(message)
    
    def start_epoch(self):
        """Reset metrics at the start of each epoch."""
        self.current_train_loss = 0.0
        self.current_train_correct = 0
        self.current_train_total = 0
        self.batch_count = 0
        
        # Calculate total batches from train_loader
        if hasattr(self, 'train_loader'):
            self.total_batches = len(self.train_loader)
            self.logger.info(f"Total batches this epoch: {self.total_batches}")
        
        self.logger.info("\nStarting new epoch...")
        self.logger.info("=" * 90)
        self.logger.info("Training Progress (Validation metrics available at epoch end)")
        self.logger.info("-" * 90)
        self.logger.info("")
        
    def update_batch(self, loss, outputs, labels, lr):
        self.current_train_loss += loss
        _, predicted = outputs.max(1)
        self.current_train_total += labels.size(0)
        self.current_train_correct += predicted.eq(labels).sum().item()
        self.batch_count += 1
        
        train_acc = 100. * self.current_train_correct / self.current_train_total
        # Print progress every 20 batches
        if self.batch_count % 20 == 0:
            self.print_batch_progress(self.batch_count, train_acc, loss, lr)
            if self.total_batches:
                progress_percent = min(100.0, (self.batch_count / self.total_batches) * 100)
                self.logger.info(f"Progress: {self.batch_count}/{self.total_batches} batches ({progress_percent:.1f}%)")
    
    def print_batch_progress(self, batch_idx, train_acc, loss, lr):
        self.logger.info(
            f"Batch {batch_idx} | Loss: {loss:.4f} | Training Acc: {train_acc:.2f}% | "
            f"LR: {lr:.6f} | Validation: (waiting for epoch end)"
        )
    
    def end_epoch(self, val_loss, val_acc, all_preds, all_labels, example_images=None, example_outputs=None):
        avg_train_loss = self.current_train_loss / self.batch_count
        train_acc = 100. * self.current_train_correct / self.current_train_total
        
        self.train_losses.append(avg_train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        self.logger.info("\nEpoch Summary:")
        self.logger.info("-" * 50)
        self.logger.info(f"Training Loss: {avg_train_loss:.4f}")
        self.logger.info(f"Training Accuracy: {train_acc:.2f}%")
        self.logger.info(f"Validation Loss: {val_loss:.4f}")
        self.logger.info(f"Validation Accuracy: {val_acc:.2f}%")
        self.logger.info("=" * 50)
        
        # Calculate and log overfitting score
        if len(self.train_accs) > 1:
            overfit_score = (self.train_accs[-1] - self.val_accs[-1]) - (self.train_accs[-2] - self.val_accs[-2])
            self.logger.info(f"\nOverfitting Score: {overfit_score:.4f}")
        
        # Plot and save confusion matrix if predictions available
        if all_preds is not None and all_labels is not None:
            self.plot_confusion_matrix(all_preds, all_labels)
        
        # Plot and save example predictions if available
        if example_images is not None:
            epoch_label = len(self.train_accs) if len(self.train_accs) > 0 else 'final'
            self.plot_example_predictions(example_images, all_preds[:5], all_labels[:5], example_outputs, epoch=epoch_label)
    
    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss over time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.val_accs, label='Val Accuracy')
        plt.title('Accuracy over time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot and close
        plt.savefig(os.path.join(self.plots_dir, 'training_metrics.png'))
        plt.close()
        
        # Save metrics to JSON
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def plot_confusion_matrix(self, preds, labels):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'))
        plt.close()
    
    def plot_example_predictions(self, images, preds, labels, raw_outputs, epoch, num_examples=5):
        # Randomly select indices for display
        num_images = len(images)
        indices = torch.randperm(num_images)[:num_examples]
        
        plt.figure(figsize=(20, 5))
        for i in range(min(num_examples, len(images))):
            plt.subplot(1, num_examples, i + 1)
            idx = indices[i]
            img = images[idx].cpu().permute(1, 2, 0)
            # Denormalize the image
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img = img * std + mean
            img = img.clamp(0, 1)  # Ensure values are within [0, 1]
            plt.imshow(img)
            
            # Get prediction and confidence
            if raw_outputs is not None:
                probs = torch.nn.functional.softmax(raw_outputs[idx], dim=0)
                confidence, pred_idx = torch.max(probs, dim=0)
            else:
                pred_idx = preds[idx] if isinstance(preds[idx], int) else preds[idx].item()
                confidence = None
            
            true_label = labels[idx] if isinstance(labels[idx], int) else labels[idx].item()
            
            # Get breed names if available
            if self.class_names is not None:
                true_breed = self.class_names[true_label]
                pred_breed = self.class_names[pred_idx]
                # Wrap text if too long
                true_breed = '\n'.join([true_breed[j:j+20] for j in range(0, len(true_breed), 20)])
                pred_breed = '\n'.join([pred_breed[j:j+20] for j in range(0, len(pred_breed), 20)])
                title = f'True: {true_breed}\nPred: {pred_breed}'
            else:
                title = f'True: {true_label}\nPred: {pred_idx}'
            
            # Add confidence if available
            if confidence is not None:
                title += f'\nConf: {confidence:.2%}'
            
            # Set color based on prediction correctness
            is_correct = pred_idx == true_label
            color = 'green' if is_correct else 'red'
            
            plt.title(title, color=color, fontsize=12, pad=20)  # Increased font size and padding
            plt.axis('off')
        
        plt.tight_layout(pad=3.0)  # Added padding between subplots
        # Save with epoch number or final
        if epoch == 0:
            filename = 'example_predictions_epoch_0.png'
        elif epoch == 'final':
            filename = 'example_predictions_final.png'
        else:
            filename = f'example_predictions_epoch_{epoch}.png'
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()
