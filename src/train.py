import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt


class Train:
    
    def __init__(self, model, train_loader, val_loader, config, device):
      
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # differential learning rates
        self.optimizer = optim.Adam([
            {'params': model.features.parameters(), 
             'lr': config.learning_rate * 0.1},
            {'params': model.classifier.parameters(), 
             'lr': config.learning_rate}
        ])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            if self.config.use_features:
                images, features, labels = batch
                images = images.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = None
            
            self.optimizer.zero_grad()
            outputs = self.model(images, features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                if self.config.use_features:
                    images, features, labels = batch
                    images = images.to(self.device)
                    features = features.to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    features = None
                
                labels = labels.to(self.device)
                outputs = self.model(images, features)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        accuracy = 100 * np.sum(all_labels == all_preds) / len(all_labels)
        
        return epoch_loss, accuracy, all_labels, all_preds
    
    def train(self):

        # checkpoint directory
        checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.config.num_epochs}')
            print('-' * 50)
            
            train_loss, train_acc = self.train_epoch()
            
            val_loss, val_acc, val_labels, val_preds = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # metrics
            cm = confusion_matrix(val_labels, val_preds)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%')
            print(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, checkpoint_dir / 'best_model.pth')
                print('✓ Saved best model')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': self.config.to_dict()
        }, checkpoint_dir / 'final_model.pth')
        
        return self.history
    
    def plot_history(self, save_path: Path = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # loss plot
        ax1.plot(self.history['train_loss'], 'o-', label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], 's-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Aacuracy plot
        ax2.plot(self.history['train_acc'], 'o-', label='Train Acc', linewidth=2)
        ax2.plot(self.history['val_acc'], 's-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Training history saved to: {save_path}")
        
        plt.close()