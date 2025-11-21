import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
import json
from pathlib import Path


class Evaluate:
    
    def __init__(self, model, test_loader, config, device):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.results_dir = Path(config.output_dir) / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self):
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluating'):
                if self.config.use_features:
                    images, features, labels = batch
                    images = images.to(self.device)
                    features = features.to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    features = None
                
                outputs = self.model(images, features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        self._plot_confusion_matrix(all_labels, all_preds)
        self._plot_roc_curve(all_labels, all_probs, metrics['auc'])
        
        self._save_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self, labels, preds, probs):
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) \
             if (precision + sensitivity) > 0 else 0
        
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        return {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1),
            'auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def _plot_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Cataract'],
                    yticklabels=['Normal', 'Cataract'],
                    annot_kws={'size': 16})
        plt.title('Confusion Matrix - Test Set', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        
        save_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Confusion matrix saved to: {save_path}")
    
    def _plot_roc_curve(self, labels, probs, roc_auc):
        fpr, tpr, _ = roc_curve(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.results_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" ROC curve saved to: {save_path}")
    
    def _save_metrics(self, metrics):
        save_path = self.results_dir / 'metrics.json'
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"✓ Metrics saved to: {save_path}")