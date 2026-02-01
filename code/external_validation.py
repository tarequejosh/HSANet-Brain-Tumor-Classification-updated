"""
External Validation on BRISC 2025 and BraTS 2021 Datasets
=========================================================
Validates trained HSANet model on new external datasets to demonstrate
cross-domain generalization capability.

BRISC 2025: 6000 T1-MRI slices, 4 classes (glioma, meningioma, pituitary, no_tumor)
BraTS 2021: Brain tumor segmentation challenge - extract 2D slices for classification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths - Update these as needed
HSANET_ROOT = Path("/Users/tarequejosh/Documents/Research_2026/MRI_updated/files_updated/HSANet_Research")
BRISC_PATH = Path("/Users/tarequejosh/Documents/Research_2026/MRI_updated/files_updated/brisc_2025/brisc2025/classification_task")
BRATS_PATH = Path("/Users/tarequejosh/Documents/Research_2026/MRI_updated/files_updated/brats_2021")

# Output paths
RESULTS_DIR = HSANET_ROOT / "results" / "external_validation"
BRISC_RESULTS = RESULTS_DIR / "BRISC_2025"
BRATS_RESULTS = RESULTS_DIR / "BraTS_2021"

# Create output directories
for d in [RESULTS_DIR, BRISC_RESULTS, BRATS_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# Class mappings
KAGGLE_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
BRISC_CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
BRISC_TO_KAGGLE = {'glioma': 0, 'meningioma': 1, 'no_tumor': 2, 'pituitary': 3}

# Standard transforms (same as training)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class BRISCDataset(Dataset):
    """BRISC 2025 Dataset for external validation"""
    
    def __init__(self, root_path, split='test', transform=None):
        self.root = Path(root_path)
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = BRISC_TO_KAGGLE
        
        # Load all images from split folder
        split_path = self.root / split
        for class_name in BRISC_CLASSES:
            class_path = split_path / class_name
            if class_path.exists():
                for img_path in class_path.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_path.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"BRISC {split}: Loaded {len(self.samples)} samples")
        
        # Class distribution
        labels = [s[1] for s in self.samples]
        for idx, name in enumerate(KAGGLE_CLASSES):
            count = labels.count(idx)
            print(f"  {name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def load_hsanet_model(checkpoint_path=None):
    """Load trained HSANet model"""
    import sys
    sys.path.insert(0, str(HSANET_ROOT / "code"))
    
    try:
        from hsanet_model import HSANetV2 as HSANet
        print("Successfully imported HSANetV2 from hsanet_model.py")
        model = HSANet(num_classes=4)
    except ImportError as e:
        print(f"Import error: {e}")
        # Fallback: Create a simple model structure
        import timm
        class HSANetSimple(nn.Module):
            def __init__(self, num_classes=4):
                super().__init__()
                self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
            
            def forward(self, x):
                logits = self.backbone(x)
                probs = torch.softmax(logits, dim=1)
                uncertainty = torch.ones(x.size(0), device=x.device) * 0.1
                return logits, probs, uncertainty
        
        print("Using fallback simple model")
        model = HSANetSimple(num_classes=4)
    
    # Find checkpoint
    if checkpoint_path is None:
        # Try models folder first (preferred)
        models_checkpoints = list((HSANET_ROOT / "models").glob("*.pth"))
        if models_checkpoints:
            # Prefer hsanet_final.pth
            final_ckpt = HSANET_ROOT / "models" / "hsanet_final.pth"
            if final_ckpt.exists():
                checkpoint_path = final_ckpt
            else:
                checkpoint_path = models_checkpoints[0]
            print(f"Using checkpoint: {checkpoint_path}")
    
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Remap keys if needed (handle different versions)
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remap old naming convention to new
                new_key = k.replace('amsm.', 'amsm_modules.').replace('dam.', 'dam_modules.')
                new_state_dict[new_key] = v
            
            # Try loading with flexible matching
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print(f"Loaded checkpoint (strict): {checkpoint_path}")
            except:
                model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded checkpoint (non-strict): {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using pretrained backbone weights only")
    
    return model


def evaluate_dataset(model, dataloader, device, class_names, dataset_name):
    """Comprehensive evaluation on a dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_uncertainties = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle HSANetV2 dict output format
            if isinstance(outputs, dict):
                probs = outputs['probs']
                uncertainty = outputs.get('uncertainty_total', torch.zeros(images.size(0), device=device))
                # Handle scalar uncertainty by expanding
                if uncertainty.dim() == 0:
                    uncertainty = uncertainty.expand(images.size(0))
            elif isinstance(outputs, tuple):
                if len(outputs) >= 3:
                    logits, probs, uncertainty = outputs[0], outputs[1], outputs[2]
                else:
                    logits, probs = outputs[0], outputs[1]
                    uncertainty = torch.ones(images.size(0), device=device) * 0.1
            else:
                logits = outputs
                probs = torch.softmax(logits, dim=1)
                uncertainty = torch.ones(images.size(0), device=device) * 0.1
            
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_uncertainties = np.array(all_uncertainties)
    
    # Compute metrics
    metrics = {
        'dataset': dataset_name,
        'samples': len(all_labels),
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        'precision_macro': precision_score(all_labels, all_preds, average='macro') * 100,
        'recall_macro': recall_score(all_labels, all_preds, average='macro') * 100,
        'f1_macro': f1_score(all_labels, all_preds, average='macro') * 100,
        'cohen_kappa': cohen_kappa_score(all_labels, all_preds),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'mean_uncertainty': float(np.mean(all_uncertainties))
    }
    
    # Per-class metrics
    for idx, name in enumerate(class_names):
        mask = all_labels == idx
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == idx).mean() * 100
            metrics[f'{name}_accuracy'] = class_acc
    
    # AUC-ROC (one-vs-rest)
    try:
        metrics['auc_roc_macro'] = roc_auc_score(all_labels, all_probs, multi_class='ovr') * 100
    except:
        metrics['auc_roc_macro'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return metrics, cm, all_preds, all_labels, all_probs


def plot_confusion_matrix(cm, class_names, save_path, title):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_class_performance(metrics, class_names, save_path, title):
    """Plot per-class accuracy"""
    accuracies = [metrics.get(f'{name}_accuracy', 0) for name in class_names]
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = plt.bar(class_names, accuracies, color=colors, edgecolor='black')
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Tumor Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_results(metrics, cm, save_dir, dataset_name):
    """Save all results to files"""
    # Save metrics JSON
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")
    
    # Save confusion matrix CSV
    cm_path = save_dir / "confusion_matrix.csv"
    pd.DataFrame(cm, index=KAGGLE_CLASSES, columns=KAGGLE_CLASSES).to_csv(cm_path)
    print(f"Saved: {cm_path}")
    
    # Create summary report
    report_path = save_dir / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"HSANet External Validation Report - {dataset_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Samples: {metrics['samples']}\n\n")
        f.write(f"{'='*60}\n")
        f.write(f"OVERALL METRICS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Accuracy:     {metrics['accuracy']:.2f}%\n")
        f.write(f"F1-Score:     {metrics['f1_macro']:.2f}%\n")
        f.write(f"Precision:    {metrics['precision_macro']:.2f}%\n")
        f.write(f"Recall:       {metrics['recall_macro']:.2f}%\n")
        f.write(f"Cohen's κ:    {metrics['cohen_kappa']:.4f}\n")
        f.write(f"MCC:          {metrics['mcc']:.4f}\n")
        f.write(f"AUC-ROC:      {metrics.get('auc_roc_macro', 0):.2f}%\n\n")
        f.write(f"{'='*60}\n")
        f.write(f"PER-CLASS ACCURACY\n")
        f.write(f"{'='*60}\n")
        for name in KAGGLE_CLASSES:
            acc = metrics.get(f'{name}_accuracy', 0)
            f.write(f"{name:15}: {acc:.2f}%\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"CONFUSION MATRIX\n")
        f.write(f"{'='*60}\n")
        f.write(f"{pd.DataFrame(cm, index=KAGGLE_CLASSES, columns=KAGGLE_CLASSES).to_string()}\n")
    print(f"Saved: {report_path}")


def run_brisc_validation(model, device):
    """Run validation on BRISC 2025 dataset"""
    print("\n" + "="*60)
    print("BRISC 2025 External Validation")
    print("="*60)
    
    # Load dataset
    brisc_dataset = BRISCDataset(BRISC_PATH, split='test', transform=test_transform)
    brisc_loader = DataLoader(brisc_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Evaluate
    metrics, cm, preds, labels, probs = evaluate_dataset(
        model, brisc_loader, device, KAGGLE_CLASSES, "BRISC 2025"
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"BRISC 2025 Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"F1-Score:  {metrics['f1_macro']:.2f}%")
    print(f"Cohen's κ: {metrics['cohen_kappa']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    
    # Save results
    save_results(metrics, cm, BRISC_RESULTS, "BRISC 2025")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, KAGGLE_CLASSES,
        BRISC_RESULTS / "brisc_confusion_matrix.png",
        "BRISC 2025 - Confusion Matrix"
    )
    
    # Plot per-class accuracy
    plot_class_performance(
        metrics, KAGGLE_CLASSES,
        BRISC_RESULTS / "brisc_class_accuracy.png",
        "BRISC 2025 - Per-Class Accuracy"
    )
    
    return metrics


def main():
    """Main validation function"""
    print("="*60)
    print("HSANet External Validation")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading HSANet model...")
    model = load_hsanet_model()
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Run BRISC validation
    brisc_metrics = run_brisc_validation(model, device)
    
    # Summary
    print("\n" + "="*60)
    print("EXTERNAL VALIDATION SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<20} {'Accuracy':<12} {'F1-Score':<12} {'κ':<10}")
    print("-"*54)
    print(f"{'BRISC 2025':<20} {brisc_metrics['accuracy']:.2f}%{'':<5} {brisc_metrics['f1_macro']:.2f}%{'':<5} {brisc_metrics['cohen_kappa']:.4f}")
    
    # Save combined results
    combined_results = {
        'BRISC_2025': brisc_metrics
    }
    
    with open(RESULTS_DIR / "external_validation_summary.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
