"""
PMRAM Bangladeshi Brain Cancer MRI - Local Validation Script
============================================================
Uses the original HSANet model definition to load trained weights.
"""

import sys
from pathlib import Path

# Add code directory to path
CODE_DIR = Path("/Users/tarequejosh/Documents/Research_2026/MRI_updated/files_updated/HSANet_Research/code")
sys.path.insert(0, str(CODE_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import cv2
import warnings
import json
warnings.filterwarnings('ignore')

# Import original model
from hsanet_model import HSANetV2

# Device setup for M2 Mac
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple M2 GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")


# =======================================================================
# Dataset Class
# =======================================================================

class PMRAMDataset(Dataset):
    """PMRAM Bangladeshi Brain Cancer Dataset"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Map PMRAM folder names to class indices
        self.class_mapping = {
            '512Glioma': 0,
            '512Meningioma': 1,
            '512Normal': 2,
            '512Pituitary': 3
        }
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        
        self.samples = []
        for folder_name, class_idx in self.class_mapping.items():
            class_dir = self.root_dir / folder_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.JPEG']:
                    for img_path in class_dir.glob(ext):
                        self.samples.append((str(img_path), class_idx))
        
        print(f"Loaded {len(self.samples)} images from PMRAM dataset")
        
        # Print class distribution
        class_counts = {i: 0 for i in range(4)}
        for _, label in self.samples:
            class_counts[label] += 1
        print("Class distribution:")
        for i, name in enumerate(self.class_names):
            print(f"  {name}: {class_counts[i]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


# =======================================================================
# Validation Function
# =======================================================================

def validate_on_pmram(model, loader, class_names, output_dir):
    model.eval()
    all_preds, all_labels, all_probs, all_epistemic, all_paths = [], [], [], [], []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            
            probs = outputs['probs']
            preds = probs.argmax(dim=1)
            u_epi = outputs['uncertainty_epistemic']
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_epistemic.extend(u_epi.cpu().numpy())
            all_paths.extend(paths)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {(batch_idx + 1) * loader.batch_size}/{len(loader.dataset)} images")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    print("\n" + "="*70)
    print("PMRAM BANGLADESHI BRAIN CANCER DATASET - EXTERNAL VALIDATION RESULTS")
    print("="*70)
    print(f"\n  Accuracy:      {accuracy*100:.2f}%")
    print(f"  F1-Score:      {f1*100:.2f}%")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    
    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Find misclassified samples
    misclassified = []
    for i, (pred, label, prob, path, epi) in enumerate(zip(all_preds, all_labels, all_probs, all_paths, all_epistemic)):
        if pred != label:
            misclassified.append({
                'index': i,
                'path': path,
                'true': class_names[label],
                'pred': class_names[pred],
                'confidence': float(prob[pred]),
                'epistemic': float(epi)
            })
    
    print(f"\nMisclassified: {len(misclassified)} / {len(all_labels)} ({len(misclassified)/len(all_labels)*100:.2f}%)")
    for m in misclassified[:20]:
        print(f"  True: {m['true']:12s} → Pred: {m['pred']:12s} (conf: {m['confidence']:.2%}, u_epi: {m['epistemic']:.3f})")
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('PMRAM Bangladeshi Dataset - Confusion Matrix\n(HSANet External Validation)', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = output_dir / 'pmram_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved: {cm_path}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'kappa': float(kappa),
        'total_samples': len(all_labels),
        'misclassified_count': len(misclassified),
        'misclassified': misclassified
    }
    
    results_path = output_dir / 'pmram_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    
    return results, all_preds, all_labels, all_paths


# =======================================================================
# GradCAM Visualization
# =======================================================================

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
    
    def generate(self, x, target_class=None):
        self.model.eval()
        x.requires_grad = True
        
        out = self.model(x)
        probs = out['probs']
        
        if target_class is None:
            target_class = probs.argmax(dim=1).item()
        
        # Use stored activations from model
        self.activations = self.model.activations
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(probs)
        one_hot[0, target_class] = 1
        probs.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients
        self.gradients = self.model.gradients
        
        if self.gradients is not None and self.activations is not None:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
            cam = cam / (cam.max() + 1e-8)
            cam = nn.functional.interpolate(cam, (224, 224), mode='bilinear')
            return cam.squeeze().detach().cpu().numpy(), probs.squeeze().detach().cpu().numpy()
        else:
            # Fallback if no gradients
            return np.zeros((224, 224)), probs.squeeze().detach().cpu().numpy()


def visualize_gradcam(model, dataset, output_dir, n_samples=12):
    print("\nGenerating GradCAM visualizations...")
    
    gradcam = GradCAM(model)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Sample evenly from dataset
    step = max(1, len(dataset) // n_samples)
    
    for i in range(min(n_samples, 12)):
        idx = i * step
        if idx >= len(dataset):
            break
            
        img, label, path = dataset[idx]
        orig = np.array(Image.open(path).convert('RGB').resize((224, 224)))
        
        try:
            cam, probs = gradcam.generate(img.unsqueeze(0).to(device))
            pred = probs.argmax()
            conf = probs[pred]
            
            if cam.max() > 0:
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
            else:
                overlay = orig
        except Exception as e:
            print(f"  GradCAM failed for sample {i}: {e}")
            overlay = orig
            with torch.no_grad():
                out = model(img.unsqueeze(0).to(device))
                probs = out['probs'].squeeze().cpu().numpy()
            pred = probs.argmax()
            conf = probs[pred]
        
        axes[i].imshow(overlay)
        color = 'green' if pred == label else 'red'
        axes[i].set_title(f"True: {dataset.class_names[label]}\nPred: {dataset.class_names[pred]} ({conf:.0%})", 
                          fontsize=10, color=color)
        axes[i].axis('off')
    
    plt.suptitle("GradCAM Visualization - PMRAM Bangladeshi Dataset\n(HSANet External Validation)", fontsize=14)
    plt.tight_layout()
    gradcam_path = output_dir / 'pmram_gradcam.png'
    plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"GradCAM visualization saved: {gradcam_path}")
    
    return gradcam_path


# =======================================================================
# Main Execution
# =======================================================================

if __name__ == "__main__":
    # Paths
    PMRAM_DIR = Path("/Users/tarequejosh/Documents/Research_2026/MRI_updated/files_updated/PMRAM Bangladeshi Brain Cancer - MRI Dataset/Raw")
    MODEL_DIR = Path("/Users/tarequejosh/Documents/Research_2026/MRI_updated/files_updated/HSANet_Research/models")
    OUTPUT_DIR = Path("/Users/tarequejosh/Documents/Research_2026/MRI_updated/files_updated/HSANet_Research/results/pmram_validation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading PMRAM dataset...")
    dataset = PMRAMDataset(PMRAM_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Load model with original architecture
    print("\nLoading HSANet model (original architecture)...")
    model = HSANetV2(num_classes=4, pretrained=True)
    
    # Load trained weights
    model_path = MODEL_DIR / 'hsanet_final.pth'
    if model_path.exists():
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            metrics = checkpoint.get('metrics', {})
            print(f"Training accuracy: {metrics.get('accuracy', 'N/A'):.2f}%")
        else:
            state_dict = checkpoint
        
        # Rename keys to match our model structure
        # Saved: amsm.0, dam.0, classifier.fc.* 
        # Model: amsm_modules.0, dam_modules.0, classifier.fc.*
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('amsm.', 'amsm_modules.')
            new_key = new_key.replace('dam.', 'dam_modules.')
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded successfully!")
    else:
        print(f"ERROR: No trained weights found at {model_path}")
        exit(1)
    
    model = model.to(device)
    model.eval()
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Run validation
    results, preds, labels, paths = validate_on_pmram(model, loader, dataset.class_names, OUTPUT_DIR)
    
    # Generate GradCAM
    gradcam_path = visualize_gradcam(model, dataset, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"F1-Score: {results['f1']*100:.2f}%")
    print(f"Misclassified: {results['misclassified_count']}/{results['total_samples']}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
