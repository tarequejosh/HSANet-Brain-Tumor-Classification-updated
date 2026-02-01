"""
PMRAM Bangladeshi Brain Cancer MRI Dataset - External Validation
================================================================
Validates HSANet model on an independent Bangladeshi dataset
from Daffodil International University / Dhaka Medical College.

Dataset: https://data.mendeley.com/datasets/m7w55sw88b/1
Raw images: 1,505 (4 classes)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, roc_auc_score, cohen_kappa_score)
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =======================================================================
# Dataset Class
# =======================================================================

class PMRAMDataset(Dataset):
    """PMRAM Bangladeshi Brain Cancer Dataset"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Map PMRAM folder names to our class indices
        self.class_mapping = {
            '512Glioma': 0,      # glioma
            '512Meningioma': 1,  # meningioma
            '512Normal': 2,      # no tumor
            '512Pituitary': 3    # pituitary
        }
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        
        self.samples = []
        for folder_name, class_idx in self.class_mapping.items():
            class_dir = self.root_dir / folder_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
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
# GradCAM Implementation
# =======================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        # Handle evidential output
        if isinstance(output, dict):
            probs = output['probs']
        else:
            probs = torch.softmax(output, dim=1)
        
        if target_class is None:
            target_class = probs.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(probs)
        one_hot[0, target_class] = 1
        probs.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear')
        return cam.squeeze().cpu().numpy(), probs.squeeze().detach().cpu().numpy()


# =======================================================================
# Validation Function
# =======================================================================

def validate_on_pmram(model, data_loader, class_names):
    """Run validation and compute metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    uncertainties = []
    
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Handle evidential output
            if isinstance(outputs, dict):
                probs = outputs['probs']
                if 'uncertainty_epistemic' in outputs:
                    uncertainties.extend(outputs['uncertainty_epistemic'].cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
            
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    print("\n" + "="*60)
    print("PMRAM BANGLADESHI DATASET - EXTERNAL VALIDATION RESULTS")
    print("="*60)
    print(f"\nAccuracy:      {accuracy*100:.2f}%")
    print(f"F1-Score:      {f1*100:.2f}%")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"AUC-ROC:       {auc:.4f}")
    
    print("\n" + "-"*60)
    print("Classification Report:")
    print("-"*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Find misclassified samples
    misclassified = []
    for i, (pred, label, prob) in enumerate(zip(all_preds, all_labels, all_probs)):
        if pred != label:
            misclassified.append({
                'index': i,
                'true': class_names[label],
                'pred': class_names[pred],
                'confidence': prob[pred]
            })
    
    print(f"\nMisclassified: {len(misclassified)} / {len(all_labels)}")
    for m in misclassified[:10]:  # Show first 10
        print(f"  True: {m['true']:12s} → Pred: {m['pred']:12s} (conf: {m['confidence']:.2%})")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'kappa': kappa,
        'auc': auc,
        'confusion_matrix': cm,
        'misclassified': misclassified
    }


# =======================================================================
# GradCAM Visualization
# =======================================================================

def visualize_gradcam_samples(model, dataset, gradcam, class_names, n_samples=12, save_path=None):
    """Generate GradCAM visualizations for sample images"""
    model.eval()
    
    # Select samples from each class
    samples_per_class = n_samples // 4
    selected_indices = []
    class_counts = {i: 0 for i in range(4)}
    
    for idx, (_, label, _) in enumerate(dataset):
        if class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if sum(class_counts.values()) >= n_samples:
            break
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_indices[:12]):
        image, label, img_path = dataset[idx]
        
        # Get original image for visualization
        orig_img = Image.open(img_path).convert('RGB')
        orig_img = orig_img.resize((224, 224))
        orig_np = np.array(orig_img)
        
        # Generate GradCAM
        input_tensor = image.unsqueeze(0).to(device)
        cam, probs = gradcam.generate(input_tensor)
        
        pred = probs.argmax()
        confidence = probs[pred]
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
        
        axes[i].imshow(overlay)
        color = 'green' if pred == label else 'red'
        axes[i].set_title(f"True: {class_names[label]}\nPred: {class_names[pred]} ({confidence:.0%})", 
                          fontsize=10, color=color)
        axes[i].axis('off')
    
    plt.suptitle("GradCAM Visualization - PMRAM Bangladeshi Dataset", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"GradCAM visualization saved to: {save_path}")
    
    plt.show()


# =======================================================================
# Main Execution
# =======================================================================

if __name__ == "__main__":
    # Paths - UPDATE THESE
    PMRAM_RAW_DIR = '/kaggle/input/pmram-bangladeshi-brain-cancer/Raw'  # Kaggle path
    MODEL_PATH = '/kaggle/working/hsanet_final.pth'  # Or your model path
    OUTPUT_DIR = '/kaggle/working/'
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = PMRAMDataset(PMRAM_RAW_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Load your HSANet model
    # model = HSANetV2(num_classes=4)
    # model.load_state_dict(torch.load(MODEL_PATH))
    # model = model.to(device)
    
    # Run validation
    # results = validate_on_pmram(model, loader, dataset.class_names)
    
    # GradCAM visualization
    # For EfficientNet-B3 backbone, target the last conv layer
    # gradcam = GradCAM(model, model.backbone._conv_head)  # Adjust based on your model
    # visualize_gradcam_samples(model, dataset, gradcam, dataset.class_names, 
    #                          save_path=f'{OUTPUT_DIR}/pmram_gradcam.png')
    
    print("\n=== Script loaded. Uncomment the model loading section to run. ===")
