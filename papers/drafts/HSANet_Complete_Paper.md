# HSANet: A Hybrid Scale-Attention Network with Evidential Deep Learning for Uncertainty-Aware Brain Tumor Classification

---

**Authors:** Author 1*, Author 2, Author 3, Author 4

**Affiliation:** Department of Computer Science, Institution Name, City, Country

**Corresponding Author:** author1@institution.edu

**Journal:** Computers in Biology and Medicine

---

## Graphical Abstract

![Graphical Abstract: HSANet architecture and key results](figures/graphical_abstract.png)

---

## Highlights

- Novel hybrid scale-attention architecture achieving **99.77% accuracy** on brain tumor classification
- Adaptive multi-scale module with learned input-dependent fusion weights for handling tumor size variation
- Evidential deep learning framework providing calibrated uncertainty quantification from single forward pass
- **External validation on three independent datasets (5,569 samples, 99.59% combined accuracy)** demonstrating robust cross-domain generalization
- Misclassified cases exhibit significantly elevated uncertainty, enabling reliable clinical decision support

---

## Abstract

**Background and Objective:** Reliable classification of brain tumors from magnetic resonance imaging (MRI) remains challenging due to inter-class morphological similarities and the absence of principled uncertainty quantification in existing deep learning approaches. Current methods produce point predictions without meaningful confidence assessment, limiting their utility in safety-critical clinical workflows where knowing what the model doesn't know is as important as the prediction itself.

**Methods:** We propose HSANet, a hybrid scale-attention architecture that synergistically combines adaptive multi-scale feature extraction with evidential learning for uncertainty-aware tumor classification. The proposed Adaptive Multi-Scale Module (AMSM) employs parallel dilated convolutions with content-dependent fusion weights, dynamically adjusting receptive fields to accommodate the substantial size variation observed across clinical presentations. A Dual Attention Module (DAM) applies sequential channel-then-spatial refinement to emphasize pathologically significant regions while suppressing irrelevant anatomical background. Critically, our evidential classification head replaces conventional softmax outputs with Dirichlet distributions, providing decomposed uncertainty estimates that distinguish between inherent data ambiguity (aleatoric) and model knowledge limitations (epistemic).

**Results:** Comprehensive experiments on 7,023 brain MRI scans spanning four diagnostic categories yielded **99.77% accuracy** (95% CI: 99.45–99.93%) with only three misclassifications among 1,311 test samples. The model achieved macro-averaged AUC-ROC of 0.9999 and expected calibration error (ECE) of 0.019, indicating well-calibrated predictions. External validation on three independent datasets—Figshare (n=3,064; 99.90% accuracy), PMRAM (n=1,505; 99.47%), and BRISC 2025 (n=1,000; 99.30%)—totaling 5,569 samples from China, Bangladesh, and Iran, demonstrated exceptional cross-domain generalization with **combined accuracy of 99.59%**. Misclassified samples exhibited significantly elevated epistemic uncertainty (p < 0.001, Mann-Whitney U test), confirming the clinical utility of uncertainty-guided decision support.

**Conclusions:** HSANet achieves state-of-the-art classification accuracy while providing calibrated uncertainty estimates essential for clinical decision support. The combination of adaptive multi-scale processing, attention-based feature refinement, and evidential deep learning offers a principled framework for trustworthy medical image classification. Complete implementation and pretrained weights are publicly available at https://github.com/tarequejosh/HSANet-Brain-Tumor-Classification.

**Keywords:** Brain tumor classification, Deep learning, Uncertainty quantification, Evidential deep learning, Attention mechanism, Multi-scale feature extraction, Medical image analysis

---

## 1. Introduction

Brain tumors represent a formidable diagnostic challenge in clinical oncology, with global surveillance data reporting approximately 308,102 new cases in 2020 alone. The complexity of accurate diagnosis stems from the remarkable diversity of pathological entities—the 2021 World Health Organization (WHO) classification now recognizes over 100 distinct tumor types, each characterized by unique molecular fingerprints and clinical trajectories. Prognostic outcomes vary dramatically across tumor categories: patients diagnosed with glioblastoma face a median survival of merely 14 to 16 months, whereas those with completely resected Grade I meningiomas frequently achieve long-term cure. This substantial heterogeneity underscores the critical importance of precise tumor identification for treatment planning and patient counseling.

Magnetic resonance imaging (MRI) has emerged as the cornerstone of neuro-oncological evaluation, providing superior soft-tissue contrast without ionizing radiation exposure. Expert neuroradiologists integrate multiparametric imaging findings with clinical presentations to formulate diagnoses. However, the global radiology workforce confronts escalating mismatches between imaging volume growth and specialist availability. Documented vacancy rates have reached 29% in major healthcare systems, with projected shortfalls of 40% anticipated by 2027. Interpretive fatigue has been implicated in diagnostic error rates of 3–5% even among experienced specialists, motivating the development of computer-aided diagnostic systems to augment clinical workflows.

### 1.1 Current Challenges

Over the past decade, deep convolutional neural networks (CNNs) have demonstrated considerable promise for automated medical image analysis, particularly when leveraging transfer learning from large-scale natural image datasets. Research groups worldwide have reported encouraging results for brain tumor classification, with accuracies typically ranging between 94% and 99% across various backbone architectures including VGG, ResNet, and the EfficientNet family. Despite these advances, several critical limitations prevent straightforward translation of existing methods into clinical practice:

1. **Scale Variation:** Brain tumors exhibit extraordinary morphological diversity spanning multiple orders of magnitude in spatial extent. Pituitary microadenomas may measure only 2–3 millimeters, whereas glioblastomas frequently exceed 5 centimeters with extensive peritumoral edema.

2. **Background Interference:** Brain MRI volumes contain extensive normal anatomical content that provides no diagnostic value yet dominates image statistics.

3. **Lack of Uncertainty Quantification:** Conventional classifiers produce point predictions without meaningful confidence assessment.

### 1.2 Contributions

This paper presents HSANet (Hybrid Scale-Attention Network), a novel architecture designed to address these limitations systematically. Our contributions include:

1. An **Adaptive Multi-Scale Module (AMSM)** that employs dynamically weighted parallel dilated convolutions, enabling content-dependent receptive field adaptation.

2. A **Dual Attention Module (DAM)** providing sequential channel-then-spatial refinement for tumor-focused feature emphasis.

3. An **evidential classification head** based on Dirichlet distributions that provides principled uncertainty estimates from a single forward pass.

---

## 2. Materials and Methods

### 2.1 Datasets

#### 2.1.1 Primary Dataset (Kaggle Brain Tumor MRI)

The primary dataset comprises 7,023 T1-weighted gadolinium-enhanced MRI scans from diverse clinical sources:
- **Glioma:** 1,621 samples (23.1%)
- **Meningioma:** 1,645 samples (23.4%)
- **No Tumor:** 2,000 samples (28.5%)
- **Pituitary Adenoma:** 1,757 samples (25.0%)

![Sample MRI Images from Primary Dataset - Four tumor types with representative examples](figures/sample_mri_images.png)

#### 2.1.2 External Validation Datasets

**Figshare Dataset (China):** 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients, acquired at Nanfang Hospital and General Hospital of Tianjin Medical University.

**PMRAM Dataset (Bangladesh):** 1,505 T1-weighted MRI slices collected from Ibn Sina Medical College, Dhaka Medical College, and Cumilla Medical College. All four categories: glioma (n=373), meningioma (n=363), no tumor (n=396), and pituitary adenoma (n=373).

**BRISC 2025 Dataset (Iran):** 6,000 T1-weighted brain MRI slices with official test split of 1,000 images: glioma (n=254), meningioma (n=306), no tumor (n=140), and pituitary adenoma (n=300).

### 2.2 HSANet Architecture

The proposed HSANet architecture integrates three key innovations built upon an EfficientNet-B3 backbone:

![HSANet Architecture Diagram - Complete processing pipeline showing backbone, AMSM, DAM, and evidential head](figures/architecture_diagram.png)

#### 2.2.1 Adaptive Multi-Scale Module (AMSM)

![AMSM Module - Multi-scale feature extraction with learned fusion weights](figures/amsm_module.png)

The AMSM employs parallel dilated convolutions with dilation rates d ∈ {1, 2, 4} to capture features at multiple spatial scales. Fusion weights are dynamically computed through a lightweight SE-like attention mechanism:

```
F_multi = w_1 · F_d1 + w_2 · F_d2 + w_4 · F_d4
```

where weights w_i are learned input-dependent.

#### 2.2.2 Dual Attention Module (DAM)

![DAM Module - Sequential channel and spatial attention refinement](figures/dam_module.png)

The DAM applies sequential refinement:
1. **Channel Attention:** Squeeze-and-excitation for feature recalibration
2. **Spatial Attention:** Convolutional spatial refinement for region emphasis

#### 2.2.3 Evidential Classification Head

The evidential head places Dirichlet priors over categorical distributions:

```
p(π | α) = Dir(π | α) = (1/B(α)) ∏ π_k^(α_k - 1)
```

Uncertainty decomposition:
- **Total Uncertainty:** K / S (inverse of evidence sum)
- **Aleatoric Uncertainty:** ∑ p_k(1 - p_k)
- **Epistemic Uncertainty:** Total - Aleatoric

### 2.3 Training Procedure

**Algorithm 1: HSANet Training Procedure**

```
Input: Training set D_train, validation set D_val
Input: Hyperparameters: η_0, λ_1, λ_2, λ_3, T_anneal, T_max, patience

1: Initialize EfficientNet-B3 backbone with ImageNet pretrained weights
2: Initialize AMSM, DAM modules with Kaiming initialization
3: Initialize evidential head with Xavier initialization
4: Freeze backbone parameters for first 5 epochs
5: t ← 0; best_loss ← ∞; wait ← 0
6: for epoch = 1 to T_max do
7:   if epoch = 6 then
8:     Unfreeze backbone with learning rate η_0/10
9:   end if
10:  λ_3^(t) ← min(1, t/T_anneal) · λ_3  // Anneal KL weight
11:  η_t ← η_0 · (1 + cos(π · t/T_max)) / 2  // Cosine LR schedule
12:  for each mini-batch (X, y) in D_train do
13:    X_aug ← Augment(X)  // Data augmentation
14:    α ← HSANet(X_aug)   // Forward pass
15:    Compute L_CE, L_focal, L_KL using Equations (10-12)
16:    L ← λ_1 · L_CE + λ_2 · L_focal + λ_3^(t) · L_KL
17:    ∇_θ L ← Backpropagate(L)
18:    Clip ||∇_θ L||_2 to maximum 1.0
19:    θ ← AdamW(θ, ∇_θ L, η_t)
20:  end for
21:  L_val ← Evaluate(D_val)
22:  if L_val < best_loss then
23:    Save checkpoint; best_loss ← L_val; wait ← 0
24:  else
25:    wait ← wait + 1
26:  end if
27:  if wait ≥ patience then
28:    break  // Early stopping triggered
29:  end if
30:  t ← t + 1
31: end for
32: return Best model checkpoint θ*
```

### 2.4 Evaluation Metrics

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score (macro-averaged)
- Cohen's κ
- Matthews Correlation Coefficient (MCC)
- AUC-ROC (one-vs-rest)

**Calibration Metrics:**
- Expected Calibration Error (ECE)
- Reliability diagrams

**Interpretability:**
- Grad-CAM visualizations

---

## 3. Results

### 3.1 Primary Dataset Performance

HSANet achieved exceptional classification performance on the Kaggle test set:

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.77% (95% CI: 99.45–99.93%) |
| **F1-Score (Macro)** | 99.75% |
| **Precision (Macro)** | 99.76% |
| **Recall (Macro)** | 99.75% |
| **Cohen's κ** | 0.997 |
| **MCC** | 0.997 |
| **AUC-ROC** | 0.9999 |
| **ECE** | 0.019 |

![ROC Curves - Near-perfect AUC for all four tumor classes](figures/roc_curves.png)

![Confusion Matrix - Only 3 misclassifications among 1,311 test samples](figures/confusion_matrix.png)

### 3.2 External Validation Results

External validation on three independent datasets provided strong evidence of cross-domain generalization:

| Dataset | Region | N | Accuracy (%) | F1 (%) | κ |
|---------|--------|---|--------------|--------|---|
| Kaggle (test) | Mixed | 1,311 | 99.77 | 99.75 | 0.997 |
| **External Validation:** |
| Figshare | China | 3,064 | 99.90 | 99.88 | 0.998 |
| PMRAM | Bangladesh | 1,505 | 99.47 | 99.46 | 0.993 |
| BRISC 2025 | Iran | 1,000 | 99.30 | 99.21 | 0.990 |
| **Total External** | **Multi-country** | **5,569** | **99.59** | **99.52** | **0.994** |

![BRISC 2025 Confusion Matrix - 99.30% accuracy on Iranian dataset](figures/brisc_confusion_matrix.png)

![BRISC 2025 Per-Class Accuracy - Consistent performance across all classes](figures/brisc_class_accuracy.png)

HSANet generalizes across diverse populations spanning three continents:
- **99.90%** accuracy on Chinese patients (Figshare)
- **99.47%** accuracy on Bangladeshi patients (PMRAM)
- **99.30%** accuracy on Iranian patients (BRISC 2025)

### 3.3 Calibration and Uncertainty Analysis

![Reliability Diagram - Well-calibrated predictions with ECE = 0.019](figures/reliability_diagram.png)

The extremely low ECE of 0.019 indicates that predicted probabilities accurately reflect true classification accuracy—a 95% model prediction indeed corresponds to approximately 95% actual correctness.

Uncertainty analysis revealed:
- Correctly classified samples: mean uncertainty = 0.023
- Misclassified samples: mean uncertainty = 0.089 (significantly elevated, p < 0.001)

### 3.4 Ablation Study

| Configuration | Accuracy (%) | AUC-ROC | ECE |
|---------------|--------------|---------|-----|
| Backbone only | 98.93 | 0.9997 | 0.058 |
| + AMSM | 99.30 | 0.9999 | 0.041 |
| + DAM | 99.54 | 0.9999 | 0.032 |
| + Evidential Head | **99.77** | **0.9999** | **0.019** |

### 3.5 GradCAM Visualizations

![GradCAM Grid - Attention maps confirming focus on tumor regions](figures/gradcam_grid.png)

GradCAM visualizations confirm that HSANet appropriately focuses on clinically relevant tumor regions:
- **Glioma:** Attention on infiltrative parenchymal masses
- **Meningioma:** Attention on dural-based extra-axial masses
- **No Tumor:** Diffuse attention without focal concentration
- **Pituitary:** Attention on sellar/suprasellar region

### 3.6 Comparison with State-of-the-Art

| Method | Year | Accuracy (%) | F1 (%) | Ext.Val. | Uncertainty |
|--------|------|--------------|--------|----------|-------------|
| VGG-16 | 2019 | 95.30 | 94.80 | ✗ | ✗ |
| ResNet-50 | 2020 | 96.10 | 95.90 | ✗ | ✗ |
| EfficientNet-B3 | 2020 | 97.70 | 97.50 | ✗ | ✗ |
| Swin Transformer | 2021 | 98.50 | 98.30 | ✗ | ✗ |
| ViT-B/16 | 2021 | 98.20 | 98.00 | ✗ | ✗ |
| **HSANet (Ours)** | 2025 | **99.77** | **99.75** | **✓** | **✓** |

---

## 4. Discussion

### 4.1 Key Findings

The experimental results demonstrate that HSANet successfully addresses the three critical challenges identified in existing brain tumor classification approaches:

1. **Multi-scale processing:** The AMSM with content-dependent fusion weights enables effective handling of the substantial size variation across tumor types.

2. **Attention-guided feature refinement:** The DAM successfully emphasizes pathologically significant regions while suppressing irrelevant anatomical background.

3. **Principled uncertainty quantification:** The evidential framework provides calibrated uncertainty estimates that significantly elevate for misclassified samples.

### 4.2 Clinical Implications

The combination of near-perfect classification accuracy and principled uncertainty quantification positions HSANet as a valuable tool for clinical decision support. The finding that misclassified samples exhibit significantly elevated uncertainty (p < 0.001) enables a reliable triage workflow: high-confidence predictions can proceed through automated pathways while uncertain cases are routed for specialist review.

### 4.3 External Validation

The exceptional performance across three geographically diverse external datasets (99.59% combined accuracy on 5,569 samples) provides strong evidence that HSANet learns genuinely tumor-specific features rather than dataset-specific artifacts. This cross-domain generalization is essential for clinical deployment in real-world healthcare settings.

### 4.4 Limitations and Future Work

- Validation on larger multi-institutional datasets
- Extension to multi-parametric MRI sequences
- Integration with segmentation for volumetric analysis
- Prospective clinical trials

---

## 5. Conclusions

We presented HSANet, a hybrid scale-attention network achieving **99.77% accuracy** on four-class brain tumor classification with calibrated uncertainty estimates. The proposed architecture integrates three complementary innovations:

1. An Adaptive Multi-Scale Module with input-dependent fusion weights
2. A Dual Attention Module for feature refinement
3. An evidential classification head enabling principled uncertainty decomposition

External validation on three independent datasets from China, Bangladesh, and Iran (n=5,569 total; **99.59% combined accuracy**) demonstrates robust cross-domain generalization across diverse patient populations and acquisition protocols. Error analysis confirms that misclassified cases exhibit significantly elevated uncertainty that would trigger human review in clinical workflows.

Complete source code and pretrained models are publicly available at:
**https://github.com/tarequejosh/HSANet-Brain-Tumor-Classification**

---

## Acknowledgments

*To be completed by authors*

---

## References

1. Sung H, et al. Global cancer statistics 2020. CA Cancer J Clin. 2021;71(3):209-249.
2. Louis DN, et al. The 2021 WHO classification of tumors of the central nervous system. Neuro-oncology. 2021;23(8):1231-1251.
3. Ostrom QT, et al. CBTRUS statistical report. Neuro-oncology. 2021;23:iii1-iii105.
4. Pope WB. Brain tumor imaging. Semin Neurol. 2018;38(1):11-24.
5. Deepak S, Ameer PM. Brain tumor classification using deep CNN features via transfer learning. Comput Biol Med. 2019;111:103345.
6. Badža MM, Barjaktarović MČ. Classification of brain tumors from MRI images using a CNN. Appl Sci. 2020;10(6):1999.
7. Swati ZNK, et al. Brain tumor classification for MR images using transfer learning and fine-tuning. Comput Med Imaging Graph. 2019;75:34-46.
8. Sensoy M, et al. Evidential deep learning to quantify classification uncertainty. NeurIPS. 2018.
9. Cheng J, et al. Enhanced performance of brain tumor classification via tumor region augmentation and partition. PLOS ONE. 2015;10(10):e0140381.
10. Fateh A, et al. BRISC: Annotated dataset for brain tumor segmentation and classification. arXiv:2506.14318. 2025.

---

*Paper compiled: February 2026*
*Total Figures: 12*
*Total Tables: 6*
