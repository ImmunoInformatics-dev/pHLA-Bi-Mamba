# pHLA-Bi-mamba: Dual-Task Peptide-HLA Binding Prediction

**pHLA-Bi-mamba** is a deep learning framework based on the **Mamba (State Space Model)** architecture for predicting the interaction between human leukocyte antigen (HLA) sequences and peptides.

It leverages a **Bi-directional Mamba** backbone to efficiently capture long-range dependencies in biological sequences and supports two prediction tasks:
1.  **Binding Affinity (BA)** regression.
2.  **Binding Probability (EL/Elution)** classification.

## 🚀 Key Features

* **Efficient Architecture**: Uses Mamba-SSM for linear-time sequence modeling, faster and more memory-efficient than Transformers on long sequences.
* **Bi-directional Modeling**: Captures context from both forward and backward directions of the peptide-HLA complex.
* **Dual-Mode Inference**: Supports predicting Binding Affinity (BA) and Binding Probability (EL) simultaneously or separately.
* **Flexible Input**: Accepts raw amino acid sequences for MHC/HLA and Peptides.

## 🛠️ Installation & Requirements

This project requires Python 3.8+ and PyTorch. The core dependency is `mamba-ssm`, which requires a GPU environment.

### 1. Clone the repository
```bash
git clone [https://github.com/YourUsername/pHLAmamba.git](https://github.com/YourUsername/pHLAmamba.git)
cd pHLAmamba
