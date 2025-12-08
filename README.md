# **pHLA-Bi-mamba: Dual-Task Peptide-HLA Binding Prediction**

**pHLA-Bi-mamba** is a deep learning framework designed to predict the interaction between human leukocyte antigen class I (HLA-I) molecules and peptides. Built upon the **Mamba (Selective State Space Model)** architecture, it efficiently models long biological sequences with linear computational complexity.

The framework utilizes a **Bi-directional Mamba backbone** to capture contextual information from both the peptide and HLA sequences and supports two prediction tasks:

1. **Binding Affinity (BA)**: Regression task predicting peptide-HLA binding strength.  
2. **Binding Probability (EL/Elution)**: Classification task predicting antigen presentation likelihood.

## **🚀 Key Features**

* **State Space Model Core**: Powered by Mamba, enabling faster inference and lower memory usage than Transformer-based models, especially for long sequences.  
* **Bi-directional Sequence Modeling**: A custom Bi-Directional Mamba Mixer captures forward and backward dependencies in peptide-HLA pairs.  
* **Dual-Task Learning**: Supports independent or joint prediction of binding affinity and binding probability.  
* **Pan-Allele Prediction**: Supports diverse HLA alleles using full-length or pseudo-sequence inputs.

## **🛠️ Installation**

This project depends on mamba-ssm, which requires a CUDA-enabled GPU.

### **1\. Clone the Repository**

git clone https://github.com/ImmunoInformatics-dev/pHLA-Bi-Mamba.git

cd pHLAmamba

### **2\. Environment Setup (Recommended: Conda)**

\# 1\. Create and activate environment  
conda create \-n phlamamba python=3.10 -y
conda activate phlamamba

\# 2\. Install PyTorch (CUDA 11.8 Example)
Make sure the CUDA version matches your GPU driver.

pip install torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/cu118\](https://download.pytorch.org/whl/cu118)

\# 3\. Install Mamba and Causal Conv1d (Essential for the backbone)  
pip install causal-conv1d\>=1.2.0  
pip install mamba-ssm

\# 4\. Install other utilities  
pip install pandas tqdm

## **📥 Model Checkpoints**

Before running inference, place the pretrained model weights in the model/ directory (or specify custom paths).

**Default Directory Structure:**

pHLA-Bi-mamba/  
│├─ model/  
│   ├─ phla_bi_mamba_ba.pt       \# Checkpoint for Binding Affinity (BA) model  
│   └─ phla_bi_mamba_el.pt     \# Checkpoint for Binding Probability (EL) model
...

## **📂 Data Preparation**

### **Input Format**

The input file must be a **CSV file** with the following required columns. Column order does not matter, but column names ** must match exactly**.

* mhc\_seq: Full amino acid sequence or HLA pseudo-sequence.  
* pep: Peptide amino acid sequence.

### **Example Input (data/test\_input.csv)**

| mhc\_seq | pep |
| :---- | :---- |
| YYSEYRNICTNTYESNLYLRYDYYTWAELAYLWY | LKCAGNEDI |
| YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY | LLFGYPVYV |
| YYAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY | MLSPASSQ |

**Note**: Only standard amino acids are supported. Padding is handled automatically.

## **⚡ Usage / Inference**

We provide Predict.py for flexible inference. You can choose to predict Binding Affinity, Binding Probability, or both.

### **1\. Default Inference (Both BA and EL)**

Predicts both binding affinity and presentation probability and saves results to ./output/result.csv.

python Predict.py \--input ./data/test.csv

### **2\. Custom Output Path & GPU Device**

Run on a specific GPU (cuda:1) and save results to a specific file.

python Predict.py \\  
    \--input ./data/validation\_set.csv \\  
    \--output ./results/my\_predictions.csv \\  
    \--device cuda:1


### **3\. Single Task Prediction**

Save computation time by predicting only what you need.

**Binding Affinity (BA) Only:**

python Predict.py \--input ./data/test.csv \--mode ba \--output ./output/ba\_only.csv

**Binding Probability (EL) Only:**

python Predict.py \--input ./data/test.csv \--mode el \--output ./output/el\_only.csv




## **📄 License**

This project is licensed under the MIT License \- see the LICENSE file for details.
