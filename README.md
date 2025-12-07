# **pHLA-Bi-mamba: Dual-Task Peptide-HLA Binding Prediction**

**pHLA-Bi-mamba** is a deep learning framework designed to predict the interaction between human leukocyte antigen (HLA) sequences and peptides. Built upon the **Mamba (State Space Model)** architecture, it efficiently models long biological sequences with linear complexity.

The model utilizes a **Bi-directional Mamba backbone** to capture context from both ends of the peptide-HLA complex and supports two concurrent prediction tasks:

1. **Binding Affinity (BA)**: Regression task predicting the binding strength.  
2. **Binding Probability (EL/Elution)**: Classification task predicting the likelihood of presentation.

## **🚀 Key Features**

* **State Space Model Core**: Leverages Mamba for faster inference and lower memory usage compared to Transformers, especially on long sequences.  
* **Bi-directional Mixing**: A custom Bi-Directional Mixer ensures robust feature extraction from both forward and backward sequence contexts.  
* **Dual-Task Inference**: Flexible architecture allows for simultaneous or independent prediction of affinity and probability.  

## **🛠️ Installation**

This project relies on mamba-ssm, which requires a CUDA-enabled GPU environment.

### **1\. Clone the Repository**

git clone https://github.com/ImmunoInformatics-dev/pHLA-Bi-Mamba.git

cd pHLAmamba

### **2\. Environment Setup**

We recommend using **Conda** to manage dependencies to avoid CUDA version conflicts.

\# 1\. Create a clean environment  
conda create \-n phlamamba python=3.10  
conda activate phlamamba

\# 2\. Install PyTorch (Ensure CUDA version matches your driver, e.g., CUDA 11.8)  
pip install torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/cu118\](https://download.pytorch.org/whl/cu118)

\# 3\. Install Mamba and Causal Conv1d (Essential for the backbone)  
pip install causal-conv1d\>=1.2.0  
pip install mamba-ssm

\# 4\. Install other utilities  
pip install pandas tqdm

## **📥 Model Checkpoints**

Before running inference, ensure your model weights are placed in the model/ directory (or specify their path via arguments).

**Default Expected Structure:**

pHLA-Bi-mamba/  
├── model/  
│   ├── phla_bi_mamba_ba.pt       \# Checkpoint for Binding Affinity (BA)  
│   └── phla_bi_mamba_el.pt     \# Checkpoint for Binding Probability (EL)  
...

## **📂 Data Preparation**

### **Input Format**

The inference script expects a **CSV file** with the following required columns. The order of columns does not matter, but the **headers must match exactly**.

* mhc\_seq: The full amino acid sequence (or pseudo-sequence) of the MHC/HLA.  
* pep: The peptide amino acid sequence.

### **Sample Data (data/test\_input.csv)**

| mhc\_seq | pep |
| :---- | :---- |
| YYSEYRNICTNTYESNLYLRYDYYTWAELAYLWY | LKCAGNEDI |
| YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY | LLFGYPVYV |
| YYAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY | MLSPASSQ |

**Note**: The tokenizer supports standard amino acids. Padding is handled automatically during inference.

## **⚡ Usage / Inference**

We provide Predict.py for flexible inference. You can choose to predict Binding Affinity, Binding Probability, or both.

### **1\. Default Run (Both BA and EL)**

Predicts both metrics using default model paths and saves to ./output/result.csv.

python Predict.py \--input ./data/test.csv

### **2\. Custom Output & Device**

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
