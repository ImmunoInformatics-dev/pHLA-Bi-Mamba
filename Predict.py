#!/usr/bin/env python

# @Time    : 2025/3/22 17:25
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Predict.py

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  

import pHLAmamba
import Tokenizer
import Dataset


def get_args():
    parser = argparse.ArgumentParser(description='Predict peptide-HLA binding with pHLA-Mamba')
    parser.add_argument('--input', type=str, required=True, help='Input CSV with columns ["mhc_seq", "pep"]')
    parser.add_argument('--output', type=str, default="./output/result.csv", help='Path to save the result CSV')
    parser.add_argument('--mode', type=str, default="all", choices=["ba", "el", "all"],
                        help='Mode: "ba" (Affinity), "el" (Probability), or "all"')
    parser.add_argument('--ba_model', type=str, default="./model/epoch20_model.pt",
                        help='Path to the Binding Affinity (BA) model checkpoint')
    parser.add_argument('--el_model', type=str, default="./model/iter29300_model.pt",
                        help='Path to the Elution/Probability (EL) model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Inference batch size')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (e.g., "cuda:0", "cpu")')

    return parser.parse_args()


def run_inference(model, loader, tokenizer, device, is_probability=False):
    model.eval()
    preds = []
    activation = nn.Sigmoid().to(device) if is_probability else None

    with torch.no_grad():
        for hla_seq, pep_seq, len_hla_seq, len_pep_seq in tqdm(loader, desc="Inferring"):
            f_toks, seg_toks, pos = tokenizer.encode(pep_seq, hla_seq, len_pep_seq, len_hla_seq)

            output = model(input_ids=f_toks, seg=seg_toks, pos=pos)
            logits = output.cls_logits

            if is_probability:
                logits = activation(logits)

            preds.extend(logits.view(-1).tolist())

    return preds


if __name__ == "__main__":
    args = get_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {args.input}...")
    tokenizer = Tokenizer.Tokenizer(add_special_token=True, device=device)
    infer_data = Dataset.HLAPEPDataset_infer(file_path=args.input)
    infer_loader = DataLoader(infer_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    infer_df = infer_data.data.copy()  # 复制一份，防止修改原数据对象

    config = pHLAmamba.MambaConfig()
    model = pHLAmamba.MambaLMHeadModel(config=config, device=device)

    # --- Binding Affinity (BA) ---
    if args.mode in ["all", "ba"]:
        print(f"Loading BA model from: {args.ba_model}")
        if os.path.exists(args.ba_model):
            model.load_state_dict(torch.load(args.ba_model, map_location=device))
            print("Predicting Binding Affinity...")
            affs = run_inference(model, infer_loader, tokenizer, device, is_probability=False)
            infer_df['binding_affinity'] = affs
        else:
            print(f"Warning: BA model path {args.ba_model} not found. Skipping.")

    # --- Elution / Probability (EL) ---
    if args.mode in ["all", "el"]:
        print(f"Loading EL model from: {args.el_model}")
        if os.path.exists(args.el_model):
            model.load_state_dict(torch.load(args.el_model, map_location=device))
            print("Predicting Binding Probability...")
            probs = run_inference(model, infer_loader, tokenizer, device, is_probability=True)
            infer_df['binding_probability'] = probs
        else:
            print(f"Warning: EL model path {args.el_model} not found. Skipping.")


    print(f"Saving results to {args.output}")
    infer_df.to_csv(args.output, index=False)
    print("Done.")

