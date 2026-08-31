# 03_zero_shot.py
# Inferencia zero-shot no BGL e HDFS

import torch
import time
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# import importlib.util
# import sys
# import os

# spec = importlib.util.spec_from_file_location("load_model", os.path.join(os.path.dirname(__file__), "02_load_model.py"))
# mod = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(mod)
# load_model = mod.load_model

# --- START: Duplicating load_model definition from 02_load_model.py (cell NW1USdohvUao) ---
# This is a workaround due to __file__ not being defined in Colab and 02_load_model.py not being a physical file.
# In a modular setup, load_model should be imported from a separate file or be globally defined in a preceding cell.

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Carregando {model_name}...")
    print("Download de ~5 GB (pode levar 3 a 8 minutos)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
    print(f"\nModelo carregado!")
    print(f"Memoria GPU em uso: {gpu_memory:.1f} GB")

    return model, tokenizer
# --- END: Duplicating load_model definition ---

SYSTEM_PROMPT = "You are a cybersecurity log analyzer. Classify log sequences as NORMAL or ANOMALOUS. Respond with only one word."

def zero_shot_inference(model, tokenizer, sequences):
    model.eval()
    preds = []

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this log sequence:\n{str(seq)[:300]}"}
            ]

            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
            )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            pred = 1 if "ANOMAL" in response.upper() else 0
            preds.append(pred)

            if (i + 1) % 100 == 0:
                print(f"  Processadas: {i+1}/{len(sequences)}")

    return preds

def calculate_metrics(y_true, y_pred, dataset_name):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n  {dataset_name}:")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    FPR:       {fpr:.4f}")

    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

if __name__ == "__main__":
    model, tokenizer = load_model()

    bgl_test = pd.read_csv("data/bgl_test.csv")
    hdfs_test = pd.read_csv("data/hdfs_test.csv")

    bgl_seqs = bgl_test['sequence'].tolist()
    bgl_labels = bgl_test['label'].tolist()
    hdfs_seqs = hdfs_test['sequence'].tolist()
    hdfs_labels = hdfs_test['label'].tolist()

    print("\n=== ZERO-SHOT BGL ===")
    start = time.time()
    bgl_preds = zero_shot_inference(model, tokenizer, bgl_seqs)
    bgl_time = time.time() - start
    bgl_metrics = calculate_metrics(bgl_labels, bgl_preds, "BGL (Zero-Shot)")
    bgl_metrics["total_time"] = float(bgl_time)

    print("\n=== ZERO-SHOT HDFS ===")
    start = time.time()
    hdfs_preds = zero_shot_inference(model, tokenizer, hdfs_seqs)
    hdfs_time = time.time() - start
    hdfs_metrics = calculate_metrics(hdfs_labels, hdfs_preds, "HDFS (Zero-Shot)")
    hdfs_metrics["total_time"] = float(hdfs_time)

    results = {"bgl": bgl_metrics, "hdfs": hdfs_metrics}
    with open("results/zero_shot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResultados salvos em results/zero_shot_results.json")
