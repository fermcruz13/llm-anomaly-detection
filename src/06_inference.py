# 06_inference.py
# Inferencia pós-treinamento com LoRA no conjunto de teste

import torch
import time
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

SYSTEM_PROMPT = "You are a cybersecurity log analyzer. Classify log sequences as NORMAL or ANOMALOUS. Respond with only one word."

def load_finetuned_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, "./lora_output/final")
    model.eval()

    print("Modelo fine-tuned carregado!")
    return model, tokenizer

def inference_lora(model, tokenizer, sequences, max_new_tokens=5):
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
                max_new_tokens=max_new_tokens,
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
    model, tokenizer = load_finetuned_model()

    bgl_test = pd.read_csv("data/bgl_test.csv")
    hdfs_test = pd.read_csv("data/hdfs_test.csv")

    bgl_seqs = bgl_test['sequence'].tolist()
    bgl_labels = bgl_test['label'].tolist()
    hdfs_seqs = hdfs_test['sequence'].tolist()
    hdfs_labels = hdfs_test['label'].tolist()

    print("\n=== INFERENCIA FINE-TUNING BGL ===")
    start = time.time()
    bgl_preds = inference_lora(model, tokenizer, bgl_seqs)
    bgl_time = time.time() - start
    bgl_metrics = calculate_metrics(bgl_labels, bgl_preds, "BGL (Fine-Tuning LoRA)")
    bgl_metrics["total_time"] = float(bgl_time)

    print("\n=== INFERENCIA FINE-TUNING HDFS ===")
    start = time.time()
    hdfs_preds = inference_lora(model, tokenizer, hdfs_seqs)
    hdfs_time = time.time() - start
    hdfs_metrics = calculate_metrics(hdfs_labels, hdfs_preds, "HDFS (Fine-Tuning LoRA)")
    hdfs_metrics["total_time"] = float(hdfs_time)

    results = {"bgl": bgl_metrics, "hdfs": hdfs_metrics}
    with open("results/finetuning_lora_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResultados salvos em results/finetuning_lora_results.json")
