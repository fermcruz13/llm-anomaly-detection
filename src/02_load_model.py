# 02_load_model.py
# Carrega o LLaMA-3-8B-Instruct com quantizacao 4-bit NF4

import torch
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

if __name__ == "__main__":
    model, tokenizer = load_model()
