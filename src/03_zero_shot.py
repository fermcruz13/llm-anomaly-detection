# ETAPA 3: Inferência Zero-Shot
# O modelo classifica logs como normal/anômalo sem exemplos prévios

import time
import torch
import json

def zero_shot_inference_instruct(model, tokenizer, sequences, max_length=512):
    """
    Zero-shot usando o template de chat do LLaMA-3-Instruct.
    """
    results = []
    
    system_prompt = """You are a cybersecurity log analyzer. Your task is to classify log sequences as NORMAL or ANOMALOUS.
ANOMALOUS sequences contain: error messages, kernel alerts, fatal errors, failed connections, authentication failures, or unexpected system behavior.
NORMAL sequences contain: routine operations, informational messages, or successful transactions.
Respond with only one word: NORMAL or ANOMALOUS."""

    model.eval()
    print(f"  Processando {len(sequences)} sequencias...")
    
    for i, seq in enumerate(sequences):
        user_message = f"Classify the following log sequence as NORMAL or ANOMALOUS:\n\n{str(seq)[:400]}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        if 'ANOMAL' in response.upper():
            prediction = 1
        else:
            prediction = 0
        
        results.append(prediction)
        
        if (i + 1) % 50 == 0:
            print(f"    Processadas: {i+1}/{len(sequences)}")
    
    return results

print("=" * 60)
print("ZERO-SHOT COM LLAMA-3-8B-INSTRUCT")
print("=" * 60)
print()

# BGL
print("Dataset BGL - Zero-Shot (Instruct):")
start_time = time.time()
bgl_zs_preds = zero_shot_inference_instruct(model, tokenizer, bgl_test_sequences)
bgl_zs_time = time.time() - start_time

print(f"  Tempo total: {bgl_zs_time:.1f}s")
print(f"  Tempo por sequencia: {bgl_zs_time/len(bgl_test_sequences)*1000:.1f}ms")
print()

# HDFS
print("Dataset HDFS - Zero-Shot (Instruct):")
start_time = time.time()
hdfs_zs_preds = zero_shot_inference_instruct(model, tokenizer, hdfs_test_sequences)
hdfs_zs_time = time.time() - start_time

print(f"  Tempo total: {hdfs_zs_time:.1f}s")
print(f"  Tempo por sequencia: {hdfs_zs_time/len(hdfs_test_sequences)*1000:.1f}ms")

# Salvar
zs_results = {
    'bgl_predictions': bgl_zs_preds,
    'bgl_labels': bgl_test_labels,
    'bgl_time': bgl_zs_time,
    'hdfs_predictions': hdfs_zs_preds,
    'hdfs_labels': hdfs_test_labels,
    'hdfs_time': hdfs_zs_time
}

with open('/content/datasets/zero_shot_instruct_results.json', 'w') as f:
    json.dump(zs_results, f)

print("\nResultados zero-shot (Instruct) salvos!")
