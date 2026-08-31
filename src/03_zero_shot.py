# ETAPA 3: Inferência Zero-Shot
# O modelo classifica logs como normal/anômalo sem exemplos prévios

import time
import torch
import json

def zero_shot_inference(model, tokenizer, sequences, max_length=512):
    """
    Realiza inferência zero-shot: classificação binária sem exemplos.
    """
    results = []
    
    zero_shot_prompt = """Analyze the following log sequence and classify it as NORMAL or ANOMALOUS.
A log sequence is ANOMALOUS if it contains error messages, kernel alerts, fatal errors, 
failed connections, or unexpected system behavior.
A log sequence is NORMAL if it contains only routine operations, informational messages,
or successful transactions.

Log sequence: {log_seq}

Classification (respond with only NORMAL or ANOMALOUS):"""
    
    model.eval()
    
    print(f"  Processando {len(sequences)} sequencias...")
    
    for i, seq in enumerate(sequences):
        prompt = zero_shot_prompt.format(log_seq=str(seq)[:400])
        
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
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        if 'ANOMAL' in response.upper():
            prediction = 1
        else:
            prediction = 0
        
        results.append(prediction)
        
        if (i + 1) % 50 == 0:
            print(f"    Processadas: {i+1}/{len(sequences)}")
    
    return results

print("=" * 60)
print("INFERENCIA ZERO-SHOT")
print("=" * 60)
print()

# Testar no BGL
print("Dataset BGL - Zero-Shot:")
bgl_test_sequences = bgl_test['sequence'].tolist()
bgl_test_labels = bgl_test['label'].tolist()

start_time = time.time()
bgl_zero_shot_preds = zero_shot_inference(model, tokenizer, bgl_test_sequences)
bgl_zs_time = time.time() - start_time

print(f"  Tempo total: {bgl_zs_time:.1f}s")
print(f"  Tempo por sequencia: {bgl_zs_time/len(bgl_test_sequences)*1000:.1f}ms")
print()

# Testar no HDFS
print("Dataset HDFS - Zero-Shot:")
hdfs_test_sequences = hdfs_test['sequence'].tolist()
hdfs_test_labels = hdfs_test['label'].tolist()

start_time = time.time()
hdfs_zero_shot_preds = zero_shot_inference(model, tokenizer, hdfs_test_sequences)
hdfs_zs_time = time.time() - start_time

print(f"  Tempo total: {hdfs_zs_time:.1f}s")
print(f"  Tempo por sequencia: {hdfs_zs_time/len(hdfs_test_sequences)*1000:.1f}ms")

# Salvar resultados
zero_shot_results = {
    'bgl_predictions': bgl_zero_shot_preds,
    'bgl_labels': bgl_test_labels,
    'bgl_time': bgl_zs_time,
    'hdfs_predictions': hdfs_zero_shot_preds,
    'hdfs_labels': hdfs_test_labels,
    'hdfs_time': hdfs_zs_time
}

with open('/content/datasets/zero_shot_results.json', 'w') as f:
    json.dump(zero_shot_results, f)

print("\nResultados zero-shot salvos!")
