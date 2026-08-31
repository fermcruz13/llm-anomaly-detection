# ETAPA 4: Inferência Few-Shot com 5 exemplos
# O modelo recebe 5 exemplos rotulados no prompt antes de classificar

import time
import torch
import json
import random
import pandas as pd

def get_few_shot_examples(train_df, n_examples=5, random_state=42):
    """
    Seleciona exemplos balanceados (normais e anômalos) do treino.
    """
    random.seed(random_state)
    
    normal_samples = train_df[train_df['label'] == 0].sample(
        n=min(n_examples // 2 + 1, len(train_df[train_df['label'] == 0])),
        random_state=random_state
    )
    anomaly_samples = train_df[train_df['label'] == 1].sample(
        n=min(n_examples // 2, len(train_df[train_df['label'] == 1])),
        random_state=random_state
    )
    
    examples = []
    for _, row in pd.concat([normal_samples, anomaly_samples]).iterrows():
        label_text = "ANOMALOUS" if row['label'] == 1 else "NORMAL"
        examples.append({
            'sequence': str(row['sequence'])[:300],
            'label': label_text
        })
    
    return examples

def few_shot_inference(model, tokenizer, sequences, examples, max_length=1024):
    """
    Realiza inferência few-shot: classificação com exemplos no prompt.
    """
    results = []
    
    examples_text = ""
    for ex in examples:
        examples_text += f"Log sequence: {ex['sequence']}\nClassification: {ex['label']}\n\n"
    
    few_shot_prompt = """Analyze the following log sequences and classify each as NORMAL or ANOMALOUS.
A log sequence is ANOMALOUS if it contains errors, kernel alerts, fatal errors, or failed connections.
A log sequence is NORMAL if it contains only routine operations or informational messages.

Here are some examples:

{examples}

Now classify this new log sequence:

Log sequence: {log_seq}

Classification (respond with only NORMAL or ANOMALOUS):"""
    
    model.eval()
    print(f"  Processando {len(sequences)} sequencias...")
    
    for i, seq in enumerate(sequences):
        prompt = few_shot_prompt.format(
            examples=examples_text,
            log_seq=str(seq)[:300]
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
print("INFERENCIA FEW-SHOT (5 exemplos)")
print("=" * 60)
print()

# BGL
print("Dataset BGL - Few-Shot:")
bgl_examples = get_few_shot_examples(bgl_train, n_examples=5)
start_time = time.time()
bgl_few_shot_preds = few_shot_inference(model, tokenizer, bgl_test_sequences, bgl_examples)
bgl_fs_time = time.time() - start_time

print(f"  Tempo total: {bgl_fs_time:.1f}s")
print(f"  Tempo por sequencia: {bgl_fs_time/len(bgl_test_sequences)*1000:.1f}ms")
print()

# HDFS
print("Dataset HDFS - Few-Shot:")
hdfs_examples = get_few_shot_examples(hdfs_train, n_examples=5)
start_time = time.time()
hdfs_few_shot_preds = few_shot_inference(model, tokenizer, hdfs_test_sequences, hdfs_examples)
hdfs_fs_time = time.time() - start_time

print(f"  Tempo total: {hdfs_fs_time:.1f}s")
print(f"  Tempo por sequencia: {hdfs_fs_time/len(hdfs_test_sequences)*1000:.1f}ms")

# Salvar
few_shot_results = {
    'bgl_predictions': bgl_few_shot_preds,
    'bgl_labels': bgl_test_labels,
    'bgl_time': bgl_fs_time,
    'hdfs_predictions': hdfs_few_shot_preds,
    'hdfs_labels': hdfs_test_labels,
    'hdfs_time': hdfs_fs_time
}

with open('/content/datasets/few_shot_results.json', 'w') as f:
    json.dump(few_shot_results, f)

print("\nResultados few-shot salvos!")
