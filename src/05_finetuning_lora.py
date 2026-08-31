# 05_finetuning_lora.py
# Fine-tuning do LLaMA-3-8B-Instruct com LoRA

import torch
import time
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from src_02_load_model import load_model

import warnings
warnings.filterwarnings('ignore')

SYSTEM_PROMPT = "You are a cybersecurity log analyzer. Classify log sequences as NORMAL or ANOMALOUS. Respond with only one word."

def prepare_training_data(train_df, dataset_name):
    formatted = []
    for _, row in train_df.iterrows():
        label = "ANOMALOUS" if row['label'] == 1 else "NORMAL"
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this log sequence:\n{str(row['sequence'])[:300]}"},
                {"role": "assistant", "content": label}
            ]
        })
    print(f"  {dataset_name}: {len(formatted)} exemplos de treino")
    return formatted

def format_for_training(examples, tokenizer, max_length=512):
    input_ids_list = []
    labels_list = []

    for ex in examples:
        prompt = tokenizer.apply_chat_template(
            ex["messages"][:2], add_generation_prompt=True, tokenize=False
        )
        full_text = tokenizer.apply_chat_template(
            ex["messages"], add_generation_prompt=False, tokenize=False
        )

        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": [[1] * len(ids) for ids in input_ids_list]
    })

if __name__ == "__main__":
    model, tokenizer = load_model()

    bgl_train = pd.read_csv("data/bgl_train.csv")
    hdfs_train = pd.read_csv("data/hdfs_train.csv")

    print("\nPreparando dados de treino...")
    bgl_train_data = prepare_training_data(bgl_train, "BGL")
    hdfs_train_data = prepare_training_data(hdfs_train, "HDFS")

    all_train_data = bgl_train_data + hdfs_train_data
    print(f"\nTotal combinado: {len(all_train_data)} exemplos")

    print("Tokenizando dados...")
    train_dataset = format_for_training(all_train_data, tokenizer)
    print(f"Dataset tokenizado: {len(train_dataset)} exemplos")

    print("\nConfigurando LoRA...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 50)
    print("INICIANDO TREINAMENTO...")
    print("=" * 50)

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    print(f"\nTreinamento concluido em {train_time:.1f}s ({train_time/60:.1f} min)")

    trainer.save_model("./lora_output/final")
    tokenizer.save_pretrained("./lora_output/final")
    print("Modelo LoRA salvo em ./lora_output/final")
