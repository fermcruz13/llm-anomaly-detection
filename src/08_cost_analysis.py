# 08_cost_analysis.py
# Analise de custo computacional das tres estrategias

import torch
import json
import os

bgl_test_size = 1100
hdfs_test_size = 1680

strategies = {
    "Zero-Shot": {
        "bgl": {"total_time": 653.9, "n_seqs": bgl_test_size},
        "hdfs": {"total_time": 736.9, "n_seqs": hdfs_test_size},
    },
    "Few-Shot": {
        "bgl": {"total_time": 942.7, "n_seqs": bgl_test_size},
        "hdfs": {"total_time": 1586.6, "n_seqs": hdfs_test_size},
    },
    "Fine-Tuning LoRA": {
        "bgl": {"total_time": 394.5, "n_seqs": bgl_test_size},
        "hdfs": {"total_time": 563.5, "n_seqs": hdfs_test_size},
    },
}

def calculate_cost_metrics(strategy_data):
    total_time = strategy_data["total_time"]
    n_seqs = strategy_data["n_seqs"]

    latency_ms = (total_time / n_seqs) * 1000
    throughput = n_seqs / total_time

    return {
        "total_time_s": round(total_time, 1),
        "n_sequences": n_seqs,
        "latency_ms_per_seq": round(latency_ms, 2),
        "throughput_seqs_per_sec": round(throughput, 2),
    }

if __name__ == "__main__":
    print("=" * 60)
    print("ANALISE DE CUSTO COMPUTACIONAL")
    print("=" * 60)

    cost_results = {}

    for strategy, datasets in strategies.items():
        cost_results[strategy] = {}
        for ds_name, data in datasets.items():
            result = calculate_cost_metrics(data)
            cost_results[strategy][ds_name] = result

            print(f"\n{strategy} - {ds_name}:")
            print(f"  Tempo total: {result['total_time_s']}s")
            print(f"  Sequencias: {result['n_sequences']}")
            print(f"  Latencia: {result['latency_ms_per_seq']} ms/seq")
            print(f"  Throughput: {result['throughput_seqs_per_sec']} seqs/s")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NVIDIA Tesla T4"
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 14.6
    gpu_allocated = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 6.1

    print(f"\n--- Memoria GPU ---")
    print(f"GPU: {gpu_name}")
    print(f"Memoria total: {gpu_total:.1f} GB")
    print(f"Memoria em uso: {gpu_allocated:.1f} GB")

    total_params = 8030261248
    trainable_params = 20971520
    trainable_pct = (trainable_params / total_params) * 100

    print(f"\n--- Parametros do Modelo ---")
    print(f"Parametros totais: {total_params:,}")
    print(f"Parametros treinaveis (LoRA): {trainable_params:,}")
    print(f"Percentual treinavel: {trainable_pct:.2f}%")

    print(f"\n{'Estrategia':<20} {'Dataset':<8} {'Tempo(s)':<10} {'Lat(ms)':<10} {'Thru(seq/s)':<12}")
    print("-" * 60)
    for strategy, datasets in cost_results.items():
        for ds_name, data in datasets.items():
            print(f"{strategy:<20} {ds_name.upper():<8} {data['total_time_s']:<10.1f} {data['latency_ms_per_seq']:<10.2f} {data['throughput_seqs_per_sec']:<12.2f}")
    print("-" * 60)

    cost_summary = {
        "gpu": {
            "name": gpu_name,
            "total_memory_gb": round(gpu_total, 1),
            "allocated_memory_gb": round(gpu_allocated, 1),
        },
        "model": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_pct": round(trainable_pct, 2),
        },
        "strategies": cost_results,
        "training": {
            "epochs_configured": 3,
            "epochs_executed": 0.58,
            "batch_size": 4,
            "gradient_accumulation": 4,
            "effective_batch_size": 16,
            "learning_rate": 2e-4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "optimizer": "paged_adamw_8bit",
            "precision": "FP16 (4-bit NF4)",
        }
    }

    with open("results/cost_analysis.json", "w") as f:
        json.dump(cost_summary, f, indent=2)

    print("\nResultados salvos em results/cost_analysis.json")
