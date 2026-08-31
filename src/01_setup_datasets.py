# 01_setup_datasets.py
# Gera os datasets sinteticos BGL e HDFS e divide em treino/teste

import random
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

random.seed(42)

bgl_normal_templates = [
    "RAS KERNEL INFO instruction cache parity error corrected",
    "RAS KERNEL INFO data cache parity error corrected",
    "RAS KERNEL INFO instruction cache tag parity error corrected",
    "RAS KERNEL INFO data cache tag parity correctable error",
    "RAS KERNEL INFO node {}:{}: MCD 0: {}",
    "RAS KERNEL INFO {} MCD 0: Fatal: permanent cpu {}",
    "RAS KERNEL INFO instruction MMU unavailable",
    "RAS KERNEL INFO handling machine check",
]

bgl_anomaly_templates = [
    "RAS KERNEL ERROR instruction cache parity error not corrected",
    "RAS KERNEL ERROR data cache parity error not corrected",
    "RAS KERNEL ERROR instruction cache tag parity uncorrectable",
    "RAS KERNEL ERROR data cache tag parity uncorrectable error",
    "RAS KERNEL FATAL: permanent cpu {} hardware failure",
    "RAS KERNEL ERROR node {}:{}: MCD 0: Fatal: permanent cpu {}",
    "RAS KERNEL ERROR L2 cache data parity error",
    "RAS KERNEL FATAL machine check exception",
]

hdfs_normal_templates = [
    "VERIFIED Complete block verification at offset {} of file {}",
    "INFO PacketResponder:VERIFY_BLOCK received from {}",
    "INFO DataNode client:[] Starting thread to transfer block {} to {}",
    "INFO DataNode:Starting thread to transfer block {} to {}",
    "INFO PacketResponder:{}:Exception writing {} bytes to mirror",
    "INFO DataNode:Starting thread to send block {} to {}",
    "INFO DataNode client:[] Got exception serving {} to {}",
]

hdfs_anomaly_templates = [
    "ERROR PacketResponder:{}:Exception writing block {} received from {}",
    "ERROR DataNode:{}:Incompatible namespaceIDs for block {}",
    "ERROR DataNode:{}:Exception transfering block {} to {}",
    "ERROR PacketResponder:{} for block {} Interrupted while waiting",
    "ERROR DataNode:{}:Exception writing block {} to mirror",
    "ERROR DataNode:{}:DataNode for block {} interrupted",
    "FATAL DataNode:{}:Block {} does not exist on the node",
]

def generate_bgl_data(n_normal=10000, n_anomaly=1000):
    logs = []
    labels = []
    for _ in range(n_normal):
        template = random.choice(bgl_normal_templates)
        log = template.format(
            random.randint(1, 128), random.randint(0, 7), random.randint(0, 31),
            random.randint(0, 3), random.randint(0, 63)
        )
        logs.append(log)
        labels.append(0)
    for _ in range(n_anomaly):
        template = random.choice(bgl_anomaly_templates)
        log = template.format(
            random.randint(1, 128), random.randint(0, 7), random.randint(0, 31),
            random.randint(0, 3), random.randint(0, 63)
        )
        logs.append(log)
        labels.append(1)
    indices = list(range(len(logs)))
    random.shuffle(indices)
    return pd.DataFrame({
        'sequence': [logs[i] for i in indices],
        'label': [labels[i] for i in indices]
    })

def generate_hdfs_data(n_normal=16000, n_anomaly=800):
    logs = []
    labels = []
    for _ in range(n_normal):
        template = random.choice(hdfs_normal_templates)
        log = template.format(
            random.randint(0, 999), f"blk_{random.randint(0, 99999)}",
            f"10.250.{random.randint(0,255)}.{random.randint(0,255)}",
            f"/tmp/hadoop-root/dfs/data/blk_{random.randint(0,99999)}"
        )
        logs.append(log)
        labels.append(0)
    for _ in range(n_anomaly):
        template = random.choice(hdfs_anomaly_templates)
        log = template.format(
            random.randint(0, 999), f"blk_{random.randint(0, 99999)}",
            f"10.250.{random.randint(0,255)}.{random.randint(0,255)}"
        )
        logs.append(log)
        labels.append(1)
    indices = list(range(len(logs)))
    random.shuffle(indices)
    return pd.DataFrame({
        'sequence': [logs[i] for i in indices],
        'label': [labels[i] for i in indices]
    })

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    bgl_df = generate_bgl_data()
    hdfs_df = generate_hdfs_data()

    print(f"BGL: {len(bgl_df)} sequencias ({bgl_df['label'].sum()} anomalias)")
    print(f"HDFS: {len(hdfs_df)} sequencias ({hdfs_df['label'].sum()} anomalias)")

    bgl_train, bgl_test = train_test_split(
        bgl_df, test_size=0.1, random_state=42, stratify=bgl_df['label']
    )
    hdfs_train, hdfs_test = train_test_split(
        hdfs_df, test_size=0.1, random_state=42, stratify=hdfs_df['label']
    )

    print(f"\nBGL treino: {len(bgl_train)} | BGL teste: {len(bgl_test)}")
    print(f"HDFS treino: {len(hdfs_train)} | HDFS teste: {len(hdfs_test)}")

    bgl_train.to_csv("data/bgl_train.csv", index=False)
    bgl_test.to_csv("data/bgl_test.csv", index=False)
    hdfs_train.to_csv("data/hdfs_train.csv", index=False)
    hdfs_test.to_csv("data/hdfs_test.csv", index=False)

    print("\nDatasets salvos em data/")
    print("Concluido!")
