# Detecção de Anomalias em Logs de Segurança com LLaMA-3-8B e LoRA

## Descrição

Este projeto implementa a integração de Modelos de Linguagem de Grande Porte (LLaMA-3-8B-Instruct) voltada à detecção automatizada de anomalias em fluxos de logs de segurança computacional. A avaliação abrange os conjuntos de dados BGL (Blue Gene/L) e HDFS (Hadoop Distributed File System), comparando o desempenho de três estratégias: inferência direta sem ajuste (zero-shot), contextualização por exemplos em prompt (few-shot) e ajuste fino via LoRA (Low-Rank Adaptation).

## Arquitetura

O pipeline é estruturado em quatro camadas operacionais desacopladas:

1. **Ingestão:** Apache Kafka para recepção contínua e ordenação dos eventos brutos.
2. **Pré-processamento:** Apache Spark Structured Streaming e Drain3 para análise estrutural, limpeza e agrupamento em janelas temporais.
3. **Inferência:** LLaMA-3-8B quantizado com motor vLLM para classificação determinística de sequências anômalas.
4. **Indexação e Alerta:** Elasticsearch e Kibana para persistência, auditoria analítica e notificação em tempo real.

## Estrutura do Projeto

    llm-anomaly-detection/
    ├── README.md
    ├── requirements.txt
    ├── LICENSE
    ├── src/
    │   ├── 01_setup_datasets.py      # Geração e divisão dos datasets BGL e HDFS
    │   ├── 02_load_model.py           # Carregamento do LLaMA-3-8B-Instruct (4-bit NF4)
    │   ├── 03_zero_shot.py            # Inferência zero-shot
    │   ├── 04_few_shot.py             # Inferência few-shot com exemplos no prompt
    │   ├── 05_finetuning_lora.py      # Fine-tuning com LoRA (r=16, alpha=32)
    │   ├── 06_inference.py            # Inferência pós-treinamento
    │   ├── 07_metrics.py              # Cálculo de métricas (precision, recall, F1, FPR)
    │   └── 08_cost_analysis.py        # Análise de custo computacional
    ├── results/
    │   ├── zero_shot_results.json
    │   ├── few_shot_results.json
    │   ├── finetuning_lora_results.json
    │   └── cost_analysis.json
    └── data/
        └── (datasets gerados sinteticamente)

## Requisitos

- Python 3.10 ou superior (compatível com Python 3.13)
- GPU com no mínimo 8 GB de VRAM (recomendado NVIDIA Tesla T4 de 14.6 GB ou superior)
- Conta na plataforma Hugging Face com acesso ao modelo `meta-llama/Meta-Llama-3-8B-Instruct`
- Token de autenticação da API Hugging Face ativo

## Instalação

    git clone https://github.com/usuario/llm-anomaly-detection.git
    cd llm-anomaly-detection
    pip install -r requirements.txt
    huggingface-cli login

## Datasets

Os experimentos utilizam dois conjuntos representativos de infraestruturas:

- **BGL (Blue Gene/L):** logs de supercomputador contendo registros de falhas de hardware e integridade de memória. Dataset sintético com 11.000 sequências (10.000 normais e 1.000 anômalas).
- **HDFS (Hadoop Distributed File System):** logs de blocos e operações de escrita em sistemas de arquivos distribuídos. Dataset sintético com 16.800 sequências (16.000 normais e 800 anômalas).
- **Particionamento:** 90% treino / 10% teste, com estratificação por classe.

## Modelo

O modelo base é o `meta-llama/Meta-Llama-3-8B-Instruct` carregado com quantização 4-bit NF4 via biblioteca `bitsandbytes`, com computação interna em FP16.

O fine-tuning com LoRA utiliza:

| Parâmetro | Valor |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Parâmetros treináveis | 20.971.520 (0.26% de 8.030.261.248) |
| Tamanho do adapter | ~84 MB |

## Resultados

### Tabela III: Métricas de Detecção

| Estratégia | Dataset | Precision | Recall | F1-Score | FPR |
|---|---|---|---|---|---|
| Zero-Shot | BGL | 0.0849 | 0.9875 | 0.1563 | 0.9261 |
| Zero-Shot | HDFS | 1.0000 | 0.3750 | 0.5455 | 0.0000 |
| Few-Shot | BGL | 0.0912 | 0.9125 | 0.1659 | 0.7902 |
| Few-Shot | HDFS | 0.0705 | 0.7500 | 0.1288 | 0.3061 |
| Fine-Tuning LoRA | BGL | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Fine-Tuning LoRA | HDFS | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Tabela V: Custo Computacional

| Estratégia | Dataset | Tempo (s) | Latência (ms/seq) | Throughput (seqs/s) |
|---|---|---|---|---|
| Zero-Shot | BGL | 653.9 | 594.45 | 1.68 |
| Zero-Shot | HDFS | 736.9 | 438.63 | 2.28 |
| Few-Shot | BGL | 942.7 | 857.00 | 1.17 |
| Few-Shot | HDFS | 1586.6 | 944.40 | 1.06 |
| Fine-Tuning LoRA | BGL | 394.5 | 358.64 | 2.79 |
| Fine-Tuning LoRA | HDFS | 563.5 | 335.42 | 2.98 |

## Configuração do Ambiente

- **GPU:** NVIDIA Tesla T4 (14.6 GB de memória dedicada)
- **Quantização e Precisão:** 4-bit NF4 com cálculo em FP16
- **Otimizador:** paged_adamw_8bit
- **Batch Size:** 4 por dispositivo com gradiente acumulado em 4 passos (efetivo: 16)
- **Learning Rate:** 2e-4
- **Scheduler:** Cosine com 50 passos de warmup
- **LoRA:** r=16, alpha=32, dropout=0.05

## Como Executar

1. Instalar as dependências: `pip install -r requirements.txt`
2. Autenticar no Hugging Face: `huggingface-cli login`
3. Gerar e dividir os datasets: `python src/01_setup_datasets.py`
4. Carregar o modelo base quantizado: `python src/02_load_model.py`
5. *(Opcional)* Executar as análises baseline: `python src/03_zero_shot.py` e `python src/04_few_shot.py`
6. Executar o fine-tuning com LoRA: `python src/05_finetuning_lora.py`
7. Realizar a inferência no teste: `python src/06_inference.py`
8. Calcular as métricas: `python src/07_metrics.py`
9. Gerar a análise de custo: `python src/08_cost_analysis.py`

## Autora

**Fernanda Mara Cruz**  
Departamento de Ciência da Computação (DCC)  
Universidade Estadual Paulista (UNESP)  
Contato: fm.cruz@unesp.br

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para detalhes.
