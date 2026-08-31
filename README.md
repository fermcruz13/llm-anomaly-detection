Detecção de Anomalias em Logs de Segurança com LLaMA-3-8B e LoRA

Descrição

Este projeto implementa a integração de Modelos de Linguagem de Grande Porte (LLaMA-3-8B-Instruct) voltada à detecção automatizada de anomalias em fluxos de logs de segurança computacional. A avaliação abrange os conjuntos de dados BGL (Blue Gene/L) e HDFS (Hadoop Distributed File System), comparando o desempenho prático de três estratégias distintas: inferência direta sem ajuste (zero-shot), contextualização por exemplos em prompt (few-shot) e ajuste fino eficiente de parâmetros via LoRA (Low-Rank Adaptation).

Arquitetura
O pipeline de processamento e análise é estruturado em quatro camadas operacionais desacopladas:
1.	Ingestão: Apache Kafka para recepção contínua e ordenação dos eventos brutos emitidos pelos sistemas.
2.	Pré-processamento: Apache Spark Structured Streaming e Drain3 para análise estrutural, limpeza e agrupamento em janelas temporais.
3.	Inferência: LLaMA-3-8B quantizado com motor vLLM para classificação determinística de sequências anômalas.
4.	Indexação e Alerta: Elasticsearch e Kibana para persistência, auditoria analítica e notificação em tempo real.

Estrutura do Projeto
llm-anomaly-detection/ ├── README.md ├── requirements.txt ├── src/ │   ├── 01_setup_datasets.py      # Geração e divisão dos datasets BGL e HDFS │   ├── 02_load_model.py           # Carregamento do LLaMA-3-8B-Instruct (4-bit NF4) │   ├── 03_zero_shot.py            # Inferência zero-shot │   ├── 04_few_shot.py             # Inferência few-shot com exemplos no prompt │   ├── 05_finetuning_lora.py      # Fine-tuning com LoRA (r=16, alpha=32) │   ├── 06_inference.py            # Inferência pós-treinamento │   ├── 07_metrics.py              # Cálculo de métricas (precision, recall, F1, FPR) │   └── 08_cost_analysis.py        # Análise de custo computacional ├── results/ │   ├── zero_shot_results.json │   ├── few_shot_results.json │   ├── finetuning_lora_results.json │   └── cost_analysis.json └── data/     └── (datasets gerados sinteticamente)

Requisitos
●	Python 3.10 ou superior (compatível com Python 3.13)
●	Placa de vídeo (GPU) com no mínimo 8 GB de VRAM (recomendado NVIDIA Tesla T4 de 14.6 GB ou superior)
●	Conta na plataforma Hugging Face com autorização de acesso ao modelo meta-llama/Meta-Llama-3-8B-Instruct
●	Token de autenticação da API Hugging Face ativo

Instalação
git clone https://github.com/usuario/llm-anomaly-detection.git cd llm-anomaly-detection pip install -r requirements.txt huggingface-cli login
Datasets
Os experimentos utilizam dois conjuntos representativos de infraestruturas críticas:
●	BGL (Blue Gene/L): logs de supercomputador contendo registros de falhas de hardware e integridade de memória. Estrutura sintética de 11.000 sequências (10.000 padrões normais e 1.000 ocorrências anômalas).
●	HDFS (Hadoop Distributed File System): logs de blocos e operações de escrita em sistemas de arquivos distribuídos. Estrutura sintética de 16.800 sequências (16.000 normais e 800 anômalas).
●	Particionamento: partição estratificada de 90% para treino e 10% para teste, mantendo a proporção de anomalias em ambas as frações.

Modelo
O modelo base adotado é o meta-llama/Meta-Llama-3-8B-Instruct carregado com quantização de 4 bits em formato NormalFloat (NF4) via biblioteca bitsandbytes, utilizando computação interna em ponto flutuante de 16 bits (FP16). O fine-tuning com LoRA opera com rank 
r = 16
, coeficiente de escala 
alpha = 32
, taxa de descarte (dropout) de 0.05 e aplicação sobre todas as matrizes lineares de atenção e projeção (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). Com essa parametrização, apenas 20.971.520 pesos são treináveis (0.26% do total de 8.030.261.248 parâmetros), gerando adaptadores de aproximadamente 84 MB.

Resultados
Tabela III: Métricas de Detecção
Estratégia	Dataset	Precision	Recall	F1-Score	FPR
Zero-Shot	BGL	0.0849	0.9875	0.1563	0.9261
Zero-Shot	HDFS	1.0000	0.3750	0.5455	0.0000
Few-Shot	BGL	0.0912	0.9125	0.1659	0.7902
Few-Shot	HDFS	0.0705	0.7500	0.1288	0.3061
Fine-Tuning LoRA	BGL	1.0000	1.0000	1.0000	0.0000
Fine-Tuning LoRA	HDFS	1.0000	1.0000	1.0000	0.0000
Tabela V: Custo Computacional
Estratégia	Dataset	Tempo Total (s)	Latência (ms/seq)	Throughput (seqs/s)
Zero-Shot	BGL	653.9	594.45	1.68
Zero-Shot	HDFS	736.9	438.63	2.28
Few-Shot	BGL	942.7	857.00	1.17
Few-Shot	HDFS	1586.6	944.40	1.06
Fine-Tuning LoRA	BGL	394.5	358.64	2.79
Fine-Tuning LoRA	HDFS	563.5	335.42	2.98
Configuração do Ambiente
●	GPU: NVIDIA Tesla T4 (14.6 GB de memória dedicada)
●	Quantização e Precisão: 4-bit NF4 com tipo de cálculo FP16
●	Otimizador: paged_adamw_8bit
●	Batch Size: 4 por dispositivo com acumulação de gradientes em 4 passos (tamanho efetivo: 16)
●	Taxa de Aprendizado (Learning Rate): 2e-4
●	Decaimento de Taxa (Scheduler): Cosine com 50 passos de aquecimento (warmup)
●	Parâmetros LoRA: rank $$r = 16$$, $$\alpha = 32$$, dropout = 0.05

Como Executar
5.	Instalar as dependências do ambiente: pip install -r requirements.txt
6.	Autenticar as credenciais no Hugging Face: huggingface-cli login
7.	Executar o particionamento e validação dos dados: python src/01_setup_datasets.py
8.	Carregar o modelo base quantizado: python src/02_load_model.py
9.	(Opcional) Executar as análises baseline de prompt engineering: python src/03_zero_shot.py e python src/04_few_shot.py
10.	Executar o fine-tuning supervisionado via adaptadores LoRA: python src/05_finetuning_lora.py
11.	Realizar a classificação do conjunto de teste: python src/06_inference.py
12.	Processar a matriz de confusão e métricas de desempenho: python src/07_metrics.py
13.	Gerar o relatório consolidado de latência e consumo de hardware: python src/08_cost_analysis.py

Autora
Fernanda Mara Cruz
Departamento de Ciência da Computação (DCC)
Universidade Estadual Paulista (UNESP)
Contato: fm.cruz@unesp.br

Licença
Este projeto é disponibilizado sob os termos da licença MIT. Para detalhes completos, consulte o arquivo LICENSE na raiz do repositório.
