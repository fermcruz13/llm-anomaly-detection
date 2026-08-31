# Datasets

Esta pasta contem os datasets gerados pelo script `src/01_setup_datasets.py`.

Os arquivos sao criados automaticamente ao executar:

    python src/01_setup_datasets.py

Arquivos gerados:

- `bgl_train.csv` - 9.900 sequencias de treino (BGL)
- `bgl_test.csv` - 1.100 sequencias de teste (BGL)
- `hdfs_train.csv` - 15.120 sequencias de treino (HDFS)
- `hdfs_test.csv` - 1.680 sequencias de teste (HDFS)

Os datasets sao sinteticos, baseados em padroes reais dos logs BGL e HDFS.
Nao e necessario versionar estes arquivos no Git, pois sao regeneraveis.
