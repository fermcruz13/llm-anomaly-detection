# 07_metrics.py
# Funcoes reutilizaveis para calculo de metricas

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred, dataset_name="Dataset"):
    """
    Calcula precision, recall, F1, FPR e matriz de confusao.

    Parametros:
        y_true: lista de labels reais (0 = normal, 1 = anomalia)
        y_pred: lista de labels preditas
        dataset_name: nome do dataset para exibicao

    Retorna:
        dict com todas as metricas
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n  {dataset_name}:")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    FPR:       {fpr:.4f}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def print_comparison_table(zero_shot, few_shot, finetuning):
    """
    Imprime tabela comparativa das tres estrategias.
    Cada argumento e um dict com chaves 'bgl' e 'hdfs'.
    """
    print("\n" + "=" * 70)
    print("COMPARACAO DAS ESTRATEGIAS")
    print("=" * 70)
    print(f"\n{'Estrategia':<20} {'Dataset':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12}")
    print("-" * 70)

    for name, results in [("Zero-Shot", zero_shot), ("Few-Shot", few_shot), ("Fine-Tuning LoRA", finetuning)]:
        for ds in ["bgl", "hdfs"]:
            r = results[ds]
            print(f"{name:<20} {ds.upper():<8} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['fpr']:<12.4f}")

    print("-" * 70)

if __name__ == "__main__":
    # Exemplo de uso
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 1, 0]
    metrics = calculate_metrics(y_true, y_pred, "Exemplo")
    print(f"\nResultado: {metrics}")
