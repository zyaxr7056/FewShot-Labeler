import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(y_true: list, y_pred: list) -> dict:
    """Calculates standard classification metrics."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    }

def compare_models(human_labels: list, llm_labels: list, finetuned_labels: list = None) -> pd.DataFrame:
    """Generates a comparison table between the Few-Shot LLM and the Fine-Tuned Model."""
    results = {}
    
    # Evaluate Few-Shot LLM
    results['Few-Shot LLM'] = calculate_metrics(human_labels, llm_labels)
    
    # Evaluate Fine-Tuned Model (if available from U3)
    if finetuned_labels:
        results['Fine-Tuned DistilBERT'] = calculate_metrics(human_labels, finetuned_labels)
    else:
        results['Fine-Tuned DistilBERT'] = {"Accuracy": "Pending U3", "Precision": "Pending U3", "Recall": "Pending U3", "F1-Score": "Pending U3"}
        
    return pd.DataFrame(results).T
    