import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
from src.config import TAXONOMY

def generate_confusion_matrix(y_true: list, y_pred: list):
    """Generates an interactive Plotly confusion matrix heatmap."""
    # Compute the matrix
    cm = confusion_matrix(y_true, y_pred, labels=TAXONOMY)
    
    # Create the Plotly figure
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Label (LLM)", y="True Label (Human)", color="Ticket Count"),
        x=TAXONOMY,
        y=TAXONOMY,
        color_continuous_scale="Blues",
        text_auto=True
    )
    
    fig.update_layout(
        title="Human vs LLM Agreement",
        xaxis_title="LLM Predicted Class",
        yaxis_title="Human Ground Truth",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def audit_label_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Isolates tickets where the LLM was highly confident but wrong."""
    # Ensure columns exist before filtering
    if 'human_ground_truth' not in df.columns or 'llm_predicted_class' not in df.columns:
        return pd.DataFrame()
        
    # Filter for mismatches with high confidence (> 0.85)
    mismatches = df[
        (df['human_ground_truth'] != df['llm_predicted_class']) & 
        (df['llm_confidence_score'] >= 0.85)
    ]
    
    return mismatches[['text', 'human_ground_truth', 'llm_predicted_class', 'llm_confidence_score']]
    