import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "unlabelled_tickets.csv"
GOLD_DATA_PATH = DATA_DIR / "gold_standard" / "human_labels.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "llm_labelled_tickets.csv"
MODEL_SAVE_PATH = BASE_DIR / "models" / "distilbert_finetuned"

# Taxonomy / Classes
TAXONOMY = ["Billing", "Password", "Bug Report", "Feature Request", "Hardware"]
ID2LABEL = {i: label for i, label in enumerate(TAXONOMY)}
LABEL2ID = {label: i for i, label in enumerate(TAXONOMY)}

# Few-Shot Examples (U1)
FEW_SHOT_EXAMPLES = [
    {"text": "I was charged twice for my pro subscription this month.", "label": "Billing"},
    {"text": "How do I reset my admin password? The email link expired.", "label": "Password"},
    {"text": "The app crashes every time I click the export to PDF button.", "label": "Bug Report"},
    {"text": "It would be great if we could schedule emails in advance.", "label": "Feature Request"},
    {"text": "My provided laptop screen is flickering and the hinge is loose.", "label": "Hardware"}
]

# LLM Configuration
HF_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # Free-tier model for API inference
MAX_TOKENS = 512