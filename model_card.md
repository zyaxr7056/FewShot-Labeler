# Model Card: DistilBERT Support Ticket Classifier (P9)

## Model Details
* **Model Name:** P9-DistilBERT-Support-Labeler
* **Architecture:** `distilbert-base-uncased` (Student Model)
* **Teacher Model:** Qwen2.5 / TinyLlama (via few-shot prompting)
* **Task:** Multi-class text classification (Support Tickets)
* **Taxonomy:** Billing, Password, Bug Report, Feature Request, Hardware.

## Intended Use
* **Primary Use Case:** Automated routing and tagging of incoming IT and customer support tickets to the appropriate department.
* **Out-of-Scope:** This model is not intended for diagnostic medical support, legal advice, or handling severely toxic/abusive text without an upstream moderation filter.

## Training Data
The model was fine-tuned using a synthetic dataset generated via a two-step process:
1. **Few-Shot Labeling:** Unlabelled raw tickets were classified by a teacher LLM to generate base training labels.
2. **Synthetic Augmentation:** Minority classes (e.g., specific Hardware issues) were augmented using LLM-generated synthetic tickets to prevent class imbalance.

## Evaluation Data & Metrics
* **Gold Standard Set:** The model's real-world accuracy is evaluated against a sequestered set of 50 human-labeled support tickets.
* **Metrics Tracked:** Accuracy, Precision, Recall, and F1-Score (accessible via the U2 evaluation module).

## Ethical Considerations & Bias (U5 Audit)
* **Systemic Labeller Bias:** The training data inherits the biases of the teacher LLM. During the U5 audit, we monitor for high-confidence mismatches—instances where the LLM was >85% confident but disagreed with the human ground truth. 
* **Label Noise:** Ambiguous tickets (e.g., a "Password" issue causing a "Billing" lockout) may confuse the teacher model. The pipeline explicitly filters out predictions below a set confidence threshold (e.g., 0.85) before they enter the DistilBERT training set to mitigate noise propagation.
* **Privacy:** Support tickets often contain Personally Identifiable Information (PII). In a production environment, a PII-redaction step (e.g., Microsoft Presidio) must be executed before sending raw text to external APIs for few-shot labeling.