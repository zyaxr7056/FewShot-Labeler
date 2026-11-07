# Few-Shot Data Labeler

## Project Overview
This project is an end-to-end Generative AI pipeline designed to label unlabelled support tickets using a Large Language Model (LLM) and use that data to fine-tune a smaller, efficient student model (DistilBERT). The pipeline includes automated synthetic data generation for rare classes and a robust bias/noise audit of the LLM's predictions.

## Features (U1 - U5 Syllabus Coverage)
* **U1 (Foundations & Prompt Engineering):** Few-shot prompting to label support tickets via the Hugging Face API, extracting logprobs to calculate a confidence score.
* **U2 (GANs & LLMs):** Baseline evaluation framework to compare the Few-Shot LLM's performance against a fine-tuned sequence classifier.
* **U3 (GPT, BERT & Multimodal):** Distillation of knowledge from the LLM into a `distilbert-base-uncased` student model for local, low-latency inference.
* **U4 (Tools & Generation):** Generative synthetic data augmentation for minority support ticket classes using persona-based prompts.
* **U5 (Responsible AI):** Interactive Plotly confusion matrix and bias audit isolating systemic labeller noise (high-confidence mismatches).

## Local Setup Instructions
1. Clone this repository.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Hugging Face API token:
   ```text
   HF_TOKEN=your_huggingface_token_here
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Tech Stack
* **UI & Hosting:** Streamlit
* **Models:** Hugging Face Inference API (Qwen/TinyLlama), DistilBERT
* **Libraries:** PyTorch, Pandas, Transformers, Scikit-learn, Plotly
