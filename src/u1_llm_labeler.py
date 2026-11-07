import pandas as pd
import math
from huggingface_hub import InferenceClient
from src.config import FEW_SHOT_EXAMPLES, HF_MODEL_ID, TAXONOMY

def build_few_shot_prompt(target_text: str) -> str:
    """Constructs a strict few-shot prompt using predefined examples."""
    prompt = "Classify the support ticket into exactly one of these categories: "
    prompt += ", ".join(TAXONOMY) + ".\n\n"
    
    for example in FEW_SHOT_EXAMPLES:
        prompt += f"Input: {example['text']}\nLabel: {example['label']}\n\n"
        
    prompt += f"Input: {target_text}\nLabel:"
    return prompt

def extract_confidence(logprob: float) -> float:
    """Converts log probability to a linear confidence score (0 to 1)."""
    if logprob is None:
        return 0.0
    return math.exp(logprob)

def run_few_shot_labeling(df: pd.DataFrame, hf_token: str, confidence_threshold: float) -> pd.DataFrame:
    """Iterates through tickets, queries the LLM, and extracts predictions and confidence."""
    client = InferenceClient(model=HF_MODEL_ID, token=hf_token)
    
    predictions = []
    confidences = []
    
    for idx, row in df.iterrows():
        prompt = build_few_shot_prompt(row['text'])
        
        try:
            # Generate text and request token details for logprobs
            response = client.text_generation(
                prompt,
                max_new_tokens=5,
                details=True,
                return_full_text=False,
                temperature=0.1
            )
            
            # Parse the predicted text and clean it
            predicted_text = response.generated_text.strip().split('\n')[0]
            
            # Extract logprob of the first generated token
            first_token_logprob = response.details.tokens[0].logprob
            confidence = extract_confidence(first_token_logprob)
            
            # Map to taxonomy or assign 'Unknown'
            matched_label = next((label for label in TAXONOMY if label.lower() in predicted_text.lower()), "Unknown")
            
        except Exception as e:
            print(f"Error processing ticket {idx}: {e}")
            matched_label = "Error"
            confidence = 0.0
            
        predictions.append(matched_label)
        confidences.append(confidence)
        
    # Create the output DataFrame
    df_labelled = df.copy()
    df_labelled['llm_predicted_class'] = predictions
    df_labelled['llm_confidence_score'] = confidences
    df_labelled['needs_review'] = df_labelled['llm_confidence_score'] < confidence_threshold
    
    return df_labelled
    