import pandas as pd
from huggingface_hub import InferenceClient
from src.config import HF_MODEL_ID

def generate_synthetic_tickets(target_class: str, num_samples: int, hf_token: str) -> pd.DataFrame:
    """Generates synthetic support tickets for minority classes using the LLM."""
    client = InferenceClient(model=HF_MODEL_ID, token=hf_token)
    
    # Kept concise to prevent the model from rambling and hitting token limits
    prompt = f"Write {num_samples} distinct, 1-sentence support tickets about a '{target_class}' issue. Do not include numbers or labels, just the ticket text on a new line."
    
    try:
        # Added do_sample=True (required when using temperature) and lowered max_new_tokens
        response = client.text_generation(
            prompt, 
            max_new_tokens=250, 
            temperature=0.7,
            do_sample=True
        )
        
        raw_text = response.replace("*", "").split('\n')
        
        # Clean up the output into a list of strings
        synthetic_texts = [text.strip("- ").strip() for text in raw_text if len(text.strip()) > 10]
        
        # Cap at requested number in case the model over-generates
        synthetic_texts = synthetic_texts[:num_samples]
        
        df_synthetic = pd.DataFrame({
            "text": synthetic_texts, 
            "llm_predicted_class": [target_class] * len(synthetic_texts)
        })
        return df_synthetic
        
    except Exception as e:
        print(f"API blocked or offline. Using fallback for demo. Error: {str(e)}")
        
        # Failsafe mock data so your video recording doesn't get ruined by Wi-Fi drops
        fallback_texts = [
            f"I am having a terrible time dealing with this {target_class} issue.",
            f"Can someone please help me fix my {target_class}?",
            f"This {target_class} problem is completely blocking my workflow.",
            f"I've tried everything to resolve this {target_class} bug and nothing works.",
            f"Who do I contact about a persistent {target_class} failure?"
        ]
        
        df_synthetic = pd.DataFrame({
            "text": fallback_texts[:num_samples], 
            "llm_predicted_class": [target_class] * min(num_samples, len(fallback_texts))
        })
        return df_synthetic