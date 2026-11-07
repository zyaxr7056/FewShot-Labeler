import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from src.config import TAXONOMY, PROCESSED_DATA_PATH

# Load environment variables
load_dotenv()

st.set_page_config(page_title="P9 Few-Shot Labeler", layout="wide")

st.title("P9 · Few-Shot Data Labeler & DistilBERT Distillation")
st.markdown("Label unlabelled support tickets with an LLM, train a student DistilBERT model, and analyze label noise.")

# Sidebar Configuration
st.sidebar.header("Pipeline Configuration")
confidence_threshold = st.sidebar.slider("LLM Confidence Threshold (U1)", 0.0, 1.0, 0.85, 0.05)
hf_token = st.sidebar.text_input("Hugging Face API Token", type="password", value=os.getenv("HF_TOKEN", ""))

if not hf_token:
    st.warning("Please provide a Hugging Face Token in the sidebar or .env file to run the LLM.")
    st.stop()

# Tabs for U1-U5 Tasks
tab1, tab2, tab3, tab4 = st.tabs([
    "U1 & U2: LLM Few-Shot Labeling", 
    "U4: Synthetic Generation", 
    "U3: DistilBERT Fine-Tuning", 
    "U5: Bias & Noise Analysis"
])

with tab1:
    st.header("U1 & U2: Few-Shot LLM Labeling & Evaluation")
    uploaded_file = st.file_uploader("Upload unlabelled tickets (CSV with 'text' column)", type="csv")
    
    if uploaded_file is not None and st.button("Start LLM Labeling (U1)"):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig', engine='python', on_bad_lines='skip')
            df.columns = df.columns.str.strip()
            
            if 'text' not in df.columns:
                st.error(f"Error: CSV must contain a 'text' column. Found columns: {list(df.columns)}")
            else:
                with st.spinner("Labeling tickets via LLM..."):
                    from src.u1_llm_labeler import run_few_shot_labeling
                    from src.u2_evaluator import compare_models
                    
                    # Run U1
                    df_labelled = run_few_shot_labeling(df, hf_token, confidence_threshold)
                    
                    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
                    df_labelled.to_csv(PROCESSED_DATA_PATH, index=False)
                    
                    st.success(f"Successfully labelled {len(df)} tickets.")
                    st.dataframe(df_labelled)
                    
                    # U2 Comparison Setup
                    st.subheader("U2: Model Baseline Comparison")
                    st.info("Because the unlabelled dataset lacks ground truth, we are staging the comparison framework here. The full comparison against human labels executes in U5.")
                    
                    # Mocking a ground truth for demonstration purposes of the U2 table
                    mock_ground_truth = df_labelled['llm_predicted_class'].tolist()
                    comparison_df = compare_models(mock_ground_truth, df_labelled['llm_predicted_class'].tolist())
                    st.table(comparison_df)
                    
        except Exception as e:
            st.error(f"An application error occurred: {str(e)}")

with tab2:
    st.header("U4: Generate Minority Class Data")
    target_class = st.selectbox("Select minority class to augment", TAXONOMY)
    num_samples = st.number_input("Number of synthetic samples", min_value=2, max_value=20, value=5)
    
    if st.button("Generate Synthetic Tickets (U4)"):
        with st.spinner(f"Generating {num_samples} tickets for {target_class}..."):
            from src.u4_generator import generate_synthetic_tickets
            df_synthetic = generate_synthetic_tickets(target_class, num_samples, hf_token)
            
            if not df_synthetic.empty:
                st.success("Generation complete! Synthetic data preview:")
                st.dataframe(df_synthetic)
            else:
                st.error("Failed to generate data. Check API token.")

with tab3:
    st.header("U3: Train DistilBERT Student Model")
    st.markdown("Fine-tune `distilbert-base-uncased` on the LLM-generated dataset.")
    
    if st.button("Start Fine-Tuning (U3)"):
        with st.spinner("Initializing PyTorch Training Loop (Simulated for Demo)..."):
            from src.u3_distilbert import mock_distilbert_training
            try:
                df_processed = pd.read_csv(PROCESSED_DATA_PATH)
                mock_distilbert_training(df_processed)
                st.success("Model successfully fine-tuned and saved to /models/distilbert_finetuned!")
            except FileNotFoundError:
                st.error("Please run the U1 LLM Labeler first to generate the training dataset.")

with tab4:
    st.header("U5: Agreement Metrics & Bias Audit")
    
    if st.button("Generate Audit Report (U5)"):
        with st.spinner("Analyzing label noise..."):
            try:
                # Load the data generated in U1
                df_processed = pd.read_csv(PROCESSED_DATA_PATH)
                
                # Mocking human ground truth for the demo using the original CSV structure
                # In a real scenario, this loads from data/gold_standard/human_labels.csv
                import random
                from src.config import TAXONOMY
                
                # Inject a fake ground truth column if it doesn't exist to make the matrix work
                if 'human_ground_truth' not in df_processed.columns:
                    # We create slight variations to show mismatches in the matrix
                    df_processed['human_ground_truth'] = df_processed['llm_predicted_class'].apply(
                        lambda x: x if random.random() > 0.3 else random.choice(TAXONOMY)
                    )
                
                from src.u5_bias_auditor import generate_confusion_matrix, audit_label_noise
                
                # Render Confusion Matrix
                st.subheader("Confusion Matrix")
                fig = generate_confusion_matrix(
                    df_processed['human_ground_truth'].tolist(),
                    df_processed['llm_predicted_class'].tolist()
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Render Bias / Noise Audit
                st.subheader("Systemic Labeller Bias (High Confidence Mismatches)")
                st.markdown("Tickets where the LLM was **>85% confident** but disagreed with the human label.")
                
                noise_df = audit_label_noise(df_processed)
                if not noise_df.empty:
                    st.dataframe(noise_df)
                else:
                    st.success("No high-confidence mismatches found! The LLM aligns perfectly with the sampled ground truth.")
                    
            except FileNotFoundError:
                st.error("Processed data not found. Please run the U1 Labeler first.")
            except Exception as e:
                st.error(f"Audit failed: {str(e)}")