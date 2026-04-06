import streamlit as st
import torch
import numpy as np
import plotly.express as px
from logic.inference import InferenceEngine

st.set_page_config(page_title="Doc Processor", layout="wide")

@st.cache_resource
def load_engine():
    return InferenceEngine()

engine = load_engine()

st.title("🚀 Odoc: Document Tagger")
st.markdown("Automated Multi-label Classification for Legal & ComputerScience Documents")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Select Model", ["trans", "lstm", "ensemble"])
    threshold = st.slider("Threshold", 0.1, 0.9, 0.3, 0.05)
    if model_choice == "ensemble":
        w_trans = st.slider("Weight Transformer", 0.0, 1.0, 0.7, 0.1)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Document")
    title = st.text_input("Title", "Commission Implementing Decision...")
    body = st.text_area("Main Body", "The Annex to Decision...", height=150)
    recitals = st.text_area("Recitals", "Having regard to the Treaty...", height=200)
    
    run_btn = st.button("Run Inference", use_container_width=True)

if run_btn:
    data = {"title": title, "main_body": body, "recitals": recitals}
    
    with st.spinner("Processing..."):
        if model_choice == "ensemble":
            res = engine.ensemble_predict(data, w_trans=w_trans, threshold=threshold)
        else:
            res = engine.predict(data, model_type=model_choice, override_thres=threshold)
            
    with col2:
        st.subheader("Results")
        st.write(f"**Inference Time:** {res['inference_time']}")
        
        if res['labels']:
            st.write("**Predicted Tags:**")
            cols = st.columns(len(res['labels']))
            for i, label in enumerate(res['labels']):
                st.info(f"Tag: {label}")
            
            # Confidence
            conf_data = {"Tags": res['labels'], "Confidence": res['confidence']}
            fig = px.bar(conf_data, x='Confidence', y='Tags', orientation='h', 
                         title="Confidence Scores", color='Confidence', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No tags detected above threshold.")

    # Attention Rollout Section
    if model_choice == "trans":
        st.divider()
        st.subheader("🧠 Attention Rollout (Interpretability)")
        # visualization 
        full_text = f"{title} [SEP] {body}"
        viz_data = engine.visualize_attention(full_text, model_type='trans')
        
        if isinstance(viz_data, list):
            tokens = [d['token'] for d in viz_data]
            weights = [d['weight'] for d in viz_data]
            
            # Heatmap
            fig_attn = px.imshow([weights], x=tokens, aspect="auto", 
                                 labels=dict(x="Tokens", color="Attention"),
                                 color_continuous_scale='Reds', title="Attention Weights per Token")
            st.plotly_chart(fig_attn, use_container_width=True)