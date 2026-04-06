import streamlit as st
import torch
import numpy as np
import plotly.express as px
from logic.inference import InferenceEngine

st.set_page_config(page_title="Doc Processor", layout="wide")
title= """Commission Implementing Decision of 14 October 2011 amending and correcting the Annex to Commission Decision 
                    2011/163/EU on the approval of plans submitted by third countries in
                    accordance with Article 29 of Council Directive 96/23/EC (notified under document C(2011) 7167) Text with EEA relevance"""

main_body= """['The Annex to Decision 2011/163/EU is replaced by the text in the Annex to this Decision.'
                            'This Decision shall apply from 1 November 2011.\nHowever, the amendment concerning the entry for Singapore shall apply from 15 March 2011.'
                            'This Decision is addressed to the Member States.']"""

recitals= """,
                    Having regard to the Treaty on the Functioning of the European Union,
                    Having regard to Council Directive 96/23/EC of 29 April 1996 on measures to monitor certain substances and residues thereof in live animals and animal products and repealing Directives 85/358/EEC and 86/469/EEC and Decisions 89/187/EEC and 91/664/EEC (1), and in particular the fourth subparagraph of Article 29(1) and Article 29(2) thereof.
                    Whereas:
                    (1) Directive 96/23/EC lays down measures to monitor the substances and groups of residues listed in Annex I thereto. Pursuant to Directive 96/23/EC, the inclusion and retention on the lists of third countries from which Member States are authorised to import animals and animal products covered by that Directive are subject to the submission by the third countries concerned of a plan setting out the guarantees which they offer as regards the monitoring of the groups of residues and substances listed in that Annex. Those plans are to be updated at the request of the Commission, particularly when certain checks render it necessary.
                    (2) Commission Decision 2011/163/EU (2) approves the plans provided for in Article 29 of Directive 96/23/EC (‘the plans’) submitted by certain third countries listed in the Annex thereto for the animals and animal products indicated in that list. Decision 2011/163/EU repealed and replaced Commission Decision 2004/432/EC of 29 April 2004 on the approval of residue monitoring plans submitted by third countries in accordance with Council Directive 96/23/EC (3).
                    (3) In the light of the recent plans submitted by certain third countries and additional information obtained by the Commission, it is necessary to update the list of third countries from which Member States are authorised to import certain animals and animal products, as provided for in Directive 96/23/EC and currently listed in the Annex to Decision 2011/163/EU (‘the list’).
                    (4) Belize is currently included in the list for aquaculture and honey. However, Belize has not provided a plan as required by Article 29 of Directive 96/23/EC. Therefore, Belize should be removed from the list.
                    (5) Ghana has submitted a plan for honey to the Commission. That plan provides sufficient guarantees and should be approved. Therefore, an entry for Ghana for honey should be included in the list.
                    (6) India has now carried out corrective measures to address the shortcomings in its residue plan for honey. That third country has submitted an improved residue plan for honey and a Commission inspection confirmed an acceptable implementation of the plan. Therefore, the entry for India in the list should include honey.
                    (7) Madagascar has submitted a plan for honey to the Commission. That plan provides sufficient guarantees and should be approved. Therefore, honey should be included in the entry for Madagascar in the list.
                    (8) Mauritius is currently included in the list for poultry but with a reference to footnote 2 in the Annex to Decision 2011/163/EU. That footnote restricts such imports to those from third countries using only raw material either from Member States or from other third countries approved for imports of such raw material to the Union, in accordance with Article 2 of that Decision. However, Mauritius has not provided the required guarantees for the plan for poultry. Therefore, the entry for that third country in the list should no longer include poultry.
                    (9) Turkey has submitted a plan for eggs to the Commission. That plan provides sufficient guarantees and should be approved. Therefore, eggs should be included in the entry for Turkey in the list.
                    (10) The entry for Singapore in the list includes aquaculture but with a reference to footnote 2 in the Annex to Decision 2011/163/EU. However, in the Annex to Decision 2004/432/EC, as amended by Commission Decision 2010/327/EU (4), there is no reference to footnote 2 as Singapore submitted an approved plan for aquaculture. The Commission has not been advised of any change since the approval of that plan. Therefore, the entry for that third country in the list should be corrected by deleting the reference to that footnote for imports of aquaculture. For reasons of legal certainty, the entry for Singapore should apply retroactively from 15 March 2011, the date of application of Decision 2011/163/EU when the error in the entry regarding Singapore occurred. The competent authorities of the Member States have been informed accordingly and no disruption to imports has been reported to the Commission.
                    (11) The Annex to Decision 2011/163/EU should therefore be amended accordingly.
                    (12) The measures provided for in this Decision are in accordance with the opinion of the Standing Committee on the Food Chain and Animal Health,"""
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
    title = st.text_input("Title", title)
    body = st.text_area("Main Body", main_body, height=150)
    recitals = st.text_area("Recitals", recitals, height=200)
    
    run_btn = st.button("Run Inference", use_container_width=True)

if run_btn:
    data = {"title": title, "main_body": main_body, "recitals": recitals}
    
    with st.spinner("Processing..."):
        if model_choice == "ensemble":
            res = engine.ensemble_predict(data, w_trans=w_trans, threshold=threshold)
        else:
            res = engine.predict(data, model_type=model_choice, thres=threshold)
            
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
    if model_choice in ["trans", "lstm"]:
        st.divider()
        st.subheader(f"🧠 {model_choice.upper()} Attention Analysis")
        viz_data = engine.visualize_attention(data, model_choice=model_choice)
        
        if viz_data:
            tokens = [d['token'] for d in viz_data]
            weights = [d['weight'] for d in viz_data]
            
            fig_attn = px.imshow(
                [weights], 
                x=tokens, 
                aspect="auto",
                title="Attention Weights per Token",
                color_continuous_scale='Reds'
            )
            fig_attn.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            fig_attn.update_xaxes(side="bottom",
                                  tickangle=45,         
                                automargin=True)
            fig_attn.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=100))
            st.plotly_chart(fig_attn, use_container_width=True)
        else:
            st.warning("No attention weights detected.")