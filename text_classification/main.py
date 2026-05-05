
from training import train_lstm as lstm
from training import train_trans as trans

import loader.trans_loader as trans_loader
import loader.lstm_loader as lstm_loader

from eda.eval import eval
from logic.inference import InferenceEngine

def compare_with_gt(preds, gt_labels):
    pred_set = set(preds['labels'])
    gt_set = set(gt_labels)
    
    hits = pred_set.intersection(gt_set)
    missed = gt_set - pred_set
    fp = pred_set - gt_set
    
    precision = len(hits) / len(pred_set) if len(pred_set) > 0 else 0
    recall = len(hits) / len(gt_set) if len(gt_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 30)
    print(f"REPORT FOR MODEL: {preds.get('model_used', 'N/A')}")
    print(f"Hits ({len(hits)}): {hits}")
    print(f"Missed ({len(missed)}): {missed}")
    print(f"False Positives ({len(fp)}): {fp}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("-" * 30)



from time import time
def inference():

    start = time()
    engine = InferenceEngine()

    # Example input text (replace with actual test samples)
    test_data = {
        'title': """Commission Implementing Decision of 14 October 2011 amending and correcting the Annex to Commission Decision 
                    2011/163/EU on the approval of plans submitted by third countries in
                    accordance with Article 29 of Council Directive 96/23/EC (notified under document C(2011) 7167) Text with EEA relevance""",

        'main_body': """['The Annex to Decision 2011/163/EU is replaced by the text in the Annex to this Decision.'
                            'This Decision shall apply from 1 November 2011.\nHowever, the amendment concerning the entry for Singapore shall apply from 15 March 2011.'
                            'This Decision is addressed to the Member States.']""",

        'recitals': """,
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
    }   
    mid = time()
    print(f"Inference Engine initialized in {mid - start:.2f}s")
    lstm_preds = engine.predict(test_data, model_type='lstm', thres=0.3)
    

    print(f"LSTM Predictions ({time() - mid:.2f}s): {lstm_preds}")
    gt_labels = ['1166', '1338', '1445', '1644', '1729', '1841', '1907', '2121', '2531', '2718', '3191', '4164', '4580', '4747', '4841', '6268', '6569']
    
    # Predict & Compare
    lstm_preds = engine.predict(test_data, model_type='lstm')
    compare_with_gt(lstm_preds, gt_labels)

    # Predict & Compare
    trans_preds = engine.predict(test_data, model_type='trans', thres=0.4)
    compare_with_gt(trans_preds, gt_labels)

def eval_zero_shot():
    import torch 
    from logic.base import base_model_eval
    from model.trans import Trans
    from loader.trans_loader import trans_loader_eur
    # Load model, tokenizer, mlb, id_to_name
    # Call eval_base_zero_shot with test_loader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Trans(device=device)
    
    from loader.trans_loader import trans_loader_eur
    train_loader, test_loader, mlb, tokenizer , id_to_title= trans_loader_eur()
    base_model_eval(model, test_loader, mlb, tokenizer,device, id_to_title
                    )


# inference()
eval_zero_shot()