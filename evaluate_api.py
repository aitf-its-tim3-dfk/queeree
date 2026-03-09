import os
import sys
import json
import asyncio
import argparse
import subprocess

def install_deps():
    print("Checking dependencies...")
    try:
        import pandas
        import sklearn
        import bert_score
        import requests
    except ImportError:
        print("Installing required dependencies in the current venv...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "scikit-learn", "bert-score", "requests"])
        print("Dependencies installed successfully.")

install_deps()

import pandas as pd
import requests
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from bert_score import score as bert_score_fn
import sseclient # pip install sseclient-py might be needed, let's just parse SSE manually for simplicity.

def get_api_result(image_path):
    url = "http://localhost:8000/api/analyze"
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
        
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"content": ""}
            response = requests.post(url, data=data, files=files, stream=True, timeout=600)
            
            final_result = None
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[6:]
                        try:
                            event_data = json.loads(json_str)
                            if event_data.get('type') == 'result':
                                final_result = event_data.get('data')
                            elif event_data.get('type') == 'error':
                                print(f"API Error: {event_data.get('data')}")
                        except json.JSONDecodeError:
                            pass
            return final_result
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def main():
    csv_path = "sample/extracted_stuff/extracted_data.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    y_true_cats = []
    y_pred_cats = []
    
    ref_analysis = []
    cand_analysis = []
    
    ref_laws = []
    cand_laws = []

    print(f"Starting evaluation on {len(df)} rows...")
    for idx, row in df.iterrows():
        img_rel_path = row.get("Screen Capture Path", "")
        img_path = os.path.join("sample", img_rel_path) if "sample" not in img_rel_path else img_rel_path
        
        # In the CSV, paths might be extracted_stuff\row_X_img.png
        # Let's handle it dynamically
        if not os.path.exists(img_path):
            img_path = os.path.join("sample", "extracted_stuff", os.path.basename(img_rel_path))
            
        print(f"[{idx+1}/{len(df)}] Processing {img_path}...")
        
        result = get_api_result(img_path)
        if result is None:
            print(f"  -> Failed to get result. Skipping.")
            continue
            
        gt_cat = str(row.get("Kategori", "")).strip()
        # API might return different casing
        pred_cats = [c.strip() for c in result.get("categories", [])]
        
        # Ground truth might have multiple if separated by comma, but in CSV it looks single
        y_true_cats.append([gt_cat] if gt_cat else [])
        y_pred_cats.append(pred_cats)
        
        # For text generation evaluation
        gt_pelanggaran = str(row.get("Analisis Pelanggaran", ""))
        gt_dampak = str(row.get("Analisis Dampak", ""))
        gt_analysis = f"{gt_pelanggaran}\n{gt_dampak}".strip()
        
        pred_analysis = result.get("final_summary", "")
        
        gt_hukum = str(row.get("Dasar Hukum", "")).strip()
        pred_hukum = str(result.get("laws_summary", "")).strip()
        
        ref_analysis.append(gt_analysis)
        cand_analysis.append(pred_analysis)
        
        ref_laws.append(gt_hukum)
        cand_laws.append(pred_hukum)
        
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS")
    print("="*50)
    
    mlb = MultiLabelBinarizer()
    all_cats = y_true_cats + y_pred_cats
    mlb.fit(all_cats)
    y_true_bin = mlb.transform(y_true_cats)
    y_pred_bin = mlb.transform(y_pred_cats)
    
    print("Classes found locally:", mlb.classes_)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    print(f"Exact Match Accuracy: {acc:.4f}")
    
    print("\nClassification Report (Micro/Macro averages):")
    print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0))
    
    print("\n" + "="*50)
    print("TEXT GENERATION METRICS (BERTScore)")
    print("="*50)
    
    print("Calculating BERTScore for Analysis (Analisis Pelanggaran + Dampak vs Final Summary)...")
    if len(cand_analysis) > 0:
        P, R, F1 = bert_score_fn(cand_analysis, ref_analysis, lang="id", verbose=True)
        print(f"Analysis BERTScore -> Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    else:
        print("No analysis text to evaluate.")

    print("\nCalculating BERTScore for Laws (Dasar Hukum vs Laws Summary)...")
    if len(cand_laws) > 0:
        P_law, R_law, F1_law = bert_score_fn(cand_laws, ref_laws, lang="id", verbose=True)
        print(f"Laws BERTScore     -> Precision: {P_law.mean():.4f}, Recall: {R_law.mean():.4f}, F1: {F1_law.mean():.4f}")
    else:
        print("No law text to evaluate.")
        
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()
