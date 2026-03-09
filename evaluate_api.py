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
        import aiohttp
        import aiofiles
    except ImportError:
        print("Installing required dependencies in the current venv...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "scikit-learn", "bert-score", "requests", "aiohttp", "aiofiles"])
        print("Dependencies installed successfully.")

install_deps()

import pandas as pd
import aiohttp
import aiofiles
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from bert_score import score as bert_score_fn

async def get_api_result_async(session, image_path, idx):
    url = "http://localhost:8000/api/analyze"
    
    if not os.path.exists(image_path):
        print(f"[{idx}] File not found: {image_path}")
        return idx, None
        
    try:
        # We need to construct a multipart payload manually for aiohttp
        data = aiohttp.FormData()
        data.add_field('content', '')
        data.add_field('image',
                       open(image_path, 'rb'),
                       filename=os.path.basename(image_path),
                       content_type='application/octet-stream')

        async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=600)) as response:
            final_result = None
            async for line in response.content:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line.startswith('data: '):
                    json_str = decoded_line[6:]
                    try:
                        event_data = json.loads(json_str)
                        if event_data.get('type') == 'result':
                            final_result = event_data.get('data')
                        elif event_data.get('type') == 'error':
                            print(f"[{idx}] API Error: {event_data.get('data')}")
                    except json.JSONDecodeError:
                        pass
            print(f"[{idx}] Finished processing {image_path}")
            return idx, final_result
    except Exception as e:
        print(f"[{idx}] Request failed for {image_path}: {e}")
        return idx, None

async def process_all_requests(df):
    results = [None] * len(df)
    
    # We will limit concurrency to avoid overwhelming the local server or API
    semaphore = asyncio.Semaphore(5)
    
    async def sem_task(session, img_path, idx):
        async with semaphore:
            return await get_api_result_async(session, img_path, idx)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in df.iterrows():
            img_rel_path = row.get("Screen Capture Path", "")
            img_path = os.path.join("sample", img_rel_path) if "sample" not in img_rel_path else img_rel_path
            
            if not os.path.exists(img_path):
                img_path = os.path.join("sample", "extracted_stuff", os.path.basename(img_rel_path))
            
            tasks.append(sem_task(session, img_path, idx))
            
        print(f"Starting {len(tasks)} concurrent API requests (max 5 parallel)...")
        completed_tasks = await asyncio.gather(*tasks)
        
        for idx, result in completed_tasks:
            results[idx] = result
            
        return results

async def main_async():
    csv_path = "sample/extracted_stuff/extracted_data.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Run all API requests concurrently
    raw_results = await process_all_requests(df)
    
    # Save raw results
    results_json_path = "evaluation_raw_results.json"
    print(f"\nSaving raw API results to {results_json_path}...")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
        
    y_true_cats = []
    y_pred_cats = []
    
    ref_analysis = []
    cand_analysis = []
    
    ref_laws = []
    cand_laws = []

    for idx, row in df.iterrows():
        result = raw_results[idx]
        if result is None:
            print(f"[{idx}] Skipping metrics due to failed request.")
            continue
            
        gt_cat = str(row.get("Kategori", "")).strip()
        pred_cats = [c.strip() for c in result.get("categories", [])]
        
        y_true_cats.append([gt_cat] if gt_cat else [])
        y_pred_cats.append(pred_cats)
        
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
    
    metrics_log_path = "evaluation_metrics.txt"
    with open(metrics_log_path, "w", encoding="utf-8") as f_out:
        def log_print(msg):
            print(msg)
            f_out.write(msg + "\n")
            
        mlb = MultiLabelBinarizer()
        all_cats = y_true_cats + y_pred_cats
        mlb.fit(all_cats)
        y_true_bin = mlb.transform(y_true_cats)
        y_pred_bin = mlb.transform(y_pred_cats)
        
        log_print(f"Classes found locally: {mlb.classes_}")
        
        acc = accuracy_score(y_true_bin, y_pred_bin)
        log_print(f"Exact Match Accuracy: {acc:.4f}\n")
        
        log_print("Classification Report (Micro/Macro averages):")
        report = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0)
        log_print(report)
        
        log_print("\n" + "="*50)
        log_print("TEXT GENERATION METRICS (BERTScore)")
        log_print("="*50)
        
        log_print("\nCalculating BERTScore for Analysis (Analisis Pelanggaran + Dampak vs Final Summary)...")
        if len(cand_analysis) > 0:
            P, R, F1 = bert_score_fn(cand_analysis, ref_analysis, lang="id", verbose=True)
            log_print(f"Analysis BERTScore -> Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
        else:
            log_print("No analysis text to evaluate.")

        log_print("\nCalculating BERTScore for Laws (Dasar Hukum vs Laws Summary)...")
        if len(cand_laws) > 0:
            P_law, R_law, F1_law = bert_score_fn(cand_laws, ref_laws, lang="id", verbose=True)
            log_print(f"Laws BERTScore     -> Precision: {P_law.mean():.4f}, Recall: {R_law.mean():.4f}, F1: {F1_law.mean():.4f}")
        else:
            log_print("No law text to evaluate.")
            
        log_print("\nEvaluation Complete.")
        print(f"\nMetrics saved to {metrics_log_path}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
