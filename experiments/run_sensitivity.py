"""
Simplified Sensitivity Analysis - 简化版敏感性分析
减少模型数量，加快实验速度
"""
import modal
import os

app = modal.App("sensitivity-simple")

volume = modal.Volume.from_name("time-utility-data", create_if_missing=True)
sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

experiment_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "numpy", "scikit-learn", "scipy", "requests")
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", "")
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)

TRAIN_RATIO = 0.7
DAYS_PER_CASE = 2.0
CONFIDENCE_DELAY_FACTOR = 50
RANDOM_SEED = 42


@app.function(
    image=experiment_image,
    volumes={"/workspace": volume},
    timeout=3600,
    cpu=4,
    memory=16384,
)
def run_task(dataset: str, T_max: int, k_budget: int):
    """运行单个敏感性分析任务."""
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from scipy.stats import spearmanr
    import warnings
    warnings.filterwarnings('ignore')
    
    task_id = f"T{T_max}_K{k_budget}_{dataset}"
    print(f"\n{'='*50}")
    print(f"Task: {task_id}")
    print(f"{'='*50}")
    
    np.random.seed(RANDOM_SEED)
    
    # Load data
    if dataset == 'creditcard':
        for path in ["/workspace/data/raw/creditcard.csv", "/workspace/data/creditcard.csv"]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        target_col = 'Class'
        time_col = 'Time'
        amount_col = 'Amount'
        feature_cols = [c for c in df.columns if c not in [target_col]]
        
    elif dataset == 'ieee_cis':
        df = pd.read_csv("/workspace/data/raw/ieee_cis_fraud.csv")
        target_col = 'isFraud'
        time_col = 'TransactionDT'
        amount_col = 'TransactionAmt'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in [target_col]]
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
    elif dataset == 'paysim':
        df = pd.read_csv("/workspace/data/raw/paysim.csv")
        target_col = 'isFraud'
        time_col = 'step'
        amount_col = 'amount'
        df['type_encoded'] = pd.factorize(df['type'])[0]
        feature_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest', 'type_encoded']
        # Sample
        fraud_df = df[df[target_col] == 1]
        non_fraud_df = df[df[target_col] == 0].sample(n=min(500000, len(df[df[target_col]==0])), random_state=RANDOM_SEED)
        df = pd.concat([fraud_df, non_fraud_df]).sort_values(time_col).reset_index(drop=True)
    
    print(f"Loaded: {len(df):,} rows, {df[target_col].sum()} frauds")
    
    # Split
    df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    val_df['transaction_day'] = (np.arange(len(val_df)) / len(val_df)) * T_max
    val_df['amount'] = df[amount_col].iloc[split_idx:].values if amount_col in df.columns else 100
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 8 models (simplified)
    models = {
        'LR-Aggressive': LogisticRegression(C=0.01, class_weight={0:1, 1:100}, max_iter=500, random_state=RANDOM_SEED),
        'LR-Balanced': LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=RANDOM_SEED),
        'RF-Balanced': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
        'RF-Deep': RandomForestClassifier(n_estimators=50, max_depth=15, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
        'RF-HighPrecision': RandomForestClassifier(n_estimators=50, class_weight={0:1, 1:50}, random_state=RANDOM_SEED, n_jobs=-1),
        'GB-Standard': GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=RANDOM_SEED),
        'GB-Aggressive': GradientBoostingClassifier(n_estimators=50, max_depth=7, learning_rate=0.1, random_state=RANDOM_SEED),
        'GB-Fast': GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED),
    }
    
    def calc_detection_time(trans_day, conf, queue_pos):
        queue_wait = queue_pos * DAYS_PER_CASE
        invest_time = CONFIDENCE_DELAY_FACTOR * (1 - conf) + 1
        return min(trans_day + queue_wait + invest_time, T_max)
    
    def calc_utility(det_time, conf):
        return max(0, T_max - det_time) * conf
    
    def evaluate_tau(eval_df, y_prob):
        eval_df = eval_df.copy()
        eval_df['prob_fraud'] = y_prob
        top_k = eval_df.nlargest(k_budget, 'prob_fraud').copy()
        
        det_times = []
        for idx, (_, row) in enumerate(top_k.iterrows()):
            det_time = calc_detection_time(row['transaction_day'], row['prob_fraud'], idx)
            det_times.append(det_time)
        top_k['det_time'] = det_times
        top_k['utility'] = [calc_utility(d, c) for d, c in zip(top_k['det_time'], top_k['prob_fraud'])]
        
        total_util = top_k['utility'].sum()
        tp = top_k[target_col].sum()
        tp_cases = top_k[top_k[target_col] == 1]
        avg_det = tp_cases['det_time'].mean() if len(tp_cases) > 0 else T_max
        
        return total_util, int(tp), avg_det
    
    results = []
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_prob = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            
            f1 = f1_score(y_val, y_pred)
            tau_util, tau_tp, tau_det = evaluate_tau(val_df, y_prob)
            
            results.append({
                'model': name,
                'f1': f1,
                'tau_util': tau_util,
                'tau_det': tau_det,
            })
        except Exception as e:
            print(f"  Error {name}: {e}")
    
    if len(results) < 2:
        return None
    
    results_df = pd.DataFrame(results)
    results_df['f1_rank'] = results_df['f1'].rank(ascending=False)
    results_df['tau_rank'] = results_df['tau_util'].rank(ascending=False)
    
    rho, _ = spearmanr(results_df['f1_rank'], results_df['tau_rank'])
    
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    best_tau = results_df.loc[results_df['tau_util'].idxmax()]
    
    lead_time = best_f1['tau_det'] - best_tau['tau_det']
    
    top3_f1 = set(results_df.nsmallest(3, 'f1_rank')['model'].values)
    top3_tau = set(results_df.nsmallest(3, 'tau_rank')['model'].values)
    top3_diff = (3 - len(top3_f1.intersection(top3_tau))) / 3 * 100
    
    c1 = rho < 0.6
    c2 = lead_time >= 15
    c3 = top3_diff >= 20
    
    print(f"✅ ρ={rho:.3f}, lead={lead_time:.1f}d, top3={top3_diff:.0f}%")
    print(f"   Best F1: {best_f1['model']} (det={best_f1['tau_det']:.1f}d)")
    print(f"   Best TAU: {best_tau['model']} (det={best_tau['tau_det']:.1f}d)")
    
    return {
        'task_id': task_id,
        'dataset': dataset,
        'T_max': T_max,
        'k_budget': k_budget,
        'spearman_rho': float(rho),
        'lead_time': float(lead_time),
        'top3_diff': float(top3_diff),
        'best_f1_model': best_f1['model'],
        'best_tau_model': best_tau['model'],
        'c1_rho': bool(c1),
        'c2_lead': bool(c2),
        'c3_top3': bool(c3),
        'all_pass': bool(c1 and c2 and c3)
    }


@app.local_entrypoint()
def main():
    import json
    
    print("=" * 60)
    print("🚀 Simplified Sensitivity Analysis")
    print("=" * 60)
    
    # Parameters to test
    T_values = [30, 60, 90, 120, 150, 180]
    K_values = [10, 20, 30, 50, 75, 100]
    datasets = ['creditcard', 'paysim', 'ieee_cis']
    
    # Build tasks: T sensitivity (k=30) + K sensitivity (T=90)
    tasks = []
    
    # T_max sensitivity
    for t in T_values:
        for d in datasets:
            tasks.append((d, t, 30))
    
    # K sensitivity (skip T=90, k=30 already covered)
    for k in K_values:
        if k != 30:
            for d in datasets:
                tasks.append((d, 90, k))
    
    print(f"\n📋 Total: {len(tasks)} tasks")
    print(f"   T_max: {T_values} (k=30)")
    print(f"   K: {K_values} (T=90)")
    
    # Run
    print("\n🔄 Running...")
    results = []
    for r in run_task.starmap(tasks):
        if r:
            results.append(r)
            status = "✅" if r['all_pass'] else "⚠️"
            print(f"  {status} {r['task_id']}: ρ={r['spearman_rho']:.2f}, lead={r['lead_time']:.1f}d")
    
    print(f"\n✅ Completed {len(results)}/{len(tasks)} tasks")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    import numpy as np
    
    print("\n📈 T_max Sensitivity (k=30):")
    print("| T_max | Lead Time | ρ | Success |")
    print("|-------|-----------|---|---------|")
    for t in T_values:
        sub = [r for r in results if r['T_max'] == t and r['k_budget'] == 30]
        if sub:
            lt = np.mean([r['lead_time'] for r in sub])
            rho = np.mean([r['spearman_rho'] for r in sub])
            sr = np.mean([r['all_pass'] for r in sub]) * 100
            print(f"| {t:5} | {lt:9.1f} | {rho:.2f} | {sr:.0f}% |")
    
    print("\n📈 K Budget Sensitivity (T=90):")
    print("| K | Lead Time | ρ | Success |")
    print("|---|-----------|---|---------|")
    for k in K_values:
        sub = [r for r in results if r['k_budget'] == k and r['T_max'] == 90]
        if sub:
            lt = np.mean([r['lead_time'] for r in sub])
            rho = np.mean([r['spearman_rho'] for r in sub])
            sr = np.mean([r['all_pass'] for r in sub]) * 100
            print(f"| {k:3} | {lt:9.1f} | {rho:.2f} | {sr:.0f}% |")
    
    # Save
    with open('/workspace/sensitivity_simple_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    volume.commit()
    
    print("\n✅ Saved to /workspace/sensitivity_simple_results.json")
    
    return results


if __name__ == "__main__":
    main()
