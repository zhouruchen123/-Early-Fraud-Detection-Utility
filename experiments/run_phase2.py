"""
Phase 2: Extended Model Configurations

完全复制 run_three_datasets_v3.py，唯一改动是模型配置从 7 个扩展到 15 个
"""
import modal
import os

app = modal.App("phase2-extended")

volume = modal.Volume.from_name("time-utility-data", create_if_missing=True)
sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

experiment_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas", "numpy", "scikit-learn", "scipy", 
        "matplotlib", "seaborn", "requests"
    )
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", "")
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)

# Experiment parameters - 与 v3 完全一致
T_MAX = 90
K_BUDGET = 30
TRAIN_RATIO = 0.7
DAYS_PER_CASE = 2.0
CONFIDENCE_DELAY_FACTOR = 50
RANDOM_SEEDS = [42, 123, 456, 789, 1024]


@app.function(
    image=experiment_image,
    volumes={"/workspace": volume},
    timeout=86400,  # 24 hours (Modal max)
    cpu=8,
    memory=49152,
    secrets=[modal.Secret.from_name("orchestra-supabase")]
)
def run_experiment():
    """Run experiments on all 3 datasets with multiple seeds."""
    import sys
    sys.path.insert(0, "/root")
    
    import json
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, precision_score, recall_score
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')
    
    from src.orchestra_sdk.experiment import Experiment
    
    exp = Experiment.init(
        name="Phase 2: Extended Models (15 configs)",
        description="基于v3代码，模型从7个扩展到15个",
        config={
            "datasets": ["creditcard", "paysim", "ieee_cis"],
            "random_seeds": RANDOM_SEEDS,
            "T_max": T_MAX,
            "k_budget": K_BUDGET,
            "n_models": 15
        }
    )
    exp.add_tags(['phase2', 'extended-models', '15-configs'])
    
    os.makedirs("/workspace/results/phase2", exist_ok=True)
    os.makedirs("/workspace/plots/phase2", exist_ok=True)
    
    print("="*70)
    print("PHASE 2: EXTENDED MODELS (15 configurations)")
    print("="*70)
    
    # 检查数据集 - 与 v3 完全一致
    dataset_paths = {
        'creditcard': ["/workspace/data/raw/creditcard.csv", "/workspace/data/creditcard.csv"],
        'paysim': ["/workspace/data/raw/paysim.csv"],
        'ieee_cis': ["/workspace/data/raw/ieee_cis_fraud.csv"]
    }
    
    datasets_available = []
    for ds, paths in dataset_paths.items():
        for path in paths:
            if os.path.exists(path):
                datasets_available.append(ds)
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"✅ {ds}: {size_mb:.1f} MB")
                break
        else:
            print(f"❌ {ds}: Not found")
    
    print(f"\nTotal: {len(datasets_available)} datasets × {len(RANDOM_SEEDS)} seeds = {len(datasets_available) * len(RANDOM_SEEDS)} runs")
    
    # =========================================================================
    # 核心函数 - 完全复制自 run_three_datasets_v3.py
    # =========================================================================
    
    def run_single_experiment(dataset_name: str, random_seed: int):
        """Helper function to run experiment on a single dataset."""
        np.random.seed(random_seed)
        
        print(f"\n   Loading {dataset_name}...")
        
        # Load data based on dataset name - 与 v3 完全一致
        if dataset_name == 'creditcard':
            for path in dataset_paths['creditcard']:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break
            target_col = 'Class'
            time_col = 'Time'
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
            feature_cols = [c for c in df.columns if c not in [target_col]]
            
        elif dataset_name == 'ieee_cis':
            df = pd.read_csv("/workspace/data/raw/ieee_cis_fraud.csv")
            target_col = 'isFraud'
            time_col = 'TransactionDT'
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in [target_col]]
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
            
        elif dataset_name == 'paysim':
            df = pd.read_csv("/workspace/data/raw/paysim.csv")
            target_col = 'isFraud'
            time_col = 'step'
            df['type_encoded'] = pd.factorize(df['type'])[0]
            feature_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                           'oldbalanceDest', 'newbalanceDest', 'type_encoded']
            df[feature_cols] = df[feature_cols].fillna(0)
            
            # Stratified sampling for PaySim (too large) - 与 v3 完全一致
            print(f"   Original size: {len(df):,} rows")
            fraud_df = df[df[target_col] == 1]
            non_fraud_df = df[df[target_col] == 0]
            sample_size = min(1000000 - len(fraud_df), len(non_fraud_df))
            non_fraud_sample = non_fraud_df.sample(n=sample_size, random_state=random_seed)
            df = pd.concat([fraud_df, non_fraud_sample]).sort_values(time_col).reset_index(drop=True)
            print(f"   Sampled to: {len(df):,} rows (keeping all {len(fraud_df)} frauds)")
        
        print(f"   Loaded: {len(df):,} rows, {df[target_col].sum()} frauds ({df[target_col].mean()*100:.2f}%)")
        
        # Sort and split - 与 v3 完全一致
        df = df.sort_values(time_col).reset_index(drop=True)
        split_idx = int(len(df) * TRAIN_RATIO)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        
        # ⭐ 关键: 模拟的 transaction_day 从 0 到 T_MAX 均匀分布 - 与 v3 完全一致
        val_df['transaction_day'] = (np.arange(len(val_df)) / len(val_df)) * T_MAX
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # ⭐ 唯一改动: 模型配置从 7 个扩展到 15 个
        print(f"   Training 15 models...")
        model_configs = [
            # 原有 7 个模型 (与 v3 完全一致)
            {"name": "LR-HighRecall", "model": LogisticRegression(C=0.01, class_weight={0:1, 1:100}, max_iter=1000, random_state=random_seed)},
            {"name": "LR-Balanced", "model": LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=random_seed)},
            {"name": "LR-HighPrecision", "model": LogisticRegression(C=10.0, class_weight={0:1, 1:10}, max_iter=1000, random_state=random_seed)},
            {"name": "RF-Balanced", "model": RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=random_seed, n_jobs=-1)},
            {"name": "RF-HighRecall", "model": RandomForestClassifier(n_estimators=100, class_weight={0:1, 1:50}, max_depth=15, random_state=random_seed, n_jobs=-1)},
            {"name": "GB-Balanced", "model": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_seed)},
            {"name": "GB-Conservative", "model": GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=random_seed)},
            # 新增 8 个模型
            {"name": "LR-VeryHighRecall", "model": LogisticRegression(C=0.001, class_weight={0:1, 1:200}, max_iter=1000, random_state=random_seed)},
            {"name": "LR-L1-Balanced", "model": LogisticRegression(C=1.0, penalty='l1', solver='saga', class_weight='balanced', max_iter=1000, random_state=random_seed)},
            {"name": "RF-Deep", "model": RandomForestClassifier(n_estimators=150, class_weight='balanced', max_depth=20, random_state=random_seed, n_jobs=-1)},
            {"name": "RF-Shallow", "model": RandomForestClassifier(n_estimators=50, class_weight='balanced', max_depth=5, random_state=random_seed, n_jobs=-1)},
            {"name": "RF-HighPrecision", "model": RandomForestClassifier(n_estimators=100, class_weight={0:1, 1:5}, max_depth=10, random_state=random_seed, n_jobs=-1)},
            {"name": "GB-Aggressive", "model": GradientBoostingClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, random_state=random_seed)},
            {"name": "GB-Deep", "model": GradientBoostingClassifier(n_estimators=100, max_depth=10, learning_rate=0.05, random_state=random_seed)},
            {"name": "GB-Fast", "model": GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.2, random_state=random_seed)},
        ]
        
        model_predictions = {}
        for config in model_configs:
            model = config['model']
            model.fit(X_train_scaled, y_train)
            y_prob = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = model.predict(X_val_scaled)
            model_predictions[config['name']] = {'prob': y_prob, 'pred': y_pred}
        
        # Evaluate - 与 v3 完全一致
        def calculate_detection_time(transaction_day, confidence, queue_position):
            queue_wait = queue_position * DAYS_PER_CASE
            investigation_time = CONFIDENCE_DELAY_FACTOR * (1 - confidence) + 1
            return min(transaction_day + queue_wait + investigation_time, T_MAX)
        
        def calculate_utility(detection_time, confidence):
            return np.maximum(0, T_MAX - detection_time) * confidence
        
        results = []
        for model_name, pred_data in model_predictions.items():
            eval_df = val_df.copy()
            eval_df['prob_fraud'] = pred_data['prob']
            eval_df['pred'] = pred_data['pred']
            
            f1 = f1_score(eval_df[target_col], pred_data['pred'], zero_division=0)
            precision = precision_score(eval_df[target_col], pred_data['pred'], zero_division=0)
            recall = recall_score(eval_df[target_col], pred_data['pred'], zero_division=0)
            
            top_k = eval_df.nlargest(K_BUDGET, 'prob_fraud').copy()
            
            detection_times = []
            for idx, (_, row) in enumerate(top_k.iterrows()):
                det_time = calculate_detection_time(row['transaction_day'], row['prob_fraud'], idx)
                detection_times.append(det_time)
            top_k['detection_time'] = detection_times
            top_k['utility'] = calculate_utility(top_k['detection_time'], top_k['prob_fraud'])
            
            total_utility = top_k['utility'].sum()
            true_positives = top_k[target_col].sum()
            
            tp_cases = top_k[top_k[target_col] == 1]
            avg_detection_time = tp_cases['detection_time'].mean() if len(tp_cases) > 0 else T_MAX
            
            results.append({
                'model_name': model_name,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'total_utility': total_utility,
                'true_positives': int(true_positives),
                'avg_detection_time': avg_detection_time
            })
        
        results_df = pd.DataFrame(results)
        results_df['f1_rank'] = results_df['f1_score'].rank(ascending=False, method='min').astype(int)
        results_df['utility_rank'] = results_df['total_utility'].rank(ascending=False, method='min').astype(int)
        
        spearman_rho, spearman_p = spearmanr(results_df['f1_rank'], results_df['utility_rank'])
        
        best_f1_model = results_df.loc[results_df['f1_score'].idxmax()]
        best_utility_model = results_df.loc[results_df['total_utility'].idxmax()]
        lead_time_diff = best_f1_model['avg_detection_time'] - best_utility_model['avg_detection_time']
        
        top3_f1 = set(results_df.nsmallest(3, 'f1_rank')['model_name'].values)
        top3_utility = set(results_df.nsmallest(3, 'utility_rank')['model_name'].values)
        top3_overlap = len(top3_f1.intersection(top3_utility))
        top3_difference_pct = (3 - top3_overlap) / 3 * 100
        
        criterion1_met = top3_difference_pct >= 20
        criterion2_met = lead_time_diff >= 15
        criterion3_met = spearman_rho < 0.6
        
        print(f"   Results: ρ={spearman_rho:.3f}, lead_time={lead_time_diff:.1f}d, top3_diff={top3_difference_pct:.0f}%")
        print(f"   Best F1: {best_f1_model['model_name']} (det={best_f1_model['avg_detection_time']:.1f}d)")
        print(f"   Best Util: {best_utility_model['model_name']} (det={best_utility_model['avg_detection_time']:.1f}d)")
        
        return {
            'dataset': dataset_name,
            'random_seed': random_seed,
            'n_samples': len(df),
            'n_frauds': int(df[target_col].sum()),
            'fraud_rate': float(df[target_col].mean()),
            'spearman_rho': float(spearman_rho),
            'spearman_p': float(spearman_p),
            'top3_difference_pct': float(top3_difference_pct),
            'lead_time_diff': float(lead_time_diff),
            'best_f1_model': best_f1_model['model_name'],
            'best_utility_model': best_utility_model['model_name'],
            'best_f1_det': float(best_f1_model['avg_detection_time']),
            'best_util_det': float(best_utility_model['avg_detection_time']),
            'criterion1_met': criterion1_met,
            'criterion2_met': criterion2_met,
            'criterion3_met': criterion3_met,
            'criteria_passed': sum([criterion1_met, criterion2_met, criterion3_met]),
            'model_results': results_df.to_dict(orient='records')
        }
    
    # =========================================================================
    # 运行所有实验 - 与 v3 完全一致
    # =========================================================================
    all_results = []
    total_runs = len(datasets_available) * len(RANDOM_SEEDS)
    current_run = 0
    
    for dataset_name in datasets_available:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        for seed in RANDOM_SEEDS:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] {dataset_name} with seed={seed}")
            
            try:
                result = run_single_experiment(dataset_name, seed)
                all_results.append(result)
                
                exp.log({
                    f"{dataset_name}_seed{seed}_spearman": result['spearman_rho'],
                    f"{dataset_name}_seed{seed}_lead_time": result['lead_time_diff'],
                    f"{dataset_name}_seed{seed}_criteria": result['criteria_passed']
                }, step=current_run)
                
            except Exception as e:
                print(f"⚠️ Error on {dataset_name} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
        
        exp.set_progress(int(current_run / total_runs * 80))
    
    # =========================================================================
    # 汇总结果 - 与 v3 完全一致
    # =========================================================================
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    
    print("\n📊 Summary by Dataset:")
    for ds in datasets_available:
        ds_results = results_df[results_df['dataset'] == ds]
        print(f"\n{ds}:")
        print(f"   Spearman ρ: {ds_results['spearman_rho'].mean():.4f} ± {ds_results['spearman_rho'].std():.4f}")
        print(f"   Lead Time: {ds_results['lead_time_diff'].mean():.2f} ± {ds_results['lead_time_diff'].std():.2f} days")
        print(f"   Top-3 Diff: {ds_results['top3_difference_pct'].mean():.1f}% ± {ds_results['top3_difference_pct'].std():.1f}%")
        print(f"   All 3 criteria met: {(ds_results['criteria_passed'] == 3).sum()}/{len(ds_results)}")
    
    overall_stats = {
        'spearman_rho_mean': float(results_df['spearman_rho'].mean()),
        'spearman_rho_std': float(results_df['spearman_rho'].std()),
        'top3_diff_mean': float(results_df['top3_difference_pct'].mean()),
        'top3_diff_std': float(results_df['top3_difference_pct'].std()),
        'lead_time_mean': float(results_df['lead_time_diff'].mean()),
        'lead_time_std': float(results_df['lead_time_diff'].std()),
        'all_criteria_met_rate': float((results_df['criteria_passed'] == 3).mean()),
        'at_least_2_criteria_rate': float((results_df['criteria_passed'] >= 2).mean())
    }
    
    print(f"\n📈 Overall Statistics:")
    for k, v in overall_stats.items():
        print(f"   {k}: {v}")
    
    # =========================================================================
    # 可视化 - 与 v3 完全一致
    # =========================================================================
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    dataset_order = sorted(datasets_available)
    colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(dataset_order)]
    
    # 1. Spearman ρ
    ax = axes[0, 0]
    for i, ds in enumerate(dataset_order):
        ds_data = results_df[results_df['dataset'] == ds]['spearman_rho']
        ax.bar(i, ds_data.mean(), yerr=ds_data.std(), color=colors[i], 
               edgecolor='black', capsize=5, alpha=0.8)
    ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Threshold (0.6)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, fontsize=11)
    ax.set_ylabel("Spearman's ρ", fontsize=12, fontweight='bold')
    ax.set_title("Rank Correlation (15 models)", fontsize=12, fontweight='bold')
    ax.legend()
    
    # 2. Lead Time
    ax = axes[0, 1]
    for i, ds in enumerate(dataset_order):
        ds_data = results_df[results_df['dataset'] == ds]['lead_time_diff']
        ax.bar(i, ds_data.mean(), yerr=ds_data.std(), color=colors[i],
               edgecolor='black', capsize=5, alpha=0.8)
    ax.axhline(y=15, color='green', linestyle='--', linewidth=2, label='Target (15 days)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, fontsize=11)
    ax.set_ylabel("Lead Time (days)", fontsize=12, fontweight='bold')
    ax.set_title("Detection Time Advantage", fontsize=12, fontweight='bold')
    ax.legend()
    
    # 3. Top-3 Difference
    ax = axes[1, 0]
    for i, ds in enumerate(dataset_order):
        ds_data = results_df[results_df['dataset'] == ds]['top3_difference_pct']
        ax.bar(i, ds_data.mean(), yerr=ds_data.std(), color=colors[i],
               edgecolor='black', capsize=5, alpha=0.8)
    ax.axhline(y=20, color='green', linestyle='--', linewidth=2, label='Target (20%)')
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, fontsize=11)
    ax.set_ylabel("Top-3 Difference (%)", fontsize=12, fontweight='bold')
    ax.set_title("Ranking Disagreement", fontsize=12, fontweight='bold')
    ax.legend()
    
    # 4. Success heatmap
    ax = axes[1, 1]
    criteria_matrix = []
    for ds in dataset_order:
        ds_results = results_df[results_df['dataset'] == ds]
        criteria_matrix.append([
            ds_results['criterion1_met'].mean() * 100,
            ds_results['criterion2_met'].mean() * 100,
            ds_results['criterion3_met'].mean() * 100,
            (ds_results['criteria_passed'] == 3).mean() * 100
        ])
    
    criteria_df = pd.DataFrame(
        criteria_matrix,
        index=dataset_order,
        columns=['Top-3≥20%', 'Lead≥15d', 'ρ<0.6', 'All Met']
    )
    
    sns.heatmap(criteria_df, annot=True, fmt='.0f', cmap='RdYlGn', 
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Success Rate (%)'})
    ax.set_title('Success Criteria by Dataset', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/workspace/plots/phase2/metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ metrics_summary.png")
    
    # Box plots - 与 v3 完全一致
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    results_df.boxplot(column='spearman_rho', by='dataset', ax=ax)
    ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2)
    ax.set_title("Spearman's ρ Distribution")
    ax.set_xlabel("Dataset")
    plt.suptitle('')
    
    ax = axes[1]
    results_df.boxplot(column='lead_time_diff', by='dataset', ax=ax)
    ax.axhline(y=15, color='green', linestyle='--', linewidth=2)
    ax.set_title("Lead Time Distribution")
    ax.set_xlabel("Dataset")
    plt.suptitle('')
    
    ax = axes[2]
    results_df.boxplot(column='top3_difference_pct', by='dataset', ax=ax)
    ax.axhline(y=20, color='green', linestyle='--', linewidth=2)
    ax.set_title("Top-3 Difference Distribution")
    ax.set_xlabel("Dataset")
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('/workspace/plots/phase2/distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ distributions.png")
    
    # =========================================================================
    # 保存结果 - 与 v3 完全一致
    # =========================================================================
    results_df.to_csv('/workspace/results/phase2/all_results.csv', index=False)
    
    full_summary = {
        'experiment_date': datetime.now().isoformat(),
        'datasets': datasets_available,
        'random_seeds': RANDOM_SEEDS,
        'parameters': {
            'T_max': T_MAX,
            'k_budget': K_BUDGET,
            'days_per_case': DAYS_PER_CASE,
            'confidence_delay_factor': CONFIDENCE_DELAY_FACTOR
        },
        'overall_statistics': overall_stats,
        'by_dataset': {
            ds: {
                'n_samples': int(results_df[results_df['dataset']==ds]['n_samples'].iloc[0]),
                'fraud_rate': float(results_df[results_df['dataset']==ds]['fraud_rate'].iloc[0]),
                'spearman_rho_mean': float(results_df[results_df['dataset']==ds]['spearman_rho'].mean()),
                'spearman_rho_std': float(results_df[results_df['dataset']==ds]['spearman_rho'].std()),
                'lead_time_mean': float(results_df[results_df['dataset']==ds]['lead_time_diff'].mean()),
                'lead_time_std': float(results_df[results_df['dataset']==ds]['lead_time_diff'].std()),
                'top3_diff_mean': float(results_df[results_df['dataset']==ds]['top3_difference_pct'].mean()),
                'criteria_all_met': int((results_df[results_df['dataset']==ds]['criteria_passed'] == 3).sum()),
                'total_runs': int(len(results_df[results_df['dataset']==ds]))
            }
            for ds in datasets_available
        }
    }
    
    with open('/workspace/results/phase2/summary.json', 'w') as f:
        json.dump(full_summary, f, indent=2, default=str)
    
    volume.commit()
    
    exp.log({
        'overall_spearman_mean': overall_stats['spearman_rho_mean'],
        'overall_lead_time_mean': overall_stats['lead_time_mean'],
        'overall_top3_diff_mean': overall_stats['top3_diff_mean'],
        'all_criteria_met_rate': overall_stats['all_criteria_met_rate']
    }, step=100)
    
    exp.set_progress(100)
    exp.finish('completed')
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return full_summary


@app.local_entrypoint()
def main():
    print("🚀 Running Phase 2: Extended Models (15 configurations)...")
    result = run_experiment.remote()
    if result:
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        for k, v in result['overall_statistics'].items():
            print(f"   {k}: {v}")
