"""
Phase 3: Baseline Comparisons - 基线方法对比实验

对比5种主流方法与我们的 Time-Aware Utility 方法:
1. Cost-Sensitive Learning (Instance-Dependent)
2. F1-Optimal Threshold Selection
3. SMOTE + Standard Classifier
4. Class-Weight Balanced Models
5. Profit-Optimal (Expected Savings) Selection

使用多进程并行执行加速实验
"""
import modal
import os

app = modal.App("phase3-baselines")

volume = modal.Volume.from_name("time-utility-data", create_if_missing=True)
sdk_path = os.environ.get('ORCHESTRA_SDK_PATH', '/root/vm_worker/src')

experiment_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas", "numpy", "scikit-learn", "scipy", 
        "matplotlib", "seaborn", "requests", "imbalanced-learn"
    )
    .env({
        "AGENT_ID": os.getenv("AGENT_ID", ""),
        "PROJECT_ID": os.getenv("PROJECT_ID", ""),
        "USER_ID": os.getenv("USER_ID", "")
    })
    .add_local_dir(sdk_path, remote_path="/root/src")
)

# 实验参数 (使用 Phase 2 确定的最佳参数)
T_MAX = 90
K_BUDGET = 30
DAYS_PER_CASE = 2.0
CONFIDENCE_DELAY_FACTOR = 50
TRAIN_RATIO = 0.7
N_SEEDS = 5  # 5个随机种子确保统计稳健性
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

# 数据集
DATASETS = ['creditcard', 'paysim', 'ieee_cis']

# Baseline 方法
BASELINE_METHODS = [
    'time_aware_utility',      # 我们的方法 (baseline for comparison)
    'cost_sensitive',          # Instance-dependent cost-sensitive
    'f1_optimal_threshold',    # F1-optimal threshold selection
    'smote_classifier',        # SMOTE + standard classifier
    'class_weight_balanced',   # Class-weight balanced models
    'profit_optimal',          # Expected savings optimization
]


@app.function(
    image=experiment_image,
    volumes={"/workspace": volume},
    timeout=14400,  # 4 hours for large datasets
    cpu=8,  # More CPU for faster training
    memory=49152,  # More memory
)
def run_baseline_task(task: dict):
    """运行单个baseline对比任务."""
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
    from scipy.stats import spearmanr
    import warnings
    warnings.filterwarnings('ignore')
    
    dataset_name = task['dataset']
    seed = task['seed']
    task_id = task['task_id']
    
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Dataset: {dataset_name}, Seed: {seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
    # =========================================================================
    # 1. 加载数据
    # =========================================================================
    print(f"Loading {dataset_name}...")
    
    if dataset_name == 'creditcard':
        for path in ["/workspace/data/raw/creditcard.csv", "/workspace/data/creditcard.csv"]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        target_col = 'Class'
        time_col = 'Time'
        amount_col = 'Amount'
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
        feature_cols = [c for c in df.columns if c not in [target_col]]
        
    elif dataset_name == 'ieee_cis':
        df = pd.read_csv("/workspace/data/raw/ieee_cis_fraud.csv")
        target_col = 'isFraud'
        time_col = 'TransactionDT'
        amount_col = 'TransactionAmt'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in [target_col]]
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
    elif dataset_name == 'paysim':
        df = pd.read_csv("/workspace/data/raw/paysim.csv")
        target_col = 'isFraud'
        time_col = 'step'
        amount_col = 'amount'
        df['type_encoded'] = pd.factorize(df['type'])[0]
        feature_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest', 'type_encoded']
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Stratified sampling for PaySim
        print(f"Original size: {len(df):,} rows")
        fraud_df = df[df[target_col] == 1]
        non_fraud_df = df[df[target_col] == 0]
        sample_size = min(1000000 - len(fraud_df), len(non_fraud_df))
        non_fraud_sample = non_fraud_df.sample(n=sample_size, random_state=seed)
        df = pd.concat([fraud_df, non_fraud_sample]).sort_values(time_col).reset_index(drop=True)
        print(f"Sampled to: {len(df):,} rows (keeping all {len(fraud_df)} frauds)")
    
    print(f"Loaded: {len(df):,} rows, {df[target_col].sum()} frauds ({df[target_col].mean()*100:.2f}%)")
    
    # Sort and split
    df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    # 设置 transaction_day
    val_df['transaction_day'] = (np.arange(len(val_df)) / len(val_df)) * T_MAX
    
    # 确保 amount 列存在
    if amount_col not in val_df.columns:
        val_df['amount'] = 100  # 默认金额
    else:
        val_df['amount'] = val_df[amount_col].fillna(100)
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # =========================================================================
    # 2. 定义模型配置 (15个模型，与Phase 2一致)
    # =========================================================================
    model_configs = [
        {"name": "LR-HighRecall", "model": LogisticRegression(C=0.01, class_weight={0:1, 1:100}, max_iter=1000, random_state=seed)},
        {"name": "LR-Balanced", "model": LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=seed)},
        {"name": "LR-HighPrecision", "model": LogisticRegression(C=10.0, class_weight={0:1, 1:10}, max_iter=1000, random_state=seed)},
        {"name": "RF-Balanced", "model": RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=seed, n_jobs=-1)},
        {"name": "RF-HighRecall", "model": RandomForestClassifier(n_estimators=100, class_weight={0:1, 1:50}, max_depth=15, random_state=seed, n_jobs=-1)},
        {"name": "GB-Balanced", "model": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed)},
        {"name": "GB-Conservative", "model": GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=seed)},
        {"name": "LR-VeryHighRecall", "model": LogisticRegression(C=0.001, class_weight={0:1, 1:200}, max_iter=1000, random_state=seed)},
        {"name": "LR-L1-Balanced", "model": LogisticRegression(C=1.0, penalty='l1', solver='saga', class_weight='balanced', max_iter=1000, random_state=seed)},
        {"name": "RF-Deep", "model": RandomForestClassifier(n_estimators=150, class_weight='balanced', max_depth=20, random_state=seed, n_jobs=-1)},
        {"name": "RF-Shallow", "model": RandomForestClassifier(n_estimators=50, class_weight='balanced', max_depth=5, random_state=seed, n_jobs=-1)},
        {"name": "RF-HighPrecision", "model": RandomForestClassifier(n_estimators=100, class_weight={0:1, 1:5}, max_depth=10, random_state=seed, n_jobs=-1)},
        {"name": "GB-Aggressive", "model": GradientBoostingClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, random_state=seed)},
        {"name": "GB-Deep", "model": GradientBoostingClassifier(n_estimators=100, max_depth=10, learning_rate=0.05, random_state=seed)},
        {"name": "GB-Fast", "model": GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.2, random_state=seed)},
    ]
    
    # =========================================================================
    # 3. 训练所有模型
    # =========================================================================
    print("Training 15 models...")
    model_predictions = {}
    for config in model_configs:
        model = config['model']
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]
        y_pred = model.predict(X_val_scaled)
        model_predictions[config['name']] = {'prob': y_prob, 'pred': y_pred, 'model': model}
    
    # =========================================================================
    # 4. 定义评估函数
    # =========================================================================
    def calculate_detection_time(transaction_day, confidence, queue_position):
        """计算检测时间."""
        queue_wait = queue_position * DAYS_PER_CASE
        investigation_time = CONFIDENCE_DELAY_FACTOR * (1 - confidence) + 1
        return min(transaction_day + queue_wait + investigation_time, T_MAX)
    
    def calculate_utility(detection_time, confidence):
        """计算时间感知效用."""
        return np.maximum(0, T_MAX - detection_time) * confidence
    
    def evaluate_time_aware_utility(eval_df, y_prob, target_col):
        """方法1: 我们的Time-Aware Utility方法."""
        eval_df = eval_df.copy()
        eval_df['prob_fraud'] = y_prob
        
        # 按概率排序，取top-k
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
        
        return {
            'total_utility': total_utility,
            'true_positives': int(true_positives),
            'avg_detection_time': avg_detection_time
        }
    
    def evaluate_cost_sensitive(eval_df, y_prob, target_col, c_f=10.0):
        """方法2: Instance-Dependent Cost-Sensitive Learning.
        
        基于 Höppner et al. (2021) 的方法:
        - 最优阈值: t* = c_f / Amount
        - 成本矩阵: C(FN) = Amount, C(FP) = c_f
        """
        eval_df = eval_df.copy()
        eval_df['prob_fraud'] = y_prob
        
        # Instance-dependent threshold
        eval_df['threshold'] = c_f / (eval_df['amount'] + 1e-6)  # 避免除零
        eval_df['threshold'] = eval_df['threshold'].clip(0, 1)
        
        # 使用 instance-dependent threshold 预测
        eval_df['pred_cs'] = (eval_df['prob_fraud'] > eval_df['threshold']).astype(int)
        
        # 选择预测为欺诈的案例，按概率排序取top-k
        fraud_pred = eval_df[eval_df['pred_cs'] == 1].nlargest(K_BUDGET, 'prob_fraud')
        
        if len(fraud_pred) == 0:
            # 如果没有预测为欺诈的，回退到top-k
            fraud_pred = eval_df.nlargest(K_BUDGET, 'prob_fraud')
        
        detection_times = []
        for idx, (_, row) in enumerate(fraud_pred.iterrows()):
            det_time = calculate_detection_time(row['transaction_day'], row['prob_fraud'], idx)
            detection_times.append(det_time)
        fraud_pred = fraud_pred.copy()
        fraud_pred['detection_time'] = detection_times
        
        # 计算 cost savings
        total_fraud_amount = eval_df[eval_df[target_col] == 1]['amount'].sum()
        detected_fraud_amount = fraud_pred[fraud_pred[target_col] == 1]['amount'].sum()
        cost_savings = detected_fraud_amount / (total_fraud_amount + 1e-6)
        
        tp_cases = fraud_pred[fraud_pred[target_col] == 1]
        avg_detection_time = tp_cases['detection_time'].mean() if len(tp_cases) > 0 else T_MAX
        
        return {
            'total_utility': cost_savings * 100,  # 转换为百分比以便比较
            'true_positives': int(fraud_pred[target_col].sum()),
            'avg_detection_time': avg_detection_time,
            'cost_savings': cost_savings
        }
    
    def evaluate_f1_optimal_threshold(eval_df, y_prob, y_val, target_col):
        """方法3: F1-Optimal Threshold Selection.
        
        找到最大化F1的阈值，然后用该阈值选择案例。
        """
        from sklearn.metrics import f1_score
        
        eval_df = eval_df.copy()
        eval_df['prob_fraud'] = y_prob
        
        # 搜索最优阈值
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            y_pred_thresh = (y_prob > thresh).astype(int)
            f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        # 使用最优阈值预测
        eval_df['pred_f1'] = (eval_df['prob_fraud'] > best_threshold).astype(int)
        
        # 选择预测为欺诈的案例
        fraud_pred = eval_df[eval_df['pred_f1'] == 1].nlargest(K_BUDGET, 'prob_fraud')
        
        if len(fraud_pred) == 0:
            fraud_pred = eval_df.nlargest(K_BUDGET, 'prob_fraud')
        
        detection_times = []
        for idx, (_, row) in enumerate(fraud_pred.iterrows()):
            det_time = calculate_detection_time(row['transaction_day'], row['prob_fraud'], idx)
            detection_times.append(det_time)
        fraud_pred = fraud_pred.copy()
        fraud_pred['detection_time'] = detection_times
        
        tp_cases = fraud_pred[fraud_pred[target_col] == 1]
        avg_detection_time = tp_cases['detection_time'].mean() if len(tp_cases) > 0 else T_MAX
        
        return {
            'total_utility': best_f1 * 100,  # F1 score as utility proxy
            'true_positives': int(fraud_pred[target_col].sum()),
            'avg_detection_time': avg_detection_time,
            'optimal_threshold': best_threshold,
            'best_f1': best_f1
        }
    
    def evaluate_smote_classifier(X_train_scaled, y_train, X_val_scaled, eval_df, target_col, seed):
        """方法4: SMOTE + Standard Classifier.
        
        使用SMOTE过采样后训练标准分类器。
        """
        from imblearn.over_sampling import SMOTE
        
        eval_df = eval_df.copy()
        
        # 应用SMOTE
        try:
            smote = SMOTE(random_state=seed, k_neighbors=min(5, sum(y_train == 1) - 1))
            X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        except:
            # 如果SMOTE失败，使用原始数据
            X_resampled, y_resampled = X_train_scaled, y_train
        
        # 训练标准LR (无class_weight)
        model_smote = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        model_smote.fit(X_resampled, y_resampled)
        
        y_prob_smote = model_smote.predict_proba(X_val_scaled)[:, 1]
        eval_df['prob_fraud'] = y_prob_smote
        
        # 按概率排序取top-k
        top_k = eval_df.nlargest(K_BUDGET, 'prob_fraud').copy()
        
        detection_times = []
        for idx, (_, row) in enumerate(top_k.iterrows()):
            det_time = calculate_detection_time(row['transaction_day'], row['prob_fraud'], idx)
            detection_times.append(det_time)
        top_k['detection_time'] = detection_times
        
        tp_cases = top_k[top_k[target_col] == 1]
        avg_detection_time = tp_cases['detection_time'].mean() if len(tp_cases) > 0 else T_MAX
        
        return {
            'total_utility': top_k[target_col].sum() / K_BUDGET * 100,  # Precision as utility proxy
            'true_positives': int(top_k[target_col].sum()),
            'avg_detection_time': avg_detection_time
        }
    
    def evaluate_class_weight_balanced(eval_df, y_prob, target_col):
        """方法5: Class-Weight Balanced Models.
        
        使用 class_weight='balanced' 的模型，标准阈值0.5。
        """
        eval_df = eval_df.copy()
        eval_df['prob_fraud'] = y_prob
        
        # 使用标准阈值 0.5
        eval_df['pred_balanced'] = (eval_df['prob_fraud'] > 0.5).astype(int)
        
        # 选择预测为欺诈的案例
        fraud_pred = eval_df[eval_df['pred_balanced'] == 1].nlargest(K_BUDGET, 'prob_fraud')
        
        if len(fraud_pred) == 0:
            fraud_pred = eval_df.nlargest(K_BUDGET, 'prob_fraud')
        
        detection_times = []
        for idx, (_, row) in enumerate(fraud_pred.iterrows()):
            det_time = calculate_detection_time(row['transaction_day'], row['prob_fraud'], idx)
            detection_times.append(det_time)
        fraud_pred = fraud_pred.copy()
        fraud_pred['detection_time'] = detection_times
        
        tp_cases = fraud_pred[fraud_pred[target_col] == 1]
        avg_detection_time = tp_cases['detection_time'].mean() if len(tp_cases) > 0 else T_MAX
        
        return {
            'total_utility': fraud_pred[target_col].sum() / len(fraud_pred) * 100 if len(fraud_pred) > 0 else 0,
            'true_positives': int(fraud_pred[target_col].sum()),
            'avg_detection_time': avg_detection_time
        }
    
    def evaluate_profit_optimal(eval_df, y_prob, target_col):
        """方法6: Profit-Optimal (Expected Savings) Selection.
        
        基于 Expected Savings 选择模型:
        Savings = (detected_fraud_amount - investigation_cost) / total_fraud_amount
        """
        eval_df = eval_df.copy()
        eval_df['prob_fraud'] = y_prob
        
        # 计算每个案例的期望收益
        c_investigation = 10.0  # 调查成本
        eval_df['expected_profit'] = eval_df['prob_fraud'] * eval_df['amount'] - c_investigation
        
        # 按期望收益排序，选择正收益的案例
        profitable = eval_df[eval_df['expected_profit'] > 0].nlargest(K_BUDGET, 'expected_profit')
        
        if len(profitable) == 0:
            profitable = eval_df.nlargest(K_BUDGET, 'prob_fraud')
        
        detection_times = []
        for idx, (_, row) in enumerate(profitable.iterrows()):
            det_time = calculate_detection_time(row['transaction_day'], row['prob_fraud'], idx)
            detection_times.append(det_time)
        profitable = profitable.copy()
        profitable['detection_time'] = detection_times
        
        # 计算 expected savings
        total_fraud_amount = eval_df[eval_df[target_col] == 1]['amount'].sum()
        detected_fraud_amount = profitable[profitable[target_col] == 1]['amount'].sum()
        investigation_cost = len(profitable) * c_investigation
        savings = (detected_fraud_amount - investigation_cost) / (total_fraud_amount + 1e-6)
        
        tp_cases = profitable[profitable[target_col] == 1]
        avg_detection_time = tp_cases['detection_time'].mean() if len(tp_cases) > 0 else T_MAX
        
        return {
            'total_utility': savings * 100,
            'true_positives': int(profitable[target_col].sum()),
            'avg_detection_time': avg_detection_time,
            'expected_savings': savings
        }
    
    # =========================================================================
    # 5. 对每个模型运行所有baseline方法
    # =========================================================================
    print("\nEvaluating all baseline methods...")
    
    all_results = []
    
    for model_name, pred_data in model_predictions.items():
        y_prob = pred_data['prob']
        y_pred = pred_data['pred']
        
        # 基础指标
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        # 方法1: Time-Aware Utility (我们的方法)
        result_tau = evaluate_time_aware_utility(val_df, y_prob, target_col)
        
        # 方法2: Cost-Sensitive
        result_cs = evaluate_cost_sensitive(val_df, y_prob, target_col)
        
        # 方法3: F1-Optimal Threshold
        result_f1opt = evaluate_f1_optimal_threshold(val_df, y_prob, y_val, target_col)
        
        # 方法4: SMOTE (需要重新训练)
        result_smote = evaluate_smote_classifier(X_train_scaled, y_train, X_val_scaled, val_df, target_col, seed)
        
        # 方法5: Class-Weight Balanced
        result_balanced = evaluate_class_weight_balanced(val_df, y_prob, target_col)
        
        # 方法6: Profit-Optimal
        result_profit = evaluate_profit_optimal(val_df, y_prob, target_col)
        
        all_results.append({
            'model_name': model_name,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            # Time-Aware Utility
            'tau_utility': result_tau['total_utility'],
            'tau_tp': result_tau['true_positives'],
            'tau_det_time': result_tau['avg_detection_time'],
            # Cost-Sensitive
            'cs_utility': result_cs['total_utility'],
            'cs_tp': result_cs['true_positives'],
            'cs_det_time': result_cs['avg_detection_time'],
            # F1-Optimal
            'f1opt_utility': result_f1opt['total_utility'],
            'f1opt_tp': result_f1opt['true_positives'],
            'f1opt_det_time': result_f1opt['avg_detection_time'],
            # SMOTE
            'smote_utility': result_smote['total_utility'],
            'smote_tp': result_smote['true_positives'],
            'smote_det_time': result_smote['avg_detection_time'],
            # Class-Weight Balanced
            'balanced_utility': result_balanced['total_utility'],
            'balanced_tp': result_balanced['true_positives'],
            'balanced_det_time': result_balanced['avg_detection_time'],
            # Profit-Optimal
            'profit_utility': result_profit['total_utility'],
            'profit_tp': result_profit['true_positives'],
            'profit_det_time': result_profit['avg_detection_time'],
        })
    
    results_df = pd.DataFrame(all_results)
    
    # =========================================================================
    # 6. 计算各方法的排名和对比指标
    # =========================================================================
    
    # 为每种方法计算排名
    results_df['f1_rank'] = results_df['f1_score'].rank(ascending=False, method='min').astype(int)
    results_df['tau_rank'] = results_df['tau_utility'].rank(ascending=False, method='min').astype(int)
    results_df['cs_rank'] = results_df['cs_utility'].rank(ascending=False, method='min').astype(int)
    results_df['f1opt_rank'] = results_df['f1opt_utility'].rank(ascending=False, method='min').astype(int)
    results_df['smote_rank'] = results_df['smote_utility'].rank(ascending=False, method='min').astype(int)
    results_df['balanced_rank'] = results_df['balanced_utility'].rank(ascending=False, method='min').astype(int)
    results_df['profit_rank'] = results_df['profit_utility'].rank(ascending=False, method='min').astype(int)
    
    # 计算各方法与F1的Spearman相关性
    spearman_tau, _ = spearmanr(results_df['f1_rank'], results_df['tau_rank'])
    spearman_cs, _ = spearmanr(results_df['f1_rank'], results_df['cs_rank'])
    spearman_f1opt, _ = spearmanr(results_df['f1_rank'], results_df['f1opt_rank'])
    spearman_smote, _ = spearmanr(results_df['f1_rank'], results_df['smote_rank'])
    spearman_balanced, _ = spearmanr(results_df['f1_rank'], results_df['balanced_rank'])
    spearman_profit, _ = spearmanr(results_df['f1_rank'], results_df['profit_rank'])
    
    # 找出每种方法的最佳模型
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    best_tau = results_df.loc[results_df['tau_utility'].idxmax()]
    best_cs = results_df.loc[results_df['cs_utility'].idxmax()]
    best_f1opt = results_df.loc[results_df['f1opt_utility'].idxmax()]
    best_smote = results_df.loc[results_df['smote_utility'].idxmax()]
    best_balanced = results_df.loc[results_df['balanced_utility'].idxmax()]
    best_profit = results_df.loc[results_df['profit_utility'].idxmax()]
    
    # 计算检测时间差异 (相对于F1最优模型)
    lead_time_tau = best_f1['tau_det_time'] - best_tau['tau_det_time']
    lead_time_cs = best_f1['cs_det_time'] - best_cs['cs_det_time']
    lead_time_f1opt = best_f1['f1opt_det_time'] - best_f1opt['f1opt_det_time']
    lead_time_smote = best_f1['smote_det_time'] - best_smote['smote_det_time']
    lead_time_balanced = best_f1['balanced_det_time'] - best_balanced['balanced_det_time']
    lead_time_profit = best_f1['profit_det_time'] - best_profit['profit_det_time']
    
    # Top-3 差异
    def calc_top3_diff(rank_col1, rank_col2):
        top3_1 = set(results_df.nsmallest(3, rank_col1)['model_name'].values)
        top3_2 = set(results_df.nsmallest(3, rank_col2)['model_name'].values)
        overlap = len(top3_1.intersection(top3_2))
        return (3 - overlap) / 3 * 100
    
    top3_diff_tau = calc_top3_diff('f1_rank', 'tau_rank')
    top3_diff_cs = calc_top3_diff('f1_rank', 'cs_rank')
    top3_diff_f1opt = calc_top3_diff('f1_rank', 'f1opt_rank')
    top3_diff_smote = calc_top3_diff('f1_rank', 'smote_rank')
    top3_diff_balanced = calc_top3_diff('f1_rank', 'balanced_rank')
    top3_diff_profit = calc_top3_diff('f1_rank', 'profit_rank')
    
    print(f"\n✅ Results Summary:")
    print(f"   Time-Aware Utility: ρ={spearman_tau:.3f}, lead_time={lead_time_tau:.1f}d, top3_diff={top3_diff_tau:.0f}%")
    print(f"   Cost-Sensitive:     ρ={spearman_cs:.3f}, lead_time={lead_time_cs:.1f}d, top3_diff={top3_diff_cs:.0f}%")
    print(f"   F1-Optimal:         ρ={spearman_f1opt:.3f}, lead_time={lead_time_f1opt:.1f}d, top3_diff={top3_diff_f1opt:.0f}%")
    print(f"   SMOTE:              ρ={spearman_smote:.3f}, lead_time={lead_time_smote:.1f}d, top3_diff={top3_diff_smote:.0f}%")
    print(f"   Class-Balanced:     ρ={spearman_balanced:.3f}, lead_time={lead_time_balanced:.1f}d, top3_diff={top3_diff_balanced:.0f}%")
    print(f"   Profit-Optimal:     ρ={spearman_profit:.3f}, lead_time={lead_time_profit:.1f}d, top3_diff={top3_diff_profit:.0f}%")
    
    return {
        'task_id': task_id,
        'dataset': dataset_name,
        'seed': seed,
        'n_samples': len(df),
        'n_frauds': int(df[target_col].sum()),
        'fraud_rate': float(df[target_col].mean()),
        # Time-Aware Utility
        'tau_spearman': float(spearman_tau),
        'tau_lead_time': float(lead_time_tau),
        'tau_top3_diff': float(top3_diff_tau),
        'tau_best_model': best_tau['model_name'],
        'tau_best_det_time': float(best_tau['tau_det_time']),
        'tau_best_tp': int(best_tau['tau_tp']),
        # Cost-Sensitive
        'cs_spearman': float(spearman_cs),
        'cs_lead_time': float(lead_time_cs),
        'cs_top3_diff': float(top3_diff_cs),
        'cs_best_model': best_cs['model_name'],
        'cs_best_det_time': float(best_cs['cs_det_time']),
        'cs_best_tp': int(best_cs['cs_tp']),
        # F1-Optimal
        'f1opt_spearman': float(spearman_f1opt),
        'f1opt_lead_time': float(lead_time_f1opt),
        'f1opt_top3_diff': float(top3_diff_f1opt),
        'f1opt_best_model': best_f1opt['model_name'],
        'f1opt_best_det_time': float(best_f1opt['f1opt_det_time']),
        'f1opt_best_tp': int(best_f1opt['f1opt_tp']),
        # SMOTE
        'smote_spearman': float(spearman_smote),
        'smote_lead_time': float(lead_time_smote),
        'smote_top3_diff': float(top3_diff_smote),
        'smote_best_model': best_smote['model_name'],
        'smote_best_det_time': float(best_smote['smote_det_time']),
        'smote_best_tp': int(best_smote['smote_tp']),
        # Class-Balanced
        'balanced_spearman': float(spearman_balanced),
        'balanced_lead_time': float(lead_time_balanced),
        'balanced_top3_diff': float(top3_diff_balanced),
        'balanced_best_model': best_balanced['model_name'],
        'balanced_best_det_time': float(best_balanced['balanced_det_time']),
        'balanced_best_tp': int(best_balanced['balanced_tp']),
        # Profit-Optimal
        'profit_spearman': float(spearman_profit),
        'profit_lead_time': float(lead_time_profit),
        'profit_top3_diff': float(top3_diff_profit),
        'profit_best_model': best_profit['model_name'],
        'profit_best_det_time': float(best_profit['profit_det_time']),
        'profit_best_tp': int(best_profit['profit_tp']),
        # F1 baseline
        'f1_best_model': best_f1['model_name'],
        'f1_best_score': float(best_f1['f1_score']),
    }


@app.function(
    image=experiment_image,
    volumes={"/workspace": volume},
    timeout=3600,
    cpu=2,
    memory=8192,
    secrets=[modal.Secret.from_name("orchestra-supabase")]
)
def aggregate_baseline_results(all_results: list):
    """汇总baseline对比结果并生成可视化."""
    import sys
    sys.path.insert(0, "/root")
    
    import json
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    from src.orchestra_sdk.experiment import Experiment
    
    exp = Experiment.init(
        name="Phase 3: Baseline Comparisons",
        description="对比5种baseline方法与Time-Aware Utility",
        config={
            "baseline_methods": BASELINE_METHODS,
            "datasets": DATASETS,
            "n_seeds": N_SEEDS,
            "T_max": T_MAX,
            "k_budget": K_BUDGET
        }
    )
    exp.add_tags(['phase3', 'baselines', 'comparison'])
    
    os.makedirs("/workspace/results/phase3_baselines", exist_ok=True)
    os.makedirs("/workspace/plots/phase3_baselines", exist_ok=True)
    
    print("="*70)
    print("AGGREGATING BASELINE COMPARISON RESULTS")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n✅ Received {len(all_results)} results")
    print(f"   Datasets: {results_df['dataset'].unique().tolist()}")
    print(f"   Seeds: {results_df['seed'].unique().tolist()}")
    
    # 保存原始结果
    results_df.to_csv('/workspace/results/phase3_baselines/all_results.csv', index=False)
    
    # =========================================================================
    # 1. 汇总统计
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS BY METHOD")
    print("="*70)
    
    methods = ['tau', 'cs', 'f1opt', 'smote', 'balanced', 'profit']
    method_names = {
        'tau': 'Time-Aware Utility (Ours)',
        'cs': 'Cost-Sensitive',
        'f1opt': 'F1-Optimal Threshold',
        'smote': 'SMOTE + Classifier',
        'balanced': 'Class-Weight Balanced',
        'profit': 'Profit-Optimal'
    }
    
    summary_data = []
    
    for method in methods:
        spearman_col = f'{method}_spearman'
        lead_time_col = f'{method}_lead_time'
        top3_diff_col = f'{method}_top3_diff'
        det_time_col = f'{method}_best_det_time'
        tp_col = f'{method}_best_tp'
        
        summary_data.append({
            'Method': method_names[method],
            'Spearman ρ (mean)': results_df[spearman_col].mean(),
            'Spearman ρ (std)': results_df[spearman_col].std(),
            'Lead Time (mean)': results_df[lead_time_col].mean(),
            'Lead Time (std)': results_df[lead_time_col].std(),
            'Top-3 Diff % (mean)': results_df[top3_diff_col].mean(),
            'Top-3 Diff % (std)': results_df[top3_diff_col].std(),
            'Avg Det Time (mean)': results_df[det_time_col].mean(),
            'Avg Det Time (std)': results_df[det_time_col].std(),
            'True Positives (mean)': results_df[tp_col].mean(),
            'True Positives (std)': results_df[tp_col].std(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('/workspace/results/phase3_baselines/method_summary.csv', index=False)
    
    print("\n📊 Method Comparison Summary:")
    print("-" * 100)
    print(f"{'Method':<30} {'Spearman ρ':>15} {'Lead Time':>15} {'Top-3 Diff':>15} {'Det Time':>15}")
    print("-" * 100)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Method']:<30} "
              f"{row['Spearman ρ (mean)']:>7.3f}±{row['Spearman ρ (std)']:<6.3f} "
              f"{row['Lead Time (mean)']:>7.1f}±{row['Lead Time (std)']:<6.1f} "
              f"{row['Top-3 Diff % (mean)']:>7.1f}±{row['Top-3 Diff % (std)']:<6.1f} "
              f"{row['Avg Det Time (mean)']:>7.1f}±{row['Avg Det Time (std)']:<6.1f}")
    
    # =========================================================================
    # 2. 按数据集分析
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS BY DATASET")
    print("="*70)
    
    for dataset in DATASETS:
        ds_data = results_df[results_df['dataset'] == dataset]
        print(f"\n📊 {dataset.upper()} (n={len(ds_data)} runs):")
        print("-" * 80)
        
        for method in methods:
            spearman = ds_data[f'{method}_spearman'].mean()
            lead_time = ds_data[f'{method}_lead_time'].mean()
            top3_diff = ds_data[f'{method}_top3_diff'].mean()
            det_time = ds_data[f'{method}_best_det_time'].mean()
            
            # 判断是否满足成功标准
            criterion1 = '✅' if top3_diff >= 20 else '❌'
            criterion2 = '✅' if lead_time >= 15 else '❌'
            criterion3 = '✅' if spearman < 0.6 else '❌'
            
            print(f"   {method_names[method]:<30}: ρ={spearman:>6.3f} {criterion3}, "
                  f"lead={lead_time:>5.1f}d {criterion2}, "
                  f"top3={top3_diff:>4.0f}% {criterion1}, "
                  f"det={det_time:>5.1f}d")
    
    # =========================================================================
    # 3. 统计显著性检验
    # =========================================================================
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    
    from scipy.stats import wilcoxon, ttest_rel
    
    # 比较 Time-Aware Utility 与其他方法的检测时间
    print("\n🔬 Paired t-test: Time-Aware Utility vs Other Methods (Detection Time)")
    print("-" * 70)
    
    tau_det_times = results_df['tau_best_det_time'].values
    
    for method in ['cs', 'f1opt', 'smote', 'balanced', 'profit']:
        other_det_times = results_df[f'{method}_best_det_time'].values
        
        # Paired t-test
        t_stat, p_value = ttest_rel(tau_det_times, other_det_times)
        
        # Wilcoxon signed-rank test (非参数)
        try:
            w_stat, w_p_value = wilcoxon(tau_det_times, other_det_times)
        except:
            w_stat, w_p_value = np.nan, np.nan
        
        mean_diff = np.mean(tau_det_times - other_det_times)
        
        sig_marker = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
        
        print(f"   TAU vs {method_names[method]:<25}: "
              f"Δ={mean_diff:>6.2f}d, t={t_stat:>6.2f}, p={p_value:.4f} {sig_marker}")
    
    # =========================================================================
    # 4. 可视化
    # =========================================================================
    print("\n📊 Creating visualizations...")
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    
    # Figure 1: 方法对比 - Spearman 相关性
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1a: Spearman ρ by method
    ax = axes[0]
    method_order = ['tau', 'cs', 'f1opt', 'smote', 'balanced', 'profit']
    spearman_data = [results_df[f'{m}_spearman'].values for m in method_order]
    bp = ax.boxplot(spearman_data, labels=[method_names[m].split('(')[0].strip() for m in method_order],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Threshold (ρ=0.6)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_ylabel("Spearman's ρ with F1", fontsize=12)
    ax.set_title('Rank Correlation with F1 Score', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right')
    
    # 1b: Lead Time by method
    ax = axes[1]
    lead_time_data = [results_df[f'{m}_lead_time'].values for m in method_order]
    bp = ax.boxplot(lead_time_data, labels=[method_names[m].split('(')[0].strip() for m in method_order],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=15, color='green', linestyle='--', linewidth=2, label='Target (15 days)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_ylabel('Lead Time Advantage (days)', fontsize=12)
    ax.set_title('Detection Time Advantage vs F1-Optimal', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right')
    
    # 1c: Detection Time by method
    ax = axes[2]
    det_time_data = [results_df[f'{m}_best_det_time'].values for m in method_order]
    bp = ax.boxplot(det_time_data, labels=[method_names[m].split('(')[0].strip() for m in method_order],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Average Detection Time (days)', fontsize=12)
    ax.set_title('Detection Time (Lower is Better)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/workspace/plots/phase3_baselines/method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ method_comparison.png")
    
    # Figure 2: 按数据集的方法对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        ds_data = results_df[results_df['dataset'] == dataset]
        
        x = np.arange(len(method_order))
        width = 0.35
        
        means = [ds_data[f'{m}_best_det_time'].mean() for m in method_order]
        stds = [ds_data[f'{m}_best_det_time'].std() for m in method_order]
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, alpha=0.7)
        
        ax.set_ylabel('Detection Time (days)', fontsize=11)
        ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in method_order], rotation=45, ha='right')
        
        # 标注最低值
        min_idx = np.argmin(means)
        ax.annotate('Best', xy=(min_idx, means[min_idx]), 
                   xytext=(min_idx, means[min_idx] - 5),
                   ha='center', fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('/workspace/plots/phase3_baselines/dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ dataset_comparison.png")
    
    # Figure 3: 热力图 - 各方法在各数据集上的表现
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['spearman', 'lead_time', 'best_det_time']
    metric_titles = ["Spearman's ρ (Lower=Better)", 'Lead Time (Higher=Better)', 'Detection Time (Lower=Better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        
        heatmap_data = []
        for dataset in DATASETS:
            ds_data = results_df[results_df['dataset'] == dataset]
            row = [ds_data[f'{m}_{metric}'].mean() for m in method_order]
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                  index=[d.upper() for d in DATASETS],
                                  columns=[m.upper() for m in method_order])
        
        # 根据指标类型选择颜色映射
        if metric == 'lead_time':
            cmap = 'RdYlGn'  # 高值好
        else:
            cmap = 'RdYlGn_r'  # 低值好
        
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap=cmap, ax=ax,
                   cbar_kws={'label': title.split('(')[0].strip()})
        ax.set_title(title, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/workspace/plots/phase3_baselines/heatmap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ heatmap_comparison.png")
    
    # Figure 4: 成功标准达成率
    fig, ax = plt.subplots(figsize=(10, 6))
    
    success_rates = []
    for method in method_order:
        # 计算每个标准的达成率
        c1 = (results_df[f'{method}_top3_diff'] >= 20).mean() * 100
        c2 = (results_df[f'{method}_lead_time'] >= 15).mean() * 100
        c3 = (results_df[f'{method}_spearman'] < 0.6).mean() * 100
        all_3 = ((results_df[f'{method}_top3_diff'] >= 20) & 
                 (results_df[f'{method}_lead_time'] >= 15) & 
                 (results_df[f'{method}_spearman'] < 0.6)).mean() * 100
        success_rates.append({
            'Method': method_names[method],
            'Top-3 Diff ≥20%': c1,
            'Lead Time ≥15d': c2,
            'Spearman ρ <0.6': c3,
            'All 3 Criteria': all_3
        })
    
    success_df = pd.DataFrame(success_rates)
    success_df.to_csv('/workspace/results/phase3_baselines/success_rates.csv', index=False)
    
    x = np.arange(len(method_order))
    width = 0.2
    
    ax.bar(x - 1.5*width, success_df['Top-3 Diff ≥20%'], width, label='Top-3 Diff ≥20%', color='#3498db')
    ax.bar(x - 0.5*width, success_df['Lead Time ≥15d'], width, label='Lead Time ≥15d', color='#2ecc71')
    ax.bar(x + 0.5*width, success_df['Spearman ρ <0.6'], width, label='Spearman ρ <0.6', color='#9b59b6')
    ax.bar(x + 1.5*width, success_df['All 3 Criteria'], width, label='All 3 Criteria', color='#e74c3c')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Criteria Achievement Rate by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([method_names[m].split('(')[0].strip() for m in method_order], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('/workspace/plots/phase3_baselines/success_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ success_rates.png")
    
    # =========================================================================
    # 5. 保存最终结果
    # =========================================================================
    
    # 计算最终汇总
    final_summary = {
        'experiment': 'Phase 3: Baseline Comparisons',
        'timestamp': datetime.now().isoformat(),
        'n_experiments': len(all_results),
        'datasets': DATASETS,
        'n_seeds': N_SEEDS,
        'parameters': {
            'T_max': T_MAX,
            'k_budget': K_BUDGET,
            'days_per_case': DAYS_PER_CASE,
            'confidence_delay_factor': CONFIDENCE_DELAY_FACTOR
        },
        'results_by_method': {}
    }
    
    for method in methods:
        final_summary['results_by_method'][method_names[method]] = {
            'spearman_rho': {
                'mean': float(results_df[f'{method}_spearman'].mean()),
                'std': float(results_df[f'{method}_spearman'].std())
            },
            'lead_time_days': {
                'mean': float(results_df[f'{method}_lead_time'].mean()),
                'std': float(results_df[f'{method}_lead_time'].std())
            },
            'top3_diff_pct': {
                'mean': float(results_df[f'{method}_top3_diff'].mean()),
                'std': float(results_df[f'{method}_top3_diff'].std())
            },
            'detection_time_days': {
                'mean': float(results_df[f'{method}_best_det_time'].mean()),
                'std': float(results_df[f'{method}_best_det_time'].std())
            },
            'success_rate_all_criteria': float(
                ((results_df[f'{method}_top3_diff'] >= 20) & 
                 (results_df[f'{method}_lead_time'] >= 15) & 
                 (results_df[f'{method}_spearman'] < 0.6)).mean() * 100
            )
        }
    
    with open('/workspace/results/phase3_baselines/final_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    # Log to experiment tracker
    exp.log({
        'tau_spearman_mean': results_df['tau_spearman'].mean(),
        'tau_lead_time_mean': results_df['tau_lead_time'].mean(),
        'tau_det_time_mean': results_df['tau_best_det_time'].mean(),
        'cs_spearman_mean': results_df['cs_spearman'].mean(),
        'f1opt_spearman_mean': results_df['f1opt_spearman'].mean(),
        'smote_spearman_mean': results_df['smote_spearman'].mean(),
    }, step=1)
    
    exp.set_progress(100)
    exp.log_text("Phase 3 baseline comparison completed successfully!")
    
    volume.commit()
    
    print("\n" + "="*70)
    print("✅ PHASE 3 BASELINE COMPARISON COMPLETED!")
    print("="*70)
    print(f"\n📁 Results saved to:")
    print(f"   - /workspace/results/phase3_baselines/all_results.csv")
    print(f"   - /workspace/results/phase3_baselines/method_summary.csv")
    print(f"   - /workspace/results/phase3_baselines/success_rates.csv")
    print(f"   - /workspace/results/phase3_baselines/final_summary.json")
    print(f"\n📊 Plots saved to:")
    print(f"   - /workspace/plots/phase3_baselines/method_comparison.png")
    print(f"   - /workspace/plots/phase3_baselines/dataset_comparison.png")
    print(f"   - /workspace/plots/phase3_baselines/heatmap_comparison.png")
    print(f"   - /workspace/plots/phase3_baselines/success_rates.png")
    
    exp.finish('completed')
    
    return final_summary


@app.local_entrypoint()
def main():
    """主入口：并行运行所有baseline对比任务."""
    import time
    
    print("="*70)
    print("PHASE 3: BASELINE COMPARISONS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Datasets: {DATASETS}")
    print(f"  - Seeds: {RANDOM_SEEDS}")
    print(f"  - T_max: {T_MAX}")
    print(f"  - K_budget: {K_BUDGET}")
    print(f"  - Baseline methods: {len(BASELINE_METHODS)}")
    
    # 创建任务列表
    tasks = []
    task_id = 0
    for dataset in DATASETS:
        for seed in RANDOM_SEEDS:
            tasks.append({
                'task_id': f"task_{task_id:03d}_{dataset}_{seed}",
                'dataset': dataset,
                'seed': seed,
            })
            task_id += 1
    
    print(f"\n📋 Total tasks: {len(tasks)}")
    print(f"   = {len(DATASETS)} datasets × {len(RANDOM_SEEDS)} seeds")
    
    start_time = time.time()
    
    # 并行执行所有任务
    print("\n🚀 Starting parallel execution...")
    results = list(run_baseline_task.map(tasks, return_exceptions=True))
    
    # 过滤成功的结果
    successful_results = [r for r in results if r is not None]
    print(f"\n✅ Completed: {len(successful_results)}/{len(tasks)} tasks")
    
    # 聚合结果
    if successful_results:
        print("\n📊 Aggregating results...")
        final_summary = aggregate_baseline_results.remote(successful_results)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
        
        return final_summary
    else:
        print("\n❌ No successful results to aggregate!")
        return None


if __name__ == "__main__":
    main()
