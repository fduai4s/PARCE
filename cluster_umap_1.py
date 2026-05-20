#在cluster_umap.py基础上，对wyckoff进行优化，具体是优化归一化处理
#之前用的standardscaler,可能把0变成负的，现在改为Maxabsscaler,按最大值缩放
import pandas as pd
import numpy as np
import matplotlib
# 设置后端为 Agg，适用于没有图形界面的远程服务器
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.metrics import silhouette_score
# [Modified] 引入 DictVectorizer 用于 Wyckoff 解析
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
# [Removed] 移除了 PCA 库的依赖，因为不再需要降维
# from sklearn.decomposition import PCA 
# 引入 UMAP
import umap
import os
import re
import warnings
from kneed import KneeLocator
import itertools

# ==========================================
# 1. Configuration & Global Settings
# ==========================================
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 0

# Dataset and output paths
DATASET_PATH = '/home2/yhchen/01-PARCE/cluster_and_model/raman2/dataset/dataset.csv'
BASE_OUTPUT_DIR = "/home2/yhchen/01-PARCE/cluster_and_model/raman2/all_cluster_results_final_optimized"

# Feature combinations definition
# 1:space_group, 2:pearson, 3:wyckoff, 4:c_a_ratio, 5:beta_angle
BASE_FEATURES = ['1', '2', '3', '4', '5']
FEATURE_COMBINATIONS = []
# Generate all combinations of 3, 4, and 5 parameters
for i in range(3, 6):
    for combo in itertools.combinations(BASE_FEATURES, i):
        FEATURE_COMBINATIONS.append("".join(sorted(combo)))

# K-means cluster range to search
CLUSTER_RANGE = range(30, 91, 5)

# ==========================================
# 2. Helper Functions
# ==========================================

def _beta_to_sin_cos(beta_deg: pd.Series) -> pd.DataFrame:
    """把 β 角（度）转成 sin/cos 两维，更符合角度的周期性。"""
    beta = pd.to_numeric(beta_deg, errors='coerce')
    beta_rad = np.deg2rad(beta)
    return pd.DataFrame({
        'beta_sin': np.sin(beta_rad),
        'beta_cos': np.cos(beta_rad),
    })

def _parse_wyckoff_string(s):
    """
    [New] 自定义 Wyckoff 解析器
    解析 "d2a" -> {'d': 2, 'a': 1} (d位置有2个原子)
    解析 "cba" -> {'c': 1, 'b': 1, 'a': 1}
    """
    if pd.isna(s):
        return {}
    s = str(s).strip().lower() # 统一转小写
    
    # 正则逻辑：
    # ([a-z]) : 捕获一个字母
    # (\d*)   : 捕获跟随的数字（可以是空的）
    matches = re.findall(r'([a-z])(\d*)', s)
    
    counts = {}
    for char, count_str in matches:
        # 如果没有数字，默认为 1；否则转为 int
        count = int(count_str) if count_str else 1
        # 累加（防止出现 d2d3 这种奇怪写法）
        counts[char] = counts.get(char, 0) + count
    return counts

def _parse_pearson_symbol(s):
    """
    [New] 自定义 Pearson 解析器
    将 "cF32" 拆解为 ("cF", 32)
    """
    s = str(s).strip()
    if len(s) < 3:
        return "Unknown", 0.0
    
    # 前两个字母是布拉维点阵 (例如 cF, mP)
    bravais = s[:2]
    # 剩下的部分是原子数
    try:
        atoms = float(s[2:])
    except:
        atoms = 0.0
    return bravais, atoms


def process_all_features(data):
    """
    [Improved] 特征工程函数
    核心改进：
    1. Wyckoff: 正则解析数量 + DictVectorizer + StandardScaler (无PCA)
    2. Pearson: 拆解为 布拉维点阵(OneHot) + 原子数(数值)
    3. 几何参数: 保持数值处理，权重在后续步骤调整
    """
    print("  Starting feature engineering (Final Optimized Version)...")

    data = data.copy()
    data['space_group_number'] = pd.to_numeric(data['space_group_number'], errors='coerce').fillna(-1)
    
    # 1) Space group：OneHot
    space_group_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        min_frequency=10,
    )
    space_group_encoded = space_group_encoder.fit_transform(data[['space_group_number']].astype(int).astype(str))

    # 2) Pearson：Split into Bravais (OneHot) + Atom Count (Scalar)
    pearson_parsed = data['pearson_symbol'].apply(_parse_pearson_symbol).tolist()
    bravais_list = [x[0] for x in pearson_parsed]
    atoms_list = [[x[1]] for x in pearson_parsed]

    # 2.1 布拉维点阵 (只有14种组合，适合OneHot)
    bravais_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    bravais_encoded = bravais_encoder.fit_transform(np.array(bravais_list).reshape(-1, 1))
    
    # 2.2 原子数 (数值标准化)
    atoms_scaler = StandardScaler()
    atoms_scaled = atoms_scaler.fit_transform(atoms_list)
    
    # 合并 Pearson 特征
    pearson_final = np.hstack([bravais_encoded, atoms_scaled])
    print(f"    Pearson features: {bravais_encoded.shape[1]} Bravais types + 1 Atom Count.")

    # 3) Wyckoff：Regex Parse + DictVectorizer + StandardScaler (No PCA)
    # [Step A] 解析字符串为字典列表
    wyckoff_dicts = data['wyckoff_sequence'].apply(_parse_wyckoff_string).tolist()
    
    # [Step B] 字典向量化 (稠密矩阵)
    vec = DictVectorizer(sparse=False, sort=True)
    wyckoff_raw_matrix = vec.fit_transform(wyckoff_dicts)
    feature_names = vec.get_feature_names_out()
    print(f"    Wyckoff Vocabulary: {len(feature_names)} unique positions (e.g., {feature_names[:5]}...).")

    # [Step C] 按最大值缩放 (MaxAbsScaler)
    # 计数特征保持非负且 0 仍为 0，避免 StandardScaler 中心化后出现负值
    wyckoff_final = MaxAbsScaler().fit_transform(wyckoff_raw_matrix)
    print(f"    Wyckoff features ready: {wyckoff_final.shape[1]} dimensions (Raw, No PCA).")

    # 4) c/a：中位数填补 + 标准化
    ca = pd.to_numeric(data['c_a_ratio'], errors='coerce')
    ca_imputed = SimpleImputer(strategy='median').fit_transform(ca.to_frame())
    ca_scaled = StandardScaler().fit_transform(ca_imputed)

    # 5) β：sin/cos 两维 + 中位数填补 + 标准化
    beta_sc = _beta_to_sin_cos(data['beta_angle'])
    beta_imputed = SimpleImputer(strategy='median').fit_transform(beta_sc)
    beta_scaled = StandardScaler().fit_transform(beta_imputed)

    print("  Feature engineering completed.")
    return {
        '1': space_group_encoded, 
        '2': pearson_final,       
        '3': wyckoff_final,       
        '4': ca_scaled,           
        '5': beta_scaled,         
    }


def get_selected_features(features_dict, selected_params):
    """
    [Weighted] 特征拼接函数
    权重平衡：几何参数 3.0，拓扑参数 1.0
    """
    PHYSICAL_WEIGHTS = {
        '1': 1.0,  # Space Group
        '2': 1.0,  # Pearson (Bravais + Atoms)
        '3': 1.0,  # Wyckoff
        '4': 3.0,  # c/a Ratio (Weight 3.0)
        '5': 3.0,  # Beta Angle (Weight 3.0)
    }

    feature_list = []
    for param_char in selected_params:
        if param_char in features_dict:
            raw_feature = features_dict[param_char]
            weight = PHYSICAL_WEIGHTS.get(param_char, 1.0)
            feature_list.append(raw_feature * weight)

    if not feature_list:
        raise ValueError(f"Feature combination '{selected_params}' is invalid or not found.")

    return np.hstack(feature_list)


def run_kmeans_for_combination(data, features_dict, combination, output_dir):
    """
    [UMAP-Based] 单个组合的完整聚类流程
    """
    print(f"\n{'='*60}")
    print(f"▶️  Processing feature combination: {combination}")
    print(f"{'='*60}")
    
    combo_output_dir = os.path.join(output_dir, combination)
    os.makedirs(combo_output_dir, exist_ok=True)
    
    print(f"  Combining features: {combination}...")
    try:
        combined_features = get_selected_features(features_dict, combination)
        print(f"  Weighted feature matrix dimensions: {combined_features.shape}")
    except ValueError as e:
        print(f"  Error: {e}")
        return None

    # Global UMAP (Optimized for Clustering)
    print("  Starting UMAP dimensionality reduction (Global)...")
    if combined_features.shape[1] > 10:
        # [MODIFIED] 这里是修改后的 UMAP 参数
        reducer = umap.UMAP(
            n_components=10,      # 保持 10 维供 K-Means 使用
            n_neighbors=50,       # [修改] 增大到 50，捕捉全局晶体家族结构
            min_dist=0.0,         # [修改] 降为 0.0，让相似材料紧密抱团
            metric='euclidean',   # 保持欧氏距离，保留 Wyckoff 强度信息
            random_state=42,
            n_jobs=1
        )
        embedding = reducer.fit_transform(combined_features)
        print(f"  UMAP completed. Reduced from {combined_features.shape[1]} to {embedding.shape[1]} dimensions.")
    else:
        print("  Low dimensionality detected. Skipping UMAP, using raw weighted features.")
        embedding = combined_features

    # K-means Loop
    clustering_results = []
    print(f"  Starting K-means clustering on UMAP embedding (cluster range: {min(CLUSTER_RANGE)} to {max(CLUSTER_RANGE)})...")
    
    for n_clusters in CLUSTER_RANGE:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)
        
        silhouette_avg = silhouette_score(embedding, cluster_labels)
        sse = kmeans.inertia_
        
        clustering_results.append({
            'n_clusters': n_clusters, 
            'silhouette_score': silhouette_avg, 
            'sse': sse
        })

    results_df = pd.DataFrame(clustering_results)
    results_csv_path = os.path.join(combo_output_dir, f'clustering_scores_{combination}.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"  Clustering scores saved to: {results_csv_path}")

    # Find Elbow
    kneedle = KneeLocator(results_df['n_clusters'], results_df['sse'], curve='convex', direction='decreasing')
    elbow_clusters = kneedle.elbow
    
    if not elbow_clusters:
        print("  ⚠️  Could not automatically find elbow point, using cluster number with highest silhouette score.")
        elbow_clusters = int(results_df.loc[results_df['silhouette_score'].idxmax()]['n_clusters'])

    elbow_silhouette = results_df.loc[results_df['n_clusters'] == elbow_clusters, 'silhouette_score'].values[0]
    print(f"  Optimal elbow point K = {elbow_clusters}, corresponding silhouette score = {elbow_silhouette:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'K-means Evaluation (UMAP+Weighted): {combination}', fontsize=16)

    ax1.plot(results_df['n_clusters'], results_df['silhouette_score'], 'b-o')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs. Number of Clusters')
    ax1.grid(True)
    ax1.axvline(elbow_clusters, color='grey', linestyle='--', label=f'Elbow K={elbow_clusters}')
    ax1.legend()

    ax2.plot(results_df['n_clusters'], results_df['sse'], 'r-o')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('SSE (Inertia)')
    ax2.set_title('SSE (Elbow Method)')
    ax2.grid(True)
    ax2.vlines(elbow_clusters, ax2.get_ylim()[0], 
               results_df.loc[results_df['n_clusters'] == elbow_clusters, 'sse'].values[0], 
               linestyles='--', color='grey', label=f'Elbow K={elbow_clusters}')
    ax2.legend()
    
    scores_plot_path = os.path.join(combo_output_dir, f'clustering_scores_plot_{combination}.png')
    plt.savefig(scores_plot_path, dpi=300)
    plt.close()
    print(f"  Score plots saved to: {scores_plot_path}")

    return {
        'combination': combination,
        'elbow_clusters': elbow_clusters,
        'silhouette_at_elbow': elbow_silhouette
    }


def main():
    print("=== K-means Clustering Analysis Pipeline (Final Optimized Version) ===")
    print(f"Features: Regex-Wyckoff (Full Dim), Split-Pearson, Weighted-Geometry")
    print(f"UMAP Settings: n_neighbors=50, min_dist=0.0")
    print(f"Number of feature combinations to analyze: {len(FEATURE_COMBINATIONS)}")
    print(f"Output root directory: {BASE_OUTPUT_DIR}")
    
    print(f"\nStep 1/3: Loading dataset: {DATASET_PATH}")
    try:
        data = pd.read_csv(DATASET_PATH)
        print(f"  Successfully loaded {len(data)} material data points.")
    except FileNotFoundError:
        print(f"  Error: Dataset file not found! Please check the path.")
        return

    features_dict = process_all_features(data)

    print(f"\nStep 2/3: Processing each feature combination...")
    summary_results = []
    
    for combo in FEATURE_COMBINATIONS:
        result = run_kmeans_for_combination(data, features_dict, combo, BASE_OUTPUT_DIR)
        if result:
            summary_results.append(result)
            print(f"✅ Feature combination {combo} completed.")
        else:
            print(f"❌ Feature combination {combo} failed.")

    print(f"\nStep 3/3: Generating final summary report...")
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df = summary_df.sort_values(by='silhouette_at_elbow', ascending=False)
        
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        summary_path = os.path.join(BASE_OUTPUT_DIR, 'elbow_summary_report.csv')
        
        summary_df.to_csv(summary_path, index=False)
        print("="*60)
        print("✅ All combinations processed! Summary report generated.")
        print(f"Report path: {summary_path}")
        print("\n--- Best Combination Summary ---")
        print(summary_df.to_string())
        print("="*60)
    else:
        print("All feature combinations failed, no summary report generated.")


if __name__ == "__main__":
    main()