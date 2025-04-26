'''
Author: cunyu277 2465899266@qq.com
Date: 2025-04-17 21:50:34
LastEditors: cunyu277 2465899266@qq.com
LastEditTime: 2025-04-17 22:50:41
FilePath: \crop_yield_prediction\cunuu\predict\rf.py
Description: 

Copyright (c) 2025 by yh, All Rights Reserved. 
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import linregress
import os
from tqdm import tqdm
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False  

def load_npz_data_with_years(directory, target=0):
    """加载NPZ数据并提取年份信息"""
    X, y, years = [], [], []
    skipped = {'missing': 0, 'invalid': 0}
    
    for filename in tqdm(os.listdir(directory)):
        if not filename.endswith('.npz'):
            continue
            
        try:
            # 从文件名提取年份
            year = int(filename.split('_')[2])
            data = np.load(os.path.join(directory, filename), allow_pickle=True)
            
            # 检查目标变量
            if target == 0:
                target_value = data['yield'][0]  # 单位面积产量
            else:
                target_value = data['yield'][1]  # 总产量
                
            if np.isnan(target_value):
                skipped['missing'] += 1
                continue
                
            # 检查特征
            features = []
            for key in data.files:
                if key != 'yield' and isinstance(data[key], np.ndarray):
                    if np.isnan(data[key]).any():
                        skipped['invalid'] += 1
                        break
                    features.append(data[key].reshape(-1))
            else:  # 所有特征有效
                X.append(np.concatenate(features))
                y.append(target_value)
                years.append(year)
                
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    print(f"\n数据加载完成，跳过记录:")
    print(f"- 目标变量缺失: {skipped['missing']}")
    print(f"- 特征数据无效: {skipped['invalid']}")
    
    return np.array(X), np.array(y), np.array(years)

# 加载数据并提取年份
X, y, years = load_npz_data_with_years(r"D:\python\crop_yield_prediction\cunuu\predict\enhanced_npz", target=0)
print(f"总样本数: {len(X)}")
print(f"特征维度: {X.shape[1]}")
print(f"包含年份: {np.unique(years)}")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 按指定年份划分训练测试集
def train_test_split_by_year(X, y, years, test_year):
    test_mask = years == test_year
    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]

# 示例：使用2010年作为测试集
test_year = 2022
X_train, X_test, y_train, y_test = train_test_split_by_year(X_scaled, y, years, test_year)
print(f"\n训练集: {len(X_train)}样本 (非{test_year}年)")
print(f"测试集: {len(X_test)}样本 ({test_year}年)")

# 训练模型
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# 评估函数（增加年份信息）
def evaluate_model(model, X_test, y_test, year=None):
    y_pred = model.predict(X_test)
    pearson_r, _ = pearsonr(y_test, y_pred)
    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MRE': mean_absolute_percentage_error(y_test, y_pred) * 100,
        'R2': r2_score(y_test, y_pred),
        'Pearson_r': pearson_r
    }
    
    if year:
        print(f"\n{year}年测试结果:")
    else:
        print("\n评估指标:")

    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")

    if year == test_year:
        # 可视化预测结果
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, label='样本点')
        # 1:1 参考线
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='1:1参考线')
        # 添加线性回归拟合线
        slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
        fit_line = slope * y_test + intercept
        plt.plot(y_test, fit_line, 'b-', linewidth=1, color = 'b',
                label=f'拟合线 (y={slope:.2f}x+{intercept:.2f}, r={r_value:.2f})')
        plt.xlabel('实际产量')
        plt.ylabel('预测产量')
        plt.title('实际产量 vs 预测产量对比')
        plt.legend()
        plt.grid(True)
        plt.show()


    return metrics

# 测试指定年份
print(f"\n评估{test_year}年:")
test_metrics = evaluate_model(rf, X_test, y_test, year=test_year)

# 按年份的留一法交叉验证
print("\n开始按年份的留一法交叉验证...")
unique_years = np.unique(years)
yearly_metrics = defaultdict(list)

for year in unique_years:
    X_train, X_test, y_train, y_test = train_test_split_by_year(X_scaled, y, years, year)
    
    if len(X_test) == 0:
        print(f"跳过{year}年（无测试样本）")
        continue
        
    rf.fit(X_train, y_train)
    metrics = evaluate_model(rf, X_test, y_test, year)
    
    for k, v in metrics.items():
        yearly_metrics[k].append(v)

# 打印各年份平均表现
print("\n各年份交叉验证结果汇总:")
for metric, values in yearly_metrics.items():
    print(f"{metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")

# 可视化各年份R2表现
plt.figure(figsize=(10, 5))
plt.bar(unique_years, yearly_metrics['R2'])
plt.xlabel('年份')
plt.ylabel('R²分数')
plt.title('各年份模型表现(R²)')
plt.xticks(unique_years)
plt.grid(True)
plt.show()