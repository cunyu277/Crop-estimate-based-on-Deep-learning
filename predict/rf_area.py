import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
import os
from tqdm import tqdm

# 数据加载和预处理函数
def parse_filename(filename):
    """解析NPZ文件名格式：省代码_市代码_年份_all.npz"""
    parts = filename.replace('.npz', '').split('_')
    if len(parts) < 3:
        raise ValueError(f"文件名格式错误: {filename}")
    return {
        'province_code': parts[0],
        'city_code': parts[1],
        'year': int(parts[2]),
        'other': '_'.join(parts[3:]) if len(parts) > 3 else None
    }

def load_data_with_geo(directory, target=0):
    """加载数据并解析地理信息"""
    X, y, meta = [], [], []
    
    for filename in tqdm(os.listdir(directory)):
        if not filename.endswith('.npz'):
            continue
        
        try:
            geo_info = parse_filename(filename)
            data = np.load(os.path.join(directory, filename), allow_pickle=True)
            
            # 获取目标变量
            target_value = data['yield'][target]
            if np.isnan(target_value):
                continue
                
            # 验证并拼接特征
            features = []
            for key in data.files:
                if key != 'yield':
                    arr = np.array(data[key])
                    if np.isnan(arr).any():
                        break
                    features.append(arr.reshape(-1))
            else:
                X.append(np.concatenate(features))
                y.append(target_value)
                meta.append(geo_info)
                
        except Exception as e:
            print(f"加载失败 {filename}: {str(e)}")
            continue
    
    df_meta = pd.DataFrame(meta)
    return np.array(X), np.array(y), df_meta

# 模型评估和结果保存函数
def get_predictions_df(model, X, df_meta, y_true):
    """生成包含预测结果的完整数据表"""
    y_pred = model.predict(X)
    df = df_meta.copy()
    df['实际产量'] = y_true
    df['预测产量'] = y_pred
    return df

def evaluate_group(y_true, y_pred):
    """计算评估指标"""
    if len(y_true) < 2:
        return {'样本数': len(y_true),
        '绝对误差': float(np.abs(y_true - y_pred).iloc[0]),  # 取第一个值并转为float
        '真实值': int(y_true.iloc[0]),
        '预测值': float(y_pred.iloc[0])}
    
    pearson_r, _ = pearsonr(y_true, y_pred)

    return {
        '样本数': len(y_true),
        #'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MRE': mean_absolute_percentage_error(y_true, y_pred)*100,
        'R2': r2_score(y_true, y_pred),
        'Pearson_r': pearson_r
    }

def generate_evaluation_results(model, X, y, df_meta, test_year, target_province):
    """生成多层级评估结果"""
    # 获取测试年数据
    test_mask = df_meta['year'] == test_year
    X_test = X[test_mask]
    y_test = y[test_mask]
    df_test = df_meta[test_mask].copy()
    
    # 生成预测结果
    df_full = get_predictions_df(model, X_test, df_test, y_test)
    
    # 定义多级索引评估 - 显式选择列
    results = {
        '各市评估': df_full.groupby(['province_code', 'city_code'])[['实际产量', '预测产量']].apply(
            lambda x: pd.Series(evaluate_group(x['实际产量'], x['预测产量']))
        ).reset_index(),
        
        '各省评估': df_full.groupby('province_code')[['实际产量', '预测产量']].apply(
            lambda x: pd.Series(evaluate_group(x['实际产量'], x['预测产量']))
        ).reset_index(),
        
        '指定省评估': df_full[df_full['province_code'] == target_province].groupby('city_code')[['实际产量', '预测产量']].apply(
            lambda x: pd.Series(evaluate_group(x['实际产量'], x['预测产量']))
        ).reset_index()
    }
    
    return df_full, results

# 主程序
if __name__ == "__main__":
    # 数据路径和参数设置
    DATA_DIR = "D:\python\crop_yield_prediction\cunuu\predict\enhanced_npz"
    TEST_YEAR = 2022
    TARGET_PROVINCE = "130000"  # 河北省代码
    
    # 加载数据
    X, y, df_meta = load_data_with_geo(DATA_DIR, target=0)
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集
    test_mask = df_meta['year'] == TEST_YEAR
    X_train, X_test = X_scaled[~test_mask], X_scaled[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    
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
    
    # 生成结果
    df_predictions, results = generate_evaluation_results(rf, X_test, y_test, 
                                                        df_meta[test_mask], 
                                                        TEST_YEAR, TARGET_PROVINCE)
    
    # 保存结果
    with pd.ExcelWriter(f"{TEST_YEAR}年产量预测评估.xlsx") as writer:
        # 各层级评估结果
        results['各市评估'].to_excel(writer, sheet_name='各市评估', index=False)
        results['各省评估'].to_excel(writer, sheet_name='各省评估', index=False)
        results['指定省评估'].to_excel(writer, sheet_name=f'{TARGET_PROVINCE}省各市评估', index=False)
        
        # 详细预测数据
        df_predictions[['province_code', 'city_code', 'year', '实际产量', '预测产量']].to_excel(
            writer, sheet_name='详细数据', index=False)
    
    print(f"""
    结果已保存为 {TEST_YEAR}年产量预测评估.xlsx
    包含以下工作表：
    1. 各市评估 - 市级行政区评估指标
    2. 各省评估 - 省级行政区评估指标
    3. {TARGET_PROVINCE}省各市评估 - 指定省份市级评估
    4. 详细数据 - 包含每个样本的真实值和预测值
    """)