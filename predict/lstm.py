import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr, linregress
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False  

# ==================== 数据加载 ====================
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

def load_data(directory, target=0):
    """加载数据并按时间序列整理特征"""
    X, y, meta = [], [], []
    skipped = {'missing': 0, 'invalid': 0}
    
    for filename in tqdm(os.listdir(directory)):
        if not filename.endswith('.npz'):
            continue
        
        try:
            geo_info = parse_filename(filename)
            data = np.load(os.path.join(directory, filename), allow_pickle=True)
            
            target_value = data['yield'][target]
            if np.isnan(target_value):
                skipped['missing'] += 1
                continue
            
            # 为每个地区的每一年，按时间步排列特征
            time_series = []
            for time_step in range(16):  # 16个时间步
                step_features = []
                for key in data.files:
                    if key != 'yield':  # 忽略目标变量
                        # 为每个时间步提取对应特征
                        step_features.append(data[key][time_step].ravel())
                # 每个时间步的特征
                time_series.append(np.concatenate(step_features))
            
            # 构建X和y，X为每年16时间步的特征，y为目标变量
            X.append(np.stack(time_series))  # 16 x n_features 的结构
            y.append(target_value)
            meta.append(geo_info)
                
        except Exception as e:
            print(f"加载失败 {filename}: {str(e)}")
            continue
    
    print(f"\n数据加载完成，跳过记录:")
    print(f"- 目标变量缺失: {skipped['missing']}")
    print(f"- 特征数据无效: {skipped['invalid']}")

    # 返回 numpy 数组和 metadata
    return np.array(X), np.array(y), pd.DataFrame(meta)

# ==================== LSTM模型 ====================
class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # 取最后一个时间步
        return self.regressor(last_output).squeeze()

# ==================== 训练函数 ====================
def train_lstm(X_train, y_train, X_val, y_val, input_size, epochs=150):
    """训练流程（添加目标变量标准化）"""
    # 标准化目标变量
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = YieldLSTM(input_size=input_size, hidden_size=256, num_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
        
        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
            val_loss = mean_squared_error(y_val, val_pred)
        scheduler.step(val_loss)  # 调整学习率
        
        # 打印信息
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Val MSE: {val_loss:.4f}")
            print("标准化预测样例:", val_pred[:5])
    
    # 返回标准化参数
    return model, y_mean, y_std

# ==================== 评估函数 ====================
def evaluate_model(model, X_test, y_test, year=None):
    """评估并可视化"""
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    
    # 计算指标
    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MRE': mean_absolute_percentage_error(y_test, y_pred) * 100,
        'R2': r2_score(y_test, y_pred),
        'Pearson_r': pearsonr(y_test, y_pred)[0]
    }
    
    # 测试年可视化
    if year == TEST_YEAR:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, label='样本点')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='1:1线')
        
        # 添加回归线
        slope, intercept, r_value, _, _ = linregress(y_test, y_pred)
        plt.plot(y_test, slope*y_test + intercept, 'b-', 
                label=f'拟合线 (y={slope:.2f}x+{intercept:.2f}, r={r_value:.2f})')
        
        plt.xlabel('实际产量')
        plt.ylabel('预测产量')
        plt.title(f'{year}年产量预测结果')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return metrics

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置参数
    DATA_DIR = r"D:\python\crop_yield_prediction\cunuu\predict\enhanced_npz"
    TEST_YEAR = 2022
    TARGET_PROVINCE = "130000"  # 河北省代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    X, y, df_meta = load_data(DATA_DIR, target=0)
    print(f"总样本数: {len(X)}, 特征维度: {X.shape[-1]}")
    print("包含年份:", sorted(df_meta['year'].unique()))
    
    # 按年份划分
    test_mask = df_meta['year'] == TEST_YEAR
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    print(f"\n训练集: {len(X_train)}样本 (非{TEST_YEAR}年)")
    print(f"测试集: {len(X_test)}样本 ({TEST_YEAR}年)")
    
    # 训练模型
    model, y_mean, y_std = train_lstm(X_train, y_train, X_test, y_test, input_size=X.shape[-1])
    
    # 评估测试年
    print("\n=== 测试年评估 ===")
    test_metrics = evaluate_model(model, X_test, y_test, year=TEST_YEAR)
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")
    
    # 按年份留一验证
    print("\n=== 按年份留一验证 ===")
    yearly_metrics = defaultdict(list)
    for year in sorted(df_meta['year'].unique()):
        year_mask = df_meta['year'] == year
        if sum(year_mask) < 5:  # 跳过样本太少的年份
            continue
            
        metrics = evaluate_model(model, X[year_mask], y[year_mask], year)
        for k, v in metrics.items():
            yearly_metrics[k].append(v)
        print(f"{year}年: R2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.1f}")
    
    # 输出各年份平均表现
    print("\n=== 各年份平均表现 ===")
    for metric, values in yearly_metrics.items():
        print(f"{metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")
    
    # 生成按省/市评估结果
    def get_predictions_df(model, X, df_meta, y_true):
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X).to(device)).cpu().numpy()
        df = df_meta.copy()
        df['实际产量'] = y_true
        df['预测产量'] = y_pred
        return df
    
    df_full = get_predictions_df(model, X_test, df_meta[test_mask], y_test)
    
    # 保存结果到Excel
    with pd.ExcelWriter(f"{TEST_YEAR}年LSTM预测评估.xlsx") as writer:
        # 各省评估
        (df_full.groupby('province_code')[['实际产量', '预测产量']]
               .apply(lambda x: pd.Series({
                   '样本数': len(x),
                   'RMSE': np.sqrt(mean_squared_error(x['实际产量'], x['预测产量'])),
                   'MAE': mean_absolute_error(x['实际产量'], x['预测产量']),
                   'R2': r2_score(x['实际产量'], x['预测产量']),
                   'Pearson_r': pearsonr(x['实际产量'], x['预测产量'])[0]
               }))
               .to_excel(writer, sheet_name='各省评估'))
        
        # 各市评估
        (df_full.groupby(['province_code', 'city_code'])[['实际产量', '预测产量']]
               .apply(lambda x: pd.Series({
                   '样本数': len(x),
                   'RMSE': np.sqrt(mean_squared_error(x['实际产量'], x['预测产量'])),
                   'MAE': mean_absolute_error(x['实际产量'], x['预测产量']),
                   'R2': r2_score(x['实际产量'], x['预测产量']),
                   'Pearson_r': pearsonr(x['实际产量'], x['预测产量'])[0]
               }))
               .to_excel(writer, sheet_name='各市评估'))
        
        # 指定省评估
        (df_full[df_full['province_code'] == TARGET_PROVINCE]
               .groupby('city_code')[['实际产量', '预测产量']]
               .apply(lambda x: pd.Series({
                   '样本数': len(x),
                   'RMSE': np.sqrt(mean_squared_error(x['实际产量'], x['预测产量'])),
                   'MAE': mean_absolute_error(x['实际产量'], x['预测产量']),
                   'R2': r2_score(x['实际产量'], x['预测产量']),
                   'Pearson_r': pearsonr(x['实际产量'], x['预测产量'])[0]
               }))
               .to_excel(writer, sheet_name=f'{TARGET_PROVINCE}省评估'))
        
        # 完整结果
        df_full.to_excel(writer, sheet_name='详细数据', index=False)
    
    print(f"\n评估结果已保存至 {TEST_YEAR}年LSTM预测评估.xlsx")

    # 可视化各年份R2表现
    plt.figure(figsize=(10, 5))
    years = [y for y in sorted(df_meta['year'].unique()) 
             if sum(df_meta['year'] == y) >= 5]  # 只显示有足够样本的年份
    r2_scores = [next(v for k,v in metrics.items() if k == 'R2') 
                for metrics in yearly_metrics.values()]
    
    plt.bar(years, r2_scores)
    plt.xlabel('年份')
    plt.ylabel('R²分数')
    plt.title('各年份模型表现')
    plt.xticks(years)
    plt.grid(True)
    plt.savefig('各年份R2表现.png')
    plt.show()

    # # 模型保存
    # torch.save({
    #     'model_state': model.state_dict(),
    #     'input_size': X.shape[-1],
    #     'metrics': test_metrics
    # }, 'final_lstm_model.pth')
    # print("\n模型已保存为 final_lstm_model.pth")
