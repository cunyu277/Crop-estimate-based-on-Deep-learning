import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, linregress
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset

# 保持原始的文件解析和数据加载逻辑
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
    """加载数据并解析地理信息"""
    X, y, meta = [], [], []
    skipped = {'missing': 0, 'invalid': 0}
    
    for filename in tqdm(os.listdir(directory)):
        if not filename.endswith('.npz'):
            continue
        
        try:
            geo_info = parse_filename(filename)
            data = np.load(os.path.join(directory, filename), allow_pickle=True)
            
            # 获取目标变量
            target_value = data['yield'][target]
            if np.isnan(target_value):
                skipped['missing'] += 1
                continue
                
            # 验证并拼接特征
            features = []
            for key in data.files:
                if key != 'yield':
                    arr = np.array(data[key])
                    if np.isnan(arr).any():
                        skipped['invalid'] += 1
                        break
                    features.append(arr.reshape(-1))
            else:
                X.append(np.concatenate(features))
                y.append(target_value)
                meta.append(geo_info)
                
        except Exception as e:
            print(f"加载失败 {filename}: {str(e)}")
            continue
    
    print(f"\n数据加载完成，跳过记录:")
    print(f"- 目标变量缺失: {skipped['missing']}")
    print(f"- 特征数据无效: {skipped['invalid']}")
    
    df_meta = pd.DataFrame(meta)
    return np.array(X), np.array(y), df_meta

def prepare_lstm_data(X, y, time_steps=1):
    """将数据转换为适合LSTM模型的形状 [samples, time_steps, features]"""
    X_lstm, y_lstm = [], []
    for i in range(len(X) - time_steps):
        X_lstm.append(X[i:i+time_steps])
        y_lstm.append(y[i+time_steps])
    
    return np.array(X_lstm), np.array(y_lstm)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 只取最后一个时间步的输出
        return out

def evaluate_model(model, X_test, y_test):
    """评估模型并可视化结果"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
    y_pred = y_pred.numpy().flatten()
    pearson_r, _ = pearsonr(y_test, y_pred)
    
    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MRE': mean_absolute_percentage_error(y_test, y_pred) * 100,
        'R2': r2_score(y_test, y_pred),
        'Pearson_r': pearson_r
    }

    print("\n评估指标:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")
    
    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='样本点')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='1:1参考线')
    slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
    fit_line = slope * y_test + intercept
    plt.plot(y_test, fit_line, 'b-', linewidth=1,
             label=f'拟合线 (y={slope:.2f}x+{intercept:.2f}, r={r_value:.2f})')
    plt.xlabel('实际产量')
    plt.ylabel('预测产量')
    plt.title('实际产量 vs 预测产量对比')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close('all')

    return metrics

def get_predictions_df(model, X, df_meta, y_true):
    """生成包含预测结果的完整数据表"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
    
    y_pred = y_pred.numpy().flatten()
    df = df_meta.copy()
    df['实际产量'] = y_true
    df['预测产量'] = y_pred
    return df

def train_lstm_model(X_train, y_train, input_size, batch_size=32, epochs=50, learning_rate=0.001):
    """训练LSTM模型"""
    model = LSTMModel(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))  # 视作一列
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    return model
def evaluate_group(y_true, y_pred):
    """计算评估指标"""
    if len(y_true) < 2:
        return {
            '样本数': len(y_true),
            '绝对误差': float(np.abs(y_true - y_pred).iloc[0]),
            '真实值': int(y_true.iloc[0]),
            '预测值': float(y_pred.iloc[0])
        }
    
    # Pearson相关系数
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    # 计算其他评估指标
    return {
        '样本数': len(y_true),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MRE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'R2': r2_score(y_true, y_pred),
        'Pearson_r': pearson_r
    }

if __name__ == "__main__":
    # 配置参数
    DATA_DIR = r"D:\python\crop_yield_prediction\cunuu\predict\enhanced_npz"
    TEST_YEAR = 2016
    TARGET_PROVINCE = "130000"  # 河北省代码
    
    # 加载数据
    X, y, df_meta = load_data(DATA_DIR, target=0)
    print(f"总样本数: {len(X)}")
    print(f"特征维度: {X.shape[1]}")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集
    test_mask = df_meta['year'] == TEST_YEAR
    X_train, X_test = X_scaled[~test_mask], X_scaled[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    
    # 转换为适合LSTM的数据格式
    time_steps = 1  # 你可以根据需求调整
    X_train_lstm, y_train_lstm = prepare_lstm_data(X_train, y_train, time_steps)
    X_test_lstm, y_test_lstm = prepare_lstm_data(X_test, y_test, time_steps)

    # 将数据转化为 PyTorch Tensor
    X_train_lstm = torch.tensor(X_train_lstm, dtype=torch.float32)
    y_train_lstm = torch.tensor(y_train_lstm, dtype=torch.float32)
    X_test_lstm = torch.tensor(X_test_lstm, dtype=torch.float32)
    y_test_lstm = torch.tensor(y_test_lstm, dtype=torch.float32)
    
    # 训练LSTM模型
    input_size = X_train_lstm.shape[2]  # 特征维度
    model = train_lstm_model(X_train_lstm, y_train_lstm, input_size)

       # 评估模型
    test_metrics = evaluate_model(model, X_test_lstm, y_test_lstm)
    
    # 生成预测结果的完整数据表
    df_predictions = get_predictions_df(model, X_test_lstm, df_meta[test_mask], y_test)
    
    # 生成多层级评估结果
    print("\n开始生成评估结果...")
    results = {
        '各市评估': df_predictions.groupby(['province_code', 'city_code'])[['实际产量', '预测产量']].apply(
            lambda x: pd.Series(evaluate_group(x['实际产量'], x['预测产量']))
        ).reset_index(),
        
        '各省评估': df_predictions.groupby('province_code')[['实际产量', '预测产量']].apply(
            lambda x: pd.Series(evaluate_group(x['实际产量'], x['预测产量']))
        ).reset_index(),
        
        '指定省评估': df_predictions[df_predictions['province_code'] == TARGET_PROVINCE].groupby('city_code')[['实际产量', '预测产量']].apply(
            lambda x: pd.Series(evaluate_group(x['实际产量'], x['预测产量']))
        ).reset_index()
    }
    
    # 保存结果到Excel
    with pd.ExcelWriter(f"{TEST_YEAR}年产量预测评估_LSTM.xlsx") as writer:
        results['各市评估'].to_excel(writer, sheet_name='各市评估', index=False)
        results['各省评估'].to_excel(writer, sheet_name='各省评估', index=False)
        results['指定省评估'].to_excel(writer, sheet_name=f'{TARGET_PROVINCE}省各市评估', index=False)
        df_predictions[['province_code', 'city_code', 'year', '实际产量', '预测产量']].to_excel(
            writer, sheet_name='详细数据', index=False)
    
    print(f"""
    结果已保存为 {TEST_YEAR}年产量预测评估_LSTM.xlsx
    包含以下工作表：
    1. 各市评估 - 市级行政区评估指标
    2. 各省评估 - 省级行政区评估指标
    3. {TARGET_PROVINCE}省各市评估 - 指定省份市级评估
    4. 详细数据 - 包含每个样本的真实值和预测值
    """)