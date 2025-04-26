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
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False 
# ==================== 数据加载部分（保持不变） ====================
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
    
    # 显式定义特征顺序和波段数
    FEATURE_STRUCTURE = [
        ('ENTIRE', 7),      # (16,7,32)
        ('EVI', 1),         # (16,1,32)
        ('FPAR', 1),
        ('GNDVI', 1),
        ('LAI', 1),
        ('NDMI', 1),
        ('NDVI', 1),
        ('SIPI', 1),
        ('Temperature', 2)  # (16,2,32)
    ]  # 总波段数 7+1+1+1+1+1+1+1+2=16
    
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
            
            # 按预定顺序拼接特征
            feature_arrays = []
            for feat_name, expected_bands in FEATURE_STRUCTURE:
                if feat_name not in data:
                    raise ValueError(f"缺失关键特征 {feat_name} in {filename}")
                
                arr = data[feat_name]
                
                # 验证形状 (time, bands, bins)
                if arr.ndim != 3 or arr.shape[1] != expected_bands:
                    raise ValueError(
                        f"特征 {feat_name} 形状错误 in {filename}: "
                        f"期望 (16,{expected_bands},32), 实际 {arr.shape}"
                    )
                
                feature_arrays.append(arr)
            
            # 沿波段维度拼接 (16,16,32)
            combined = np.concatenate(feature_arrays, axis=1)
            
            # 验证最终形状
            if combined.shape != (16, 16, 32):
                raise ValueError(
                    f"特征拼接后形状错误 in {filename}: 期望 (16,16,32), 实际 {combined.shape}"
                )
            
            # 展平为 (16*16*32=8192,)
            X.append(combined.reshape(-1))
            y.append(target_value)
            meta.append(geo_info)
            
        except Exception as e:
            print(f"加载失败 {filename}: {str(e)}")
            skipped['invalid'] += 1
            continue
    
    print(f"\n数据加载完成，跳过记录:")
    print(f"- 目标变量缺失: {skipped['missing']}")
    print(f"- 特征数据无效: {skipped['invalid']}")
    
    df_meta = pd.DataFrame(meta)
    return np.array(X), np.array(y), df_meta

# ==================== 3D CNN + BiLSTM 模型 ====================
# ==================== 3D CNN + BiLSTM 模型（修正版本） ====================
class Conv3DLSTMModel(nn.Module):
    def __init__(self, input_shape=(16, 16, 32)):
        super().__init__()
        
        # 增强特征提取能力
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3)),  # 增加时间维度卷积
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),  # 改用LeakyReLU避免梯度消失
            nn.Dropout3d(0.3),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout3d(0.3),
            nn.AdaptiveAvgPool3d((4, 4, 4))  # 自适应池化解决尺寸问题
        )
        
        # 动态计算LSTM输入尺寸（修正部分）
        with torch.no_grad():
            dummy = torch.randn(1, 1, *input_shape)
            cnn_out = self.cnn(dummy)
            # 正确计算每个时间步的特征数：C*H*W
            C = cnn_out.size(1)  # 通道数
            H = cnn_out.size(3)  # 高度
            W = cnn_out.size(4)  # 宽度
            lstm_input_size = C * H * W
        
        # 增强时序建模（input_size修正为2048）
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=128,
            num_layers=3,
            bidirectional=True,  # 增加双向LSTM
            batch_first=True
        )
        
        # 改进回归头（保持原状）
        self.regressor = nn.Sequential(
            nn.Linear(128*2, 128),  # 双向LSTM需*2
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入形状验证
        assert x.dim() == 4, f"输入应为4D Tensor (batch, T, B, bins), 实际为{x.shape}"
        
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch, 1, T, B, bins)
        
        # CNN处理
        cnn_out = self.cnn(x)  # (batch, C, D, H, W)
        
        # 调整维度用于LSTM（关键修正）
        batch_size, C, D, H, W = cnn_out.size()
        lstm_input = cnn_out.permute(0, 2, 1, 3, 4)  # (batch, D, C, H, W)
        lstm_input = lstm_input.reshape(batch_size, D, -1)  # (batch, D, C*H*W)
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)
        
        # 回归预测（关键修正：保留最后一个维度）
        output = self.regressor(lstm_out[:, -1])  # [batch, 1]
        return output.squeeze(-1)  # 安全压缩最后一个维度 [batch]

# ==================== 数据预处理 ====================
def reshape_for_3dcnn(X, time_steps=16, bands=16, bins=32):
    """将展平的特征重新整形为3D CNN输入格式"""
    features_per_timestep = bands * bins
    n_samples = len(X)
    total_features = time_steps * features_per_timestep
    if X.shape[1] != total_features:
        raise ValueError(f"特征维度不匹配，期望{total_features}，实际{X.shape[1]}")
    return X.reshape(n_samples, time_steps, bands, bins)

# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=100, device='cpu'):
    # 改用MSELoss提供更强梯度
    criterion = nn.MSELoss()  
    
    # 使用AdamW优化器+权重衰减
    optimizer = optim.AdamW(model.parameters(), 
                          lr=1e-4, 
                          weight_decay=1e-5)
    
    # 改用余弦退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # 确保输出和目标的形状匹配
            if outputs.shape != y_batch.shape:
                raise ValueError(f"输出形状 {outputs.shape} 与目标形状 {y_batch.shape} 不匹配")
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val)
                
                # 确保预测和目标的形状匹配
                if preds.shape != y_val.shape:
                    raise ValueError(f"预测形状 {preds.shape} 与目标形状 {y_val.shape} 不匹配")
                
                val_loss += criterion(preds, y_val).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()
        
        print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_3dcnn_lstm_model.pth')
    
    print(f'Training complete. Best Val Loss: {best_loss:.4f}')
    
    # 绘制损失下降曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss')
    # plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

# ==================== 评估函数 ====================
def evaluate_model(model, x_test, y_test_raw, df_meta, y_scaler, device='cpu'):
    """改进的分地区评估函数（添加反标准化）"""
    model.eval()
    y_pred = np.zeros_like(y_test_raw)
    
    # 确保输入数据为numpy格式
    if isinstance(x_test, torch.Tensor):
        X_test = x_test.cpu().numpy()
    else:
        X_test = x_test
    
    df_meta = df_meta.reset_index(drop=True)
    
    # 获取唯一地区列表
    region_groups = df_meta.groupby(['province_code', 'city_code'])
    
    for (province_code, city_code), group in tqdm(region_groups, desc="Processing Regions"):
        # 生成地区mask
        region_mask = (df_meta['province_code'] == province_code) & (df_meta['city_code'] == city_code)
        
        # 转换为Tensor并送到设备
        X_region = torch.FloatTensor(X_test[region_mask]).to(device)
        
        with torch.no_grad():
            # 预测并反标准化
            preds = model(X_region).cpu().numpy().flatten()
            preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_pred[region_mask.values] = preds  # 使用.values保证索引正确
    
    # 使用原始真实值计算指标
    pearson_r, _ = pearsonr(y_test_raw, y_pred)

    metrics = {
        'MSE': mean_squared_error(y_test_raw, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_raw, y_pred)),
        'MAE': mean_absolute_error(y_test_raw, y_pred),
        'MRE': mean_absolute_percentage_error(y_test_raw, y_pred) * 100,
        'R2': r2_score(y_test_raw, y_pred),
        'Pearson_r': pearson_r
    }
    # 美化输出
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        if metric == 'MRE':  # 百分比格式
            print(f"{metric:<6}: {value:.2f}%")
        else:  # 其他指标保留4位小数
            print(f"{metric:<6}: {value:.4f}")

    # 可视化部分使用原始值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_raw, y_pred, alpha=0.5, label='样本点')
    plt.plot([min(y_test_raw), max(y_test_raw)], [min(y_test_raw), max(y_test_raw)], 'r--', label='1:1参考线')
    slope, intercept, r_value, p_value, std_err = linregress(y_test_raw, y_pred)
    fit_line = slope * y_test_raw + intercept
    plt.plot(y_test_raw, fit_line, 'b-', linewidth=1,
             label=f'拟合线 (y={slope:.2f}x+{intercept:.2f}, r={r_value:.2f})')
    plt.xlabel('实际产量')
    plt.ylabel('预测产量')
    plt.title('实际产量 vs 预测产量对比（原始量纲）')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close('all')

    return metrics

# ==================== 生成预测结果 ====================
def get_predictions_df(model, X, df_meta, y_true_raw, y_scaler, device='cpu'):
    """生成包含预测结果的完整数据表（添加反标准化）"""
    model.eval()
    y_pred = []
    regions = df_meta[['province_code', 'city_code']].drop_duplicates().reset_index(drop=True)
    
    # 确保输入数据为numpy格式
    if isinstance(X, torch.Tensor):
        X_numpy = X.cpu().numpy()
    else:
        X_numpy = X
    
    for _, region in regions.iterrows():
        region_mask = (df_meta['province_code'] == region['province_code']) & (df_meta['city_code'] == region['city_code'])
        X_region = torch.FloatTensor(X_numpy[region_mask]).to(device)
        
        with torch.no_grad():
            # 预测并反标准化
            preds = model(X_region).cpu().numpy().flatten()
            preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_pred.append(preds)
    
    y_pred = np.concatenate(y_pred)
    
    df = df_meta.copy().reset_index(drop=True)
    df['实际产量'] = y_true_raw  # 原始真实值
    df['预测产量'] = y_pred     # 反标准化后的预测值
    return df

# ==================== 计算评估指标 ====================
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

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置参数
    DATA_DIR = r"D:\python\crop_yield_prediction\cunuu\predict\enhanced_npz"
    TEST_YEAR = 2022
    TARGET_PROVINCE = "130000"  # 河北省代码
    TIME_STEPS = 16  # 时间步数
    BANDS = 16      # 总波段数
    BINS = 32       # 直方图bins数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    X, y, df_meta = load_data(DATA_DIR, target=0)
    print(f"总样本数: {len(X)}")
    
    # 正确流程（在数据划分后进行标准化）
    test_mask = df_meta['year'] == TEST_YEAR

    # 划分原始数据
    X_train_raw, X_test_raw = X[~test_mask], X[test_mask]
    y_train_raw, y_test_raw = y[~test_mask], y[test_mask]

    # 仅用训练集拟合scaler
    x_scaler = StandardScaler().fit(X_train_raw)
    y_scaler = StandardScaler().fit(y_train_raw.reshape(-1,1))

    # 应用标准化
    X_train = x_scaler.transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_train = y_scaler.transform(y_train_raw.reshape(-1,1)).flatten()
    y_test = y_scaler.transform(y_test_raw.reshape(-1,1)).flatten()

    # 重塑为3D输入
    X_train_3d = reshape_for_3dcnn(X_train)
    X_test_3d = reshape_for_3dcnn(X_test)
    
    # 转换为PyTorch Tensor
    X_train_tensor = torch.FloatTensor(X_train_3d).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_3d).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 初始化模型
    model = Conv3DLSTMModel(input_shape=(TIME_STEPS, BANDS, BINS))
    model = model.to(device)
    print(model)
    
    # 训练模型
    train_model(model, train_loader, val_loader, epochs=300, device=device)
    
    # 评估模型
    y_test_numpy = y_test_tensor.cpu().numpy()
    # 在训练后调用评估时：
    test_metrics = evaluate_model(
        model=model,
        x_test=X_test_tensor,
        y_test_raw=y_test_raw,  # 使用原始真实值
        df_meta=df_meta[test_mask].reset_index(drop=True),
        y_scaler=y_scaler,       # 传入scaler用于反标准化
        device=device
    )

    
    # 生成预测结果的完整数据表
    df_predictions = get_predictions_df(
        model=model,
        X=X_test_tensor,
        df_meta=df_meta[test_mask].reset_index(drop=True),
        y_true_raw=y_test_raw,  # 原始真实值
        y_scaler=y_scaler,      # 传入scaler用于反标准化
        device=device
    )
    
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
    with pd.ExcelWriter(f"{TEST_YEAR}年产量预测评估_3DCNN_BiLSTM.xlsx") as writer:
        results['各市评估'].to_excel(writer, sheet_name='各市评估', index=False)
        results['各省评估'].to_excel(writer, sheet_name='各省评估', index=False)
        results['指定省评估'].to_excel(writer, sheet_name=f'{TARGET_PROVINCE}省各市评估', index=False)
        df_predictions[['province_code', 'city_code', 'year', '实际产量', '预测产量']].to_excel(
            writer, sheet_name='详细数据', index=False)
    
    print(f"""
    结果已保存为 {TEST_YEAR}年产量预测评估_3DCNN_LSTM.xlsx
    包含以下工作表：
    1. 各市评估 - 市级行政区评估指标
    2. 各省评估 - 省级行政区评估指标
    3. {TARGET_PROVINCE}省各市评估 - 指定省份市级评估
    4. 详细数据 - 包含每个样本的真实值和预测值
    """)