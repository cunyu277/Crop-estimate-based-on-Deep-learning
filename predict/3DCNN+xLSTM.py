import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# ==================== 数据加载部分 ====================
def parse_filename(filename):
    parts = filename.replace('.npz', '').split('_')
    return {
        'province_code': parts[0],
        'city_code': parts[1],
        'year': int(parts[2]),
        'other': '_'.join(parts[3:]) if len(parts) > 3 else None
    }

def load_data(directory, target=0):
    X, y, meta = [], [], []
    skipped = {'missing': 0, 'invalid': 0}
    
    FEATURE_STRUCTURE = [
        ('ENTIRE', 7), ('EVI', 1), ('FPAR', 1), ('GNDVI', 1),
        ('LAI', 1), ('NDMI', 1), ('NDVI', 1), ('SIPI', 1), ('Temperature', 2)
    ]
    
    for filename in tqdm(os.listdir(directory)):
        if not filename.endswith('.npz'):
            continue
        
        try:
            data = np.load(os.path.join(directory, filename), allow_pickle=True)
            geo_info = parse_filename(filename)
            
            target_value = data['yield'][target]
            if np.isnan(target_value):
                skipped['missing'] += 1
                continue
            
            combined = np.concatenate([data[feat[0]] for feat in FEATURE_STRUCTURE], axis=1)
            X.append(combined.reshape(-1))
            y.append(target_value)
            meta.append(geo_info)
            
        except Exception as e:
            skipped['invalid'] += 1
    
    print(f"加载完成，跳过{skipped['missing']}缺失值，{skipped['invalid']}无效数据")
    return np.array(X), np.array(y), pd.DataFrame(meta)

# ==================== 数据增强模块 ====================
class CropNoiseAugment:
    def __init__(self, time_steps=16, bands=16, bins=32):
        self.time_steps = time_steps
        self.bands = bands
        self.bins = bins
        
    def __call__(self, x):
        if np.random.rand() < 0.8:  # 时间裁剪
            crop_len = int(self.time_steps*0.95)
            start = np.random.randint(0, self.time_steps-crop_len)
            x = x[start:start+crop_len]
            x = np.pad(x, ((0, self.time_steps-crop_len), (0,0), (0,0)), mode='constant')
        
        if np.random.rand() < 0.5:  # 波段遮罩
            mask = np.random.rand(self.bands) < 0.1
            x[:, mask] = 0
        
        if np.random.rand() < 0.5:  # 高斯噪声
            x += np.random.normal(0, 0.01, x.shape)
        
        if np.random.rand() < 0.5:  # 时间翻转
            x = np.flip(x, axis=0).copy()
        
        return x

class AugmentedDataset(TensorDataset):
    def __init__(self, *tensors, augment=None):
        super().__init__(*tensors)
        self.augment = augment
        
    def __getitem__(self, index):
        x, y = self.tensors[0][index], self.tensors[1][index]
        if self.augment:
            x_np = x.numpy() if isinstance(x, torch.Tensor) else x
            x = torch.FloatTensor(self.augment(x_np))
        return x, y

# ==================== xLSTM核心模块 ====================
class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size-1)*dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            padding=self.padding, dilation=dilation, **kwargs)
        
    def forward(self, x):
        return self.conv(x)[:, :, :-self.padding] if self.padding else self.conv(x)

class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super().__init__()
        block_out = out_features // num_blocks
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, block_out) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        return torch.cat([block(x) for block in self.blocks], dim=-1)

class mLSTMBlock(nn.Module):
    def __init__(self, input_size, head_size, num_heads, proj_factor=2):
        super(mLSTMBlock, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.hidden_size = head_size * num_heads
        self.num_heads = num_heads
        self.proj_factor = proj_factor

        assert proj_factor > 0

        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, self.hidden_size)
        self.down_proj = nn.Linear(self.hidden_size, input_size)

        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), self.hidden_size)

        self.Wq = BlockDiagonal(int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wi = nn.Linear(int(input_size * proj_factor), self.hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), self.hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), self.hidden_size)

        self.group_norm = nn.GroupNorm(num_heads, self.hidden_size)

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state

        h_prev = h_prev.to(x.device)
        c_prev = c_prev.to(x.device)
        n_prev = n_prev.to(x.device)
        m_prev = m_prev.to(x.device)

        assert x.size(-1) == self.input_size
        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)

        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)

        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_up_left)

        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * (v * k)  # v @ k.T
        n_t = f * n_prev + i * k
        h_t = o * (c_t * q) / torch.max(torch.abs(n_t.T @ q), 1)[0]  # o * (c @ q) / max{|n.T @ q|, 1}

        output = h_t
        output_norm = self.group_norm(output)
        output = output_norm + x_skip
        output = output * F.silu(x_up_right)
        output = self.down_proj(output)
        final_output = output + x

        return final_output, (h_t, c_t, n_t, m_t)


class sLSTMBlock(nn.Module):
    def __init__(self, input_size, head_size, num_heads, proj_factor=4 / 3):
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.hidden_size = head_size * num_heads
        self.num_heads = num_heads
        self.proj_factor = proj_factor

        assert proj_factor > 0

        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)

        self.Wz = BlockDiagonal(input_size, self.hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, self.hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, self.hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, self.hidden_size, num_heads)

        self.Rz = BlockDiagonal(self.hidden_size, self.hidden_size, num_heads)
        self.Ri = BlockDiagonal(self.hidden_size, self.hidden_size, num_heads)
        self.Rf = BlockDiagonal(self.hidden_size, self.hidden_size, num_heads)
        self.Ro = BlockDiagonal(self.hidden_size, self.hidden_size, num_heads)

        self.group_norm = nn.GroupNorm(num_heads, self.hidden_size)

        self.up_proj_left = nn.Linear(self.hidden_size, int(self.hidden_size * proj_factor))
        self.up_proj_right = nn.Linear(self.hidden_size, int(self.hidden_size * proj_factor))
        self.down_proj = nn.Linear(int(self.hidden_size * proj_factor), input_size)

    def forward(self, x, prev_state):
        assert x.size(-1) == self.input_size
        h_prev, c_prev, n_prev, m_prev = prev_state

        h_prev = h_prev.to(x.device)
        c_prev = c_prev.to(x.device)
        n_prev = n_prev.to(x.device)
        m_prev = m_prev.to(x.device)

        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        z = torch.tanh(self.Wz(x_norm) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x_norm) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t

        output = h_t
        output_norm = self.group_norm(output)
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = F.gelu(output_right)
        output = output_left * output_gated
        output = self.down_proj(output)
        final_output = output + x

        return final_output, (h_t, c_t, n_t, m_t)


class xLSTM(nn.Module):
    def __init__(self, input_size, head_size, num_heads, layers, batch_first=False, proj_factor_slstm=4 / 3,
                 proj_factor_mlstm=2):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.hidden_size = head_size * num_heads
        self.num_heads = num_heads
        self.layers = layers
        self.num_layers = len(layers)
        self.batch_first = batch_first
        self.proj_factor_slstm = proj_factor_slstm
        self.proj_factor_mlstm = proj_factor_mlstm

        self.layers = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMBlock(input_size, head_size, num_heads, proj_factor_slstm)
            elif layer_type == 'm':
                layer = mLSTMBlock(input_size, head_size, num_heads, proj_factor_mlstm)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)

    def forward(self, x, state=None):
        assert x.ndim == 3
        if self.batch_first: x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        if state is not None:
            state = torch.stack(list(state)).to(x.device)
            assert state.ndim == 4
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4
            assert state_num_layers == self.num_layers
            assert state_batch_size == batch_size
            assert state_input_size == self.input_size
            state = state.transpose(0, 1)
        else:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size, device=x.device)

        output = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))
            output.append(x_t)

        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        state = tuple(state.transpose(0, 1))
        return output, state

# ==================== 3D CNN + xLSTM模型 ====================
class Conv3DxLSTMModel(nn.Module):
    def __init__(self, input_shape=(16, 16, 32)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 64, (3,3,3)), 
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout3d(0.3),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, (3,3,3)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout3d(0.3),
            nn.AdaptiveAvgPool3d((4,4,4))
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, *input_shape)
            cnn_out = self.cnn(dummy)
            self.lstm_input_size = cnn_out.size(1) * cnn_out.size(3) * cnn_out.size(4)
        
        self.xlstm = xLSTM(
            input_size=self.lstm_input_size,
            head_size=32,
            num_heads=4,
            layers=['m', 's'],
            batch_first=True
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(self.lstm_input_size, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        cnn_out = self.cnn(x)
        batch_size, C, D, H, W = cnn_out.size()
        lstm_input = cnn_out.permute(0, 2, 1, 3, 4).reshape(batch_size, D, -1)
        lstm_out, _ = self.xlstm(lstm_input)
        return self.regressor(lstm_out[:, -1]).squeeze(-1)

# ==================== 训练评估函数 ====================
def reshape_for_3dcnn(X, time_steps=16, bands=16, bins=32):
    return X.reshape(-1, time_steps, bands, bins)

def train_model(model, train_loader, val_loader, epochs=100, device='cpu'):
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X.to(device)).cpu()
                val_loss += criterion(pred, y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train:.4f}, Val Loss {avg_val:.4f}")
        if avg_val < best_loss:
            torch.save(model.state_dict(), 'best_model.pth')
            best_loss = avg_val
    
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.show()

def evaluate_model(model, x_test, y_test_raw, df_meta, y_scaler, device='cpu'):
    model.eval()
    y_pred = np.zeros_like(y_test_raw)
    # 修正：确保张量先移动到CPU再转换为numpy
    X_test = x_test.cpu().numpy() if isinstance(x_test, torch.Tensor) else x_test
    
    for (prov, city), group in tqdm(df_meta.groupby(['province_code', 'city_code'])):
        idx = group.index
        X_part = torch.FloatTensor(X_test[idx]).to(device)  # 转换为张量并移至设备
        with torch.no_grad():
            pred = model(X_part).cpu().numpy()  # 预测结果移至CPU再转换为numpy
            pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
            y_pred[idx] = pred
    
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

# ==================== 主程序流程 ====================
if __name__ == "__main__":
    # 配置参数
    DATA_DIR = r"D:\python\crop_yield_prediction\cunuu\predict\enhanced_npz"
    TEST_YEAR = 2016
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
    
    # 修改数据加载器部分（替换原来的错误代码）
    train_dataset = AugmentedDataset(
        torch.FloatTensor(X_train_3d),
        torch.FloatTensor(y_train),
        augment=CropNoiseAugment()  # 启用数据增强
    )
    val_dataset = AugmentedDataset(
        torch.FloatTensor(X_test_3d),
        torch.FloatTensor(y_test),
        augment=None  # 验证时不增强
    )
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化优化模型
    model = Conv3DxLSTMModel(input_shape=(TIME_STEPS, BANDS, BINS)).to(device)
    
    model = model.to(device)
    print(model)
    
    # 训练模型
    train_model(model, train_loader, val_loader, epochs=20, device=device)
    
    # 评估模型
    y_test_numpy = y_test_tensor.cpu().numpy()
    # 在训练后调用评估时：
    test_metrics = evaluate_model(
        model=model,
        x_test=X_test_tensor.cpu(),  # 确保传入CPU张量
        y_test_raw=y_test_raw,
        df_meta=df_meta[test_mask].reset_index(drop=True),
        y_scaler=y_scaler,
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
    # 结果保存（保持原有格式）
    with pd.ExcelWriter(f"{TEST_YEAR}年产量预测评估_3DCNN_xLSTM_without_50.xlsx") as writer:
        results['各市评估'].to_excel(writer, sheet_name='各市评估', index=False)
        results['各省评估'].to_excel(writer, sheet_name='各省评估', index=False)
        results['指定省评估'].to_excel(writer, sheet_name=f'{TARGET_PROVINCE}省各市评估', index=False)
        df_predictions[['province_code','city_code','year','实际产量','预测产量']].to_excel(
            writer, sheet_name='详细数据', index=False)