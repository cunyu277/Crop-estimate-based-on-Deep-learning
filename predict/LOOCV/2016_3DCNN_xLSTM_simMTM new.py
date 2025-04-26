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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="libpng warning: iCCP")

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
# ==================== SimMTM时间掩码模块 ====================
class SimMTMMask:
    def __init__(self, 
                 base_ratio=0.15,
                 dynamic_range=(0.4, 0.6),
                 time_steps=16,
                 bands=16,
                 min_scale=3,
                 max_scale=4):
        """
        改进的SimMTM风格时间掩码策略
        参数：
            base_ratio: 基础掩码比例 (0-0.3)
            dynamic_range: 动态调整比例范围 (min, max)
            time_steps: 时间步数
            bands: 波段数
            min_scale: 最小时间尺度（块长度）
            max_scale: 最大时间尺度
        """
        self.base_ratio = base_ratio
        self.dynamic_range = dynamic_range
        self.time_steps = time_steps
        self.bands = bands
        self.scales = list(range(min_scale, max_scale+1))
        
    def _compute_similarity(self, x):
        """计算时间维度相似性矩阵"""
        # 滑动窗口局部相似性计算
        similarity = np.zeros((self.time_steps, self.time_steps))
        for i in range(self.time_steps):
            for j in range(i, self.time_steps):
                # 使用余弦相似度
                win_size = min(3, self.time_steps - j)
                a = x[i:i+win_size].flatten()
                b = x[j:j+win_size].flatten()
                if len(a) == 0 or len(b) == 0:
                    similarity[i,j] = 0
                else:
                    similarity[i,j] = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8)
        return similarity

    def _adaptive_masking(self, x):
        """动态调整掩码策略"""
        # 1. 计算时序相似性
        sim_matrix = self._compute_similarity(x)
        
        # 2. 识别高冗余区域
        redundancy = np.mean(sim_matrix, axis=1)
        dynamic_ratio = np.clip(
            self.base_ratio * (1 + redundancy.mean()), 
            *self.dynamic_range
        )
        
        # 3. 多尺度掩码生成
        total_mask = np.zeros((self.time_steps, 1))
        remaining = int(self.time_steps * dynamic_ratio)
        
        # 优先掩码高冗余区域
        while remaining > 0:
            scale = np.random.choice(self.scales)
            scale = min(scale, remaining)
            
            # 选择最冗余的起始点
            start = np.argmax(redundancy)
            start = max(0, start - scale//2)
            end = min(start + scale, self.time_steps)
            
            # 应用掩码并更新剩余配额
            mask_length = end - start
            total_mask[start:end] = 1
            remaining -= mask_length
            
            # 更新冗余度（避免重叠）
            redundancy[start:end] = -np.inf
            
        return total_mask

    def __call__(self, x):
        """
        x: 输入时序数据 (T, Bands, Bins)
        返回: 掩码后的数据 (同维度)
        """
        # 步骤1：生成基础掩码模板
        base_mask = self._adaptive_masking(x[..., 5])  # 取第一个波段计算
        
        # 步骤2：跨波段协同（随机选择部分波段应用）
        band_mask = np.random.rand(self.bands) < 0.5  # 70%波段应用相同掩码
        final_mask = base_mask * band_mask.reshape(1, -1, 1)
        
        # 步骤3：应用掩码并添加高斯噪声
        masked_x = x * (1 - final_mask)
        noise = np.random.normal(0, 0.03, x.shape) * final_mask
        return np.clip(masked_x + noise, 0, 1)

# ==================== 数据增强模块 ====================
class CropNoiseAugment:
    def __init__(self, time_steps=16, bands=16, bins=32, use_simmtm=True):
        self.time_steps = time_steps
        self.bands = bands
        self.bins = bins
        self.simmtm_mask = SimMTMMask() if use_simmtm else None
        
    def __call__(self, x):
        # 应用SimMTM时间掩码 (80%概率)
        if self.simmtm_mask and np.random.rand() < 0.9:
            # 初始化（参数需根据实际数据调整）
            masker = SimMTMMask(
                base_ratio=0.25,  # 基础掩码比例
                dynamic_range=(0.1, 0.3),  # 动态调整范围
                time_steps=16,
                bands=16,
                min_scale=3,
                max_scale=6
            )
            y = masker(x)
            # # 绘制原始数据和掩码后数据的对比图（横向排列）
            # plt.figure(figsize=(15, 6))  # 调整画布大小

            # # 子图1：原始时序数据
            # plt.subplot(121)  # 1行2列的第1个子图
            # plt.imshow(x[:, :, 0].T)  # 转置使图像方向正确，使用viridis配色
            # plt.colorbar(label='数值范围', shrink=0.8)  # 添加颜色条
            # plt.title("原始时序数据", fontsize=12, pad=15)  # 添加标题和间距
            # plt.xlabel("时间维度", fontsize=10)  # X轴标签
            # plt.ylabel("空间维度", fontsize=10)  # Y轴标签

            # # 子图2：掩码后数据
            # plt.subplot(122)  # 1行2列的第2个子图
            # plt.imshow(y[:, :, 0].T)  # 保持与原始数据相同的配色方案
            # plt.colorbar(label='数值范围', shrink=0.8)
            # plt.title("SimMTM掩码后数据", fontsize=12, pad=15)
            # plt.xlabel("时间维度", fontsize=10)
            # plt.ylabel("空间维度", fontsize=10)

            # # 添加整体标题和调整布局
            # plt.suptitle("时序数据掩码前后对比", fontsize=14, y=1.02)  # 主标题
            # plt.tight_layout()  # 自动调整子图间距
            # plt.show()

            # 更新数据
            x = y

        if np.random.rand() < 0.8:  # 时间裁剪
            crop_len = int(self.time_steps*0.95)
            start = np.random.randint(0, self.time_steps-crop_len)
            x = x[start:start+crop_len]
            x = np.pad(x, ((0, self.time_steps-crop_len), (0,0), (0,0)), mode='constant')
        
        # if np.random.rand() < 0.4:  # 波段遮罩
        #     mask = np.random.rand(self.bands) < 0.1
        #     x[:, mask] = 0
        
        # if np.random.rand() < 0.4:  # 高斯噪声
        #     x += np.random.normal(0, 0.01, x.shape)
        
        # if np.random.rand() < 0.4:  # 时间翻转
        #     x = np.flip(x, axis=0).copy()
        
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
    

# ==================== 通道注意力 ====================  
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.shape
        # 沿时间维度平均
        weights = self.fc(x.mean(dim=1))  # [B, C]
        return x * weights.unsqueeze(1)  # [B, T, C]
    
# ==================== 时间注意力 ====================  
class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # [B, T, C] * 3
        
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)  # [B, T, T]
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)  # [B, T, C]
        return self.proj(out)
    
# ==================== 因果时间注意力 ====================  
class CausalTemporalAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
        # 创建可训练的位置偏置
        self.pos_bias = nn.Parameter(torch.Tensor(1, heads, 1, dim//heads))
        nn.init.xavier_uniform_(self.pos_bias)
        
        self.register_buffer("mask", None)

    def get_mask(self, n, device):
        if self.mask is None or self.mask.shape[-1] < n:
            # 生成下三角掩码（含对角线）
            mask = torch.tril(torch.ones(n, n, device=device))
            self.mask = mask.view(1, 1, n, n)
        return self.mask[:, :, :n, :n]

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # 分割多头并添加位置偏置
        q = q.view(B, T, self.heads, -1).transpose(1, 2) + self.pos_bias
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 应用因果掩码
        mask = self.get_mask(T, x.device)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        
        return self.to_out(out)
    
# ==================== 频率增强通道注意力 ====================
class FECAM(nn.Module):
    def __init__(self, channels, reduction=8, use_freq=True):
        super().__init__()
        self.use_freq = use_freq
        self.channels = channels
        self.reduction = reduction
        
        # 共享的特征变换层
        self.transform = nn.Sequential(
            nn.Conv1d(channels, channels//reduction, 1),
            nn.GELU(),
            nn.Conv1d(channels//reduction, channels, 1)
        )
        
        # 频域路径
        if use_freq:
            self.freq_conv = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                nn.GELU()
            )
            self.freq_gate = nn.Parameter(torch.ones(1, channels, 1))
            
        # 空域路径
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)
        
        # 融合层
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)  # [B, C, T]
        
        # 空域分支
        spatial_att = self.spatial_pool(x)  # [B, C, 1]
        spatial_att = self.transform(spatial_att)
        
        if self.use_freq:
            # 频域分支
            x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
            x_fft = self.freq_conv(x_fft.abs())
            freq_att = self.transform(x_fft.mean(dim=-1, keepdim=True))
            freq_att = freq_att * self.freq_gate
            
            # 特征融合
            combined = spatial_att + freq_att
        else:
            combined = spatial_att
            
        # 生成注意力权重
        att = self.sigmoid(combined)
        
        return (x * att).transpose(1, 2)

class ChannelAttention3D(nn.Module):
    """3D通道注意力，适用于CNN特征图"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, _, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1, 1)
        return x * y

class SpatioTemporalAttention3D(nn.Module):
    """3D时空注意力，同时关注空间和时间维度"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 空间注意力
        self.spatial_conv = nn.Conv3d(channels, 1, kernel_size=1)
        
        # 时间注意力
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(channels, channels//8, (3,1,1), padding=(1,0,0)),
            nn.GELU(),
            nn.Conv3d(channels//8, 1, (1,1,1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 空间注意力 [B,1,D,H,W]
        spatial_att = self.spatial_conv(x).sigmoid()
        
        # 时间注意力 [B,1,D,H,W]
        temporal_att = self.temporal_conv(x)
        
        # 组合注意力
        combined_att = spatial_att * temporal_att
        return x * combined_att
    
# ==================== 回归器 ====================  
class EnhancedRegressor(nn.Module):
    def __init__(self, lstm_input_size):
        super().__init__()
        # 输入投影层
        self.proj = nn.Sequential(
            nn.Linear(lstm_input_size, 256),
            nn.LayerNorm(256)
        )
        
        # # 注意力模块
        # self.temp_attn = CausalTemporalAttention(256)
        # self.fecam = FECAM(256)  # 替换原有通道注意力
        
        # 预测头
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),  # 使用SiLU激活函数
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 修改为使用ReLU的增益计算（与SiLU最接近的官方支持选项）
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    
    def forward(self, x):
        x = self.proj(x)  # [B, T, 256]
        x = x + self.temp_attn(x)  # 残差连接
        x = self.fecam(x)  # 频率增强注意力
        return self.head(x[:, -1])  # 取最后时间步
    
# ==================== 3D CNN + xLSTM模型 ====================
class Conv3DxLSTMModel(nn.Module):
    def __init__(self, input_shape=(16, 16, 32)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 64, (3,3,3)), 
            nn.BatchNorm3d(64),
            nn.SiLU(),
            # ChannelAttention3D(64),  # 新增3D通道注意力
            nn.MaxPool3d(2),
            nn.Dropout3d(0.5),

            # 第二卷积块 + 时空注意力
            nn.Conv3d(64, 128, (3,3,3), padding=1),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            # SpatioTemporalAttention3D(128),  # 新增3D时空注意力
            nn.AdaptiveAvgPool3d((4,4,4))
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, *input_shape)
            cnn_out = self.cnn(dummy)
            self.lstm_input_size = cnn_out.size(1) * cnn_out.size(3) * cnn_out.size(4)
        
        self.xlstm = xLSTM(
            input_size=self.lstm_input_size,
            head_size=64,
            num_heads=2,
            layers=['m', 's'],
            batch_first=True
        )
        
        self.regressor = EnhancedRegressor(self.lstm_input_size)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # x: [B, T=16, Bands=16, Bins=32]
        x = x.unsqueeze(1)  # [B, 1, T, Bands, Bins]
        
        # 3DCNN处理
        cnn_out = self.cnn(x)  # [B, C, T', H', W']
        B, C, D, H, W = cnn_out.shape
        
        # 准备xLSTM输入
        lstm_input = cnn_out.permute(0, 2, 1, 3, 4).reshape(B, D, -1)  # [B, T', C*H'*W']
        
        # xLSTM处理
        lstm_out, _ = self.xlstm(lstm_input)  # [B, T', features]
        
        # 回归预测
        return self.regressor(lstm_out).squeeze(-1)

# ==================== 训练评估函数 ====================
def reshape_for_3dcnn(X, time_steps=16, bands=16, bins=32):
    return X.reshape(-1, time_steps, bands, bins)

def frequency_augment(x, p=0.5):
    """频域数据增强"""
    if np.random.random() > p:
        return x
    
    # 确保输入是3D张量 [B, T, C]
    if x.dim() == 5:  # 3DCNN输入 [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(x.size(0), x.size(1), -1)  # [B, T, C*H*W]
    
    # 执行FFT
    x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
    
    # 随机滤波 (保留低频成分)
    mask = torch.ones_like(x_fft)
    freq_cutoff = int(x_fft.size(1) * np.random.uniform(0.5, 0.9))
    mask[:, freq_cutoff:] = 0
    
    # 应用滤波并转换回时域
    x_fft = x_fft * mask
    return torch.fft.irfft(x_fft, n=x.size(1), dim=1)

def train_model(model, train_loader, val_loader, epochs=100, device='cpu'):
    #criterion = nn.MSELoss()
    #criterion = nn.HuberLoss()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), 
                          lr=1e-4, 
                          weight_decay=1e-5,
                          betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    # 创建漂亮的表格头
    header = f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'Status':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # 使用更美观的进度条
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}", 
                 bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device).float()  # 确保y是float

                # # 频域数据增强
                # if model.training:
                #     # 先保存原始数据用于恢复
                #     X_original = X.clone()
                #     try:
                #         X = frequency_augment(X)
                #     except Exception as e:
                #         print(f"Augmentation failed: {e}, using original data")
                #         X = X_original

                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()

                 # 添加梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X.to(device)).cpu()
                val_loss += criterion(pred, y).item()
        
        # 计算平均loss
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        scheduler.step()
        
        # 漂亮的epoch总结
        status = "✓ Improved" if avg_val < best_loss else ""
        if avg_val < best_loss:
            torch.save(model.state_dict(), 'best_model.pth')
            best_loss = avg_val
        
        print(f"{epoch+1:<8}{avg_train:<15.4f}{avg_val:<15.4f}{status:<10}")
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    # plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Training Progress', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return history


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
    plt.title('实际产量 vs 预测产量对比（kg/ha）')
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
    
    # # 修改数据加载器部分（替换原来的错误代码）
    train_dataset = AugmentedDataset(
        torch.FloatTensor(X_train_3d),
        torch.FloatTensor(y_train),
        augment=CropNoiseAugment()  # 启用数据增强
    )
    # train_dataset = AugmentedDataset(
    #     torch.FloatTensor(X_train_3d),
    #     torch.FloatTensor(y_train),
    #     augment=None  # 启用数据增强
    # )
    val_dataset = AugmentedDataset(
        torch.FloatTensor(X_test_3d),
        torch.FloatTensor(y_test),
        augment=None  # 验证时不增强
    )
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化优化模型
    model = Conv3DxLSTMModel(input_shape=(TIME_STEPS, BANDS, BINS)).to(device)
    
    model = model.to(device)
    print(model)
    
    # 训练模型
    train_model(model, train_loader, val_loader, epochs=200, device=device)
    
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
    with pd.ExcelWriter(f"{TEST_YEAR}年产量预测评估_3DCNN_xLSTM.xlsx") as writer:
        results['各市评估'].to_excel(writer, sheet_name='各市评估', index=False)
        results['各省评估'].to_excel(writer, sheet_name='各省评估', index=False)
        results['指定省评估'].to_excel(writer, sheet_name=f'{TARGET_PROVINCE}省各市评估', index=False)
        df_predictions[['province_code','city_code','year','实际产量','预测产量']].to_excel(
            writer, sheet_name='详细数据', index=False)
    print(f"""结果已保存为 {TEST_YEAR}年产量预测评估_3DCNN_xLSTM.xlsx""")
    # 保存完整模型（结构和参数）
    torch.save({
        'model_state_dict': model.state_dict(),
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'input_shape': (TIME_STEPS, BANDS, BINS)
    }, f'{TEST_YEAR}_crop_yield_model_without.pth')

    print(f"模型已保存为 {TEST_YEAR}_crop_yield_model_without.pth")
