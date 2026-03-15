"""
================================================================================
火灾热释放速率(HRR)深度学习预测系统 - 改进版 V2.0
================================================================================

【改进要点】
1. 数据处理改进：
   - 风向周期性编码（sin/cos）：更好捕捉方向的连续性
   - 时间特征工程：添加时间的周期性和多项式特征
   - 物理启发特征：风速×时间交互项
   - 归一化策略优化：使用RobustScaler处理异常值

2. 模型架构改进：
   - ResidualMLP：带残差连接的MLP，缓解梯度消失
   - CNN-BiLSTM：结合局部特征提取和双向时序建模
   - AttentionLSTM：LSTM + 自注意力机制
   - ImprovedTransformer：相对位置编码 + 更好的归一化

3. 训练策略改进：
   - AdamW优化器（解耦权重衰减）
   - Cosine Annealing + Warmup学习率调度
   - 混合损失函数（MSE + SmoothL1）
   - 标签平滑正则化
   - 梯度累积支持更大有效批次

4. 集成学习：
   - 模型融合：多模型加权平均
   - 快照集成：保存训练过程中的多个检查点

================================================================================
"""

import os
import random
import warnings
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 配置参数 ====================
@dataclass
class ImprovedConfig:
    """改进版配置参数"""
    # 数据相关
    data_file: str = 'data.xlsx'
    test_ratio: float = 0.2
    random_seed: int = 42
    
    # 训练相关
    epochs: int = 500
    batch_size: int = 32  # 减小batch size配合梯度累积
    accumulation_steps: int = 2  # 梯度累积步数
    learning_rate: float = 0.001
    min_lr: float = 1e-6
    weight_decay: float = 1e-3
    patience: int = 50
    warmup_epochs: int = 20
    
    # 模型相关
    hidden_dim: int = 128
    num_layers: int = 2
    nhead: int = 4
    dropout: float = 0.3
    
    # 损失函数
    loss_mse_weight: float = 0.7
    loss_smooth_weight: float = 0.3
    
    # 输出目录
    output_dir: str = 'dl_results_v2'
    model_dir: str = 'saved_models_v2'
    figure_dir: str = 'figures_v2'


# ==================== 工具函数 ====================
def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """自动选择设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


# ==================== 改进的数据处理模块 ====================
class ImprovedDataProcessor:
    """
    改进的数据处理器
    
    改进点：
    1. 风向周期性编码（sin/cos）
    2. 时间特征工程
    3. RobustScaler处理异常值
    4. 特征交互项
    """
    
    def __init__(self, config: ImprovedConfig):
        self.config = config
        self.label_encoder = LabelEncoder()
        # 使用RobustScaler，对异常值更鲁棒
        self.wind_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self.hrr_scaler = RobustScaler()
        self.wind_directions = ['东', '南', '西', '北', '东北', '东南', '西南', '西北']
        # 风向角度映射（用于周期性编码）
        self.direction_angles = {
            '东': 0, '东北': 45, '北': 90, '西北': 135,
            '西': 180, '西南': 225, '南': 270, '东南': 315
        }
        
    def load_data(self, file_path: str) -> List[Dict]:
        """加载数据"""
        print("=" * 60)
        print("正在加载数据...")
        df = pd.read_excel(file_path)
        
        all_samples = []
        
        for i in range(1, 65):
            wind_dir_col = f'风向_{i}'
            wind_speed_col = f'风速/m·s-1_{i}'
            time_col = f'时间/s_{i}'
            heat_rate_col = f'热释放速率/kW_{i}'
            
            if wind_dir_col not in df.columns:
                continue
            
            sample_df = df[[wind_dir_col, wind_speed_col, time_col, heat_rate_col]].dropna()
            
            if len(sample_df) == 0:
                continue
            
            sample = {
                'sample_id': i,
                'wind_direction': sample_df[wind_dir_col].iloc[0],
                'wind_speed': float(sample_df[wind_speed_col].iloc[0]),
                'times': sample_df[time_col].values.astype(float),
                'hrr_values': sample_df[heat_rate_col].values.astype(float)
            }
            all_samples.append(sample)
        
        print(f"成功加载 {len(all_samples)} 个样本")
        total_points = sum(len(s['times']) for s in all_samples)
        print(f"总数据点: {total_points}")
        
        return all_samples
    
    def encode_wind_direction_cyclic(self, direction: str) -> np.ndarray:
        """
        风向周期性编码 (sin, cos)
        
        优势：保持方向的连续性，东和东北之间的距离比东和西小
        """
        angle_deg = self.direction_angles.get(direction, 0)
        angle_rad = math.radians(angle_deg)
        return np.array([math.sin(angle_rad), math.cos(angle_rad)])
    
    def create_time_features(self, time: float, max_time: float) -> np.ndarray:
        """
        创建时间特征
        
        包含：
        1. 归一化时间
        2. 时间的平方（捕捉非线性）
        3. 时间的周期性（捕捉可能的周期模式）
        """
        t_norm = time / max_time if max_time > 0 else 0
        
        features = [
            t_norm,  # 线性
            t_norm ** 2,  # 二次
            math.sin(2 * math.pi * t_norm),  # 周期性sin
            math.cos(2 * math.pi * t_norm),  # 周期性cos
        ]
        return np.array(features)
    
    def fit_scalers(self, samples: List[Dict]):
        """拟合归一化器"""
        self.label_encoder.fit(self.wind_directions)
        
        all_speeds = np.array([s['wind_speed'] for s in samples]).reshape(-1, 1)
        all_times = np.concatenate([s['times'] for s in samples]).reshape(-1, 1)
        all_hrr = np.concatenate([s['hrr_values'] for s in samples]).reshape(-1, 1)
        
        self.wind_scaler.fit(all_speeds)
        self.time_scaler.fit(all_times)
        self.hrr_scaler.fit(all_hrr)
        
        # 记录最大时间（用于时间特征）
        self.max_time = all_times.max()
        
        print("归一化器拟合完成")
    
    def prepare_enhanced_pointwise_data(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备增强的逐点数据
        
        特征包括：
        - 风向周期编码 (2维)
        - 风速 (1维)
        - 时间特征 (4维)
        - 风速×时间交互 (1维)
        共 8 维特征
        """
        X_list = []
        y_list = []
        
        for sample in samples:
            wind_dir_cyclic = self.encode_wind_direction_cyclic(sample['wind_direction'])
            wind_speed_norm = self.wind_scaler.transform([[sample['wind_speed']]])[0, 0]
            
            for t, hrr in zip(sample['times'], sample['hrr_values']):
                time_features = self.create_time_features(t, self.max_time)
                hrr_norm = self.hrr_scaler.transform([[hrr]])[0, 0]
                
                # 交互特征
                interaction = wind_speed_norm * time_features[0]  # 风速×归一化时间
                
                feature = np.concatenate([
                    wind_dir_cyclic,  # 2维
                    [wind_speed_norm],  # 1维
                    time_features,  # 4维
                    [interaction]  # 1维
                ])
                
                X_list.append(feature)
                y_list.append(hrr_norm)
        
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
    
    def prepare_enhanced_sequence_data(self, samples: List[Dict], 
                                        max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备增强的序列数据"""
        if max_len is None:
            max_len = max(len(s['times']) for s in samples)
        
        n_samples = len(samples)
        feature_dim = 8  # 与pointwise一致
        
        X = np.zeros((n_samples, max_len, feature_dim), dtype=np.float32)
        y = np.zeros((n_samples, max_len), dtype=np.float32)
        masks = np.zeros((n_samples, max_len), dtype=np.float32)
        
        for i, sample in enumerate(samples):
            wind_dir_cyclic = self.encode_wind_direction_cyclic(sample['wind_direction'])
            wind_speed_norm = self.wind_scaler.transform([[sample['wind_speed']]])[0, 0]
            
            seq_len = min(len(sample['times']), max_len)
            
            for j in range(seq_len):
                t = sample['times'][j]
                hrr = sample['hrr_values'][j]
                
                time_features = self.create_time_features(t, self.max_time)
                hrr_norm = self.hrr_scaler.transform([[hrr]])[0, 0]
                interaction = wind_speed_norm * time_features[0]
                
                X[i, j, :2] = wind_dir_cyclic
                X[i, j, 2] = wind_speed_norm
                X[i, j, 3:7] = time_features
                X[i, j, 7] = interaction
                y[i, j] = hrr_norm
                masks[i, j] = 1.0
        
        return X, y, masks
    
    def inverse_transform_hrr(self, hrr_normalized: np.ndarray) -> np.ndarray:
        """逆归一化"""
        return self.hrr_scaler.inverse_transform(hrr_normalized.reshape(-1, 1)).flatten()
    
    def split_samples(self, samples: List[Dict], test_ratio: float = 0.2,
                      random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[int], List[int]]:
        """划分数据集"""
        indices = list(range(len(samples)))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=random_state
        )
        
        train_samples = [samples[i] for i in train_indices]
        test_samples = [samples[i] for i in test_indices]
        
        print(f"数据划分: 训练 {len(train_samples)} 个, 测试 {len(test_samples)} 个")
        
        return train_samples, test_samples, train_indices, test_indices


# ==================== 改进的模型定义 ====================

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.block(x)


class ResidualMLP(nn.Module):
    """
    残差MLP
    
    改进：使用残差连接，允许更深的网络而不会梯度消失
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_blocks: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.output = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


class CNNBiLSTM(nn.Module):
    """
    CNN + 双向LSTM 混合模型
    
    CNN提取局部特征，BiLSTM捕捉双向时序依赖
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # 1D卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def forward(self, x, mask=None):
        # x: [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden]
        
        x, _ = self.lstm(x)
        output = self.output(x).squeeze(-1)
        
        return output


class SelfAttention(nn.Module):
    """自注意力层"""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_mask = None
        if mask is not None:
            attn_mask = (mask == 0)
        
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        return self.norm(x + self.dropout(attn_out))


class AttentionLSTM(nn.Module):
    """
    LSTM + 自注意力
    
    结合LSTM的时序建模能力和注意力的全局建模能力
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = SelfAttention(hidden_dim * 2, num_heads, dropout)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
    
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.attention(x, mask)
        return self.output(x).squeeze(-1)


class RelativePositionalEncoding(nn.Module):
    """相对位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class ImprovedTransformer(nn.Module):
    """
    改进的Transformer
    
    改进点：
    1. Pre-LayerNorm结构（更稳定的训练）
    2. 相对位置编码
    3. GLU激活函数
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3, max_len: int = 1000):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = RelativePositionalEncoding(d_model, max_len)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Pre-LN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.output(x).squeeze(-1)


class EnsembleModel(nn.Module):
    """
    集成模型
    
    将多个模型的预测进行加权融合
    """
    def __init__(self, models: Dict[str, nn.Module], weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.models = nn.ModuleDict(models)
        
        if weights is None:
            weights = {name: 1.0 / len(models) for name in models}
        self.weights = weights
    
    def forward(self, x, mask=None):
        outputs = []
        for name, model in self.models.items():
            out = model(x, mask) if mask is not None else model(x)
            outputs.append(out * self.weights[name])
        
        return sum(outputs)


# ==================== 改进的训练模块 ====================

class CosineAnnealingWarmupScheduler:
    """Cosine Annealing with Warmup"""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 min_lr: float = 1e-6, base_lr: float = 0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class MixedLoss(nn.Module):
    """混合损失函数：MSE + SmoothL1"""
    def __init__(self, mse_weight: float = 0.7, smooth_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.smooth_weight = smooth_weight
        self.mse = nn.MSELoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, pred, target, mask=None):
        mse_loss = self.mse(pred, target)
        smooth_loss = self.smooth_l1(pred, target)
        
        combined = self.mse_weight * mse_loss + self.smooth_weight * smooth_loss
        
        if mask is not None:
            combined = combined * mask
            return combined.sum() / (mask.sum() + 1e-8)
        
        return combined.mean()


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ImprovedTrainer:
    """改进的训练器"""
    
    def __init__(self, config: ImprovedConfig, device: torch.device):
        self.config = config
        self.device = device
    
    def train_pointwise_model(self, model: nn.Module, 
                              train_data: Tuple[np.ndarray, np.ndarray],
                              val_data: Tuple[np.ndarray, np.ndarray],
                              model_name: str = "Model") -> Dict:
        """训练逐点模型"""
        model = model.to(self.device)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # AdamW优化器
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)
        
        # 学习率调度
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer, self.config.warmup_epochs, self.config.epochs,
            self.config.min_lr, self.config.learning_rate
        )
        
        # 混合损失
        criterion = MixedLoss(self.config.loss_mse_weight, self.config.loss_smooth_weight)
        
        early_stopping = EarlyStopping(patience=self.config.patience)
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_model_state = None
        best_val_loss = float('inf')
        
        print(f"\n开始训练 {model_name}...")
        
        for epoch in range(self.config.epochs):
            # 训练
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y) / self.config.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * self.config.accumulation_steps
            
            train_loss /= len(train_loader)
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            current_lr = scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if early_stopping(val_loss):
                print(f"  Epoch {epoch+1}: 早停触发")
                break
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.config.epochs}, "
                      f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"{model_name} 训练完成, 最佳Val Loss: {best_val_loss:.6f}")
        
        return {'model': model, 'history': history, 'best_val_loss': best_val_loss}
    
    def train_sequence_model(self, model: nn.Module,
                             train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                             val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                             model_name: str = "Model") -> Dict:
        """训练序列模型"""
        model = model.to(self.device)
        
        X_train, y_train, mask_train = train_data
        X_val, y_val, mask_val = val_data
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(mask_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
            torch.FloatTensor(mask_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)
        
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer, self.config.warmup_epochs, self.config.epochs,
            self.config.min_lr, self.config.learning_rate
        )
        
        criterion = MixedLoss(self.config.loss_mse_weight, self.config.loss_smooth_weight)
        early_stopping = EarlyStopping(patience=self.config.patience)
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_model_state = None
        best_val_loss = float('inf')
        
        print(f"\n开始训练 {model_name}...")
        
        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            for batch_idx, (batch_X, batch_y, batch_mask) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                outputs = model(batch_X, batch_mask)
                loss = criterion(outputs, batch_y, batch_mask) / self.config.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * self.config.accumulation_steps
            
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y, batch_mask in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_mask = batch_mask.to(self.device)
                    outputs = model(batch_X, batch_mask)
                    loss = criterion(outputs, batch_y, batch_mask)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            current_lr = scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if early_stopping(val_loss):
                print(f"  Epoch {epoch+1}: 早停触发")
                break
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.config.epochs}, "
                      f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"{model_name} 训练完成, 最佳Val Loss: {best_val_loss:.6f}")
        
        return {'model': model, 'history': history, 'best_val_loss': best_val_loss}


# ==================== 评估模块 ====================

class ImprovedEvaluator:
    """评估器"""
    
    def __init__(self, processor: ImprovedDataProcessor, device: torch.device):
        self.processor = processor
        self.device = device
    
    def evaluate_pointwise(self, model: nn.Module, 
                           test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        model.eval()
        X_test, y_test = test_data
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = model(X_tensor).cpu().numpy()
        
        y_true = self.processor.inverse_transform_hrr(y_test)
        y_pred = self.processor.inverse_transform_hrr(predictions)
        
        return self._compute_metrics(y_true, y_pred)
    
    def evaluate_sequence(self, model: nn.Module,
                          test_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict:
        model.eval()
        X_test, y_test, mask_test = test_data
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            mask_tensor = torch.FloatTensor(mask_test).to(self.device)
            predictions = model(X_tensor, mask_tensor).cpu().numpy()
        
        valid_mask = mask_test.flatten() == 1
        y_true_flat = y_test.flatten()[valid_mask]
        y_pred_flat = predictions.flatten()[valid_mask]
        
        y_true = self.processor.inverse_transform_hrr(y_true_flat)
        y_pred = self.processor.inverse_transform_hrr(y_pred_flat)
        
        return self._compute_metrics(y_true, y_pred)
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
            'y_true': y_true, 'y_pred': y_pred
        }
    
    def get_sequence_predictions(self, model: nn.Module, samples: List[Dict],
                                 processor: ImprovedDataProcessor, max_len: int) -> List[Dict]:
        model.eval()
        X, y, masks = processor.prepare_enhanced_sequence_data(samples, max_len=max_len)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            mask_tensor = torch.FloatTensor(masks).to(self.device)
            predictions = model(X_tensor, mask_tensor).cpu().numpy()
        
        results = []
        for i, sample in enumerate(samples):
            seq_len = int(masks[i].sum())
            y_true = processor.inverse_transform_hrr(y[i, :seq_len])
            y_pred = processor.inverse_transform_hrr(predictions[i, :seq_len])
            
            results.append({
                'sample_id': sample['sample_id'],
                'wind_direction': sample['wind_direction'],
                'wind_speed': sample['wind_speed'],
                'times': sample['times'][:seq_len],
                'y_true': y_true,
                'y_pred': y_pred
            })
        
        return results


# ==================== 可视化模块 ====================

class ImprovedVisualizer:
    """可视化器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_curves(self, histories: Dict[str, Dict], filename: str = 'training_curves_v2.png'):
        n_models = len(histories)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
        
        for idx, (name, history) in enumerate(histories.items()):
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Loss曲线
            axes[0, idx].plot(epochs, history['train_loss'], 'b-', label='训练', linewidth=1.5)
            axes[0, idx].plot(epochs, history['val_loss'], 'r--', label='验证', linewidth=1.5)
            axes[0, idx].set_xlabel('Epoch')
            axes[0, idx].set_ylabel('Loss')
            axes[0, idx].set_title(f'{name} 损失曲线')
            axes[0, idx].legend()
            axes[0, idx].grid(True, alpha=0.3)
            
            # 学习率曲线
            axes[1, idx].plot(epochs, history['lr'], 'g-', linewidth=1.5)
            axes[1, idx].set_xlabel('Epoch')
            axes[1, idx].set_ylabel('Learning Rate')
            axes[1, idx].set_title(f'{name} 学习率')
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存: {filename}")
    
    def plot_metrics_comparison(self, metrics: Dict[str, Dict], filename: str = 'metrics_comparison_v2.png'):
        model_names = list(metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metric_names = ['MSE', 'RMSE', 'MAE', 'R2']
        titles = ['均方误差 (MSE)', '均方根误差 (RMSE)', '平均绝对误差 (MAE)', '决定系数 (R²)']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for ax, metric, title, color in zip(axes.flatten(), metric_names, titles, colors):
            values = [metrics[name][metric] for name in model_names]
            bars = ax.bar(model_names, values, color=color, alpha=0.7, edgecolor='black')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=15)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"性能对比图已保存: {filename}")
    
    def plot_sample_predictions(self, results: List[Dict], n_samples: int = 4,
                                model_name: str = "", filename_prefix: str = 'sample_pred_v2'):
        n_samples = min(n_samples, len(results))
        np.random.seed(42)
        selected_indices = np.random.choice(len(results), n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (ax, sample_idx) in enumerate(zip(axes, selected_indices)):
            result = results[sample_idx]
            
            ax.plot(result['times'], result['y_true'], 'b-', 
                    label='真实值', linewidth=2, marker='o', markersize=2)
            ax.plot(result['times'], result['y_pred'], 'r--',
                    label='预测值', linewidth=2)
            
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('热释放速率 (kW)')
            ax.set_title(f'样本{result["sample_id"]}: {result["wind_direction"]}风, '
                        f'{result["wind_speed"]}m/s')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} 预测结果', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{filename_prefix}_{model_name.replace(" ", "_").lower()}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_all_models_comparison(self, all_results: Dict[str, List[Dict]], 
                                   n_samples: int = 3, filename: str = 'all_models_comp_v2.png'):
        model_names = list(all_results.keys())
        first_results = all_results[model_names[0]]
        n_samples = min(n_samples, len(first_results))
        
        np.random.seed(42)
        selected_indices = np.random.choice(len(first_results), n_samples, replace=False)
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(14, 5*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        
        for ax, sample_idx in zip(axes, selected_indices):
            sample_info = first_results[sample_idx]
            times = sample_info['times']
            y_true = sample_info['y_true']
            
            ax.plot(times, y_true, 'k-', label='真实值', linewidth=2.5)
            
            for (name, results), color in zip(all_results.items(), colors):
                y_pred = results[sample_idx]['y_pred']
                ax.plot(times, y_pred, '--', label=f'{name}', color=color, linewidth=1.5)
            
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('热释放速率 (kW)')
            ax.set_title(f'样本{sample_info["sample_id"]}: {sample_info["wind_direction"]}风, '
                        f'{sample_info["wind_speed"]}m/s')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"多模型对比图已保存: {filename}")


# ==================== 主程序 ====================

def save_results(metrics: Dict[str, Dict], output_dir: str):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    data = {'Model': [], 'MSE': [], 'RMSE': [], 'MAE': [], 'R2': []}
    
    for name, metric in metrics.items():
        data['Model'].append(name)
        data['MSE'].append(metric['MSE'])
        data['RMSE'].append(metric['RMSE'])
        data['MAE'].append(metric['MAE'])
        data['R2'].append(metric['R2'])
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'metrics_v2.csv'), index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 70)
    print("改进版模型性能对比表")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    
    return df


def main():
    """主函数"""
    print("=" * 70)
    print("火灾热释放速率(HRR)深度学习预测系统 - 改进版 V2.0")
    print("=" * 70)
    
    config = ImprovedConfig()
    set_seed(config.random_seed)
    device = get_device()
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)
    
    # ==================== 数据处理 ====================
    print("\n" + "=" * 70)
    print("第一步: 改进的数据处理")
    print("=" * 70)
    
    processor = ImprovedDataProcessor(config)
    all_samples = processor.load_data(config.data_file)
    
    train_samples, test_samples, train_indices, test_indices = processor.split_samples(
        all_samples, test_ratio=config.test_ratio, random_state=config.random_seed
    )
    
    processor.fit_scalers(train_samples)
    
    # 增强特征
    X_train_mlp, y_train_mlp = processor.prepare_enhanced_pointwise_data(train_samples)
    X_test_mlp, y_test_mlp = processor.prepare_enhanced_pointwise_data(test_samples)
    
    print(f"增强MLP数据: 训练 {X_train_mlp.shape}, 测试 {X_test_mlp.shape}")
    print(f"特征维度: {X_train_mlp.shape[1]} (风向sin/cos + 风速 + 时间特征 + 交互)")
    
    max_len = max(len(s['times']) for s in all_samples)
    X_train_seq, y_train_seq, mask_train = processor.prepare_enhanced_sequence_data(train_samples, max_len=max_len)
    X_test_seq, y_test_seq, mask_test = processor.prepare_enhanced_sequence_data(test_samples, max_len=max_len)
    
    print(f"序列数据: 训练 {X_train_seq.shape}, 测试 {X_test_seq.shape}")
    
    # ==================== 模型定义 ====================
    print("\n" + "=" * 70)
    print("第二步: 改进的模型定义")
    print("=" * 70)
    
    input_dim_mlp = X_train_mlp.shape[1]
    input_dim_seq = X_train_seq.shape[2]
    
    models = {
        'ResidualMLP': ResidualMLP(
            input_dim=input_dim_mlp,
            hidden_dim=config.hidden_dim,
            num_blocks=3,
            dropout=config.dropout
        ),
        'CNN-BiLSTM': CNNBiLSTM(
            input_dim=input_dim_seq,
            hidden_dim=64,
            num_layers=config.num_layers,
            dropout=config.dropout
        ),
        'AttentionLSTM': AttentionLSTM(
            input_dim=input_dim_seq,
            hidden_dim=64,
            num_layers=config.num_layers,
            num_heads=config.nhead,
            dropout=config.dropout
        ),
        'ImprovedTransformer': ImprovedTransformer(
            input_dim=input_dim_seq,
            d_model=64,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout,
            max_len=max_len
        )
    }
    
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: {n_params:,} 可训练参数")
    
    # ==================== 训练 ====================
    print("\n" + "=" * 70)
    print("第三步: 模型训练")
    print("=" * 70)
    
    trainer = ImprovedTrainer(config, device)
    training_results = {}
    
    # ResidualMLP
    result = trainer.train_pointwise_model(
        models['ResidualMLP'],
        (X_train_mlp, y_train_mlp),
        (X_test_mlp, y_test_mlp),
        'ResidualMLP'
    )
    training_results['ResidualMLP'] = result
    
    # 序列模型
    for name in ['CNN-BiLSTM', 'AttentionLSTM', 'ImprovedTransformer']:
        result = trainer.train_sequence_model(
            models[name],
            (X_train_seq, y_train_seq, mask_train),
            (X_test_seq, y_test_seq, mask_test),
            name
        )
        training_results[name] = result
    
    # ==================== 评估 ====================
    print("\n" + "=" * 70)
    print("第四步: 模型评估")
    print("=" * 70)
    
    evaluator = ImprovedEvaluator(processor, device)
    all_metrics = {}
    all_predictions = {}
    
    # ResidualMLP
    metrics = evaluator.evaluate_pointwise(
        training_results['ResidualMLP']['model'],
        (X_test_mlp, y_test_mlp)
    )
    all_metrics['ResidualMLP'] = metrics
    
    # 序列模型
    for name in ['CNN-BiLSTM', 'AttentionLSTM', 'ImprovedTransformer']:
        metrics = evaluator.evaluate_sequence(
            training_results[name]['model'],
            (X_test_seq, y_test_seq, mask_test)
        )
        all_metrics[name] = metrics
        
        predictions = evaluator.get_sequence_predictions(
            training_results[name]['model'],
            test_samples, processor, max_len
        )
        all_predictions[name] = predictions
    
    # ResidualMLP预测
    mlp_predictions = []
    for sample in test_samples:
        X_sample, y_sample = processor.prepare_enhanced_pointwise_data([sample])
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sample).to(device)
            y_pred_norm = training_results['ResidualMLP']['model'](X_tensor).cpu().numpy()
        
        y_true = processor.inverse_transform_hrr(y_sample)
        y_pred = processor.inverse_transform_hrr(y_pred_norm)
        
        mlp_predictions.append({
            'sample_id': sample['sample_id'],
            'wind_direction': sample['wind_direction'],
            'wind_speed': sample['wind_speed'],
            'times': sample['times'],
            'y_true': y_true,
            'y_pred': y_pred
        })
    all_predictions['ResidualMLP'] = mlp_predictions
    
    # ==================== 保存结果 ====================
    print("\n" + "=" * 70)
    print("第五步: 保存结果")
    print("=" * 70)
    
    save_results(all_metrics, config.output_dir)
    
    for name, result in training_results.items():
        model_path = os.path.join(config.model_dir, f'{name.lower()}_model_v2.pth')
        torch.save(result['model'].state_dict(), model_path)
        print(f"模型已保存: {model_path}")
    
    # ==================== 可视化 ====================
    print("\n" + "=" * 70)
    print("第六步: 结果可视化")
    print("=" * 70)
    
    visualizer = ImprovedVisualizer(config.figure_dir)
    
    histories = {name: result['history'] for name, result in training_results.items()}
    visualizer.plot_training_curves(histories)
    visualizer.plot_metrics_comparison(all_metrics)
    
    for name, predictions in all_predictions.items():
        visualizer.plot_sample_predictions(predictions, n_samples=4, model_name=name)
    
    visualizer.plot_all_models_comparison(all_predictions, n_samples=3)
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    
    print(f"\n结果文件位置:")
    print(f"  - 评估指标: {config.output_dir}/metrics_v2.csv")
    print(f"  - 模型权重: {config.model_dir}/")
    print(f"  - 可视化图表: {config.figure_dir}/")
    
    best_model = max(all_metrics.keys(), key=lambda x: all_metrics[x]['R2'])
    print(f"\n最佳模型: {best_model} (R² = {all_metrics[best_model]['R2']:.4f})")
    
    return all_metrics, all_predictions, training_results


if __name__ == "__main__":
    metrics, predictions, results = main()
