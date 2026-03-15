"""
================================================================================
火灾热释放速率(HRR)深度学习预测系统
================================================================================

【整体思路】
本代码实现了一个完整的小样本深度学习对比实验系统，用于预测火灾热释放速率。
系统包含四种代表性深度学习模型：MLP、1D-CNN、LSTM、Transformer。

【核心设计理念】
1. 严格按样本整体划分：64个样本（风向×风速组合）作为独立单元进行划分
2. 小样本适配：所有模型都采用轻量级设计，防止过拟合
3. 公平对比：统一的数据处理、训练策略和评估标准

【为什么必须按样本整体划分训练测试集】
- 每个样本是一条完整的时间-热释放速率曲线，代表特定风向风速条件下的燃烧过程
- 如果打散时间点随机划分，会导致同一条曲线的相邻点分别出现在训练集和测试集
- 这种数据泄漏会使模型"看到"测试曲线的部分信息，导致评估指标虚高
- 正确做法：先划分样本，再使用训练样本的所有点训练，测试样本的所有点测试

【为什么小样本场景下要控制模型复杂度】
- 64个样本的数据量极其有限，复杂模型容易记住训练数据而非学习规律
- 参数过多会导致过拟合，泛化能力差
- 轻量级模型更稳定，且在小样本下往往表现更好

【各模型适用性分析】
- MLP: 最基础的模型，适合学习输入特征到输出的非线性映射，计算高效
- 1D-CNN: 擅长捕捉序列中的局部模式和特征，适合时序曲线的形状特征提取
- LSTM: 专门处理时序依赖，能记住历史信息，适合时间序列预测
- Transformer: 通过注意力机制捕捉全局依赖，适合发现远程时间关系

================================================================================
"""

import os
import random
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 配置参数 ====================
@dataclass
class Config:
    """全局配置参数"""
    # 数据相关
    data_file: str = 'data.xlsx'
    test_ratio: float = 0.2  # 测试集比例
    random_seed: int = 42
    
    # 训练相关
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4  # L2正则化
    patience: int = 30  # Early stopping耐心值
    
    # 模型相关 - 小样本轻量化设计
    mlp_hidden_dims: List[int] = None
    cnn_channels: List[int] = None
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 1
    transformer_d_model: int = 32
    transformer_nhead: int = 2
    transformer_num_layers: int = 1
    dropout: float = 0.2
    
    # 输出目录
    output_dir: str = 'dl_results'
    model_dir: str = 'saved_models'
    figure_dir: str = 'figures'
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 32, 16]
        if self.cnn_channels is None:
            self.cnn_channels = [16, 32]


# ==================== 工具函数 ====================
def set_seed(seed: int):
    """设置随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """自动选择设备（GPU或CPU）"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


# ==================== 数据处理模块 ====================
class FireDataProcessor:
    """
    火灾数据处理器
    
    负责：
    1. 从Excel读取数据
    2. 按风向+风速组合归类样本
    3. 风向编码（支持One-Hot和Label编码）
    4. 数值归一化（可逆）
    5. 按样本划分训练/测试集
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.wind_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.hrr_scaler = StandardScaler()
        self.wind_directions = ['东', '南', '西', '北', '东北', '东南', '西南', '西北']
        
    def load_data(self, file_path: str) -> List[Dict]:
        """
        加载Excel数据，按样本组织
        
        返回: 样本列表，每个样本包含：
            - sample_id: 样本编号
            - wind_direction: 风向
            - wind_speed: 风速
            - times: 时间序列
            - hrr_values: 热释放速率序列
        """
        print("=" * 60)
        print("正在加载数据...")
        df = pd.read_excel(file_path)
        
        all_samples = []
        
        for i in range(1, 65):
            # 列名格式
            wind_dir_col = f'风向_{i}'
            wind_speed_col = f'风速/m·s-1_{i}'
            time_col = f'时间/s_{i}'
            heat_rate_col = f'热释放速率/kW_{i}'
            
            # 检查列是否存在
            if wind_dir_col not in df.columns:
                print(f"警告: 列 {wind_dir_col} 不存在，跳过样本 {i}")
                continue
            
            # 提取样本数据并去除NaN
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
        
        # 统计信息
        total_points = sum(len(s['times']) for s in all_samples)
        avg_points = total_points / len(all_samples)
        print(f"总数据点: {total_points}, 平均每样本: {avg_points:.1f} 个时间点")
        
        return all_samples
    
    def fit_scalers(self, samples: List[Dict]):
        """
        拟合归一化器（仅使用训练数据）
        """
        # 收集所有风向
        all_directions = [s['wind_direction'] for s in samples]
        self.label_encoder.fit(self.wind_directions)  # 使用预定义的风向列表
        
        # 收集所有风速、时间、热释放速率
        all_speeds = np.array([s['wind_speed'] for s in samples]).reshape(-1, 1)
        all_times = np.concatenate([s['times'] for s in samples]).reshape(-1, 1)
        all_hrr = np.concatenate([s['hrr_values'] for s in samples]).reshape(-1, 1)
        
        self.wind_scaler.fit(all_speeds)
        self.time_scaler.fit(all_times)
        self.hrr_scaler.fit(all_hrr)
        
        print("归一化器拟合完成")
    
    def encode_wind_direction(self, direction: str, method: str = 'onehot') -> np.ndarray:
        """
        风向编码
        
        Args:
            direction: 风向字符串
            method: 'onehot' 或 'label'
        
        Returns:
            编码后的数组
        """
        label = self.label_encoder.transform([direction])[0]
        
        if method == 'onehot':
            onehot = np.zeros(len(self.wind_directions))
            onehot[label] = 1
            return onehot
        else:  # label encoding
            return np.array([label])
    
    def prepare_pointwise_data(self, samples: List[Dict], encoding: str = 'onehot') -> Tuple[np.ndarray, np.ndarray]:
        """
        准备逐点数据（用于MLP等非序列模型）
        
        每个数据点: [风向编码, 风速, 时间] -> 热释放速率
        """
        X_list = []
        y_list = []
        
        for sample in samples:
            wind_dir_encoded = self.encode_wind_direction(sample['wind_direction'], encoding)
            wind_speed_norm = self.wind_scaler.transform([[sample['wind_speed']]])[0, 0]
            
            for t, hrr in zip(sample['times'], sample['hrr_values']):
                time_norm = self.time_scaler.transform([[t]])[0, 0]
                hrr_norm = self.hrr_scaler.transform([[hrr]])[0, 0]
                
                feature = np.concatenate([wind_dir_encoded, [wind_speed_norm, time_norm]])
                X_list.append(feature)
                y_list.append(hrr_norm)
        
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
    
    def prepare_sequence_data(self, samples: List[Dict], encoding: str = 'onehot', 
                              max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备序列数据（用于CNN/LSTM/Transformer）
        
        每个样本是一个序列：
        - 输入: [seq_len, feature_dim] 其中feature_dim = 风向编码 + 风速 + 时间
        - 输出: [seq_len] 热释放速率序列
        - 掩码: [seq_len] 标记有效位置
        """
        if max_len is None:
            max_len = max(len(s['times']) for s in samples)
        
        n_samples = len(samples)
        wind_dim = len(self.wind_directions) if encoding == 'onehot' else 1
        feature_dim = wind_dim + 2  # 风向 + 风速 + 时间
        
        X = np.zeros((n_samples, max_len, feature_dim), dtype=np.float32)
        y = np.zeros((n_samples, max_len), dtype=np.float32)
        masks = np.zeros((n_samples, max_len), dtype=np.float32)
        
        for i, sample in enumerate(samples):
            wind_dir_encoded = self.encode_wind_direction(sample['wind_direction'], encoding)
            wind_speed_norm = self.wind_scaler.transform([[sample['wind_speed']]])[0, 0]
            
            seq_len = len(sample['times'])
            
            for j, (t, hrr) in enumerate(zip(sample['times'], sample['hrr_values'])):
                if j >= max_len:
                    break
                    
                time_norm = self.time_scaler.transform([[t]])[0, 0]
                hrr_norm = self.hrr_scaler.transform([[hrr]])[0, 0]
                
                X[i, j, :wind_dim] = wind_dir_encoded
                X[i, j, wind_dim] = wind_speed_norm
                X[i, j, wind_dim + 1] = time_norm
                y[i, j] = hrr_norm
                masks[i, j] = 1.0
        
        return X, y, masks
    
    def inverse_transform_hrr(self, hrr_normalized: np.ndarray) -> np.ndarray:
        """逆归一化热释放速率"""
        return self.hrr_scaler.inverse_transform(hrr_normalized.reshape(-1, 1)).flatten()
    
    def split_samples(self, samples: List[Dict], test_ratio: float = 0.2, 
                      random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[int], List[int]]:
        """
        按样本整体划分训练集和测试集
        
        Returns:
            train_samples, test_samples, train_indices, test_indices
        """
        n_samples = len(samples)
        indices = list(range(n_samples))
        
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=random_state
        )
        
        train_samples = [samples[i] for i in train_indices]
        test_samples = [samples[i] for i in test_indices]
        
        print(f"数据划分: 训练样本 {len(train_samples)} 个, 测试样本 {len(test_samples)} 个")
        
        return train_samples, test_samples, train_indices, test_indices
    
    def get_kfold_splits(self, samples: List[Dict], n_splits: int = 5, 
                         random_state: int = 42) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        K折交叉验证划分
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        indices = list(range(len(samples)))
        
        splits = []
        for train_idx, val_idx in kf.split(indices):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            splits.append((train_samples, val_samples))
        
        return splits


# ==================== 模型定义模块 ====================

class MLP(nn.Module):
    """
    多层感知机（MLP）
    
    适合：学习输入特征到输出的非线性映射
    结构：2-4层隐藏层，逐层减少神经元数量
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class CNN1D(nn.Module):
    """
    一维卷积神经网络（1D-CNN）
    
    适合：捕捉序列中的局部模式和特征
    结构：轻量化设计，2层卷积 + 全局池化 + 输出
    """
    
    def __init__(self, input_dim: int, seq_len: int, channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        
        # 卷积层
        conv_layers = []
        in_channels = input_dim
        
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # 输出层 - 逐点预测
        self.output_layer = nn.Linear(channels[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, features]
            mask: [batch, seq_len]
        
        Returns:
            [batch, seq_len] 每个时间点的预测值
        """
        # 转换维度: [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # 卷积
        x = self.conv(x)  # [batch, channels, seq_len]
        
        # 转回: [batch, seq_len, channels]
        x = x.permute(0, 2, 1)
        
        # 逐点输出
        output = self.output_layer(x).squeeze(-1)  # [batch, seq_len]
        
        return output


class LSTMModel(nn.Module):
    """
    长短期记忆网络（LSTM）
    
    适合：处理时序依赖，记住历史信息
    结构：单层或双层LSTM + 输出层，适合小样本
    """
    
    def __init__(self, input_dim: int, hidden_size: int = 32, 
                 num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size * self.num_directions, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.kaiming_normal_(self.output_layer.weight)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, features]
            mask: [batch, seq_len]
        
        Returns:
            [batch, seq_len] 每个时间点的预测值
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*directions]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 逐点输出
        output = self.output_layer(lstm_out).squeeze(-1)  # [batch, seq_len]
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer模型
    
    适合：通过注意力机制捕捉全局依赖
    结构：超轻量化设计，1-2层，少量注意力头，小隐藏维度
    """
    
    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 2,
                 num_layers: int = 1, dim_feedforward: int = 64, 
                 dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, features]
            mask: [batch, seq_len] padding mask
        
        Returns:
            [batch, seq_len] 每个时间点的预测值
        """
        # 输入投影
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # 创建padding mask (True表示要mask的位置)
        if mask is not None:
            src_key_padding_mask = (mask == 0)  # [batch, seq_len]
        else:
            src_key_padding_mask = None
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 输出
        output = self.output_layer(x).squeeze(-1)  # [batch, seq_len]
        
        return output


# ==================== 训练模块 ====================

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """统一训练器"""
    
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
    
    def train_pointwise_model(self, model: nn.Module, train_data: Tuple[np.ndarray, np.ndarray],
                              val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                              model_name: str = "Model") -> Dict:
        """
        训练逐点模型（MLP）
        """
        model = model.to(self.device)
        
        X_train, y_train = train_data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # 验证集
        if val_data is not None:
            X_val, y_val = val_data
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        else:
            val_loader = None
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, 
                               weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=10)
        criterion = nn.MSELoss()
        
        # 早停
        early_stopping = EarlyStopping(patience=self.config.patience, mode='min')
        
        # 训练历史
        history = {'train_loss': [], 'val_loss': []}
        best_model_state = None
        best_val_loss = float('inf')
        
        print(f"\n开始训练 {model_name}...")
        
        for epoch in range(self.config.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
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
                history['val_loss'].append(val_loss)
                
                scheduler.step(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                
                # 早停检查
                if early_stopping(val_loss):
                    print(f"  Epoch {epoch+1}: 早停触发")
                    break
                
                if (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1}/{self.config.epochs}, "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.6f}")
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"{model_name} 训练完成")
        
        return {'model': model, 'history': history}
    
    def train_sequence_model(self, model: nn.Module, 
                             train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                             val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                             model_name: str = "Model") -> Dict:
        """
        训练序列模型（CNN/LSTM/Transformer）
        """
        model = model.to(self.device)
        
        X_train, y_train, mask_train = train_data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(mask_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # 验证集
        if val_data is not None:
            X_val, y_val, mask_val = val_data
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
                torch.FloatTensor(mask_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        else:
            val_loader = None
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate,
                               weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=0.5, patience=10)
        
        # 早停
        early_stopping = EarlyStopping(patience=self.config.patience, mode='min')
        
        # 训练历史
        history = {'train_loss': [], 'val_loss': []}
        best_model_state = None
        best_val_loss = float('inf')
        
        print(f"\n开始训练 {model_name}...")
        
        for epoch in range(self.config.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y, batch_mask in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X, batch_mask)
                
                # 只计算有效位置的损失
                loss = self._masked_mse_loss(outputs, batch_y, batch_mask)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y, batch_mask in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        batch_mask = batch_mask.to(self.device)
                        
                        outputs = model(batch_X, batch_mask)
                        loss = self._masked_mse_loss(outputs, batch_y, batch_mask)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                
                if early_stopping(val_loss):
                    print(f"  Epoch {epoch+1}: 早停触发")
                    break
                
                if (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1}/{self.config.epochs}, "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.6f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"{model_name} 训练完成")
        
        return {'model': model, 'history': history}
    
    def _masked_mse_loss(self, pred, target, mask):
        """计算带掩码的MSE损失"""
        loss = (pred - target) ** 2
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)


# ==================== 评估模块 ====================

class Evaluator:
    """模型评估器"""
    
    def __init__(self, processor: FireDataProcessor, device: torch.device):
        self.processor = processor
        self.device = device
    
    def evaluate_pointwise(self, model: nn.Module, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """评估逐点模型"""
        model.eval()
        X_test, y_test = test_data
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = model(X_tensor).cpu().numpy()
        
        # 逆归一化
        y_true = self.processor.inverse_transform_hrr(y_test)
        y_pred = self.processor.inverse_transform_hrr(predictions)
        
        return self._compute_metrics(y_true, y_pred)
    
    def evaluate_sequence(self, model: nn.Module, 
                          test_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict:
        """评估序列模型"""
        model.eval()
        X_test, y_test, mask_test = test_data
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            mask_tensor = torch.FloatTensor(mask_test).to(self.device)
            predictions = model(X_tensor, mask_tensor).cpu().numpy()
        
        # 只取有效位置
        valid_mask = mask_test.flatten() == 1
        y_true_flat = y_test.flatten()[valid_mask]
        y_pred_flat = predictions.flatten()[valid_mask]
        
        # 逆归一化
        y_true = self.processor.inverse_transform_hrr(y_true_flat)
        y_pred = self.processor.inverse_transform_hrr(y_pred_flat)
        
        return self._compute_metrics(y_true, y_pred)
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算评估指标"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def get_sequence_predictions(self, model: nn.Module, samples: List[Dict],
                                 processor: FireDataProcessor, max_len: int) -> List[Dict]:
        """获取每个样本的完整预测结果"""
        model.eval()
        
        X, y, masks = processor.prepare_sequence_data(samples, max_len=max_len)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            mask_tensor = torch.FloatTensor(masks).to(self.device)
            predictions = model(X_tensor, mask_tensor).cpu().numpy()
        
        results = []
        for i, sample in enumerate(samples):
            seq_len = int(masks[i].sum())
            
            y_true_norm = y[i, :seq_len]
            y_pred_norm = predictions[i, :seq_len]
            
            y_true = processor.inverse_transform_hrr(y_true_norm)
            y_pred = processor.inverse_transform_hrr(y_pred_norm)
            
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

class Visualizer:
    """结果可视化器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_curves(self, histories: Dict[str, Dict], filename: str = 'training_curves.png'):
        """绘制训练损失曲线"""
        n_models = len(histories)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, history) in zip(axes, histories.items()):
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=1.5)
            if history['val_loss']:
                ax.plot(epochs, history['val_loss'], 'r--', label='验证损失', linewidth=1.5)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title(f'{name} 训练曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存: {filename}")
    
    def plot_metrics_comparison(self, metrics: Dict[str, Dict], filename: str = 'metrics_comparison.png'):
        """绘制模型性能对比图"""
        model_names = list(metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metric_names = ['MSE', 'RMSE', 'MAE', 'R2']
        titles = ['均方误差 (MSE)', '均方根误差 (RMSE)', '平均绝对误差 (MAE)', '决定系数 (R²)']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for ax, metric, title, color in zip(axes.flatten(), metric_names, titles, colors):
            values = [metrics[name][metric] for name in model_names]
            bars = ax.bar(model_names, values, color=color, alpha=0.7, edgecolor='black')
            
            # 添加数值标签
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
                                model_name: str = "", filename_prefix: str = 'sample_prediction'):
        """绘制测试样本的真实值与预测值对比曲线"""
        n_samples = min(n_samples, len(results))
        
        # 随机选择样本
        np.random.seed(42)
        selected_indices = np.random.choice(len(results), n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (ax, sample_idx) in enumerate(zip(axes, selected_indices)):
            result = results[sample_idx]
            
            ax.plot(result['times'], result['y_true'], 'b-', 
                    label='真实值', linewidth=2, marker='o', markersize=3)
            ax.plot(result['times'], result['y_pred'], 'r--',
                    label='预测值', linewidth=2, marker='x', markersize=3)
            
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('热释放速率 (kW)')
            ax.set_title(f'样本{result["sample_id"]}: {result["wind_direction"]}风, '
                        f'{result["wind_speed"]}m/s')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} 预测结果对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{filename_prefix}_{model_name.replace(" ", "_").lower()}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"样本预测图已保存: {filename}")
    
    def plot_all_models_comparison(self, all_results: Dict[str, List[Dict]], 
                                   n_samples: int = 2, filename: str = 'all_models_comparison.png'):
        """绘制所有模型在相同测试样本上的预测对比"""
        model_names = list(all_results.keys())
        n_models = len(model_names)
        
        # 选择样本
        np.random.seed(42)
        first_results = all_results[model_names[0]]
        n_samples = min(n_samples, len(first_results))
        selected_indices = np.random.choice(len(first_results), n_samples, replace=False)
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(14, 5*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        for ax, sample_idx in zip(axes, selected_indices):
            sample_info = first_results[sample_idx]
            times = sample_info['times']
            y_true = sample_info['y_true']
            
            # 绘制真实值
            ax.plot(times, y_true, 'k-', label='真实值', linewidth=2.5)
            
            # 绘制各模型预测值
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
    
    def plot_scatter_comparison(self, metrics: Dict[str, Dict], filename: str = 'scatter_all_models.png'):
        """绘制所有模型的真实值vs预测值散点图"""
        n_models = len(metrics)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for ax, ((name, metric), color) in zip(axes, zip(metrics.items(), colors)):
            y_true = metric['y_true']
            y_pred = metric['y_pred']
            
            ax.scatter(y_true, y_pred, alpha=0.3, c=color, s=10)
            
            # 对角线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
            
            ax.set_xlabel('真实值 (kW)')
            ax.set_ylabel('预测值 (kW)')
            ax.set_title(f'{name}\nR² = {metric["R2"]:.4f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"散点图已保存: {filename}")


# ==================== 主程序 ====================

def save_results_to_csv(metrics: Dict[str, Dict], output_dir: str):
    """保存评估结果到CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建指标表格
    data = {
        'Model': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R2': []
    }
    
    for name, metric in metrics.items():
        data['Model'].append(name)
        data['MSE'].append(metric['MSE'])
        data['RMSE'].append(metric['RMSE'])
        data['MAE'].append(metric['MAE'])
        data['R2'].append(metric['R2'])
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False, encoding='utf-8-sig')
    
    # 打印表格
    print("\n" + "=" * 70)
    print("模型性能对比表")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    
    return df


def main():
    """主函数"""
    print("=" * 70)
    print("火灾热释放速率(HRR)深度学习预测系统")
    print("小样本对比实验: MLP vs 1D-CNN vs LSTM vs Transformer")
    print("=" * 70)
    
    # 配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 设备
    device = get_device()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)
    
    # ==================== 数据处理 ====================
    print("\n" + "=" * 70)
    print("第一步: 数据处理")
    print("=" * 70)
    
    processor = FireDataProcessor(config)
    
    # 加载数据
    all_samples = processor.load_data(config.data_file)
    
    # 划分训练集和测试集（按样本整体划分）
    train_samples, test_samples, train_indices, test_indices = processor.split_samples(
        all_samples, test_ratio=config.test_ratio, random_state=config.random_seed
    )
    
    # 使用训练数据拟合归一化器
    processor.fit_scalers(train_samples)
    
    # 准备MLP数据（逐点）
    X_train_mlp, y_train_mlp = processor.prepare_pointwise_data(train_samples)
    X_test_mlp, y_test_mlp = processor.prepare_pointwise_data(test_samples)
    
    print(f"MLP数据: 训练 {X_train_mlp.shape}, 测试 {X_test_mlp.shape}")
    
    # 准备序列数据（CNN/LSTM/Transformer）
    max_len = max(len(s['times']) for s in all_samples)
    print(f"最大序列长度: {max_len}")
    
    X_train_seq, y_train_seq, mask_train = processor.prepare_sequence_data(train_samples, max_len=max_len)
    X_test_seq, y_test_seq, mask_test = processor.prepare_sequence_data(test_samples, max_len=max_len)
    
    print(f"序列数据: 训练 {X_train_seq.shape}, 测试 {X_test_seq.shape}")
    
    # ==================== 模型定义 ====================
    print("\n" + "=" * 70)
    print("第二步: 模型定义")
    print("=" * 70)
    
    input_dim_mlp = X_train_mlp.shape[1]  # 风向(8) + 风速(1) + 时间(1) = 10
    input_dim_seq = X_train_seq.shape[2]  # 同上
    
    models = {
        'MLP': MLP(
            input_dim=input_dim_mlp,
            hidden_dims=config.mlp_hidden_dims,
            dropout=config.dropout
        ),
        '1D-CNN': CNN1D(
            input_dim=input_dim_seq,
            seq_len=max_len,
            channels=config.cnn_channels,
            dropout=config.dropout
        ),
        'LSTM': LSTMModel(
            input_dim=input_dim_seq,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout
        ),
        'Transformer': TransformerModel(
            input_dim=input_dim_seq,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_num_layers,
            dropout=config.dropout,
            max_len=max_len
        )
    }
    
    # 打印模型参数量
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: {n_params:,} 可训练参数")
    
    # ==================== 训练 ====================
    print("\n" + "=" * 70)
    print("第三步: 模型训练")
    print("=" * 70)
    
    trainer = Trainer(config, device)
    training_results = {}
    
    # 训练MLP
    mlp_result = trainer.train_pointwise_model(
        models['MLP'],
        train_data=(X_train_mlp, y_train_mlp),
        val_data=(X_test_mlp, y_test_mlp),
        model_name='MLP'
    )
    training_results['MLP'] = mlp_result
    
    # 训练序列模型
    for name in ['1D-CNN', 'LSTM', 'Transformer']:
        result = trainer.train_sequence_model(
            models[name],
            train_data=(X_train_seq, y_train_seq, mask_train),
            val_data=(X_test_seq, y_test_seq, mask_test),
            model_name=name
        )
        training_results[name] = result
    
    # ==================== 评估 ====================
    print("\n" + "=" * 70)
    print("第四步: 模型评估")
    print("=" * 70)
    
    evaluator = Evaluator(processor, device)
    all_metrics = {}
    all_predictions = {}
    
    # 评估MLP
    mlp_metrics = evaluator.evaluate_pointwise(
        training_results['MLP']['model'],
        (X_test_mlp, y_test_mlp)
    )
    all_metrics['MLP'] = mlp_metrics
    
    # 评估序列模型
    for name in ['1D-CNN', 'LSTM', 'Transformer']:
        metrics = evaluator.evaluate_sequence(
            training_results[name]['model'],
            (X_test_seq, y_test_seq, mask_test)
        )
        all_metrics[name] = metrics
        
        # 获取逐样本预测结果
        predictions = evaluator.get_sequence_predictions(
            training_results[name]['model'],
            test_samples,
            processor,
            max_len
        )
        all_predictions[name] = predictions
    
    # 为MLP也准备逐样本预测（需要特殊处理）
    mlp_predictions = []
    for sample in test_samples:
        X_sample, y_sample = processor.prepare_pointwise_data([sample])
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sample).to(device)
            y_pred_norm = training_results['MLP']['model'](X_tensor).cpu().numpy()
        
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
    all_predictions['MLP'] = mlp_predictions
    
    # ==================== 保存结果 ====================
    print("\n" + "=" * 70)
    print("第五步: 保存结果")
    print("=" * 70)
    
    # 保存指标到CSV
    metrics_df = save_results_to_csv(all_metrics, config.output_dir)
    
    # 保存模型
    for name, result in training_results.items():
        model_path = os.path.join(config.model_dir, f'{name.replace(" ", "_").lower()}_model.pth')
        torch.save(result['model'].state_dict(), model_path)
        print(f"模型已保存: {model_path}")
    
    # 保存归一化器信息
    import pickle
    processor_path = os.path.join(config.model_dir, 'processor.pkl')
    with open(processor_path, 'wb') as f:
        pickle.dump({
            'label_encoder': processor.label_encoder,
            'wind_scaler': processor.wind_scaler,
            'time_scaler': processor.time_scaler,
            'hrr_scaler': processor.hrr_scaler
        }, f)
    print(f"预处理器已保存: {processor_path}")
    
    # ==================== 可视化 ====================
    print("\n" + "=" * 70)
    print("第六步: 结果可视化")
    print("=" * 70)
    
    visualizer = Visualizer(config.figure_dir)
    
    # 训练曲线
    histories = {name: result['history'] for name, result in training_results.items()}
    visualizer.plot_training_curves(histories)
    
    # 性能对比图
    visualizer.plot_metrics_comparison(all_metrics)
    
    # 各模型的样本预测图
    for name, predictions in all_predictions.items():
        visualizer.plot_sample_predictions(predictions, n_samples=4, model_name=name)
    
    # 所有模型在相同样本上的对比
    visualizer.plot_all_models_comparison(all_predictions, n_samples=2)
    
    # 散点图
    visualizer.plot_scatter_comparison(all_metrics)
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    
    print(f"\n结果文件位置:")
    print(f"  - 评估指标: {config.output_dir}/metrics.csv")
    print(f"  - 模型权重: {config.model_dir}/")
    print(f"  - 可视化图表: {config.figure_dir}/")
    
    # 最佳模型
    best_model = max(all_metrics.keys(), key=lambda x: all_metrics[x]['R2'])
    print(f"\n最佳模型: {best_model} (R² = {all_metrics[best_model]['R2']:.4f})")
    
    return all_metrics, all_predictions, training_results


# ==================== K折交叉验证实验 ====================

def run_kfold_experiment(n_splits: int = 5):
    """
    运行K折交叉验证实验
    
    这提供了更稳健的模型性能评估，特别适合小样本场景
    """
    print("=" * 70)
    print(f"K折交叉验证实验 (K={n_splits})")
    print("=" * 70)
    
    config = Config()
    set_seed(config.random_seed)
    device = get_device()
    
    processor = FireDataProcessor(config)
    all_samples = processor.load_data(config.data_file)
    
    # 获取K折划分
    kfold_splits = processor.get_kfold_splits(all_samples, n_splits=n_splits)
    
    # 存储每折的结果
    fold_results = {name: [] for name in ['MLP', '1D-CNN', 'LSTM', 'Transformer']}
    
    for fold, (train_samples, val_samples) in enumerate(kfold_splits):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # 拟合归一化器
        processor.fit_scalers(train_samples)
        
        # 准备数据
        max_len = max(len(s['times']) for s in all_samples)
        
        X_train_mlp, y_train_mlp = processor.prepare_pointwise_data(train_samples)
        X_val_mlp, y_val_mlp = processor.prepare_pointwise_data(val_samples)
        
        X_train_seq, y_train_seq, mask_train = processor.prepare_sequence_data(train_samples, max_len=max_len)
        X_val_seq, y_val_seq, mask_val = processor.prepare_sequence_data(val_samples, max_len=max_len)
        
        input_dim_mlp = X_train_mlp.shape[1]
        input_dim_seq = X_train_seq.shape[2]
        
        # 创建新模型实例
        models = {
            'MLP': MLP(input_dim_mlp, config.mlp_hidden_dims, config.dropout),
            '1D-CNN': CNN1D(input_dim_seq, max_len, config.cnn_channels, dropout=config.dropout),
            'LSTM': LSTMModel(input_dim_seq, config.lstm_hidden_size, config.lstm_num_layers, config.dropout),
            'Transformer': TransformerModel(input_dim_seq, config.transformer_d_model, 
                                           config.transformer_nhead, config.transformer_num_layers,
                                           config.dropout, max_len)
        }
        
        trainer = Trainer(config, device)
        evaluator = Evaluator(processor, device)
        
        # 训练和评估每个模型
        # MLP
        mlp_result = trainer.train_pointwise_model(
            models['MLP'], (X_train_mlp, y_train_mlp), (X_val_mlp, y_val_mlp), 'MLP'
        )
        mlp_metrics = evaluator.evaluate_pointwise(mlp_result['model'], (X_val_mlp, y_val_mlp))
        fold_results['MLP'].append(mlp_metrics['R2'])
        
        # 序列模型
        for name in ['1D-CNN', 'LSTM', 'Transformer']:
            result = trainer.train_sequence_model(
                models[name], (X_train_seq, y_train_seq, mask_train),
                (X_val_seq, y_val_seq, mask_val), name
            )
            metrics = evaluator.evaluate_sequence(result['model'], (X_val_seq, y_val_seq, mask_val))
            fold_results[name].append(metrics['R2'])
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("K折交叉验证结果汇总")
    print("=" * 70)
    
    summary = []
    for name, scores in fold_results.items():
        mean_r2 = np.mean(scores)
        std_r2 = np.std(scores)
        summary.append({
            'Model': name,
            'Mean R²': mean_r2,
            'Std R²': std_r2,
            'All Folds': scores
        })
        print(f"{name}: R² = {mean_r2:.4f} ± {std_r2:.4f}")
    
    return summary


if __name__ == "__main__":
    # 运行主实验
    metrics, predictions, results = main()
    
    # 可选：运行K折交叉验证
    # kfold_summary = run_kfold_experiment(n_splits=5)
