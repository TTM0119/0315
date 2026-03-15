"""
================================================================================
火灾热释放速率(HRR)深度学习模型评估与可视化系统
================================================================================

功能：
1. 加载已训练的深度学习模型
2. 对所有64个样本（训练集+测试集）进行预测
3. 生成多维度评价指标和可视化图表
4. 保存预测数据到本地表格
5. 生成Parity Plot（预测值 vs 实验值散点图）
6. 汇总所有评价指标到txt文件

================================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime

import torch
import torch.nn as nn

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 从improved文件导入模型定义 ====================
# 需要重新定义模型类以便加载权重

import math
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler, LabelEncoder


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
    """残差MLP"""
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
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


class CNNBiLSTM(nn.Module):
    """CNN + 双向LSTM"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
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
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.output(x).squeeze(-1)


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
    """LSTM + 自注意力"""
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
    """改进的Transformer"""
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3, max_len: int = 1000):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = RelativePositionalEncoding(d_model, max_len)
        self.input_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.output(x).squeeze(-1)


# ==================== 数据处理类 ====================
class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.wind_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self.hrr_scaler = RobustScaler()
        self.wind_directions = ['东', '南', '西', '北', '东北', '东南', '西南', '西北']
        self.direction_angles = {
            '东': 0, '东北': 45, '北': 90, '西北': 135,
            '西': 180, '西南': 225, '南': 270, '东南': 315
        }
        self.max_time = None
        
    def load_data(self, file_path: str) -> List[Dict]:
        """加载数据"""
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
        return all_samples
    
    def encode_wind_direction_cyclic(self, direction: str) -> np.ndarray:
        """风向周期性编码"""
        angle_deg = self.direction_angles.get(direction, 0)
        angle_rad = math.radians(angle_deg)
        return np.array([math.sin(angle_rad), math.cos(angle_rad)])
    
    def create_time_features(self, time: float, max_time: float) -> np.ndarray:
        """创建时间特征"""
        t_norm = time / max_time if max_time > 0 else 0
        features = [
            t_norm,
            t_norm ** 2,
            math.sin(2 * math.pi * t_norm),
            math.cos(2 * math.pi * t_norm),
        ]
        return np.array(features)
    
    def fit_scalers(self, samples: List[Dict]):
        """拟合归一化器"""
        all_speeds = np.array([s['wind_speed'] for s in samples]).reshape(-1, 1)
        all_times = np.concatenate([s['times'] for s in samples]).reshape(-1, 1)
        all_hrr = np.concatenate([s['hrr_values'] for s in samples]).reshape(-1, 1)
        
        self.wind_scaler.fit(all_speeds)
        self.time_scaler.fit(all_times)
        self.hrr_scaler.fit(all_hrr)
        self.max_time = all_times.max()
    
    def prepare_pointwise_data(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """准备逐点数据"""
        X_list = []
        y_list = []
        
        for sample in samples:
            wind_dir_cyclic = self.encode_wind_direction_cyclic(sample['wind_direction'])
            wind_speed_norm = self.wind_scaler.transform([[sample['wind_speed']]])[0, 0]
            
            for t, hrr in zip(sample['times'], sample['hrr_values']):
                time_features = self.create_time_features(t, self.max_time)
                hrr_norm = self.hrr_scaler.transform([[hrr]])[0, 0]
                interaction = wind_speed_norm * time_features[0]
                
                feature = np.concatenate([
                    wind_dir_cyclic,
                    [wind_speed_norm],
                    time_features,
                    [interaction]
                ])
                
                X_list.append(feature)
                y_list.append(hrr_norm)
        
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
    
    def prepare_sequence_data(self, samples: List[Dict], max_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备序列数据"""
        n_samples = len(samples)
        feature_dim = 8
        
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


# ==================== 评估与可视化系统 ====================
class ModelEvaluator:
    """模型评估与可视化系统"""
    
    def __init__(self, output_dir: str = 'evaluation_results'):
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录结构
        self.dirs = {
            'root': output_dir,
            'figures': os.path.join(output_dir, 'figures'),
            'sample_plots': os.path.join(output_dir, 'sample_plots'),
            'parity_plots': os.path.join(output_dir, 'parity_plots'),
            'data': os.path.join(output_dir, 'data'),
            'reports': os.path.join(output_dir, 'reports')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"评估结果将保存到: {output_dir}")
    
    def load_models(self, model_dir: str, max_len: int) -> Dict[str, nn.Module]:
        """加载已训练的模型"""
        print("\n正在加载模型...")
        
        models = {}
        
        # ResidualMLP
        model_path = os.path.join(model_dir, 'residualmlp_model_v2.pth')
        if os.path.exists(model_path):
            model = ResidualMLP(input_dim=8, hidden_dim=128, num_blocks=3, dropout=0.3)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            models['ResidualMLP'] = model
            print(f"  已加载: ResidualMLP")
        
        # CNN-BiLSTM
        model_path = os.path.join(model_dir, 'cnn-bilstm_model_v2.pth')
        if os.path.exists(model_path):
            model = CNNBiLSTM(input_dim=8, hidden_dim=64, num_layers=2, dropout=0.3)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            models['CNN-BiLSTM'] = model
            print(f"  已加载: CNN-BiLSTM")
        
        # AttentionLSTM
        model_path = os.path.join(model_dir, 'attentionlstm_model_v2.pth')
        if os.path.exists(model_path):
            model = AttentionLSTM(input_dim=8, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.3)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            models['AttentionLSTM'] = model
            print(f"  已加载: AttentionLSTM")
        
        # ImprovedTransformer
        model_path = os.path.join(model_dir, 'improvedtransformer_model_v2.pth')
        if os.path.exists(model_path):
            model = ImprovedTransformer(input_dim=8, d_model=64, nhead=4, num_layers=2, dropout=0.3, max_len=max_len)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            models['ImprovedTransformer'] = model
            print(f"  已加载: ImprovedTransformer")
        
        print(f"共加载 {len(models)} 个模型")
        return models
    
    def predict_sample(self, model: nn.Module, sample: Dict, processor: DataProcessor, 
                       max_len: int, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """对单个样本进行预测"""
        model.eval()
        
        if model_type == 'ResidualMLP':
            # 逐点预测
            X, y_norm = processor.prepare_pointwise_data([sample])
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                pred_norm = model(X_tensor).cpu().numpy()
            y_true = processor.inverse_transform_hrr(y_norm)
            y_pred = processor.inverse_transform_hrr(pred_norm)
        else:
            # 序列预测
            X, y_norm, masks = processor.prepare_sequence_data([sample], max_len)
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                mask_tensor = torch.FloatTensor(masks).to(self.device)
                pred_norm = model(X_tensor, mask_tensor).cpu().numpy()
            
            seq_len = int(masks[0].sum())
            y_true = processor.inverse_transform_hrr(y_norm[0, :seq_len])
            y_pred = processor.inverse_transform_hrr(pred_norm[0, :seq_len])
        
        return y_true, y_pred
    
    def evaluate_all_samples(self, models: Dict[str, nn.Module], all_samples: List[Dict],
                             processor: DataProcessor, train_indices: List[int], 
                             test_indices: List[int], max_len: int) -> Dict:
        """评估所有样本"""
        print("\n开始评估所有样本...")
        
        results = {name: {'samples': [], 'all_true': [], 'all_pred': []} for name in models.keys()}
        
        for sample_idx, sample in enumerate(all_samples):
            is_training = sample_idx in train_indices
            sample_type = "训练集" if is_training else "测试集"
            
            for model_name, model in models.items():
                model_type = 'ResidualMLP' if model_name == 'ResidualMLP' else 'sequence'
                y_true, y_pred = self.predict_sample(model, sample, processor, max_len, model_type)
                
                # 计算该样本的指标
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                # MAPE (避免除零)
                mask = y_true != 0
                if mask.sum() > 0:
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                else:
                    mape = np.nan
                
                sample_result = {
                    'sample_id': sample['sample_id'],
                    'wind_direction': sample['wind_direction'],
                    'wind_speed': sample['wind_speed'],
                    'is_training': is_training,
                    'sample_type': sample_type,
                    'times': sample['times'][:len(y_true)],
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape
                }
                
                results[model_name]['samples'].append(sample_result)
                results[model_name]['all_true'].extend(y_true.tolist())
                results[model_name]['all_pred'].extend(y_pred.tolist())
            
            print(f"  已评估样本 {sample['sample_id']:2d} ({sample_type})")
        
        return results
    
    def compute_overall_metrics(self, results: Dict) -> Dict:
        """计算总体指标"""
        overall_metrics = {}
        
        for model_name, model_results in results.items():
            y_true_all = np.array(model_results['all_true'])
            y_pred_all = np.array(model_results['all_pred'])
            
            # 分离训练集和测试集
            train_true, train_pred = [], []
            test_true, test_pred = [], []
            
            for sample in model_results['samples']:
                if sample['is_training']:
                    train_true.extend(sample['y_true'].tolist())
                    train_pred.extend(sample['y_pred'].tolist())
                else:
                    test_true.extend(sample['y_true'].tolist())
                    test_pred.extend(sample['y_pred'].tolist())
            
            train_true, train_pred = np.array(train_true), np.array(train_pred)
            test_true, test_pred = np.array(test_true), np.array(test_pred)
            
            def calc_metrics(y_true, y_pred, prefix=''):
                if len(y_true) == 0:
                    return {}
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                mask = y_true != 0
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
                return {
                    f'{prefix}MSE': mse,
                    f'{prefix}RMSE': rmse,
                    f'{prefix}MAE': mae,
                    f'{prefix}R2': r2,
                    f'{prefix}MAPE': mape
                }
            
            metrics = {}
            metrics.update(calc_metrics(y_true_all, y_pred_all, 'Overall_'))
            metrics.update(calc_metrics(train_true, train_pred, 'Train_'))
            metrics.update(calc_metrics(test_true, test_pred, 'Test_'))
            
            overall_metrics[model_name] = metrics
        
        return overall_metrics
    
    def plot_sample_comparison(self, results: Dict):
        """绘制每个样本的预测对比图"""
        print("\n生成样本预测对比图...")
        
        for model_name, model_results in results.items():
            model_dir = os.path.join(self.dirs['sample_plots'], model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            for sample_result in model_results['samples']:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                times = sample_result['times']
                y_true = sample_result['y_true']
                y_pred = sample_result['y_pred']
                
                ax.plot(times, y_true, 'b-', label='真实值', linewidth=2)
                ax.plot(times, y_pred, 'r--', label='预测值', linewidth=2)
                
                ax.set_xlabel('时间 (s)', fontsize=12)
                ax.set_ylabel('热释放速率 (kW)', fontsize=12)
                
                title = f'{model_name} - 样本 {sample_result["sample_id"]} ({sample_result["sample_type"]})\n'
                title += f'{sample_result["wind_direction"]}风, {sample_result["wind_speed"]}m/s\n'
                title += f'$R^2$ = {sample_result["R2"]:.4f}, RMSE = {sample_result["RMSE"]:.2f}, MAE = {sample_result["MAE"]:.2f}'
                ax.set_title(title, fontsize=11)
                
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # 填充误差区域
                ax.fill_between(times, y_true, y_pred, alpha=0.2, color='gray', label='误差区域')
                
                filename = f'sample_{sample_result["sample_id"]:02d}_{sample_result["sample_type"]}.png'
                filepath = os.path.join(model_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"  样本对比图已保存到: {self.dirs['sample_plots']}")
    
    def plot_parity_plots(self, results: Dict, overall_metrics: Dict):
        """绘制Parity Plot（预测值 vs 实验值散点图）"""
        print("\n生成Parity Plot...")
        
        # 1. 每个模型单独的Parity Plot
        for model_name, model_results in results.items():
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # 收集数据
            train_true, train_pred = [], []
            test_true, test_pred = [], []
            
            for sample in model_results['samples']:
                if sample['is_training']:
                    train_true.extend(sample['y_true'].tolist())
                    train_pred.extend(sample['y_pred'].tolist())
                else:
                    test_true.extend(sample['y_true'].tolist())
                    test_pred.extend(sample['y_pred'].tolist())
            
            all_true = np.array(model_results['all_true'])
            all_pred = np.array(model_results['all_pred'])
            train_true, train_pred = np.array(train_true), np.array(train_pred)
            test_true, test_pred = np.array(test_true), np.array(test_pred)
            
            metrics = overall_metrics[model_name]
            
            # 训练集
            ax = axes[0]
            ax.scatter(train_true, train_pred, alpha=0.3, s=5, c='blue', label='训练集')
            min_val, max_val = min(train_true.min(), train_pred.min()), max(train_true.max(), train_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='理想线')
            ax.set_xlabel('实验值 (kW)', fontsize=11)
            ax.set_ylabel('预测值 (kW)', fontsize=11)
            ax.set_title(f'{model_name} - 训练集\n$R^2$ = {metrics["Train_R2"]:.4f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # 测试集
            ax = axes[1]
            ax.scatter(test_true, test_pred, alpha=0.3, s=5, c='red', label='测试集')
            min_val, max_val = min(test_true.min(), test_pred.min()), max(test_true.max(), test_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='理想线')
            ax.set_xlabel('实验值 (kW)', fontsize=11)
            ax.set_ylabel('预测值 (kW)', fontsize=11)
            ax.set_title(f'{model_name} - 测试集\n$R^2$ = {metrics["Test_R2"]:.4f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # 全部数据
            ax = axes[2]
            ax.scatter(train_true, train_pred, alpha=0.3, s=5, c='blue', label='训练集')
            ax.scatter(test_true, test_pred, alpha=0.3, s=5, c='red', label='测试集')
            min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='理想线')
            ax.set_xlabel('实验值 (kW)', fontsize=11)
            ax.set_ylabel('预测值 (kW)', fontsize=11)
            ax.set_title(f'{model_name} - 全部数据\n$R^2$ = {metrics["Overall_R2"]:.4f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            filepath = os.path.join(self.dirs['parity_plots'], f'parity_{model_name}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 所有模型对比的Parity Plot
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for idx, (model_name, model_results) in enumerate(results.items()):
            ax = axes[idx]
            
            # 分离训练集和测试集
            for sample in model_results['samples']:
                color = 'blue' if sample['is_training'] else 'red'
                alpha = 0.2 if sample['is_training'] else 0.4
                ax.scatter(sample['y_true'], sample['y_pred'], alpha=alpha, s=3, c=color)
            
            all_true = np.array(model_results['all_true'])
            all_pred = np.array(model_results['all_pred'])
            
            min_val, max_val = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
            
            metrics = overall_metrics[model_name]
            ax.set_xlabel('实验值 (kW)', fontsize=10)
            ax.set_ylabel('预测值 (kW)', fontsize=10)
            ax.set_title(f'{model_name}\n$R^2$ = {metrics["Overall_R2"]:.4f}', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.dirs['parity_plots'], 'parity_all_models.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Parity Plot已保存到: {self.dirs['parity_plots']}")
    
    def plot_metrics_comparison(self, overall_metrics: Dict):
        """绘制指标对比图"""
        print("\n生成指标对比图...")
        
        model_names = list(overall_metrics.keys())
        
        # 1. 测试集指标对比
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metric_configs = [
            ('Test_MSE', '均方误差 (MSE)', '#3498db'),
            ('Test_RMSE', '均方根误差 (RMSE)', '#2ecc71'),
            ('Test_MAE', '平均绝对误差 (MAE)', '#e74c3c'),
            ('Test_R2', '决定系数 ($R^2$)', '#9b59b6'),
            ('Test_MAPE', '平均绝对百分比误差 (MAPE %)', '#f39c12'),
        ]
        
        for idx, (metric_key, title, color) in enumerate(metric_configs):
            ax = axes.flatten()[idx]
            values = [overall_metrics[name].get(metric_key, 0) for name in model_names]
            
            bars = ax.bar(model_names, values, color=color, alpha=0.7, edgecolor='black')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'测试集 - {title}', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_key.split('_')[1])
            ax.tick_params(axis='x', rotation=15)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 隐藏多余的子图
        axes.flatten()[5].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.dirs['figures'], 'metrics_test_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 训练集 vs 测试集 R² 对比
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        train_r2 = [overall_metrics[name]['Train_R2'] for name in model_names]
        test_r2 = [overall_metrics[name]['Test_R2'] for name in model_names]
        
        bars1 = ax.bar(x - width/2, train_r2, width, label='训练集', color='#3498db', alpha=0.7)
        bars2 = ax.bar(x + width/2, test_r2, width, label='测试集', color='#e74c3c', alpha=0.7)
        
        for bar, val in zip(bars1, train_r2):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, test_r2):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('$R^2$', fontsize=12)
        ax.set_title('训练集 vs 测试集 $R^2$ 对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.dirs['figures'], 'r2_train_test_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  指标对比图已保存到: {self.dirs['figures']}")
    
    def plot_residual_analysis(self, results: Dict):
        """绘制残差分析图"""
        print("\n生成残差分析图...")
        
        for model_name, model_results in results.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            all_true = np.array(model_results['all_true'])
            all_pred = np.array(model_results['all_pred'])
            residuals = all_true - all_pred
            
            # 1. 残差散点图
            ax = axes[0, 0]
            ax.scatter(all_pred, residuals, alpha=0.3, s=5, c='blue')
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
            ax.set_xlabel('预测值 (kW)')
            ax.set_ylabel('残差 (kW)')
            ax.set_title('残差 vs 预测值')
            ax.grid(True, alpha=0.3)
            
            # 2. 残差分布直方图
            ax = axes[0, 1]
            ax.hist(residuals, bins=50, color='blue', alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
            ax.set_xlabel('残差 (kW)')
            ax.set_ylabel('频次')
            ax.set_title(f'残差分布 (均值={residuals.mean():.2f}, 标准差={residuals.std():.2f})')
            ax.grid(True, alpha=0.3)
            
            # 3. Q-Q图（简化版）
            ax = axes[1, 0]
            sorted_residuals = np.sort(residuals)
            n = len(sorted_residuals)
            theoretical_quantiles = np.linspace(0.001, 0.999, n)
            from scipy import stats
            theoretical_values = stats.norm.ppf(theoretical_quantiles) * residuals.std() + residuals.mean()
            ax.scatter(theoretical_values, sorted_residuals, alpha=0.3, s=5, c='blue')
            min_val = min(theoretical_values.min(), sorted_residuals.min())
            max_val = max(theoretical_values.max(), sorted_residuals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
            ax.set_xlabel('理论分位数')
            ax.set_ylabel('样本分位数')
            ax.set_title('Q-Q图')
            ax.grid(True, alpha=0.3)
            
            # 4. 残差百分比分布
            ax = axes[1, 1]
            mask = all_true != 0
            residual_pct = (residuals[mask] / all_true[mask]) * 100
            ax.hist(residual_pct, bins=50, color='green', alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
            ax.set_xlabel('残差百分比 (%)')
            ax.set_ylabel('频次')
            ax.set_title(f'残差百分比分布')
            ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{model_name} - 残差分析', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filepath = os.path.join(self.dirs['figures'], f'residual_analysis_{model_name}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  残差分析图已保存到: {self.dirs['figures']}")
    
    def save_predictions_to_excel(self, results: Dict):
        """保存预测数据到Excel - 每个样本单独分列"""
        print("\n保存预测数据到Excel...")
        
        for model_name, model_results in results.items():
            # 获取最大时间点数量
            max_points = max(len(sample['times']) for sample in model_results['samples'])
            
            # 按样本分列创建数据
            # 每个样本占5列: 时间、真实值、预测值、残差、相对误差
            columns_data = {}
            
            for sample in model_results['samples']:
                sample_id = sample['sample_id']
                prefix = f"样本{sample_id}_{sample['wind_direction']}_{sample['wind_speed']}m/s"
                
                # 初始化列数据
                times = list(sample['times']) + [np.nan] * (max_points - len(sample['times']))
                y_true = list(sample['y_true']) + [np.nan] * (max_points - len(sample['y_true']))
                y_pred = list(sample['y_pred']) + [np.nan] * (max_points - len(sample['y_pred']))
                residuals = []
                rel_errors = []
                
                for t_val, p_val in zip(sample['y_true'], sample['y_pred']):
                    residuals.append(t_val - p_val)
                    rel_errors.append(((t_val - p_val) / t_val * 100) if t_val != 0 else np.nan)
                
                residuals += [np.nan] * (max_points - len(residuals))
                rel_errors += [np.nan] * (max_points - len(rel_errors))
                
                columns_data[f'{prefix}_时间(s)'] = times
                columns_data[f'{prefix}_真实值(kW)'] = y_true
                columns_data[f'{prefix}_预测值(kW)'] = y_pred
                columns_data[f'{prefix}_残差(kW)'] = residuals
                columns_data[f'{prefix}_相对误差(%)'] = rel_errors
            
            df = pd.DataFrame(columns_data)
            filepath = os.path.join(self.dirs['data'], f'predictions_{model_name}.xlsx')
            df.to_excel(filepath, index=False)
            
            # 创建样本汇总表
            summary_data = []
            for sample in model_results['samples']:
                summary_data.append({
                    '样本ID': sample['sample_id'],
                    '风向': sample['wind_direction'],
                    '风速(m/s)': sample['wind_speed'],
                    '数据类型': sample['sample_type'],
                    '数据点数': len(sample['y_true']),
                    'MSE': sample['MSE'],
                    'RMSE': sample['RMSE'],
                    'MAE': sample['MAE'],
                    'R2': sample['R2'],
                    'MAPE(%)': sample['MAPE']
                })
            
            df_summary = pd.DataFrame(summary_data)
            filepath = os.path.join(self.dirs['data'], f'sample_summary_{model_name}.xlsx')
            df_summary.to_excel(filepath, index=False)
        
        print(f"  预测数据已保存到: {self.dirs['data']}")
    
    def save_metrics_report(self, results: Dict, overall_metrics: Dict):
        """保存评价指标汇总报告"""
        print("\n生成评价指标汇总报告...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
================================================================================
火灾热释放速率(HRR)深度学习模型评估报告
================================================================================
生成时间: {timestamp}
================================================================================

一、总体评估指标汇总
--------------------------------------------------------------------------------
"""
        
        # 总体指标表格
        report += "\n1.1 测试集指标\n"
        report += "-" * 80 + "\n"
        report += f"{'模型名称':<20} {'MSE':>15} {'RMSE':>12} {'MAE':>12} {'R²':>10} {'MAPE(%)':>10}\n"
        report += "-" * 80 + "\n"
        
        for model_name, metrics in overall_metrics.items():
            report += f"{model_name:<20} {metrics['Test_MSE']:>15.2f} {metrics['Test_RMSE']:>12.2f} "
            report += f"{metrics['Test_MAE']:>12.2f} {metrics['Test_R2']:>10.4f} {metrics['Test_MAPE']:>10.2f}\n"
        
        report += "-" * 80 + "\n"
        
        report += "\n1.2 训练集指标\n"
        report += "-" * 80 + "\n"
        report += f"{'模型名称':<20} {'MSE':>15} {'RMSE':>12} {'MAE':>12} {'R²':>10} {'MAPE(%)':>10}\n"
        report += "-" * 80 + "\n"
        
        for model_name, metrics in overall_metrics.items():
            report += f"{model_name:<20} {metrics['Train_MSE']:>15.2f} {metrics['Train_RMSE']:>12.2f} "
            report += f"{metrics['Train_MAE']:>12.2f} {metrics['Train_R2']:>10.4f} {metrics['Train_MAPE']:>10.2f}\n"
        
        report += "-" * 80 + "\n"
        
        report += "\n1.3 全部数据指标\n"
        report += "-" * 80 + "\n"
        report += f"{'模型名称':<20} {'MSE':>15} {'RMSE':>12} {'MAE':>12} {'R²':>10} {'MAPE(%)':>10}\n"
        report += "-" * 80 + "\n"
        
        for model_name, metrics in overall_metrics.items():
            report += f"{model_name:<20} {metrics['Overall_MSE']:>15.2f} {metrics['Overall_RMSE']:>12.2f} "
            report += f"{metrics['Overall_MAE']:>12.2f} {metrics['Overall_R2']:>10.4f} {metrics['Overall_MAPE']:>10.2f}\n"
        
        report += "-" * 80 + "\n"
        
        # 最佳模型
        best_model = max(overall_metrics.keys(), key=lambda x: overall_metrics[x]['Test_R2'])
        report += f"\n最佳模型(按测试集R²): {best_model} (R² = {overall_metrics[best_model]['Test_R2']:.4f})\n"
        
        # 各样本详细指标
        report += """
================================================================================

二、各样本详细评估指标
--------------------------------------------------------------------------------
"""
        
        for model_name, model_results in results.items():
            report += f"\n{model_name}\n"
            report += "=" * 80 + "\n"
            report += f"{'样本ID':>6} {'风向':>6} {'风速':>6} {'类型':>8} {'MSE':>12} {'RMSE':>10} {'MAE':>10} {'R²':>8}\n"
            report += "-" * 80 + "\n"
            
            for sample in model_results['samples']:
                report += f"{sample['sample_id']:>6} {sample['wind_direction']:>6} "
                report += f"{sample['wind_speed']:>6.1f} {sample['sample_type']:>8} "
                report += f"{sample['MSE']:>12.2f} {sample['RMSE']:>10.2f} "
                report += f"{sample['MAE']:>10.2f} {sample['R2']:>8.4f}\n"
            
            # 分组统计
            train_samples = [s for s in model_results['samples'] if s['is_training']]
            test_samples = [s for s in model_results['samples'] if not s['is_training']]
            
            report += "-" * 80 + "\n"
            if train_samples:
                avg_r2_train = np.mean([s['R2'] for s in train_samples])
                avg_rmse_train = np.mean([s['RMSE'] for s in train_samples])
                report += f"训练集平均 ({len(train_samples)}个样本): R² = {avg_r2_train:.4f}, RMSE = {avg_rmse_train:.2f}\n"
            if test_samples:
                avg_r2_test = np.mean([s['R2'] for s in test_samples])
                avg_rmse_test = np.mean([s['RMSE'] for s in test_samples])
                report += f"测试集平均 ({len(test_samples)}个样本): R² = {avg_r2_test:.4f}, RMSE = {avg_rmse_test:.2f}\n"
            
            report += "\n"
        
        report += """
================================================================================

三、评价指标说明
--------------------------------------------------------------------------------
MSE  (均方误差): 预测值与真实值差值的平方的平均值，对大误差敏感
RMSE (均方根误差): MSE的平方根，与原始数据单位一致
MAE  (平均绝对误差): 预测值与真实值差值的绝对值的平均值
R²   (决定系数): 模型解释方差的比例，越接近1越好
MAPE (平均绝对百分比误差): 相对误差的平均值，以百分比表示

================================================================================
报告结束
================================================================================
"""
        
        # 保存报告
        filepath = os.path.join(self.dirs['reports'], 'evaluation_report.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  评估报告已保存到: {filepath}")
        
        # 同时保存为CSV格式的汇总表
        summary_data = []
        for model_name, metrics in overall_metrics.items():
            row = {'模型': model_name}
            for key, value in metrics.items():
                row[key] = value
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        csv_filepath = os.path.join(self.dirs['reports'], 'metrics_summary.csv')
        df_summary.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        
        print(f"  指标汇总CSV已保存到: {csv_filepath}")


def main():
    """主函数"""
    print("=" * 70)
    print("火灾热释放速率(HRR)深度学习模型评估与可视化系统")
    print("=" * 70)
    
    # 配置
    data_file = 'data.xlsx'
    model_dir = 'saved_models_v2'
    output_dir = 'evaluation_results_v2'
    random_seed = 42
    
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 '{model_dir}' 不存在！")
        print("请先运行 fire_hrr_deep_learning_improved.py 训练模型。")
        return
    
    # 初始化评估器
    evaluator = ModelEvaluator(output_dir)
    
    # 加载数据
    processor = DataProcessor()
    all_samples = processor.load_data(data_file)
    
    # 划分数据集（使用相同的随机种子）
    indices = list(range(len(all_samples)))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=random_seed
    )
    
    train_samples = [all_samples[i] for i in train_indices]
    
    print(f"训练样本: {len(train_indices)} 个, 测试样本: {len(test_indices)} 个")
    
    # 拟合归一化器
    processor.fit_scalers(train_samples)
    
    # 获取最大序列长度
    max_len = max(len(s['times']) for s in all_samples)
    
    # 加载模型
    models = evaluator.load_models(model_dir, max_len)
    
    if not models:
        print("错误: 未能加载任何模型！")
        return
    
    # 评估所有样本
    results = evaluator.evaluate_all_samples(
        models, all_samples, processor, train_indices, test_indices, max_len
    )
    
    # 计算总体指标
    overall_metrics = evaluator.compute_overall_metrics(results)
    
    # 生成可视化
    evaluator.plot_sample_comparison(results)
    evaluator.plot_parity_plots(results, overall_metrics)
    evaluator.plot_metrics_comparison(overall_metrics)
    evaluator.plot_residual_analysis(results)
    
    # 保存数据
    evaluator.save_predictions_to_excel(results)
    evaluator.save_metrics_report(results, overall_metrics)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)
    
    print(f"\n所有结果已保存到: {output_dir}/")
    print(f"  - figures/: 指标对比图、残差分析图")
    print(f"  - sample_plots/: 各模型的样本预测对比图")
    print(f"  - parity_plots/: Parity Plot (预测值 vs 实验值)")
    print(f"  - data/: 预测数据Excel表格")
    print(f"  - reports/: 评估报告和指标汇总")
    
    print("\n" + "-" * 70)
    print("模型性能汇总 (测试集):")
    print("-" * 70)
    for model_name, metrics in overall_metrics.items():
        print(f"{model_name}: R² = {metrics['Test_R2']:.4f}, RMSE = {metrics['Test_RMSE']:.2f}")
    
    best_model = max(overall_metrics.keys(), key=lambda x: overall_metrics[x]['Test_R2'])
    print(f"\n最佳模型: {best_model}")


if __name__ == "__main__":
    main()
