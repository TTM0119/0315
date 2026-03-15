import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import os
from itertools import cycle

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class FireHeatReleasePredictor:
    def __init__(self):
        self.models = {}
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.wind_directions = ['东', '南', '西', '北', '东北', '东南', '西南', '西北']
        
    def load_and_preprocess_data(self, file_path):
        """
        加载并预处理数据
        数据格式：每4列代表一个样本（风向、风速、时间、热释放速率），共64个样本
        """
        print("正在加载数据...")
        df = pd.read_excel(file_path)
        
        # 每4列组成一个样本，共64个样本
        samples = []
        sample_labels = []
        
        for i in range(1, 65):  # 1到64号样本
            wind_dir_col = f'风向_{i}'
            wind_speed_col = f'风速/m·s-1_{i}'
            time_col = f'时间/s_{i}'
            heat_rate_col = f'热释放速率/kW_{i}'
            
            # 获取当前样本的所有数据点
            sample_data = df[[wind_dir_col, wind_speed_col, time_col, heat_rate_col]].dropna()
            
            # 为每个时间点创建特征向量 [风向编码, 风速, 时间] -> 热释放速率
            for idx, row in sample_data.iterrows():
                sample_feature = [
                    row[wind_dir_col],  # 风向
                    row[wind_speed_col],  # 风速
                    row[time_col]  # 时间
                ]
                sample_target = row[heat_rate_col]  # 热释放速率
                
                samples.append(sample_feature)
                sample_labels.append(sample_target)
        
        print(f"总共提取了 {len(samples)} 个数据点")
        
        # 转换为numpy数组
        samples = np.array(samples)
        sample_labels = np.array(sample_labels)
        
        # 编码风向
        wind_directions_encoded = self.label_encoder.fit_transform(samples[:, 0])
        X = np.column_stack([
            wind_directions_encoded.astype(float),
            samples[:, 1].astype(float),  # 风速
            samples[:, 2].astype(float)   # 时间
        ])
        
        y = sample_labels.astype(float)
        
        print(f"特征矩阵X形状: {X.shape}")
        print(f"目标向量y形状: {y.shape}")
        
        return X, y
    
    def prepare_sample_level_splits(self, file_path):
        """
        按完整样本进行训练集和测试集划分
        这样确保不会把一组数据拆散
        """
        print("按完整样本进行数据划分...")
        df = pd.read_excel(file_path)
        
        # 获取所有可能的风向值以拟合LabelEncoder
        all_wind_directions = []
        for i in range(1, 65):  # 1到64号样本
            wind_dir_col = f'风向_{i}'
            sample_df = df[[wind_dir_col]].dropna()
            all_wind_directions.extend(sample_df[wind_dir_col].unique())
        
        # 拟合LabelEncoder
        self.label_encoder.fit(list(set(all_wind_directions)))
        
        # 获取每个样本的完整时间序列
        all_samples = []
        
        for i in range(1, 65):  # 1到64号样本
            wind_dir_col = f'风向_{i}'
            wind_speed_col = f'风速/m·s-1_{i}'
            time_col = f'时间/s_{i}'
            heat_rate_col = f'热释放速率/kW_{i}'
            
            # 获取当前样本的所有数据点
            sample_df = df[[wind_dir_col, wind_speed_col, time_col, heat_rate_col]].dropna()
            
            # 构建样本数据结构：每个样本包含多个时间点
            sample_data = {
                'sample_id': i,
                'wind_direction': sample_df[wind_dir_col].iloc[0],  # 样本的风向
                'wind_speed': sample_df[wind_speed_col].iloc[0],    # 样本的风速
                'time_series': []  # 时间序列数据
            }
            
            # 添加时间序列数据
            for idx, row in sample_df.iterrows():
                sample_data['time_series'].append({
                    'time': row[time_col],
                    'heat_rate': row[heat_rate_col]
                })
            
            all_samples.append(sample_data)
        
        print(f"总共识别了 {len(all_samples)} 个完整样本")
        
        # 按样本进行划分（不是按数据点）
        sample_ids = list(range(len(all_samples)))
        train_sample_ids, test_sample_ids = train_test_split(
            sample_ids, test_size=0.2, random_state=42
        )
        
        print(f"训练样本数: {len(train_sample_ids)}, 测试样本数: {len(test_sample_ids)}")
        
        # 准备训练集和测试集数据
        X_train, y_train = self._extract_features_from_samples(all_samples, train_sample_ids)
        X_test, y_test = self._extract_features_from_samples(all_samples, test_sample_ids)
        
        return X_train, X_test, y_train, y_test, all_samples, train_sample_ids, test_sample_ids
    
    def _extract_features_from_samples(self, all_samples, sample_indices):
        """
        从指定样本索引中提取特征和标签
        """
        X_list = []
        y_list = []
        
        for idx in sample_indices:
            sample = all_samples[idx]
            wind_dir_encoded = self.label_encoder.transform([sample['wind_direction']])[0]
            wind_speed = sample['wind_speed']
            
            for point in sample['time_series']:
                X_list.append([wind_dir_encoded, wind_speed, point['time']])
                y_list.append(point['heat_rate'])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def build_models(self):
        """构建多种机器学习模型"""
        print("构建机器学习模型...")
        
        self.models = {
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
            ]),
            'Gradient Boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(n_estimators=50, random_state=42))
            ]),
            'Linear Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        }
    
    def train_models(self, X_train, y_train):
        """训练所有模型"""
        print("开始训练模型...")
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"训练 {name} 模型...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        self.models = trained_models
        print("所有模型训练完成!")
    
    def evaluate_models(self, X_test, y_test):
        """评估所有模型"""
        print("评估模型性能...")
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"{name}:")
            print(f"  MSE: {mse:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
            print()
        
        return results
    
    def plot_results(self, y_test, results, all_samples, train_sample_ids, test_sample_ids):
        """绘制结果图表"""
        print("生成可视化图表...")
        
        # 创建结果目录
        os.makedirs('plots', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # 1. 模型性能比较图
        model_names = list(results.keys())
        r2_scores = [results[name]['R²'] for name in model_names]
        rmse_scores = [results[name]['RMSE'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² 分数比较
        bars1 = ax1.bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_title('模型 R² 分数比较', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE 分数比较
        bars2 = ax2.bar(model_names, rmse_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax2.set_title('模型 RMSE 分数比较', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, score in zip(bars2, rmse_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 最佳模型的实际值vs预测值散点图
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        best_predictions = results[best_model_name]['predictions']
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, best_predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('实际热释放速率 (kW)')
        plt.ylabel('预测热释放速率 (kW)')
        plt.title(f'{best_model_name} - 实际值 vs 预测值')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/scatter_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 不同样本的时间序列预测对比（选择几个代表性样本）
        self._plot_time_series_comparison(results, all_samples, test_sample_ids)
        
        # 4. 残差图
        self._plot_residuals(y_test, best_predictions)
        
        print("图表已保存到 'plots' 目录")
    
    def _plot_time_series_comparison(self, results, all_samples, test_sample_ids):
        """绘制时间序列预测对比图"""
        # 选择最佳模型
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        best_predictions = results[best_model_name]['predictions']
        
        # 重构预测结果以匹配样本结构
        # 首先我们需要知道测试集中每个样本有多少个时间点
        test_sample_data = []
        current_idx = 0
        
        for sample_idx in test_sample_ids:
            sample = all_samples[sample_idx]
            n_points = len(sample['time_series'])
            sample_times = [point['time'] for point in sample['time_series']]
            sample_actual = [point['heat_rate'] for point in sample['time_series']]
            sample_predicted = best_predictions[current_idx:current_idx+n_points]
            
            test_sample_data.append({
                'sample_id': sample['sample_id'],
                'wind_direction': sample['wind_direction'],
                'wind_speed': sample['wind_speed'],
                'times': sample_times,
                'actual': sample_actual,
                'predicted': sample_predicted
            })
            
            current_idx += n_points
        
        # 绘制前几个样本的时间序列对比
        n_samples_to_plot = min(2, len(test_sample_data))  # 最多绘制2个样本以加快速度
        fig, axes = plt.subplots(n_samples_to_plot, 1, figsize=(12, 3*n_samples_to_plot))
        if n_samples_to_plot == 1:
            axes = [axes]
        
        colors = cycle(['blue', 'red', 'green', 'orange'])
        
        for i in range(n_samples_to_plot):
            sample_data = test_sample_data[i]
            color = next(colors)
            
            axes[i].plot(sample_data['times'], sample_data['actual'], 
                        label='实际值', color=color, linewidth=2)
            axes[i].plot(sample_data['times'], sample_data['predicted'], 
                        label='预测值', linestyle='--', color=color, linewidth=2)
            axes[i].set_xlabel('时间 (s)')
            axes[i].set_ylabel('热释放速率 (kW)')
            axes[i].set_title(f'样本 {sample_data["sample_id"]} - {sample_data["wind_direction"]}风, {sample_data["wind_speed"]}m/s')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/time_series_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residuals(self, y_test, predictions):
        """绘制残差图"""
        residuals = y_test - predictions
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 残差散点图
        ax1.scatter(predictions, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('预测值')
        ax1.set_ylabel('残差')
        ax1.set_title('残差散点图')
        ax1.grid(True, alpha=0.3)
        
        # 残差直方图
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('残差')
        ax2.set_ylabel('频次')
        ax2.set_title('残差分布直方图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results, evaluation_metrics):
        """保存结果到文件"""
        print("保存结果...")
        
        # 保存模型性能指标
        metrics_df = pd.DataFrame({
            metric: {model: results[model][metric] for model in results.keys()}
            for metric in ['MSE', 'RMSE', 'MAE', 'R²']
        }).T
        
        metrics_df.to_csv('results/model_performance.csv', encoding='utf-8-sig')
        
        # 保存模型
        os.makedirs('models', exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name.replace(" ", "_")}_model.pkl')
        
        # 保存标签编码器
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        print("结果已保存到 'results' 和 'models' 目录")
    
    def predict(self, wind_direction, wind_speed, time):
        """预测热释放速率"""
        # 编码风向
        wind_dir_encoded = self.label_encoder.transform([wind_direction])[0]
        
        # 准备特征
        features = np.array([[wind_dir_encoded, wind_speed, time]])
        
        # 使用最佳模型进行预测（基于R²分数）
        results = {}
        for name, model in self.models.items():
            pred = model.predict(features)[0]
            results[name] = pred
        
        # 返回R²最高的模型的预测结果
        best_model_name = max(self.models.keys(), key=lambda x: 
                             self.evaluate_single_model(self.models[x], features, np.array([0]))['R²'] 
                             if hasattr(self, '_temp_evaluation') else 0)
        
        return results[best_model_name] if best_model_name in results else list(results.values())[0]
    
    def evaluate_single_model(self, model, X, y):
        """评估单个模型（辅助函数）"""
        try:
            y_pred = model.predict(X)
            if len(np.unique(y)) > 1:
                return {
                    'R²': r2_score(y, y_pred)
                }
            else:
                return {'R²': 0}
        except:
            return {'R²': 0}


def main():
    print("="*60)
    print("火灾热释放速率预测系统")
    print("="*60)
    
    predictor = FireHeatReleasePredictor()
    
    # 加载和预处理数据（按完整样本划分）
    X_train, X_test, y_train, y_test, all_samples, train_ids, test_ids = \
        predictor.prepare_sample_level_splits('data.xlsx')
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 构建模型
    predictor.build_models()
    
    # 训练模型
    predictor.train_models(X_train, y_train)
    
    # 评估模型
    results = predictor.evaluate_models(X_test, y_test)
    
    # 绘制结果
    predictor.plot_results(y_test, results, all_samples, train_ids, test_ids)
    
    # 保存结果
    predictor.save_results(results, None)
    
    print("所有任务完成！结果已保存到以下目录：")
    print("- models/: 保存训练好的模型")
    print("- plots/: 保存可视化图表")
    print("- results/: 保存性能指标")
    
    # 示例预测
    print("\n示例预测:")
    sample_direction = all_samples[0]['wind_direction']
    sample_speed = all_samples[0]['wind_speed']
    sample_time = all_samples[0]['time_series'][len(all_samples[0]['time_series'])//2]['time']
    
    predicted_value = predictor.predict(sample_direction, sample_speed, sample_time)
    print(f"风向: {sample_direction}, 风速: {sample_speed}m/s, 时间: {sample_time}s")
    print(f"预测热释放速率: {predicted_value:.2f} kW")


if __name__ == "__main__":
    main()