#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/28 15:06
# @Author  : zdj
# @FileName: ESM2_LIL.py
# @Software: PyCharm
import argparse
import logging
import os
import time
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from Utils import extract_embeddings_in_batches, device

# 计算标签共现矩阵
def compute_cooccurrence_matrix(label_matrix):
    """
    计算标签的共现矩阵和标签频次。
    Args:
        label_matrix (np.ndarray): 标签-多肽矩阵，形状为 (num_samples, num_labels)。
    Returns:
        co_occurrence (np.ndarray): 共现矩阵 (num_labels x num_labels)。
        label_counts (np.ndarray): 标签频次 (num_labels,)。
    """
    # 标签共现矩阵 (LxL)
    co_occurrence = np.dot(label_matrix.T, label_matrix)  # 计算 S_i 和 S_j 同时出现的样本数
    # 标签频次 (L,)
    label_counts = np.sum(label_matrix, axis=0)  # 每个标签出现的样本数
    return co_occurrence, label_counts

# 计算条件概率矩阵
def compute_conditional_probability_matrix(co_occurrence, label_counts):
    """
    根据等式 (5) 计算条件概率矩阵。
    Args:
        co_occurrence (np.ndarray): 共现矩阵 (num_labels x num_labels)。
        label_counts (np.ndarray): 标签频次 (num_labels,)。
    Returns:
        np.ndarray: 条件概率矩阵 A (num_labels x num_labels)。
    """
    num_labels = co_occurrence.shape[0]
    conditional_matrix = np.zeros_like(co_occurrence, dtype=float)

    for i in range(num_labels):
        for j in range(num_labels):
            if i == j:
                conditional_matrix[i, j] = 1  # 对角线元素
            else:
                conditional_matrix[i, j] = co_occurrence[i, j] / label_counts[j] if label_counts[j] > 0 else 0
    return conditional_matrix

# 构建标签相关矩阵
def build_label_correlation_matrix(label_matrix, tau=0.1, p=0.5):
    """
    根据等式 (5)、(6)、(7) 构建标签相关矩阵。
    Args:
        label_matrix (np.ndarray): 标签-多肽矩阵，形状为 (num_samples, num_labels)。
        tau (float): 阈值 τ，用于筛选相关性。
        p (float): 等式 (7) 中的超参数。
    Returns:
        torch.Tensor: 标签相关矩阵 A~。
    """
    # 计算共现矩阵和标签频次
    co_occurrence, label_counts = compute_cooccurrence_matrix(label_matrix)

    # 等式 (5)：计算条件概率矩阵
    conditional_matrix = compute_conditional_probability_matrix(co_occurrence, label_counts)
    # 保存条件概率矩阵为.csv文件
    # np.savetxt("result/LIL/conditional_matrix.csv", conditional_matrix)
    # 绘制条件概率矩阵分布
    # plt.hist(conditional_matrix.flatten(), bins=50)
    # plt.title("Conditional Probability Matrix Distribution")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.show()

    # 等式 (6)：应用阈值 τ
    filtered_matrix = np.where(conditional_matrix > tau, conditional_matrix, 0)

    # 等式 (7)：构建最终矩阵 A~
    num_labels = filtered_matrix.shape[0]
    correlation_matrix = np.zeros_like(filtered_matrix, dtype=float)
    for i in range(num_labels):
        for j in range(num_labels):
            if i != j:  # 非对角线元素
                denominator = np.sum(filtered_matrix[i, :])
                correlation_matrix[i, j] = p / denominator if denominator > 0 else 0
            else:  # 对角线元素
                correlation_matrix[i, j] = 1 - p

    return torch.tensor(correlation_matrix, dtype=torch.float32)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_labels, hidden_dim=256, dropout=0.3):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LIL(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, dropout, correlation_matrix, num_layers=1):
        """
        初始化 LIL 模块
        Args:
            input_dim (int): 输入特征维度
            label_dim (int): 标签数量
            correlation_matrix (torch.Tensor): 标签相关性矩阵 (LxL)
            num_layers (int): 残差聚合的层数 (c)
            hidden_dims (list of int, optional): MLP 的隐藏层维度列表
        """
        super(LIL, self).__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.num_layers = num_layers

        # 标签相关性矩阵
        self.correlation_matrix = nn.Parameter(correlation_matrix, requires_grad=False)

        self.fc_raw = Classifier(input_dim, label_dim, hidden_dim, dropout)

        self.fc_layers = nn.ModuleList([
            Classifier(input_dim, label_dim, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入特征 (N x D)
        Returns:
            torch.Tensor: 最终的标签预测 (N x L)
        """
        # 初始输出 (LIL(0))
        raw_output = self.fc_raw(x)  # N x L

        # 残差聚合
        aggregated_output = raw_output
        for fc in self.fc_layers:
            neighbor_interaction = torch.matmul(aggregated_output, self.correlation_matrix)  # 邻近标签交互
            aggregated_output = aggregated_output + fc(x) * neighbor_interaction  # 残差累积

        return aggregated_output

def predict(model, data_loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            probs = torch.sigmoid(model(batch_x)).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

def main(args):
    # 加载标签数据
    try:
        data = pd.read_csv(args.data_path)
        if 'Sequences' not in data.columns:
            logging.error("数据文件缺少 'Sequences' 列")
            return
    except Exception as e:
        logging.error(f"读取数据文件失败: {str(e)}")
        return

    # Extract sequences and labels
    sequences = data['Sequences'].values
    labels = data.iloc[:, 1:].values
    label_names = data.columns[1:]

    # Convert sequences to the format required for ESM2 (as tuples of index and sequence)
    sequences = [(i, seq) for i, seq in enumerate(sequences)]

    # Extract embeddings using ESM2 model
    embeddings = extract_embeddings_in_batches(sequences)

    # Convert embeddings and labels to tensors
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)


    # 构建标签相关矩阵
    label_correlation_matrix = build_label_correlation_matrix(labels, tau=args.tau, p=args.p)
    logging.info(f"标签相关矩阵形状：{label_correlation_matrix.shape}")

    # Set parameters for training
    input_dim = X.shape[1]
    n_labels = y.shape[1]

    # Initialize KFold cross-validator
    kf = KFold(n_splits=args.n_fold, shuffle=True, random_state=42)
    all_predictions = np.zeros_like(y.cpu().numpy(), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logging.info(f"处理第 {fold + 1}/{args.n_fold} 折")

        # Create data loaders for training and validation
        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        test_dataset = TensorDataset(X[test_idx], y[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        # Initialize model, loss function, and optimizer
        model = LIL(input_dim, n_labels, args.hidden_dim, args.dropout, label_correlation_matrix, num_layers=args.num_layers).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train the model
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(f'Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / len(train_loader):.4f}')

        # predictions
        test_predictions = predict(model, test_loader, device)
        all_predictions[test_idx] = test_predictions

        # 保存预测结果
    result_df = pd.DataFrame({
        'Sequences': data['Sequences'],
        **{label_names[i]: all_predictions[:, i] for i in range(n_labels)}
    })
    result_path = os.path.join(args.output_dir, "original_data_predictions.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    result_df.to_csv(result_path, index=False)
    logging.info(f"已保存预测结果: {result_path}")

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="ESM2 with Label Interaction Learning (LIL)")
    parser.add_argument('--data_path', type=str, default="datasets/label_matrix.csv",
                        help='Path to the label matrix CSV file')
    parser.add_argument('--output_dir', type=str, default="result_orginal_data/LIL_result",
                        help='Path to save the results')
    parser.add_argument('--n_fold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the classifier')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--tau', type=float, default=0.01, help='Threshold for filtering correlations')
    parser.add_argument('--p', type=float, default=0.3, help='Hyperparameter for correlation matrix')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in LIL module')
    parser.add_argument('--log_file', type=str, default="log_file/ESM2_LIL.log", help='Path to save the log file')

    # 解析参数
    args = parser.parse_args()

    # 配置日志
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Start timing the process
    start_time = time.time()
    main(args)
    end_time = time.time()
    logging.info(f'Total time: {end_time - start_time:.2f} seconds')