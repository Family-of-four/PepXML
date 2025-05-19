#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/28 10:16
# @Author  : zdj
# @FileName: ESM2_InteractionAttention.py
# @Software: PyCharm
import argparse
import logging
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from Utils import extract_embeddings_in_batches, device


def build_label_cooccurrence_matrix(label_matrix):
    """
    label_matrix: A binary matrix where rows represent peptides and columns represent labels.
    """
    cooccurrence_matrix = np.dot(label_matrix.T, label_matrix)
    np.fill_diagonal(cooccurrence_matrix, 0)  # Remove self-co-occurrence
    return cooccurrence_matrix

def get_label_embeddings(cooccurrence_matrix, dimensions=1280, walk_length=10, num_walks=10, p=1, q=1):
    # Create graph from co-occurrence matrix
    G = nx.from_numpy_array(cooccurrence_matrix)  # Use from_numpy_array

    # Run Node2Vec algorithm (no window_size argument)
    node2vec_model = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q)
    model = node2vec_model.fit()

    # Extract embeddings for each label
    label_embeddings = np.array([model.wv[str(i)] for i in range(cooccurrence_matrix.shape[0])])
    return torch.tensor(label_embeddings, dtype=torch.float32).to(device)

# 交互注意力机制
class InteractionAttention(nn.Module):
    def __init__(self, r, device):
        super(InteractionAttention, self).__init__()
        self.r = r
        self.device = device
        self.W_I = nn.Parameter(torch.randn(r, r).to(device))

    def forward(self, H, L):
        H = H.to(self.device)
        L = L.to(self.device)
        n = H.size(0)
        k = L.size(0)
        L = L.T  # (r, k)
        Q = torch.matmul(self.W_I, L)  # (r, k)
        M_I = torch.matmul(H, Q)  # (n, k)
        A_I = torch.softmax(M_I, dim=1)  # (n, k)
        C_I = torch.matmul(A_I, L.T)  # (n, r)
        return C_I, A_I

# 分类器
class ClassifierWithAttention(nn.Module):
    def __init__(self, r, n_labels, hidden_dim=256, dropout=0.3):
        super(ClassifierWithAttention, self).__init__()
        self.fc1 = nn.Linear(r + r, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_labels)

    def forward(self, H, L):
        interaction_attention = InteractionAttention(r=H.size(1), device=device)
        C_I, A_I = interaction_attention(H, L)
        combined_input = torch.cat([H, C_I], dim=1)
        x = self.fc1(combined_input)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output, A_I

# 预测
def predict(model, data_loader, L, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            output, _ = model(batch_x, L)
            probs = torch.sigmoid(output).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

# 主函数
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

    # 提取序列和标签
    logging.info("提取序列和标签")
    sequences = data['Sequences'].values
    labels = data.iloc[:, 1:].values
    label_names = data.columns[1:]

    # 转换为 ESM-2 需要的格式
    sequences = [(i, seq) for i, seq in enumerate(sequences)]

    # 提取序列嵌入
    embeddings = extract_embeddings_in_batches(sequences, batch_size=args.batch_size)

    # 转换为张量
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    # 构建标签共现矩阵
    logging.info("构建标签共现矩阵")
    cooccurrence_matrix = build_label_cooccurrence_matrix(labels)
    # 获取标签嵌入
    L = get_label_embeddings(cooccurrence_matrix)


    input_dim = X.shape[1]
    n_labels = y.shape[1]

    # 五折交叉验证
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    all_predictions = np.zeros_like(y.cpu().numpy(), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logging.info(f"处理第 {fold + 1}/{args.n_folds} 折")

        # 创建数据加载器
        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        test_dataset = TensorDataset(X[test_idx], y[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # 初始化模型和损失函数
        model = ClassifierWithAttention(r=input_dim, n_labels=n_labels, hidden_dim=args.hidden_dim, dropout=args.dropout)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model.to(device)

        # 训练循环
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output, _ = model(batch_x, L)  # 使用全部标签嵌入
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            logging.info(f"Fold {fold+1}, Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss:.4f}")

        # 预测
        test_predictions = predict(model, test_loader, L, device)
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
    parser = argparse.ArgumentParser(description="ESM2 Interaction Attention Model")
    parser.add_argument('--data_path', type=str, default="datasets/label_matrix.csv",
                        help='Path to the label matrix CSV file')
    parser.add_argument('--output_dir', type=str, default="result_orginal_data/IA_result",
                        help='Path to save the results')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the classifier')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--log_file', type=str, default="log_file/ESM2_InteractionAttention.log",
                        help='Path to save the log file')

    # 解析参数
    args = parser.parse_args()

    # 配置日志
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    start_time = time.time()
    # 运行主函数
    main(args)
    end_time = time.time()
    logging.info(f"总运行时间: {end_time - start_time:.2f} 秒")

