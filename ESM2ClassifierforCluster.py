#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 19:25
# @Author  : zdj
# @FileName: ESM2ClassifierforCluster.py
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
from Utils import extract_embeddings_in_batches, select_hard_negatives, device

# 预测
def predict(model, data_loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            probs = torch.sigmoid(model(batch_x)).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

# Classifier model
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

# 主函数
def main(args):
    # 遍历标签文件
    for cluster_file in os.listdir(args.cluster_dir):
        if cluster_file.startswith("cluster_") and cluster_file.endswith("_matrix.csv"):
            cluster_id = cluster_file.split('_')[1]
            cluster_path = os.path.join(args.cluster_dir, cluster_file)
            logging.info(f"处理簇 {cluster_id}")

            # 加载标签数据
            try:
                data = pd.read_csv(cluster_path)
                if 'Sequences' not in data.columns:
                    logging.warning(f"警告: {cluster_file} 缺少 'Sequences' 列，跳过")
                    continue
            except Exception as e:
                logging.error(f"读取 {cluster_file} 失败: {str(e)}")
                continue

            # 提取序列和标签
            sequences = data['Sequences'].values
            labels = data.iloc[:, 1:].values
            label_names = data.columns[1:]

            # 转换为 ESM-2 需要的格式
            sequences = [(i, seq) for i, seq in enumerate(sequences)]

            # 提取嵌入
            embeddings = extract_embeddings_in_batches(sequences, batch_size=args.batch_size)

            # 转换为张量
            X = torch.tensor(embeddings, dtype=torch.float32).to(device)
            y = torch.tensor(labels, dtype=torch.float32).to(device)
            input_dim = X.shape[1]
            n_labels = y.shape[1]

            # 五折交叉验证
            kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
            all_predictions = np.zeros_like(y.cpu().numpy(), dtype=np.float32)

            for fold, (train_idx, test_id) in enumerate(kf.split(X)):
                logging.info(f"处理簇 {cluster_id}，第 {fold + 1}/{args.n_folds} 折")

                # 创建数据加载器
                train_dataset = TensorDataset(X[train_idx], y[train_idx])
                test_dataset = TensorDataset(X[test_id], y[test_id])
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                # 初始化模型和损失函数
                model = Classifier(input_dim=input_dim, n_labels=n_labels, hidden_dim=args.hidden_dim, dropout=args.dropout)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                model.to(device)

                # 训练循环
                for epoch in range(args.epochs):
                    model.train()
                    total_loss = 0

                    if epoch % args.e_hard_negatives == 0 and epoch > 0:
                        logging.info(f"Epoch {epoch}: 挖掘硬负样本...")
                        hard_neg_indices = select_hard_negatives(
                            model, train_loader, k=args.k_hard_negatives, confidence_threshold=0.5
                        )
                        if len(hard_neg_indices) > 0:
                            original_indices = torch.arange(len(train_dataset))
                            combined_indices = torch.unique(torch.cat([original_indices, hard_neg_indices]))
                            new_dataset = TensorDataset(
                                X[combined_indices], y[combined_indices]
                            )
                            train_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True)
                            logging.info(f"添加 {len(hard_neg_indices)} 硬负样本")
                        else:
                            logging.info("未找到硬负样本")

                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        output = model(batch_x)
                        loss = criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                    avg_train_loss = total_loss / len(train_loader)
                    logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss:.4f}")

                # 预测验证集
                val_predictions = predict(model, test_loader, device)
                all_predictions[test_id] = val_predictions

                # 保存模型权重
                cluster_model_dir = os.path.join(args.model_dir, f"cluster_{cluster_id}")
                os.makedirs(cluster_model_dir, exist_ok=True)
                model_path = os.path.join(cluster_model_dir, f"fold_{fold}_model.pth")
                torch.save(model.state_dict(), model_path)
                logging.info(f"已保存模型: {model_path}")

            # 保存预测结果
            result_df = pd.DataFrame({
                'Sequences': data['Sequences'],
                **{label_names[i]: all_predictions[:, i] for i in range(n_labels)}
            })
            result_path = os.path.join(args.output_dir, f"cluster_{cluster_id}_predictions.csv")
            os.makedirs(args.output_dir, exist_ok=True)
            result_df.to_csv(result_path, index=False)
            logging.info(f"已保存预测结果: {result_path}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="ESM2 Classifier with Hard Negative Sampling for Clusters")
    parser.add_argument('--cluster_dir', type=str, default="generate_data/cluster_data",
                        help='Path to the cluster data directory')
    parser.add_argument('--output_dir', type=str, default="generate_data/cluster_pre_data",
                        help='Path to save the prediction results')
    parser.add_argument('--model_dir', type=str, default="model", help='Path to save the model weights')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the classifier')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--k_hard_negatives', type=int, default=1, help='Number of hard negatives to mine')
    parser.add_argument('--e_hard_negatives', type=int, default=10, help='Epoch interval for hard negative mining')
    parser.add_argument('--log_file', type=str, default="log_file/ESM2ClassifierforCluster.log",
                        help='Path to save the log file')

    # 解析参数
    args = parser.parse_args()

    # 配置日志
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 记录开始时间
    start_time = time.time()
    main(args)
    # 记录结束时间
    end_time = time.time()
    logging.info(f"总运行时间: {end_time - start_time:.2f} 秒")