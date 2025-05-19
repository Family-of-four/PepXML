#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 17:21
# @Author  : zdj
# @FileName: ESM2_MLP.py
# @Software: PyCharm
import argparse
import logging
import time
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from Utils import extract_embeddings_in_batches, device

# Classifier model
class Classifier(nn.Module):
    def __init__(self, input_dim, n_labels, hidden_dim=256, dropout=0.3):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = -self.alpha * (1 - p_t) ** self.gamma * targets * torch.log(inputs + 1e-8) - \
               (1 - self.alpha) * (1 - p_t) ** self.gamma * (1 - targets) * torch.log(1 - inputs + 1e-8)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Rank Loss
class RankLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(RankLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, targets):
        pos_scores = outputs * targets
        neg_scores = outputs * (1 - targets)
        loss_matrix = torch.clamp(self.margin - pos_scores.unsqueeze(2) + neg_scores.unsqueeze(1), min=0)
        return loss_matrix.mean()

def train_model(model, train_loader, val_loader, criterion, device, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
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
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss / len(val_loader):.4f}')

def predict(model, data_loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            batch_probs = torch.sigmoid(model(batch_x)).cpu().numpy()
            all_probs.append(batch_probs)
    return np.vstack(all_probs)

def main(args):
    # 加载数据
    try:
        data = pd.read_csv(args.data_path)
        if 'Sequences' not in data.columns:
            logging.error("数据文件缺少 'Sequences' 列")
    except Exception as e:
        logging.error(f"读取数据失败: {str(e)}")
        return

    sequences = data['Sequences'].values
    label_names = [col for col in data.columns if col != 'Sequences']
    labels = data[label_names].values
    n_labels = len(label_names)

    # 转换为 ESM-2 所需的序列格式
    sequences = [(i, seq) for i, seq in enumerate(sequences)]

    # 提取特征
    logging.info("提取 ESM-2 特征...")
    embeddings = extract_embeddings_in_batches(sequences, batch_size=args.batch_size)
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    # 五折交叉验证
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_predictions = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logging.info(f"处理第 {fold + 1}/{args.n_folds} 折")

        # 创建数据加载器
        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        test_dataset = TensorDataset(X[test_idx], y[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # 初始化模型和损失函数
        model = Classifier(input_dim=X.shape[1], n_labels=n_labels, hidden_dim=args.hidden_dim, dropout=args.dropout)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')  # 可选：Focal Loss
        # criterion = RankLoss(margin=1.0)  # 可选：Rank Loss

        # 训练模型
        train_model(model, train_loader, test_loader, criterion, device, epochs=args.epochs, lr=args.lr)

        # 预测验证集
        val_predictions = predict(model, test_loader, device)

        # 保存预测结果
        os.makedirs(args.output_dir, exist_ok=True)
        pred_df = pd.DataFrame({
            'Sequences': data.iloc[test_idx]['Sequences'].values,
            **{label_names[i]: val_predictions[:, i] for i in range(n_labels)}
        })
        pred_path = os.path.join(args.output_dir, f"cluster_{fold}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        logging.info(f"折 {fold + 1} 预测结果已保存至: {pred_path}")
        fold_predictions.append(pred_df)

    # 合并预测结果
    merged_df = pd.DataFrame({'Sequences': data['Sequences']})
    for label in label_names:
        merged_df[label] = 0.0

    for pred_df in fold_predictions:
        for _, row in pred_df.iterrows():
            seq = row['Sequences']
            merged_df.loc[merged_df['Sequences'] == seq, label_names] = row[label_names].values

    os.makedirs(os.path.dirname(args.merged_output_path), exist_ok=True)
    merged_df.to_csv(args.merged_output_path, index=False)
    logging.info(f"合并预测结果已保存至: {args.merged_output_path}")



if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="ESM2 MLP Model")
    parser.add_argument('--data_path', type=str, default="datasets/label_matrix.csv",
                        help='Path to the label matrix CSV file')
    parser.add_argument('--output_dir', type=str, default="result_orginal_data/cluster_data",
                        help='Path to save the fold predictions')
    parser.add_argument('--merged_output_path', type=str, default="result_orginal_data/merged_predictions.csv",
                        help='Path to save the merged predictions')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the classifier')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--log_file', type=str, default="log_file/ESM2_MLP.log", help='Path to save the log file')

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