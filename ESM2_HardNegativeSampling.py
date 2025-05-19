#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/28 14:42
# @Author  : zdj
# @FileName: ESM2_HardNegativeSampling.py
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
from Utils import extract_embeddings_in_batches, select_hard_negatives, device

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
            logging.error("错误: 数据文件缺少 'Sequences' 列")
            return
    except Exception as e:
        logging.error(f"读取数据文件失败: {str(e)}")
        return

    # Extract sequences and labels
    sequences = data['Sequences'].values
    labels = data.iloc[:, 1:].values
    label_names = data.columns[1:]

    # Convert sequences to the format required for ESM2
    sequences = [(i, seq) for i, seq in enumerate(sequences)]

    # Extract embeddings using ESM2 model
    logging.info("提取序列的嵌入...")
    embeddings = extract_embeddings_in_batches(sequences, batch_size=args.batch_size)

    # Convert embeddings and labels to tensors
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    input_dim = X.shape[1]
    n_labels = y.shape[1]


    # Initialize KFold cross-validator
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    all_predictions = np.zeros_like(y.cpu().numpy(), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logging.info(f"处理第 {fold+1}/{args.n_folds} 折")

        # Create training and validation datasets
        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        test_dataset = TensorDataset(X[test_idx], y[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        # Initialize model, criterion, and optimizer
        model = Classifier(input_dim, n_labels, args.hidden_dim, args.dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0

            # Perform hard negative mining every 10 epochs after the first epoch
            if epoch % args.e_hard_negatives == 0 and epoch > 0:
                logging.info(f"Epoch {epoch}: Mining hard negatives...")
                hard_neg_indices = select_hard_negatives(
                    model,
                    train_loader,
                    k=args.k_hard_negatives,
                    confidence_threshold=0.5
                )

                if len(hard_neg_indices) > 0:
                    # Create new training dataset with original data plus hard negatives
                    original_indices = torch.arange(len(train_dataset))
                    combined_indices = torch.cat([original_indices, hard_neg_indices])
                    # 添加去重
                    combined_indices = torch.unique(combined_indices)

                    # Update training loader
                    new_dataset = TensorDataset(X[combined_indices], y[combined_indices])
                    train_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True)
                    logging.info(f"Added {len(hard_neg_indices)} hard negative samples")
                else:
                    logging.info("No hard negative samples found!")

            # Training on the updated train_loader
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            logging.info(f"Fold {fold}, Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss:.4f}")

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
    parser = argparse.ArgumentParser(description="ESM2 with Hard Negative Sampling")
    parser.add_argument('--data_path', type=str, default="datasets/label_matrix.csv",
                        help='Path to the label matrix CSV file')
    parser.add_argument('--output_dir', type=str, default="result_orginal_data/HNS_result",
                        help='Path to save the results')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the classifier')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--k_hard_negatives', type=int, default=1, help='Number of hard negatives to mine')
    parser.add_argument('--e_hard_negatives', type=int, default=10, help='Epoch interval for hard negative mining')
    parser.add_argument('--log_file', type=str, default="log_file/ESM2_HardNegativeSampling.log",
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

    # Start timing the process
    start_time = time.time()
    main(args)

    end_time = time.time()
    logging.info(f'Total time: {end_time - start_time:.2f} seconds')