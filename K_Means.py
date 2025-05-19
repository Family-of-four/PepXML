#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/22 14:56
# @Author  : zdj
# @FileName: K_Means.py
# @Software: PyCharm
import argparse
import logging
from sklearn.cluster import KMeans
import pandas as pd
import time

def main(args):
    # 读取标签嵌入数据
    logging.info(f"Reading label embeddings from {args.labels_embedding_file}")
    df = pd.read_csv(args.labels_embedding_file)
    labels = df['Label'].values
    embeddings = df.drop(columns=['Label']).values
    print(f"Label: {len(labels)}, embeddings with shape: {embeddings.shape}")
    logging.info(f"Label: {len(labels)}, embeddings with shape: {embeddings.shape}")

    # 执行K-Means聚类
    logging.info(f"Performing K-Means clustering with k={args.optimal_n}")
    kmeans = KMeans(n_clusters=args.optimal_n, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    logging.info(f"Clustering completed.")

    # 创建包含标签名称和对应聚类标签的DataFrame
    logging.info("Creating DataFrame for labels and cluster labels")
    labels_cluster_df = pd.DataFrame({
        'Label': labels,
        'Cluster': cluster_labels
    })

    # 保存文件
    labels_cluster_df.to_csv(args.labels_clusters_file, index=False)
    logging.info(f"Cluster labels saved to {args.labels_clusters_file}")

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="K-Means Clustering")
    parser.add_argument('--labels_embedding_file', type=str, default="generate_data/label_embeddings.csv",
                        help='Path to the label embeddings CSV file')
    parser.add_argument('--labels_clusters_file', type=str, default="generate_data/labels_clusters.csv",
                        help='Path to save the clustered labels CSV file')
    parser.add_argument('--optimal_n', type=int, default=10,
                        help='Number of clusters for K-Means')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for K-Means')
    parser.add_argument('--log_file', type=str, default='log_file/kmeans.log',
                        help='Path to save the log file')
    # 解析参数
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    start_time = time.time()
    main(args)
    end_time = time.time()
    run_time = end_time - start_time
    logging.info(f"Run time: {run_time:.2f}s")
