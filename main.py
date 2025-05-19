#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/19 16:11
# @Author  : zdj
# @FileName: main.py
# @Software: PyCharm

import argparse
import logging
import os
import time
from K_Means import main as kmeans_main
from EachClusterSeqLabMatrix import generate_cluster_files
from ESM2ClassifierforCluster import main as esm2_classifier_main
from MergePredictions import merge_predictions
from Evaluation import evaluate_predictions


def setup_logging(log_file):
    """配置日志记录"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main(args):
    """主函数，依次调用五个脚本"""
    start_time = time.time()
    logging.info("Starting main.py...")

    # 1. 执行 K-Means 聚类
    logging.info("Performing K-Means clustering...")
    kmeans_main(args)
    logging.info("K-Means clustering completed.")

    # 2. 为每个簇生成多肽-标签矩阵
    logging.info("Generating peptide-label matrix for each cluster...")
    generate_cluster_files(args)
    logging.info("Peptide-label matrix generation completed.")

    # 3. 执行 ESM2 分类器训练与预测
    logging.info("Training and predicting with ESM2 classifier...")
    esm2_classifier_main(args)
    logging.info("ESM2 classifier training and prediction completed.")

    # 4. 合并预测结果
    logging.info("Merge forecast results...")
    merge_predictions(args)
    logging.info("The forecast results are merged.")

    # 5. 评估预测结果
    logging.info("Evaluate the predictions...")
    evaluate_predictions(args)
    logging.info("The evaluation of the prediction results is complete.")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Main script to run K-Means, ESM2 classifier, and evaluation.")

    # K-Means 参数
    parser.add_argument('--labels_embedding_file', type=str, default="generate_data/label_embeddings.csv",
                        help='Label embeddings CSV file path')
    parser.add_argument('--labels_clusters_file', type=str, default="generate_data/labels_clusters.csv",
                        help='Path to save the clustered labels CSV file')
    parser.add_argument('--optimal_n', type=int, default=10,
                        help='Number of clusters for K-Means')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for K-Means')

    # EachClusterSeqLabMatrix 参数
    parser.add_argument('--label_matrix_file', type=str, default="datasets/label_matrix.csv",
                        help='Path to the label matrix CSV file')
    parser.add_argument('--output_dir', type=str, default="generate_data/cluster_data",
                        help='Output directory for cluster data')

    # ESM2ClassifierforCluster 参数
    parser.add_argument('--cluster_dir', type=str, default="generate_data/cluster_data",
                        help='Cluster data directory')
    parser.add_argument('--model_dir', type=str, default="model",
                        help='Path to the ESM2 model directory')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of the classifier model')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate for the classifier model')
    parser.add_argument('--k_hard_negatives', type=int, default=1,
                        help='Number of hard negatives to mine for each positive sample')
    parser.add_argument('--e_hard_negatives', type=int, default=10,
                        help='Epoch interval for hard negative mining')

    # MergePredictions 参数
    parser.add_argument('--prediction_dir', type=str, default="generate_data/cluster_pre_data",
                        help='Directory containing the prediction files for each cluster')
    parser.add_argument('--output_path', type=str, default="generate_data/merged_predictions.csv",
                        help='Path to save the merged predictions file')

    # Evaluation 参数
    parser.add_argument('--pred_path', type=str, default="generate_data/merged_predictions.csv",
                        help='The path to the merged predictions file')
    parser.add_argument('--true_path', type=str, default="datasets/label_matrix.csv",
                        help='The path to the true labels file')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5],
                        help='List of k values for evaluation metrics')

    # 日志文件参数
    parser.add_argument('--log_file', type=str, default="log_file/main.log",
                        help='Path to save the log file for the main script')

    # 解析参数
    args = parser.parse_args()

    # 配置日志
    setup_logging(args.log_file)

    # 执行主函数
    main(args)