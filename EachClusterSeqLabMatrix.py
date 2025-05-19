#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/22 17:08
# @Author  : zdj
# @FileName: EachClusterSeqLabMatrix.py
# @Software: PyCharm
import argparse
import logging
import os
import time
import pandas as pd
from io import StringIO

# 实现 loadFileData 函数，用于读取本地 CSV 文件内容
def loadFileData(filename):
    """
    读取指定文件名的 CSV 文件内容并返回字符串。

    参数:
        filename (str): 要读取的 CSV 文件名

    返回:
        str: 文件内容的字符串表示
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {filename} 未找到")
    except Exception as e:
        raise Exception(f"读取文件 {filename} 时发生错误: {str(e)}")


# 主逻辑：为每个簇生成多肽-标签文件，确保多肽序列唯一
def generate_cluster_files(args):
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    # 加载 CSV 文件
    logging.info("Loading CSV files...")
    labels_clusters_csv = loadFileData(args.labels_clusters_file)
    label_matrix_csv = loadFileData(args.label_matrix_file)

    # 解析 labels_clusters.csv，创建标签到簇的映射
    # 使用 StringIO 将字符串转换为 DataFrame
    logging.info("Parsing labels_clusters.csv...")
    labels_clusters_df = pd.read_csv(StringIO(labels_clusters_csv))
    label_to_cluster = dict(zip(labels_clusters_df['Label'], labels_clusters_df['Cluster']))

    # 解析 label_matrix.csv，获取多肽及其标签关联
    logging.info("Parsing label_matrix.csv...")
    label_matrix_df = pd.read_csv(StringIO(label_matrix_csv))
    labels = label_matrix_df.columns[1:]  # 排除 'Sequences' 列

    # 初始化字典，键为簇的编号（获取所有唯一的簇编号），值为一个空字典，用于存储该簇的多肽及其标签数据
    logging.info("Grouping peptides-labels by clusters...")
    cluster_data = {i: {} for i in labels_clusters_df['Cluster'].unique()}
    # 遍历每个多肽，收集值为 1 的标签并按簇分组
    for _, row in label_matrix_df.iterrows():
        peptide = row['Sequences']
        for label in labels:
            if label in label_to_cluster:  # 只处理 labels_clusters.csv 中的标签
                cluster = label_to_cluster[label]
                if peptide not in cluster_data[cluster]:
                    cluster_data[cluster][peptide] = {'Sequences': peptide}
                cluster_data[cluster][peptide][label] = row[label]

    # 为每个簇生成单独的 CSV 文件
    logging.info("Generating cluster files...")
    for cluster, peptide_dict in cluster_data.items():
        if peptide_dict:  # 仅在簇有数据时生成文件
            # 筛选出至少有一个标签值为 1 的多肽
            data = []
            for peptide, peptide_row in peptide_dict.items():
                # 检查该多肽是否与簇中的任何标签关联（值为 1）
                cluster_labels = [label for label in labels if
                                  label in label_to_cluster and label_to_cluster[label] == cluster]
                if any(peptide_row.get(label, 0) == 1 for label in cluster_labels):
                    data.append(peptide_row)

            if data:  # 确保有数据才生成文件
                # 创建簇的 DataFrame
                cluster_df = pd.DataFrame(data)
                # 仅保留该簇的标签
                columns = ['Sequences'] + cluster_labels
                cluster_df = cluster_df[columns]
                # 填充缺失的标签值为 0
                cluster_df = cluster_df.fillna(0).astype({label: int for label in cluster_labels})

                # 将 DataFrame 保存为 CSV 文件
                output_csv = os.path.join(args.output_dir, f"cluster_{cluster}_matrix.csv")
                cluster_df.to_csv(output_csv, index=False)
                logging.info(f"Cluster {cluster} data saved to {output_csv}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Generate per cluster peptide-label files")
    parser.add_argument('--labels_clusters_file', type=str, default="generate_data/labels_clusters.csv",
                        help='Path to the labels clusters CSV file')
    parser.add_argument('--label_matrix_file', type=str, default="datasets/label_matrix.csv",
                        help='Path to the label matrix CSV file')
    parser.add_argument('--output_dir', type=str, default="generate_data/cluster_data",
                        help='Output directory for cluster directory')
    parser.add_argument('--log_file', type=str, default='log_file/eachClusterSeqLabMatrix.log',
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
    generate_cluster_files(args)
    end_time = time.time()
    run_time = end_time - start_time
    logging.info(f"Run time: {run_time:.4f}")