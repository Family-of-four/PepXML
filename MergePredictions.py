#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 10:53
# @Author  : zdj
# @FileName: MergePredictions.py
# @Software: PyCharm
import argparse
import logging
import os
import time

import pandas as pd


def merge_predictions(args):
    """
    合并所有簇的预测结果文件，生成包含所有多肽及其标签预测概率的单一文件，保证多肽唯一。
    假设每个簇的标签互补，预测文件不包含 true_{标签} 列，直接合并各簇的预测值。

    参数:
        prediction_dir (str): 预测结果文件目录
        label_matrix_file (str): 原始标签矩阵文件路径
        output_path (str): 合并预测结果输出路径
    """
    # 读取原始标签矩阵，获取所有多肽和标签
    try:
        label_df = pd.read_csv(args.label_matrix_file)
        if 'Sequences' not in label_df.columns:
            raise ValueError(f"{args.label_matrix_file} 缺少 'Sequences' 列")
    except Exception as e:
        logging.error(f"读取 {args.label_matrix_file} 失败: {str(e)}")
        return

    # 获取所有多肽和标签
    all_sequences = label_df['Sequences'].unique()
    label_cols = [col for col in label_df.columns if col != 'Sequences']

    # 初始化合并结果 DataFrame
    merged_df = pd.DataFrame({'Sequences': all_sequences})
    for label_col in label_cols:
        merged_df[label_col] = 0.0  # 默认预测概率为 0

    # 收集所有预测
    prediction_files = [f for f in os.listdir(args.prediction_dir) if
                        f.startswith("cluster_") and f.endswith("_predictions.csv")]
    if not prediction_files:
        logging.warning(f"警告: {args.prediction_dir} 中未找到预测文件")
        return

    processed_labels = set()

    for pred_file in prediction_files:
        pred_path = os.path.join(args.prediction_dir, pred_file)
        try:
            pred_df = pd.read_csv(pred_path)
            if 'Sequences' not in pred_df.columns:
                logging.warning(f"警告: {pred_path} 缺少 'Sequences' 列，跳过")
                continue
        except Exception as e:
            logging.error(f"读取 {pred_path} 失败: {str(e)}")
            continue

        # 提取预测列（仅保留 label_cols 中存在的列）
        pred_cols = [col for col in pred_df.columns if col in label_cols]
        if not pred_cols:
            logging.warning(f"警告: {pred_path} 无有效预测列，跳过")
            continue

        # 检查标签互补性
        if processed_labels & set(pred_cols):
            logging.warning(f"警告: {pred_path} 包含已处理的标签 {processed_labels & set(pred_cols)}，可能违反标签互补性")
        processed_labels.update(pred_cols)

        pred_df = pred_df[['Sequences'] + pred_cols]
        if not pred_df.empty:
            # 合并到 merged_df
            merged_df = merged_df.merge(pred_df, on='Sequences', how='left', suffixes=('', '_new'))
            for col in pred_cols:
                merged_df[col] = merged_df[f"{col}_new"].fillna(merged_df[col])
                merged_df = merged_df.drop(columns=f"{col}_new")
            logging.info(f"处理 {pred_path}: 包含 {len(pred_df)} 个多肽，{len(pred_cols)} 个标签")
        else:
            logging.warning(f"警告: {pred_path} 为空文件，跳过")

    # 验证标签覆盖
    missing_labels = set(label_cols) - processed_labels
    if missing_labels:
        logging.warning(f"警告: 以下标签未在任何预测文件中出现: {missing_labels}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 保存合并结果
    merged_df.to_csv(args.output_path, index=False)
    logging.info(f"合并预测结果已保存至: {args.output_path}")
    logging.info(f"总多肽数: {len(merged_df)}, 标签数: {len(label_cols)}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Merge predictions from multiple clusters into a single file.")
    parser.add_argument('--prediction_dir', type=str, default="generate_data/cluster_pre_data",
                        help='The directory of the prediction results files.')
    parser.add_argument('--label_matrix_file', type=str, default="datasets/label_matrix.csv",
                        help='The path to the original label matrix file, which contains all peptides and labels.')
    parser.add_argument('--output_path', type=str, default="generate_data/merged_predictions.csv",
                        help='The path to save the merged predictions file.')
    parser.add_argument('--log_file', type=str, default="log_file/MergePredictions.log",
                        help='The path to save the log file.')

    # 解析参数
    args = parser.parse_args()

    # 配置日志
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    start_time = time.time()
    merge_predictions(args)
    end_time = time.time()
    logging.info(f"合并预测结果完成，耗时: {end_time - start_time:.2f} 秒")
