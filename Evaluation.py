#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 14:22
# @Author  : zdj
# @FileName: Evaluation.py
# @Software: PyCharm
import argparse
import logging
import os
import time

import pandas as pd
import torch

def p_at_k(y_true, y_score, k):
    """
    计算精确率@k

    Args:
        y_true (torch.Tensor): 真实标签
        y_score (torch.Tensor): 预测分数
        k (int): 前k个预测

    Returns:
        torch.Tensor: 精确率
    """
    # 获取前k个最高分预测的索引
    _, top_k_indices = torch.topk(y_score, k)

    # 选择这些索引对应的真实标签
    top_k_true = torch.gather(y_true, 0, top_k_indices)

    # 计算正类数量
    num_positive = torch.sum(top_k_true)

    # 计算精确率
    precision = num_positive / k

    return precision

def compute_precision(y_true, y_score, k):
    """
    计算所有样本的平均P@k

    Args:
        y_true (torch.Tensor): 真实标签矩阵
        y_score (torch.Tensor): 预测分数矩阵
        k (int): 前k个预测

    Returns:
        torch.Tensor: 平均精确率
    """
    device = y_true.device
    precision_scores = torch.tensor([p_at_k(y_true[i], y_score[i], k) for i in range(y_true.shape[0])], device=device)
    return torch.mean(precision_scores)

def ndcg_at_k(y_true, y_score, k):
    """
    计算归一化折损累积增益@k

    Args:
        y_true (torch.Tensor): 真实标签
        y_score (torch.Tensor): 预测分数
        k (int): 前k个预测

    Returns:
        torch.Tensor: nDCG@k 值
    """
    # 获取前k个最高分预测的索引
    _, top_k_indices = torch.topk(y_score, k)

    # 选择这些索引对应的真实标签
    top_k_true = torch.gather(y_true, 0, top_k_indices)

    # 计算DCG
    discounts = torch.log2(torch.arange(2, k + 2, device=y_true.device).float())
    dcg = torch.sum(top_k_true / discounts)

    # 计算理想DCG
    sorted_true, _ = torch.sort(y_true, descending=True)
    idcg = torch.sum(sorted_true[:k] / discounts)

    # 避免除零
    if idcg == 0:
        return torch.tensor(0.0, device=y_true.device)

    return dcg / idcg

def compute_ndcg(y_true, y_score, k):
    """
    计算所有样本的平均nDCG@k

    Args:
        y_true (torch.Tensor): 真实标签矩阵
        y_score (torch.Tensor): 预测分数矩阵
        k (int): 前k个预测

    Returns:
        torch.Tensor: 平均nDCG
    """
    device = y_true.device
    ndcg_scores = torch.tensor([ndcg_at_k(y_true[i], y_score[i], k) for i in range(y_true.shape[0])], device=device)
    return torch.mean(ndcg_scores)

def evaluate_predictions(args):
    """
    评估预测结果的P@k和nDCG@k，并保存到文件

    参数:
        pred_path (str): 预测结果文件路径
        true_path (str): 真实标签文件路径
        ks (list): 要计算的k值列表
        output_path (str): 评估结果输出文件路径
    """
    # 读取预测和真实标签文件
    if args.ks is None:
        args.ks = [1, 3, 5]
    try:
        pred_df = pd.read_csv(args.pred_path)
        true_df = pd.read_csv(args.true_path)
        if 'Sequences' not in pred_df.columns or 'Sequences' not in true_df.columns:
            raise ValueError("预测或真实标签文件缺少 'Sequences' 列")
    except Exception as e:
        logging.error(f"读取文件失败: {str(e)}")
        return

    # 确保序列一致
    common_sequences = pred_df['Sequences'].isin(true_df['Sequences'])
    pred_df = pred_df[common_sequences].reset_index(drop=True)
    true_df = true_df[true_df['Sequences'].isin(pred_df['Sequences'])].reset_index(drop=True)

    # 确保序列顺序一致
    pred_df = pred_df.sort_values('Sequences').reset_index(drop=True)
    true_df = true_df.sort_values('Sequences').reset_index(drop=True)

    # 提取标签列
    label_cols = [col for col in true_df.columns if col != 'Sequences']
    pred_cols = [col for col in pred_df.columns if col != 'Sequences']

    # 验证标签列一致性
    if set(label_cols) != set(pred_cols):
        logging.warning(f"警告: 预测和真实标签的列不一致")
        logging.info(f"真实标签: {set(label_cols)}")
        logging.info(f"预测标签: {set(pred_cols)}")
        return

    # 转换为张量
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    y_true = torch.tensor(true_df[label_cols].values, dtype=torch.float32, device=device)
    y_score = torch.tensor(pred_df[label_cols].values, dtype=torch.float32, device=device)

    # 计算P@k和nDCG@k
    results = []
    for k in args.ks:
        precision = compute_precision(y_true, y_score, k)
        ndcg = compute_ndcg(y_true, y_score, k)
        results.append(f"P@{k}: {precision:.3f}")
        results.append(f"nDCG@{k}: {ndcg:.3f}")
        print(f"P@{k}: {precision:.3f}")
        logging.info(f"P@{k}: {precision:.3f}")
        print(f"nDCG@{k}: {ndcg:.3f}")
        logging.info(f"nDCG@{k}: {ndcg:.3f}")

    # 保存结果到文件
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'a+') as f:
        f.write(f"\nESM2 + KMeans + Hard Negative Sampling: \n")
        f.write('\n'.join(results))
        f.write("\n")
    logging.info(f"评估结果已保存至: {args.output_path}")

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Evaluate predictions using P@k and nDCG@k metrics.")
    parser.add_argument('--pred_path', type=str, default="generate_data/merged_predictions.csv",
                        help='The path to the prediction results file.')
    parser.add_argument('--true_path', type=str, default="datasets/label_matrix.csv",
                        help='The path to the true labels file.')
    parser.add_argument('--output_path', type=str, default="generate_data/evaluation.txt",
                        help='The path to save the evaluation results file.')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5],
                        help='The list of k values for evaluation. Example: 1 3 5')
    parser.add_argument('--log_file', type=str, default="log_file/Evaluation.log",
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
    evaluate_predictions(args)
    end_time = time.time()
    logging.info(f"总运行时间: {end_time - start_time:.2f} 秒")