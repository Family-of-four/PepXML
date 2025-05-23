#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/22 10:33
# @Author  : zdj
# @FileName: Node2Vec.py
# @Software: PyCharm
import argparse
import logging
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import warnings
import time
import random

warnings.filterwarnings('ignore')


def build_label_co_occurrence_graph(label_matrix_path):
    df = pd.read_csv(label_matrix_path)

    labels = df.columns[1:].tolist()

    G = nx.Graph()

    G.add_nodes_from(labels)

    for _, row in df.iterrows():
        present_labels = [label for label in labels if row[label] == 1]
        for i in range(len(present_labels)):
            for j in range(i + 1, len(present_labels)):
                G.add_edge(present_labels[i], present_labels[j])

    return G


def node2vec_embedding(G, dimensions=100, walk_length=5, num_walks=100, workers=4, seed=42):
    if G is None or G.number_of_nodes() == 0:
        raise ValueError("If G is None or G is empty, please check the input graph. ")
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, workers=workers)

    model = node2vec.fit()

    embeddings = {}
    for label in G.nodes():
        if label in model.wv:
            embeddings[label] = model.wv[label]
        else:
            embeddings[label] = np.zeros(dimensions)

    embedding_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embedding_df.index.name = 'Label'
    embedding_df.reset_index(inplace=True)

    return embedding_df


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.info("1. 开始构建标签共现图...")
    G = build_label_co_occurrence_graph(args.label_matrix_path)

    logging.info(f"图信息：节点数 {G.number_of_nodes()}，边数 {G.number_of_edges()}")

    logging.info("2. 生成Node2Vec嵌入...")
    embeddings = node2vec_embedding(G,
                                    dimensions=args.dimensions,
                                    walk_length=args.walk_length,
                                    num_walks=args.num_walks,
                                    workers=args.workers,
                                    seed=args.seed)

    logging.info("3. 保存嵌入结果...")
    embeddings.to_csv(args.output_path, index=False)

    logging.info(f"嵌入已保存到 {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Node2Vec for Label Co-occurrence Graph")
    parser.add_argument('--label_matrix_path', type=str, default='datasets/label_matrix.csv',
                        help='Path to the peptide label matrix CSV file')
    parser.add_argument('--output_path', type=str, default='generate_data/label_embeddings.csv',
                        help='Path to save the label embeddings CSV file')
    parser.add_argument('--dimensions', type=int, default=100,
                        help='Dimensions of the embedding')
    parser.add_argument('--walk_length', type=int, default=5,
                        help='Length of each random walk')
    parser.add_argument('--num_walks', type=int, default=100,
                        help='Number of walks per node')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_file', type=str, default='log_file/node2vec.log',
                        help='Path to save the log file')
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
    logging.info(f"程序运行时间: {run_time:.2f}秒")
