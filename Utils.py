#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/28 19:58
# @Author  : zdj
# @FileName: Utils.py
# @Software: PyCharm
import esm
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# Load the pretrained ESM2 model
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device)

def extract_embeddings_in_batches(sequences, batch_size=32):
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33])

        for j, (_, seq) in enumerate(batch_sequences):
            emb = results['representations'][33][j, 1:len(seq) + 1].mean(0)
            embeddings.append(emb.cpu().numpy())

        del batch_tokens, results
        torch.cuda.empty_cache()

    return embeddings

# 硬负样本挖掘
def select_hard_negatives(model, data_loader, k=2, confidence_threshold=0.5):
    """
    Select hard negative samples based on model predictions.

    Args:
        model (nn.Module): Current trained model
        data_loader (DataLoader): DataLoader containing samples for negative mining
        k (int): Number of hard negative samples to select per class
        confidence_threshold (float): Threshold for selecting hard negatives

    Returns:
        torch.Tensor: Indices of selected hard negative samples
    """
    model.eval()
    device = next(model.parameters()).device     # 获取模型的设备
    batch_size = data_loader.batch_size or 1     # 获取batch_size，若未设置，则默认为1

    all_scores = []
    all_indices = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
            try:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Get model predictions
                logits = model(batch_x)
                probs = torch.sigmoid(logits)

                # Find negative samples (where label is 0)
                negative_mask = (batch_y == 0)

                # Calculate scores for negative samples
                # Higher score means the model wrongly predicted positive with high confidence
                wrong_positive_probs = probs * negative_mask

                # Get maximum prediction score across all classes for each sample
                sample_scores = wrong_positive_probs.max(dim=1)[0]

                # Only keep samples above the confidence threshold
                valid_samples = sample_scores > confidence_threshold
                if valid_samples.any():
                    batch_indices = torch.arange(batch_x.size(0),
                                                 device=device) + batch_idx * batch_size

                    all_scores.append(sample_scores[valid_samples].cpu())
                    all_indices.append(batch_indices[valid_samples].cpu())
                    all_labels.append(batch_y[valid_samples].cpu())

            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

    if not all_scores:
        print("No hard negative samples found!")
        return torch.tensor([], dtype=torch.long)

    # Combine all batches
    all_scores = torch.cat(all_scores)
    all_indices = torch.cat(all_indices)
    all_labels = torch.cat(all_labels)

    # Select hard negatives for each class
    selected_indices = []
    n_classes = all_labels.size(1)

    for class_idx in range(n_classes):
        # Get negative samples for current class
        class_negative_mask = (all_labels[:, class_idx] == 0)
        class_scores = all_scores[class_negative_mask]
        class_indices = all_indices[class_negative_mask]

        if len(class_scores) > 0:
            # Select top k samples
            k_actual = min(k, len(class_scores))
            _, hard_neg_idx = torch.topk(class_scores, k_actual)
            selected_indices.append(class_indices[hard_neg_idx])

    # Combine and deduplicate selected indices
    if selected_indices:
        selected_indices = torch.cat(selected_indices)
        selected_indices = torch.unique(selected_indices)
        return selected_indices
    else:
        return torch.tensor([], dtype=torch.long)
