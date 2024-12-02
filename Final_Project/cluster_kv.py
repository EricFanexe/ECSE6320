import math
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from torch.nn.functional import cosine_similarity
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )
    # (BS, head, query, chunks)
    # print(f"Selected top-k chunks (pages): {topk}")
    topk_return = topk
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom, topk_return


def cosine_similarity_matrix(keys, centers):
    """矩阵化计算余弦相似度"""
    keys_norm = np.linalg.norm(keys, axis=1, keepdims=True)
    centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
    similarity = np.dot(keys, centers.T) / (keys_norm * centers_norm.T)
    return similarity

def linear_initialization(keys, num_clusters):
    """
    线性初始化聚类中心
    :param keys: 待聚类的 key 向量
    :param num_clusters: 需要的聚类中心数量
    :return: 初始化的聚类中心
    """
    min_key = np.min(keys, axis=0)  # 找到每个维度的最小值
    max_key = np.max(keys, axis=0)  # 找到每个维度的最大值
    centers = np.linspace(min_key, max_key, num_clusters)  # 在线性区间内生成 num_clusters 个中心
    return centers


def balanced_cosine_kmeans_linear(kv_pairs, num_pages, page_size=16, max_iter=100):
    """
    基于余弦相似度的平衡 K-Means 聚类，使用 Linear Initialization 方法初始化聚类中心。
    """
    keys = kv_pairs[:, 0]
    values = kv_pairs[:, 1]

    # 使用线性初始化
    centers = linear_initialization(keys, num_pages)

    for _ in range(max_iter):
        # 矩阵化计算所有 keys 和 centers 的相似度
        similarities = cosine_similarity_matrix(keys, centers)
        cluster_indices = np.argmax(similarities, axis=1)

        # 按索引分配到簇中
        clusters = {i: [] for i in range(num_pages)}
        for idx, cluster_id in enumerate(cluster_indices):
            clusters[cluster_id].append((keys[idx], values[idx]))

        # 平衡簇大小
        for cluster_id in range(num_pages):
            while len(clusters[cluster_id]) > page_size:
                excess_key, excess_value = clusters[cluster_id].pop()
                other_clusters = [i for i in range(num_pages) if i != cluster_id]
                best_cluster = min(other_clusters, key=lambda x: len(clusters[x]))
                clusters[best_cluster].append((excess_key, excess_value))

        # 更新中心
        new_centers = np.array([np.mean([pair[0] for pair in clusters[k]], axis=0) for k in range(num_pages)])
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return clusters



# 使用平衡 K-Means 聚类后的 KV pairs
def cluster_keys_into_pages(keys, values, num_pages, b, h, page_size=16):
    batch_size, num_heads, seq_len, head_dim = keys.shape
    keys_to_cluster = keys[b, h, :, :].reshape(seq_len, head_dim)
    values_to_cluster = values[b, h, :, :].reshape(seq_len, head_dim)

    # 将 keys 和 values 配对成 KV pairs
    kv_pairs = np.array(list(zip(keys_to_cluster, values_to_cluster)))

    # 使用平衡余弦 K-Means 聚类
    final_pages = balanced_cosine_kmeans_linear(kv_pairs, num_pages, page_size)

    # 将每个页面的数据按顺序重新排列
    rearranged_keys, rearranged_values = [], []
    for page_kv_pairs in final_pages.values():
        rearranged_keys.extend([pair[0] for pair in page_kv_pairs])
        rearranged_values.extend([pair[1] for pair in page_kv_pairs])

    rearranged_keys = np.array(rearranged_keys).reshape(seq_len, head_dim)
    rearranged_values = np.array(rearranged_values).reshape(seq_len, head_dim)
    return rearranged_keys, rearranged_values


def calculate_cosine_similarity_in_page(key_states, value_states, chunk_size, layer_id):
    seq_len, head_dim = key_states.shape[2], key_states.shape[-1]
    key_states_np = key_states.cpu().numpy()
    value_states_np = value_states.cpu().numpy()

    num_pages = (seq_len + chunk_size - 1) // chunk_size  # 计算需要多少个 pages
    batch_size, num_heads, seq_len, head_dim = key_states_np.shape
    rearranged_keys, rearranged_values = [], []
    for h in range(key_states_np.shape[1]):
        final_keys, final_values = cluster_keys_into_pages(
            key_states_np, value_states_np, num_pages, batch_size-1, h, chunk_size
        )

        clustered_key_states = torch.tensor(final_keys, device=key_states.device).unsqueeze(0)
        clustered_value_states = torch.tensor(final_values, device=value_states.device).unsqueeze(0)
        rearranged_keys.append(clustered_key_states)
        rearranged_values.append(clustered_value_states)

    rearranged_keys_tensor = torch.stack(rearranged_keys, dim=1)
    rearranged_values_tensor = torch.stack(rearranged_values, dim=1)
    return rearranged_keys_tensor, rearranged_values_tensor


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 调用计算每个 page 内 keys 相似度的函数
    rearranged_key_states, rearranged_value_states = calculate_cosine_similarity_in_page(key_states, value_states, self.chunk_size, self.layer_id)

    # 确保 key_states 和 query_states 的数据类型一致
    key_states = rearranged_key_states.to(query_states.dtype)
    value_states = rearranged_value_states.to(value_states.dtype)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    postive_query = query_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )
    

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        quantized_weight = quantized_weight + attention_mask
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )

    token_budget = min(kv_seq_len, self.token_budget)

    attn_weights_for_selection = quantized_weight
    

    if token_budget > 0:
        mask_bottom, selected_pages = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, self.chunk_size
        )  # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

    mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


global layer_id
layer_id = 32


def enable_quest_attention_eval(model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_eval(
                module,
                args,
            )

        global layer_id
        if isinstance(module, LlamaAttention):
            # For longchat model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                forward, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
        elif module.__class__.__name__ == "LlamaAttention":
            # For yarn model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                forward_yarn, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
