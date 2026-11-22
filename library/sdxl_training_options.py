"""Local fp16 reductions for the optional SDXL training precision policy."""

import torch
from torch.nn import functional as F


def layer_norm(hidden, norm, strict):
    if strict and hidden.dtype == torch.float16:
        weight = norm.weight.float() if norm.weight is not None else None
        bias = norm.bias.float() if norm.bias is not None else None
        return F.layer_norm(hidden.float(), norm.normalized_shape, weight, bias, norm.eps).to(hidden.dtype)
    return norm(hidden)


def spatial_group_norm(hidden, norm, strict):
    if strict and hidden.dtype == torch.float16:
        weight = norm.weight.float() if norm.weight is not None else None
        bias = norm.bias.float() if norm.bias is not None else None
        return F.group_norm(hidden.float(), norm.num_groups, weight, bias, norm.eps).to(hidden.dtype)
    return norm(hidden)


def attention_softmax(scores, strict):
    if strict and scores.dtype == torch.float16:
        return scores.float().softmax(dim=-1).to(scores.dtype)
    return scores.softmax(dim=-1)
