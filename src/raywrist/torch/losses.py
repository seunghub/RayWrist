"""Conditioning alignment at the native post-normalization, post-attention interface."""

import torch


def alignment_terms(student, teacher, valid_tokens):
    """Return squared-error sum and element count for one prefix-length group."""
    if student.ndim != 3 or teacher.shape != student.shape:
        raise ValueError("Conditioning must share [batch, tokens, width]")
    if valid_tokens.shape != student.shape[:2]:
        raise ValueError("Expected a native conditioning-token mask")
    mask = valid_tokens.bool()[..., None]
    difference = student.float() - teacher.detach().float()
    numerator = torch.where(mask, difference.square(), 0.0).sum()
    denominator = valid_tokens.bool().sum(dtype=torch.float32) * student.shape[-1]
    return numerator, denominator


def conditioning_alignment(student, teacher, valid_tokens):
    """Mean FP32 error over valid native tokens; the teacher is detached."""
    numerator, denominator = alignment_terms(student, teacher, valid_tokens)
    return numerator / denominator.clamp_min(1.0)


def grouped_conditioning_alignment(groups):
    """Combine unpadded prefix-length groups before normalization."""
    terms = [alignment_terms(*group) for group in groups]
    if not terms:
        raise ValueError("At least one conditioning group is required")
    return sum(n for n, _ in terms) / sum(d for _, d in terms).clamp_min(1.0)
