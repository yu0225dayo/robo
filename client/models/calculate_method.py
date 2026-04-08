"""
座標変換・損失計算ユーティリティ
- z_rotation_matrix: Z軸周り回転行列の生成
"""

import torch
import numpy as np


def z_rotation_matrix(angle_vec):
    """
    Z軸周りの回転行列を生成する (CPU版)

    Args:
        angle_vec: (B, N, 2) — (cosθ, sinθ) の正規化ベクトル

    Returns:
        R: (B, N, 3, 3) 回転行列
    """
    cos_theta = angle_vec[:, :, 0]
    sin_theta = angle_vec[:, :, 1]

    B, N, _ = angle_vec.shape

    R = torch.zeros((B, N, 3, 3))

    R[:, :, 0, 0] = cos_theta
    R[:, :, 0, 1] = -sin_theta
    R[:, :, 1, 0] = sin_theta
    R[:, :, 1, 1] = cos_theta
    R[:, :, 2, 2] = 1.0

    return R


def normalize_pointcloud(point_set: np.ndarray) -> np.ndarray:
    """
    点群を正規化する (中心化 + 単位球スケーリング)

    Args:
        point_set: (N, 3) numpy array

    Returns:
        正規化済み (N, 3) numpy array
    """
    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
    bbox_extents = point_set.max(axis=0) - point_set.min(axis=0)
    dist = np.max(bbox_extents) / 2.0
    point_set = point_set / dist
    return point_set
