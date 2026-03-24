"""
座標変換ユーティリティ

SAM-3D の正規化点群と SAM-6D の 6DoF pose を使って
Shape2Gesture の正規化座標系 → カメラ座標系 へ変換する。

座標変換の流れ:
    1. SAM-3D → reference mesh (正規化点群, unit sphere, center=0)
    2. SAM-6D → 6DoF pose (R, t): 物体座標系 → カメラ座標系
    3. scale 推定: 深度画像と正規化点群の bbox を比較
    4. p_cam = R @ (p_norm * scale) + t
    5. GraspGenerator → 把持姿勢 (正規化座標系)
    6. 把持姿勢も同じ変換でカメラ座標系へ
    7. カメラ座標 → 2D画素: u = fx * X/Z + cx
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraIntrinsics:
    """RealSense カメラ内部パラメータ"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480


@dataclass
class ObjectPose:
    """
    物体の姿勢情報

    SAM-6D が推定した 6DoF pose を格納する。
    R が None の場合は回転なし (中心+スケールのみ) として扱う。

    center_3d: カメラ座標系での物体中心 [m] (3,)  ← SAM-6D の t と同義
    scale:     正規化座標系 → メートルのスケール係数
    R:         (3,3) 回転行列 (物体座標系→カメラ座標系), None=単位行列
    """
    center_3d: np.ndarray          # (3,)
    scale: float = 1.0
    R: Optional[np.ndarray] = None  # (3,3) or None


def normalized_to_camera(
    pts_normalized: np.ndarray,
    pose: ObjectPose,
) -> np.ndarray:
    """
    正規化座標系の点をカメラ座標系に変換する

    R あり (6DoF): p_cam = R @ (p_norm * scale) + center_3d
    R なし (中心のみ): p_cam = p_norm * scale + center_3d

    Args:
        pts_normalized: (N, 3) 正規化座標系の点
        pose:           ObjectPose

    Returns:
        (N, 3) カメラ座標系の点 [メートル]
    """
    scaled = pts_normalized * pose.scale
    if pose.R is not None:
        return (pose.R @ scaled.T).T + pose.center_3d
    return scaled + pose.center_3d


def estimate_scale_from_depth(
    depth: np.ndarray,
    mask_u: int,
    mask_v: int,
    intrinsics: CameraIntrinsics,
    mesh_pts: np.ndarray,
    search_radius: int = 50,
) -> float:
    """
    深度画像と SAM-3D 正規化点群の bbox を比較してスケールを推定する

    Args:
        depth:          (H, W) float32 深度画像 [m]
        mask_u, mask_v: SAM マスク重心ピクセル
        intrinsics:     カメラ内部パラメータ
        mesh_pts:       (N, 3) 正規化点群 (unit sphere)
        search_radius:  探索範囲 [px]

    Returns:
        scale [m]
    """
    h, w = depth.shape
    v0 = max(0, mask_v - search_radius)
    v1 = min(h, mask_v + search_radius)
    u0 = max(0, mask_u - search_radius)
    u1 = min(w, mask_u + search_radius)

    patch = depth[v0:v1, u0:u1]
    ys, xs = np.where(patch > 0.01)
    if len(ys) < 10:
        return 0.15  # fallback

    zs = patch[ys, xs]
    Xs = (xs + u0 - intrinsics.cx) / intrinsics.fx * zs
    Ys = (ys + v0 - intrinsics.cy) / intrinsics.fy * zs
    pts_3d = np.stack([Xs, Ys, zs], axis=-1)

    center = pts_3d.mean(axis=0)
    real_radius = float(np.percentile(
        np.linalg.norm(pts_3d - center, axis=1), 90
    ))

    norm_radius = float(np.max(
        np.linalg.norm(mesh_pts - mesh_pts.mean(0), axis=1)
    ))
    if norm_radius < 1e-6:
        return real_radius

    scale = real_radius / norm_radius
    print(f"[CoordTransform] scale={scale:.4f} m  "
          f"(実半径={real_radius:.3f} m, 正規化半径={norm_radius:.3f})")
    return scale


def project_to_image(
    points_3d: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    カメラ座標系の3D点を画像ピクセル座標に投影する

    Returns:
        (N, 2) int32 ピクセル座標 (u, v)
    """
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    valid = Z > 0.01
    u = np.where(valid, intrinsics.fx * X / Z + intrinsics.cx, -1)
    v = np.where(valid, intrinsics.fy * Y / Z + intrinsics.cy, -1)
    return np.stack([u, v], axis=-1).astype(np.int32)
