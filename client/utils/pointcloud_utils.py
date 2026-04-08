"""
点群処理ユーティリティ

PLY/PCD/CSVファイルの読み書き、前処理、変換など
実世界データの取り扱いに関するユーティリティを提供する。
"""

import numpy as np
import os
from typing import Optional


def load_pointcloud_csv(filepath: str) -> np.ndarray:
    """
    CSV形式の点群ファイルを読み込む

    Args:
        filepath: CSVファイルパス (各行が "x,y,z" 形式、ヘッダーなし)

    Returns:
        (N, 3) float32 numpy array
    """
    import pandas as pd
    data = pd.read_csv(filepath, header=None).values.astype(np.float32)
    print(f"[PointCloud] CSV読み込み: {filepath} ({len(data)} points)")
    return data


def load_pointcloud_ply(filepath: str, target_points: Optional[int] = None) -> np.ndarray:
    """
    PLY形式の点群ファイルを読み込む

    Args:
        filepath:      PLYファイルパス
        target_points: ダウンサンプリング後の点数 (None: 全点)

    Returns:
        (N, 3) float32 numpy array
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points).astype(np.float32)
    except ImportError:
        from plyfile import PlyData
        ply_data = PlyData.read(filepath)
        vertex = ply_data["vertex"]
        points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    print(f"[PointCloud] PLY読み込み: {filepath} ({len(points)} points)")

    if target_points is not None and len(points) > target_points:
        choice = np.random.choice(len(points), target_points, replace=False)
        points = points[choice]
        print(f"[PointCloud] ダウンサンプリング: {target_points} points")

    return points


def save_pointcloud_csv(points: np.ndarray, filepath: str):
    """
    点群をCSV形式で保存する

    Args:
        points:   (N, 3) numpy array
        filepath: 保存先CSVファイルパス
    """
    import pandas as pd
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    pd.DataFrame(points).to_csv(filepath, header=False, index=False)
    print(f"[PointCloud] CSV保存: {filepath} ({len(points)} points)")


def save_pointcloud_ply(points: np.ndarray, filepath: str):
    """
    点群をPLY形式で保存する

    Args:
        points:   (N, 3) numpy array
        filepath: 保存先PLYファイルパス
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"[PointCloud] PLY保存: {filepath} ({len(points)} points)")


def normalize_pointcloud(points: np.ndarray) -> np.ndarray:
    """
    点群を正規化する (中心化 + 単位球スケーリング)

    Shape2Gestureモデルへの入力前処理として使用。

    Args:
        points: (N, 3) numpy array

    Returns:
        正規化済み (N, 3) numpy array
    """
    centered = points - np.expand_dims(np.mean(points, axis=0), 0)
    bbox_extents = centered.max(axis=0) - centered.min(axis=0)
    dist = np.max(bbox_extents) / 2.0
    return (centered / dist).astype(np.float32)


def resample_pointcloud(points: np.ndarray, target_num: int) -> np.ndarray:
    """
    点群を指定点数にリサンプリングする

    Args:
        points:     (N, 3) numpy array
        target_num: リサンプリング後の点数

    Returns:
        (target_num, 3) numpy array
    """
    n = len(points)
    choice = np.random.choice(n, target_num, replace=(n < target_num))
    return points[choice]


def estimate_object_scale(points: np.ndarray) -> tuple:
    """
    点群から物体の実スケールと中心を推定する

    正規化前の点群 (メートル単位) を使って、
    把持姿勢を実世界座標系に変換するために使用。

    Args:
        points: (N, 3) numpy array [メートル単位]

    Returns:
        (scale, center): scale=最大半径 [m], center=(3,) 中心座標 [m]
    """
    center = np.mean(points, axis=0)
    centered = points - center
    scale = np.max(np.sqrt(np.sum(centered ** 2, axis=1)))
    return float(scale), center.astype(np.float32)


def filter_depth_range(points: np.ndarray, z_min: float = 0.1, z_max: float = 2.0) -> np.ndarray:
    """
    深度範囲でフィルタリングする

    Args:
        points: (N, 3) numpy array
        z_min:  最小深度 [メートル]
        z_max:  最大深度 [メートル]

    Returns:
        フィルタリング済み (M, 3) numpy array
    """
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    filtered = points[mask]
    print(f"[PointCloud] 深度フィルタ ({z_min:.2f}m - {z_max:.2f}m): "
          f"{len(points)} → {len(filtered)} points")
    return filtered


def remove_statistical_outliers(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """
    統計的外れ値除去 (Statistical Outlier Removal)

    Args:
        points:       (N, 3) numpy array
        nb_neighbors: 近傍点数
        std_ratio:    標準偏差倍率

    Returns:
        外れ値除去後の点群
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        filtered = points[ind]
        print(f"[PointCloud] 外れ値除去: {len(points)} → {len(filtered)} points")
        return filtered
    except ImportError:
        print("[PointCloud] open3d が見つかりません。外れ値除去をスキップします。")
        return points
