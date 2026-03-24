"""
把持姿勢・点群の可視化ユーティリティ

matplotlib を用いた3D可視化と、
open3d を用いたインタラクティブ可視化を提供する。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

# Windows で日本語フォントを設定
rcParams["font.family"] = "MS Gothic"

# 手のスケルトン定義 (23関節の接続関係)
# 0: 手首, 1-4: 人差し指, 5-8: 中指, 9-12: 薬指, 13-16: 小指, 17-22: 親指
HAND_SKELETON = [
    0, 1, 2, 3, 4, 18,
    0, 5, 6, 7, 19,
    0, 8, 9, 10, 20,
    0, 11, 12, 13, 21,
    0, 14, 15, 16, 17, 22
]


def draw_hand(hand: np.ndarray, ax, color: str = "orange"):
    """
    手のスケルトンを3Dプロットに描画する

    Args:
        hand:  (23, 3) 関節座標
        ax:    matplotlib 3D axes
        color: 描画色
    """
    hx, hy, hz = hand[:, 0], hand[:, 1], hand[:, 2]
    s = 0

    for i in range(4):
        if i == 0:
            for k in range(5):
                j = k
                x = np.array([hx[HAND_SKELETON[j]], hx[HAND_SKELETON[j + 1]]])
                y = np.array([hy[HAND_SKELETON[j]], hy[HAND_SKELETON[j + 1]]])
                z = np.array([hz[HAND_SKELETON[j]], hz[HAND_SKELETON[j + 1]]])
                ax.plot(x, y, z, c=color)
            s += 6
        if i == 3:
            for k in range(5):
                j = s + k
                x = np.array([hx[HAND_SKELETON[j]], hx[HAND_SKELETON[j + 1]]])
                y = np.array([hy[HAND_SKELETON[j]], hy[HAND_SKELETON[j + 1]]])
                z = np.array([hz[HAND_SKELETON[j]], hz[HAND_SKELETON[j + 1]]])
                ax.plot(x, y, z, c=color)
        else:
            for k in range(4):
                j = s + k
                x = np.array([hx[HAND_SKELETON[j]], hx[HAND_SKELETON[j + 1]]])
                y = np.array([hy[HAND_SKELETON[j]], hy[HAND_SKELETON[j + 1]]])
                z = np.array([hz[HAND_SKELETON[j]], hz[HAND_SKELETON[j + 1]]])
                ax.plot(x, y, z, c=color)
            s += 5


def draw_pointcloud(points: np.ndarray, ax, color: str = "green", size: float = 5.0):
    """
    点群を3Dプロットに描画する

    Args:
        points: (N, 3) 点群
        ax:     matplotlib 3D axes
        color:  点の色
        size:   点のサイズ
    """
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=size)


def draw_segmented_pointcloud(points: np.ndarray, labels: np.ndarray, ax):
    """
    セグメンテーション結果を色分けして3D表示する

    ラベル:
        0 (緑): 非接触領域
        1 (青): 右手接触領域
        2 (赤): 左手接触領域

    Args:
        points: (N, 3) 点群
        labels: (N,) ラベル配列 (0/1/2)
        ax:     matplotlib 3D axes
    """
    for label, color, desc in [(0, "green", "非接触"), (1, "blue", "右手"), (2, "red", "左手")]:
        mask = labels == label
        if mask.sum() > 0:
            ax.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                c=color, s=5, label=desc
            )


def visualize_grasp_result(
    points: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    labels: np.ndarray = None,
    title: str = "把持姿勢生成結果",
    block: bool = True,
):
    """
    点群と生成された把持姿勢を3D表示する

    Args:
        points:     (N, 3) 物体点群
        left_hand:  (23, 3) 左手関節座標
        right_hand: (23, 3) 右手関節座標
        labels:     (N,) セグメンテーションラベル (None の場合は単色表示)
        title:      ウィンドウタイトル
        block:      ブロッキング表示 (True: 閉じるまで停止)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title(title)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_axis_off()

    # 点群表示
    if labels is not None:
        draw_segmented_pointcloud(points, labels, ax)
    else:
        draw_pointcloud(points, ax, color="green")

    # 把持姿勢表示
    draw_hand(left_hand, ax, color="orange")   # 左手: オレンジ
    draw_hand(right_hand, ax, color="purple")  # 右手: 紫

    plt.tight_layout()
    plt.show(block=block)
    return fig, ax


def visualize_multiple_grasps(
    points: np.ndarray,
    grasp_results: list,
    labels: np.ndarray = None,
):
    """
    複数の把持候補を比較表示する

    Args:
        points:        (N, 3) 物体点群
        grasp_results: [(left_hand, right_hand), ...] のリスト
        labels:        (N,) セグメンテーションラベル
    """
    n = len(grasp_results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 6 * rows))
    plt.suptitle(f"把持姿勢候補 ({n} samples)", fontsize=14)

    for i, (left_hand, right_hand) in enumerate(grasp_results):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"Sample {i + 1}")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_axis_off()

        if labels is not None:
            draw_segmented_pointcloud(points, labels, ax)
        else:
            draw_pointcloud(points, ax, color="green")

        draw_hand(left_hand, ax, color="orange")
        draw_hand(right_hand, ax, color="purple")

    plt.tight_layout()
    plt.show(block=True)


def live_visualize_setup():
    """
    リアルタイム表示用のセットアップ (インタラクティブモード)

    Returns:
        (fig, ax) matplotlib オブジェクト
    """
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("RealSense → SAM3D → Shape2Gesture")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax


def live_visualize_update(
    fig,
    ax,
    points: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    labels: np.ndarray = None,
    title: str = "RealSense → SAM3D → Shape2Gesture",
):
    """
    リアルタイム表示を更新する

    Args:
        fig, ax:    live_visualize_setup() の返り値
        points:     (N, 3) 物体点群
        left_hand:  (23, 3) 左手
        right_hand: (23, 3) 右手
        labels:     (N,) セグメンテーションラベル
    """
    ax.cla()
    ax.set_title(title)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_axis_off()

    if labels is not None:
        draw_segmented_pointcloud(points, labels, ax)
    else:
        draw_pointcloud(points, ax, color="green")

    draw_hand(left_hand, ax, color="orange")
    draw_hand(right_hand, ax, color="purple")

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ============================================================
# カメラ画像への手形状投影
# ============================================================

import cv2 as _cv2
from utils.coord_transform import CameraIntrinsics, ObjectPose, normalized_to_camera, project_to_image

# 手スケルトン: Shape2Gestureの関節定義に基づく接続リスト
# handinf=[0,1,2,3,4,18,      親指
#          0,5,6,7,19,         人差し指
#          0,8,9,10,20,        中指
#          0,11,12,13,21,      薬指
#          0,14,15,16,17,22]   小指
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),  (4, 18),   # 親指
    (0, 5),  (5, 6),  (6, 7),  (7, 19),             # 人差し指
    (0, 8),  (8, 9),  (9, 10), (10, 20),            # 中指
    (0, 11), (11, 12),(12, 13),(13, 21),             # 薬指
    (0, 14), (14, 15),(15, 16),(16, 17),(17, 22),    # 小指
]


def project_hands_on_image(
    bgr: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    object_pose: ObjectPose,
    intrinsics: CameraIntrinsics,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Shape2Gesture の把持姿勢をカメラ座標系に変換してRGB画像に重ね描きする

    Args:
        bgr:          (H, W, 3) カメラ画像 (BGR)
        left_hand:    (23, 3) 左手関節位置 (正規化座標系)
        right_hand:   (23, 3) 右手関節位置 (正規化座標系)
        object_pose:  カメラ座標系でのオブジェクト姿勢
        intrinsics:   カメラ内部パラメータ
        alpha:        重ね描きの透明度 (0.0〜1.0)

    Returns:
        (H, W, 3) 手形状を重ね描きした画像
    """
    overlay = bgr.copy()

    def draw_hand_on_image(joints_norm, color_joint, color_bone):
        # 正規化座標 → カメラ座標 → 画像座標
        joints_cam = normalized_to_camera(joints_norm, object_pose)  # (23, 3)
        print(f"  [Proj] Z range: {joints_cam[:,2].min():.3f} ~ {joints_cam[:,2].max():.3f} m")
        joints_2d = project_to_image(joints_cam, intrinsics)          # (23, 2)
        print(f"  [Proj] u range: {joints_2d[:,0].min()} ~ {joints_2d[:,0].max()}")
        print(f"  [Proj] v range: {joints_2d[:,1].min()} ~ {joints_2d[:,1].max()}")
        h, w = bgr.shape[:2]

        # ボーン (骨格線) を描画
        for p_idx, c_idx in HAND_CONNECTIONS:
            p = joints_2d[p_idx]
            c = joints_2d[c_idx]
            # 画像内に収まっている場合のみ描画
            if (0 <= p[0] < w and 0 <= p[1] < h and
                    0 <= c[0] < w and 0 <= c[1] < h):
                _cv2.line(overlay, tuple(p), tuple(c), color_bone, 2, _cv2.LINE_AA)

        # 関節点を描画
        for j in joints_2d:
            if 0 <= j[0] < w and 0 <= j[1] < h:
                _cv2.circle(overlay, tuple(j), 4, color_joint, -1, _cv2.LINE_AA)

    draw_hand_on_image(left_hand,  (0, 0, 255), (0, 0, 200))     # 左手: 赤
    draw_hand_on_image(right_hand, (255, 0, 0), (200, 0, 0))     # 右手: 青

    # 半透明合成
    result = _cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    return result
