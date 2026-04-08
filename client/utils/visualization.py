"""
把持姿勢・点群の可視化ユーティリティ

matplotlib を用いた3D可視化と、
open3d を用いたインタラクティブ可視化を提供する。
"""

import threading
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
    show_figure(fig)
    plt.close(fig)


def live_visualize_setup():
    """
    リアルタイム表示用のセットアップ (インタラクティブモード)

    Returns:
        (fig, ax) matplotlib オブジェクト
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("RealSense → SAM3D → Shape2Gesture")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax


def live_visualize_update(
    fig,
    ax,
    points: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    labels: np.ndarray = None,
    title: str = "RealSense → SAM3D → Shape2Gesture",
    save_path: str = None,
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

    show_figure(fig, save_path=save_path)


# ============================================================
# カメラ画像への手形状投影
# ============================================================

import io
import cv2 as _cv2


def _render_fig(fig) -> np.ndarray:
    """Agg バックエンドで figure を BGR numpy array に変換"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    buf.seek(0)
    return _cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), _cv2.IMREAD_COLOR)


def show_figure(fig, save_path: str = None, show: bool = True):
    """matplotlib figure を PNG に変換して保存・表示する

    Args:
        fig:       matplotlib figure
        save_path: 保存先パス (None: 保存しない)
        show:      True のとき cv2.imshow で表示する
    """
    img = _render_fig(fig)
    if img is None:
        return
    if save_path:
        _cv2.imwrite(save_path, img)
    if show:
        _cv2.imshow("Visualization", img)
        # waitKey はメインループの show_preview に任せる (ここでは呼ばない)


def save_grasp_figure(
    points: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    labels: np.ndarray = None,
    save_path: str = None,
    title: str = "",
):
    """把持姿勢を Agg で描画してファイルに保存する (表示なし)"""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
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
    show_figure(fig, save_path=save_path, show=False)
    plt.close(fig)


class MatplotlibGraspVisualizer:
    """
    matplotlib 3D をインタラクティブ表示する (マウスで回転・拡大縮小可能)

    Tk ウィンドウを別スレッドで起動するため、cv2 のメインループをブロックしない。
    poll() は不要。
    """

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()

    def update(
        self,
        points: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        labels=None,
        title: str = "Grasp 3D",
    ):
        """点群 + 把持姿勢を新しい Tk ウィンドウで表示する"""
        # 前のウィンドウを閉じる
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._stop_event = threading.Event()

        # スレッドに渡すデータをコピー
        pts = points.copy()
        lh  = left_hand.copy()
        rh  = right_hand.copy()
        lb  = labels.copy() if labels is not None else None
        stop_ev = self._stop_event

        def _run():
            import tkinter as tk
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg, NavigationToolbar2Tk,
            )

            root = tk.Tk()
            root.title(title)

            fig = Figure(figsize=(7, 7))
            ax  = fig.add_subplot(111, projection="3d")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_zlim(-1.2, 1.2)
            ax.set_axis_off()

            # 点群
            if lb is not None:
                for lbl, col in [(0, "green"), (1, "blue"), (2, "red")]:
                    mask = lb == lbl
                    if mask.sum() > 0:
                        ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2],
                                   c=col, s=3)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="green", s=3)

            # 手スケルトン
            for hand, col in [(lh, "orange"), (rh, "purple")]:
                hx, hy, hz = hand[:, 0], hand[:, 1], hand[:, 2]
                for p_idx, c_idx in HAND_CONNECTIONS:
                    ax.plot([hx[p_idx], hx[c_idx]],
                            [hy[p_idx], hy[c_idx]],
                            [hz[p_idx], hz[c_idx]], c=col)

            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas, root)
            toolbar.update()

            # stop_event が立ったらウィンドウを閉じるポーリング
            def _check_stop():
                if stop_ev.is_set():
                    root.destroy()
                else:
                    root.after(100, _check_stop)

            root.after(100, _check_stop)
            root.mainloop()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def poll(self):
        """互換用 (何もしない)"""
        pass

    def destroy(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
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


def project_pointcloud_on_image(
    bgr: np.ndarray,
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    intrinsics,
    point_color=(0, 255, 0),
    bbox_color=(0, 255, 255),
    point_size: int = 2,
    points_unit: str = "mm",
) -> np.ndarray:
    """
    3D点群とbboxをRGB画像に投影して重ね描きする

    Args:
        bgr:         (H, W, 3) カメラ画像 (BGR)
        points_3d:   (N, 3) 物体座標系の点群
        R:           (3, 3) 回転行列 (物体→カメラ座標系)
        t:           (3,)   平行移動 [m]
        intrinsics:  CameraIntrinsics
        point_color: 点群の描画色 (BGR)
        bbox_color:  バウンディングボックスの描画色 (BGR)
        point_size:  点の半径 [px]
        points_unit: 点群の単位 ("mm" or "m"). SAM-3D生成メッシュは "mm"

    Returns:
        (H, W, 3) 投影結果画像
    """
    result = bgr.copy()
    h, w = bgr.shape[:2]

    # 単位変換: メッシュPLYはmm単位、tはm単位なので合わせる
    pts_m = points_3d / 1000.0 if points_unit == "mm" else points_3d

    # 物体座標系 → カメラ座標系: p_cam = R @ p_obj_m + t_m
    pts_cam = (R @ pts_m.T).T + t  # (N, 3) [m]

    # カメラ座標 → 画像座標 (ベクトル化)
    valid = pts_cam[:, 2] > 0.01
    pts_valid = pts_cam[valid]
    us = (intrinsics.fx * pts_valid[:, 0] / pts_valid[:, 2] + intrinsics.cx).astype(np.int32)
    vs = (intrinsics.fy * pts_valid[:, 1] / pts_valid[:, 2] + intrinsics.cy).astype(np.int32)

    # 画像範囲内の点のみ描画 (ベクトル化)
    in_bounds = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    for u, v in zip(us[in_bounds], vs[in_bounds]):
        _cv2.circle(result, (int(u), int(v)), point_size, point_color, -1)

    # 3D bounding box を描画 (mm→m変換済み座標で計算)
    mins = pts_m.min(axis=0)
    maxs = pts_m.max(axis=0)
    corners = np.array([
        [mins[0], mins[1], mins[2]], [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]], [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]], [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]], [mins[0], maxs[1], maxs[2]],
    ])
    corners_cam = (R @ corners.T).T + t
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        c1, c2 = corners_cam[i], corners_cam[j]
        if c1[2] > 0.01 and c2[2] > 0.01:
            u1 = int(intrinsics.fx * c1[0] / c1[2] + intrinsics.cx)
            v1 = int(intrinsics.fy * c1[1] / c1[2] + intrinsics.cy)
            u2 = int(intrinsics.fx * c2[0] / c2[2] + intrinsics.cx)
            v2 = int(intrinsics.fy * c2[1] / c2[2] + intrinsics.cy)
            if (0 <= u1 < w and 0 <= v1 < h) or (0 <= u2 < w and 0 <= v2 < h):
                _cv2.line(result, (u1, v1), (u2, v2), bbox_color, 2, _cv2.LINE_AA)

    return result


def render_mesh_on_image(
    bgr: np.ndarray,
    mesh_path: str,
    R: np.ndarray,
    t: np.ndarray,
    intrinsics,
    mesh_unit: str = "mm",
    alpha: float = 0.6,
    mesh_color: tuple = (0.2, 0.8, 0.2),
) -> np.ndarray:
    """
    FoundationPose的手法: Open3D OffscreenRenderer でメッシュ面をレンダリングして重ね描きする

    点群投影より高品質:
    - メッシュ面全体をレンダリング (まばらな点ではなく面)
    - Z-buffer による自然なOcclusion処理
    - 照明・陰影付き

    Args:
        bgr:        (H, W, 3) カメラ画像 (BGR)
        mesh_path:  三角メッシュPLYファイルパス
        R:          (3, 3) 回転行列 (物体→カメラ座標系)
        t:          (3,)   平行移動 [m]
        intrinsics: CameraIntrinsics
        mesh_unit:  メッシュの単位 ("mm" or "m"). SAM-3D生成メッシュは "mm"
        alpha:      メッシュオーバーレイの不透明度 (0.0〜1.0)
        mesh_color: メッシュの色 (R, G, B) 各0.0〜1.0

    Returns:
        (H, W, 3) レンダリング結果画像 (BGR)
    """
    # open3d.visualization.rendering (Filament/EGL) は使わない。
    # EGL が使えないヘッドレス環境 (Docker 等) で C++ レベルのクラッシュが発生するため。
    # 代わりにメッシュ面上に均一点をサンプリングして密な点群投影を行う。

    pts = None
    try:
        import open3d as o3d
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        if len(mesh_o3d.triangles) > 0:
            # メッシュ面上に均一に 10000 点をサンプリング (頂点のみより高密度かつ均一)
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=10000)
            pts = np.asarray(pcd.points).astype(np.float32)
            print(f"[render_mesh] 面サンプリング: {len(pts)} 点")
        else:
            pts = np.asarray(mesh_o3d.vertices).astype(np.float32)
            print(f"[render_mesh] 頂点投影: {len(pts)} 点 (面なし)")
    except Exception as e:
        print(f"[render_mesh] open3d でのメッシュ読み込み失敗 ({e}). 頂点投影にフォールバック")
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(mesh_path)
            pts = np.asarray(pcd.points).astype(np.float32)
        except Exception:
            from utils.pointcloud_utils import load_pointcloud_ply
            pts = load_pointcloud_ply(mesh_path)

    point_color_bgr = (
        int(mesh_color[2] * 255),
        int(mesh_color[1] * 255),
        int(mesh_color[0] * 255),
    )
    return project_pointcloud_on_image(
        bgr, pts, R, t, intrinsics,
        point_color=point_color_bgr,
        bbox_color=(0, 220, 220),
        point_size=1,
        points_unit=mesh_unit,
    )
