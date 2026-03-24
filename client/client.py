"""
ローカルPC クライアント

RGB + 深度画像を取得して計算機サーバへ送り、
6DoF 姿勢推定結果を受け取り、把持姿勢を画像に可視化する。

依存パッケージ:
    pip install requests opencv-python numpy scipy

カメラ:
    - RealSense D4xx/L5xx (pyrealsense2 がある場合)
    - ファイル入力 (--rgb / --depth で指定)

使用例:
    # RealSense からリアルタイム取得
    python client.py --server http://<計算機IP>:8080 --realsense

    # ファイルから
    python client.py --server http://<計算機IP>:8080 \
        --rgb rgb.png --depth depth_mm.png \
        --fx 615.0 --fy 615.0 --cx 320.0 --cy 240.0
"""

import argparse
import json
import sys
import time
import numpy as np
import cv2
import requests
from pathlib import Path
from typing import Optional, Tuple

# ===== カメラ取得 =====

def capture_realsense() -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    RealSense から RGB + 深度フレームを1枚取得する

    Returns:
        rgb      : (H, W, 3) uint8
        depth_mm : (H, W) uint16  [mm単位]
        intrinsics: {"fx","fy","cx","cy"}
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[ERROR] pyrealsense2 がインストールされていません。"
              "  pip install pyrealsense2")
        sys.exit(1)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    # 自動露出安定のため数フレーム捨てる
    for _ in range(30):
        pipeline.wait_for_frames()

    align = rs.align(rs.stream.color)
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    pipeline.stop()

    bgr = np.asanyarray(color_frame.get_data())
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
    return rgb, depth_mm, intrinsics


def load_from_files(rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """ファイルから RGB + 深度を読み込む"""
    bgr = cv2.imread(rgb_path)
    if bgr is None:
        raise FileNotFoundError(f"RGB 画像が見つかりません: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(f"深度画像が見つかりません: {depth_path}")
    return rgb, depth_mm.astype(np.uint16)


# ===== サーバ通信 =====

def call_full_pipeline(
    server_url: str,
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: dict,
    click_xy: Optional[Tuple[int, int]] = None,
    det_score_thresh: float = 0.2,
    timeout: float = 600.0,
) -> dict:
    """
    /full_pipeline エンドポイントを呼び出す

    Returns:
        dict with keys: points, ply_path, R, t, mask_center_u, mask_center_v, ...
    """
    # RGB → PNG エンコード
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, rgb_buf  = cv2.imencode(".png", bgr)
    _, dep_buf  = cv2.imencode(".png", depth_mm)

    click_x, click_y = click_xy if click_xy else (-1, -1)

    files = {
        "image": ("rgb.png",   rgb_buf.tobytes(), "image/png"),
        "depth": ("depth.png", dep_buf.tobytes(), "image/png"),
    }
    data = {
        "intrinsics_json": json.dumps(intrinsics),
        "click_x":         click_x,
        "click_y":         click_y,
        "det_score_thresh": det_score_thresh,
    }

    url = f"{server_url.rstrip('/')}/full_pipeline"
    print(f"[Client] POST {url} ...")
    t0 = time.time()
    resp = requests.post(url, files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    print(f"[Client] 完了 ({time.time()-t0:.1f}s)")
    return resp.json()


# ===== 把持姿勢の生成と可視化 =====

def generate_grasp_candidates(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    n_grasps: int = 5,
) -> list:
    """
    点群に対して単純な把持候補を生成する (z軸方向への接近を仮定)

    Args:
        points : (N, 3) 物体モデル点群 (物体座標系)
        R      : (3, 3) 物体→カメラ座標系 回転行列
        t      : (3,)   物体→カメラ座標系 平行移動 [m]
        n_grasps: 生成する把持候補数

    Returns:
        list of dict:
            "center_cam" : カメラ座標系での把持中心 (3,)
            "approach_cam": 接近方向ベクトル (3,)
    """
    # 物体モデルの PCA 主軸を把持軸の候補とする
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes = eigvecs[:, ::-1]  # 分散大順

    grasps = []
    for i in range(min(n_grasps, 3)):
        axis = axes[:, i % 3]
        for sign in [1.0, -1.0]:
            approach_obj = sign * axis          # 物体座標系での接近方向
            approach_cam = R @ approach_obj     # カメラ座標系へ変換
            center_cam   = R @ centroid + t     # 把持中心 (カメラ座標系)
            grasps.append({
                "center_cam":  center_cam,
                "approach_cam": approach_cam,
                "axis_index":  i,
            })
    return grasps[:n_grasps]


def project_point(pt_cam: np.ndarray, K: np.ndarray) -> Tuple[int, int]:
    """カメラ座標系の 3D 点を画像座標へ射影"""
    x, y, z = pt_cam
    if z <= 0:
        return None
    u = int(K[0, 0] * x / z + K[0, 2])
    v = int(K[1, 1] * y / z + K[1, 2])
    return (u, v)


def draw_pose_and_grasps(
    rgb: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    intrinsics: dict,
    points: Optional[np.ndarray] = None,
    axis_len: float = 0.05,
) -> np.ndarray:
    """
    画像に物体座標系の軸と把持候補を描画する

    Args:
        rgb        : (H, W, 3) RGB 画像
        R, t       : 姿勢 (物体→カメラ)
        intrinsics : {"fx","fy","cx","cy"}
        points     : (N, 3) 物体モデル点群 (あれば把持候補も描画)
        axis_len   : 座標軸の長さ [m]

    Returns:
        描画済み BGR 画像
    """
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # 物体原点をカメラ座標系へ
    origin_cam = t.copy()
    origin_px = project_point(origin_cam, K)

    if origin_px is None:
        print("[Vis] 物体が視野外です")
        return vis

    # X / Y / Z 軸 (それぞれ赤 / 緑 / 青)
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    axis_dirs   = np.eye(3)   # 物体座標系の単位ベクトル
    for i, (color, d_obj) in enumerate(zip(axis_colors, axis_dirs)):
        tip_cam = R @ (d_obj * axis_len) + t
        tip_px  = project_point(tip_cam, K)
        if tip_px:
            cv2.arrowedLine(vis, origin_px, tip_px, color, 2, tipLength=0.3)
            label = ["X", "Y", "Z"][i]
            cv2.putText(vis, label, (tip_px[0]+4, tip_px[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 原点マーカー
    cv2.circle(vis, origin_px, 5, (255, 255, 0), -1)

    # 把持候補 (点群がある場合)
    if points is not None:
        grasps = generate_grasp_candidates(points, R, t)
        for g in grasps:
            c_px = project_point(g["center_cam"], K)
            if c_px is None:
                continue
            gripper_len = 0.04  # グリッパー幅の半分 [m]
            tip_a = project_point(g["center_cam"] + g["approach_cam"] * gripper_len, K)
            tip_b = project_point(g["center_cam"] - g["approach_cam"] * gripper_len, K)
            cv2.circle(vis, c_px, 6, (0, 255, 255), 2)
            if tip_a:
                cv2.line(vis, c_px, tip_a, (0, 200, 200), 2)
            if tip_b:
                cv2.line(vis, c_px, tip_b, (0, 200, 200), 2)

    # 姿勢情報テキスト
    t_mm = t * 1000
    info = f"t=[{t_mm[0]:.1f},{t_mm[1]:.1f},{t_mm[2]:.1f}]mm"
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis


# ===== メイン =====

def main():
    parser = argparse.ArgumentParser(description="SAM-3D + SAM-6D クライアント")
    parser.add_argument("--server", default="http://localhost:8080",
                        help="計算機サーバの URL")
    # カメラ入力
    parser.add_argument("--realsense", action="store_true",
                        help="RealSense カメラを使用")
    parser.add_argument("--rgb",   default=None, help="RGB 画像パス (ファイル入力時)")
    parser.add_argument("--depth", default=None, help="深度画像パス (uint16 PNG, mm単位)")
    # 内部パラメータ (ファイル入力時)
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    # SAM プロンプト
    parser.add_argument("--click-x", type=int, default=-1)
    parser.add_argument("--click-y", type=int, default=-1)
    # 出力
    parser.add_argument("--save", default="result.png", help="可視化結果の保存パス")
    parser.add_argument("--show", action="store_true", help="ウィンドウ表示")
    args = parser.parse_args()

    # ---- 画像取得 ----
    if args.realsense:
        print("[Client] RealSense から取得中...")
        rgb, depth_mm, intrinsics = capture_realsense()
    elif args.rgb and args.depth:
        print(f"[Client] ファイルから読み込み: {args.rgb}, {args.depth}")
        rgb, depth_mm = load_from_files(args.rgb, args.depth)
        intrinsics = {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy}
    else:
        parser.error("--realsense または --rgb / --depth を指定してください。")

    print(f"[Client] 画像サイズ: {rgb.shape[:2]}, 内部パラメータ: {intrinsics}")

    # ---- サーバへ送信 ----
    click_xy = (args.click_x, args.click_y) if args.click_x >= 0 else None
    result = call_full_pipeline(
        server_url=args.server,
        rgb=rgb,
        depth_mm=depth_mm,
        intrinsics=intrinsics,
        click_xy=click_xy,
    )

    R = np.array(result["R"], dtype=np.float32)
    t = np.array(result["t"], dtype=np.float32)
    points = np.array(result["points"], dtype=np.float32)

    print(f"[Client] t = {t*1000} mm")
    print(f"[Client] 点群: {len(points)} 点")
    print(f"[Client] PLY: {result.get('ply_path')}")

    # ---- 可視化 ----
    vis = draw_pose_and_grasps(rgb, R, t, intrinsics, points)

    cv2.imwrite(args.save, vis)
    print(f"[Client] 可視化を保存: {args.save}")

    if args.show:
        cv2.imshow("Pose & Grasp", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
