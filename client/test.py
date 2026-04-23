"""
高さ指定による姿勢推定テスト

物体の高さ（実測値）を手動指定して SAM-6D 姿勢推定を実行し、
IMU 自動高さ推定の有効性検証に使用する。

使用方法:
    # 物体高さを指定して姿勢推定
    python test.py --data-dir saved_data/test_20240101_120000 --object-size 12.5

    # クリック座標も指定する場合
    python test.py --data-dir saved_data/test_20240101_120000 \
        --object-size 12.5 --click-x 320 --click-y 240
"""

import argparse
import json
import os
import sys
import numpy as np
import cv2
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# 設定・ロード
# ===========================================================================

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_intrinsics(args, img_w: int, img_h: int):
    from utils.coord_transform import CameraIntrinsics
    if args.cam_json:
        with open(args.cam_json, "r") as f:
            cam = json.load(f)
        K = cam["cam_K"]
        fx, cx = K[0], K[2]
        fy, cy = K[4], K[5]
        if "depth_scale" in cam and args.depth_scale == 0.001:
            args.depth_scale = cam["depth_scale"]
        if "gravity" in cam and args.gravity is None:
            args.gravity = cam["gravity"]
            print(f"[intrinsics] cam.json から gravity 読み込み: {[f'{v:.4f}' for v in args.gravity]}")
        print(f"[intrinsics] fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  depth_scale={args.depth_scale}")
    else:
        fx, fy = args.fx, args.fy
        cx = args.cx if args.cx > 0 else img_w / 2
        cy = args.cy if args.cy > 0 else img_h / 2
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=img_w, height=img_h)


def load_depth(depth_path: str, depth_scale: float) -> np.ndarray:
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"深度画像が読み込めません: {depth_path}")
    depth_m = raw.astype(np.float32) * depth_scale
    print(f"[depth] shape={raw.shape}  range=[{depth_m.min():.3f}, {depth_m.max():.3f}] m")
    return depth_m


# ===========================================================================
# 高さ推定ユーティリティ（比較表示用）
# ===========================================================================

def get_points_3d(depth_m, mask, fx, fy, cx, cy, min_d=0.05, max_d=5.0):
    mask_bool = mask.astype(bool)
    vs, us = np.where(mask_bool)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    d = depth_m[vs, us].astype(np.float64)
    valid = (d > min_d) & (d < max_d)
    vs, us, d = vs[valid], us[valid], d[valid]
    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    X = (us - cx) * d / fx
    Y = (vs - cy) * d / fy
    return np.stack([X, Y, d], axis=1)


def calc_height(pts, gravity_vec):
    proj = pts @ gravity_vec
    return float(proj.max() - proj.min())


# ===========================================================================
# メイン
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="高さ指定による姿勢推定テスト")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-show", action="store_true")

    # データ入力
    parser.add_argument("--data-dir", default=None,
                        help="save_data_IMU.py で保存したフォルダ (rgb.png / depth.png / cam.json を自動解決)")
    parser.add_argument("--rgb",       default=None)
    parser.add_argument("--depth",     default=None)
    parser.add_argument("--cam-json",  default=None)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.0)
    parser.add_argument("--cx", type=float, default=-1)
    parser.add_argument("--cy", type=float, default=-1)

    # 物体サイズ（必須）
    parser.add_argument("--object-size", type=float, required=True,
                        help="物体の高さ [cm]（実測値）")

    # SAM クリック
    parser.add_argument("--click-x",    type=int, default=-1)
    parser.add_argument("--click-y",    type=int, default=-1)
    parser.add_argument("--interactive", action="store_true", default=True)

    # mesh 出力先
    parser.add_argument("--mesh-out", default="meshes/test_object.ply")

    # 重力ベクトル（IMU 推定との比較用）
    parser.add_argument("--gravity", type=float, nargs=3, default=None,
                        metavar=("GX", "GY", "GZ"))

    args = parser.parse_args()

    # --data-dir の自動展開
    if args.data_dir:
        if args.rgb is None:
            args.rgb = os.path.join(args.data_dir, "rgb.png")
        if args.depth is None:
            args.depth = os.path.join(args.data_dir, "depth.png")
        if args.cam_json is None:
            args.cam_json = os.path.join(args.data_dir, "cam.json")

    if not args.rgb or not args.depth:
        print("エラー: --data-dir または --rgb/--depth を指定してください。")
        sys.exit(1)

    config = load_config(args.config)
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})

    from pipeline.sam6d_detector import SAM6DClient

    # ---- RGB / 深度 ロード ----
    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[test] RGB: {args.rgb}  shape={rgb.shape}")

    depth = load_depth(args.depth, args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    intrinsics = load_intrinsics(args, w, h)

    object_size_mm = args.object_size * 10.0
    print(f"\n[設定] 指定高さ: {args.object_size:.1f} cm = {object_size_mm:.0f} mm")

    # ---- SAM-3D メッシュ生成 ----
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mesh_path  = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")
    click_x, click_y = args.click_x, args.click_y

    print("\n[Step 1] SAM-3D でメッシュ生成中...")
    if click_x >= 0 and click_y >= 0:
        _, masks, scores = client.save_reference_mesh(
            rgb, mesh_path,
            click_x=click_x, click_y=click_y,
            mesh_method=mesh_method,
            object_size_mm=object_size_mm,
        )
    elif args.interactive:
        _, click_x, click_y, masks, scores = client.save_reference_mesh_interactive(
            rgb, mesh_path, mesh_method=mesh_method,
        )
        client._object_size_mm = object_size_mm
    else:
        _, masks, scores = client.save_reference_mesh(
            rgb, mesh_path,
            mesh_method=mesh_method,
            object_size_mm=object_size_mm,
        )
    print(f"[Step 1完了] mesh: {mesh_path}")

    # ---- IMU 自動推定との比較（参考） ----
    gravity_vec = None
    if args.gravity is not None:
        gravity_vec = np.array(args.gravity, dtype=np.float64)
        gravity_vec = gravity_vec / np.linalg.norm(gravity_vec)

    if gravity_vec is not None and masks:
        best_idx = int(np.argmax(scores)) if scores else 0
        best_mask = masks[best_idx]
        if best_mask.shape[:2] != depth.shape[:2]:
            best_mask = cv2.resize(best_mask, (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        mask_area = int(np.count_nonzero(best_mask))
        img_area  = best_mask.shape[0] * best_mask.shape[1]
        if mask_area > img_area * 0.02:
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            best_mask = cv2.erode(best_mask, erode_kernel, iterations=1)

        pts = get_points_3d(depth, best_mask,
                            intrinsics.fx, intrinsics.fy,
                            intrinsics.cx, intrinsics.cy)
        if len(pts) >= 10:
            est_h = calc_height(pts, gravity_vec)
            print(f"\n[比較] IMU 自動推定高さ : {est_h*100:.1f} cm ({est_h*1000:.0f} mm)")
            print(f"[比較] 指定高さ         : {args.object_size:.1f} cm ({object_size_mm:.0f} mm)")
            diff = abs(est_h*100 - args.object_size)
            print(f"[比較] 差分             : {diff:.1f} cm")

        # SAM マスク保存
        os.makedirs("output/test", exist_ok=True)
        mask_vis = cv2.applyColorMap(masks[best_idx], cv2.COLORMAP_JET)
        mask_overlay = cv2.addWeighted(rgb, 0.6, mask_vis, 0.4, 0)
        cv2.imwrite("output/test/sam_mask.png", mask_overlay)
        print("[Step 1] SAM マスク保存: output/test/sam_mask.png")

    # ---- Step 2: SAM-6D 姿勢推定 ----
    client._object_size_mm = object_size_mm
    print(f"\n[Step 2] SAM-6D pose 推定中 (object_size={object_size_mm:.0f} mm)...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics,
        click_x=click_x, click_y=click_y,
    )
    print(f"[Step 2完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    os.makedirs("output/test", exist_ok=True)
    if img_pose is not None:
        cv2.imwrite("output/test/server_pointcloud.png", img_pose)
        print("[Step 2] 点群投影画像: output/test/server_pointcloud.png")
        if not args.no_show:
            cv2.imshow("Pose: pointcloud", img_pose)
    if img_mesh is not None:
        cv2.imwrite("output/test/server_mesh.png", img_mesh)
        print("[Step 2] メッシュ投影画像: output/test/server_mesh.png")
        if not args.no_show:
            cv2.imshow("Pose: mesh", img_mesh)

    if not args.no_show and (img_pose is not None or img_mesh is not None):
        print("何かキーを押すと終了...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ---- サマリー ----
    print("\n" + "=" * 50)
    print(f"  指定高さ : {args.object_size:.1f} cm ({object_size_mm:.0f} mm)")
    print(f"  pose t   : [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print("=" * 50)


if __name__ == "__main__":
    main()
