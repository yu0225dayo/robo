"""
IMU を用いた自動高さ推定付きパイプラインテスト

test_demo.py の拡張版。--object-size の手入力の代わりに、
RealSense IMU（加速度センサー）で重力方向を取得し、
SAM マスク + 深度画像 + カメラパラメータから物体高さを自動計算する。

使用方法:
    # フルモード（IMU自動取得）
    python test_demo_w_IMU.py --mode full \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --click-x 400 --click-y 280

    # 重力ベクトルを手動指定（IMU不使用 / カメラなし環境）
    python test_demo_w_IMU.py --mode full \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --gravity 0 -1 0

    # 高さ推定をスキップ（従来の --object-size 指定）
    python test_demo_w_IMU.py --mode full \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --object-size 15.5
"""

import argparse
import os
import sys
import yaml
import numpy as np
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# 設定ロード
# ===========================================================================

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_intrinsics(args, img_w: int, img_h: int):
    """intrinsics を読み込む。cam.json に gravity が含まれていれば args.gravity に設定する。"""
    import json as _json
    from utils.coord_transform import CameraIntrinsics

    if args.cam_json:
        with open(args.cam_json, "r") as f:
            cam = _json.load(f)
        K = cam["cam_K"]
        if len(K) == 9:
            fx, cx = K[0], K[2]
            fy, cy = K[4], K[5]
        else:
            raise ValueError(f"cam_K の形式が不正: {K}")
        if "depth_scale" in cam and args.depth_scale == 0.001:
            args.depth_scale = cam["depth_scale"]
        # gravity が JSON に含まれていて --gravity 未指定なら自動設定
        if "gravity" in cam and args.gravity is None:
            args.gravity = cam["gravity"]
            print(f"[intrinsics] cam_json から gravity 読み込み: {args.gravity}")
        print(f"[intrinsics] cam_json から読み込み: fx={fx} fy={fy} cx={cx} cy={cy}  depth_scale={args.depth_scale}")
    else:
        fx = args.fx
        fy = args.fy
        cx = args.cx if args.cx > 0 else img_w / 2
        cy = args.cy if args.cy > 0 else img_h / 2

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=img_w, height=img_h)


def load_depth(depth_path: str, depth_scale: float = 1.0) -> np.ndarray:
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"深度画像が読み込めません: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * depth_scale
    print(f"[test] 深度画像ロード: {depth_path}  shape={depth_raw.shape}  dtype={depth_raw.dtype}  "
          f"range=[{depth_m.min():.3f}, {depth_m.max():.3f}] m")
    return depth_m


# ===========================================================================
# IMU・高さ計算ユーティリティ
# ===========================================================================

def get_gravity_imu(n_samples: int = 30) -> np.ndarray:
    """
    RealSense 加速度センサーから重力方向の単位ベクトルを取得する。

    Returns:
        gravity_vec: (3,) 重力方向の単位ベクトル（RealSense カメラ座標系）
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise RuntimeError(
            "pyrealsense2 がインストールされていません。\n"
            "  pip install pyrealsense2\n"
            "または --gravity で重力ベクトルを手動指定してください。"
        )

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)

    print(f"[IMU] RealSense IMU を起動して重力ベクトルを取得中 ({n_samples} サンプル)...")
    profile = pipeline.start(cfg)
    samples = []
    collected = 0
    try:
        while collected < n_samples:
            frames = pipeline.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            if not accel_frame:
                continue
            motion = accel_frame.as_motion_frame().get_motion_data()
            samples.append([motion.x, motion.y, motion.z])
            collected += 1
    finally:
        pipeline.stop()

    g = np.mean(samples, axis=0)
    g = g / np.linalg.norm(g)
    print(f"[IMU] 重力ベクトル g = [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]")
    return g.astype(np.float64)


def get_points_3d_from_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
) -> np.ndarray:
    """
    マスク領域内の深度画像を 3D 点群に変換する（ベクトル化・高速）。

    Args:
        depth_m: (H, W) float32, メートル単位の深度画像
        mask:    (H, W) bool または uint8 (>0 が物体領域)
        fx, fy, cx, cy: カメラ内部パラメータ

    Returns:
        points_3d: (N, 3) float64, カメラ座標系の 3D 点群 [m]
                   有効点がない場合は shape (0, 3)
    """
    mask_bool = mask.astype(bool)
    vs, us = np.where(mask_bool)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    d = depth_m[vs, us].astype(np.float64)
    valid = (d > min_depth) & (d < max_depth)
    vs, us, d = vs[valid], us[valid], d[valid]

    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    X = (us - cx) * d / fx
    Y = (vs - cy) * d / fy
    Z = d
    return np.stack([X, Y, Z], axis=1)


def calc_height_from_points(points_3d: np.ndarray, gravity_vec: np.ndarray) -> float:
    """
    3D 点群を重力方向に射影して物体の高さを計算する。

    Args:
        points_3d:   (N, 3) カメラ座標系の 3D 点群 [m]
        gravity_vec: (3,)   重力方向の単位ベクトル

    Returns:
        height_m: 高さ [m]
    """
    projections = points_3d @ gravity_vec  # (N,)
    return float(projections.max() - projections.min())


def estimate_height_from_depth_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    gravity_vec: np.ndarray,
) -> float:
    """
    深度画像 + マスク + 重力ベクトル → 物体高さ [m]

    Returns:
        height_m: 高さ [m]。有効点が不足している場合は 0.0。
    """
    pts = get_points_3d_from_mask(depth_m, mask, fx, fy, cx, cy)
    if len(pts) < 10:
        print(f"[高さ推定] 有効な深度点が不足 ({len(pts)} 点)。高さ推定をスキップ。")
        return 0.0
    h = calc_height_from_points(pts, gravity_vec)
    print(f"[高さ推定] 点数={len(pts)}  高さ={h:.4f} m ({h*100:.1f} cm)")
    return h


# ===========================================================================
# モード別実行関数
# ===========================================================================

def run_offline_mesh(args, config):
    """RGB ファイル → サーバ → reference mesh 保存"""
    from pipeline.sam6d_detector import SAM6DClient

    sam_cfg = config["sam3d"]
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
    )

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[test] RGB ロード: {args.rgb}  shape={rgb.shape}")

    mesh_path = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")

    if args.click_x >= 0 and args.click_y >= 0:
        client.save_reference_mesh(rgb, mesh_path,
                                   click_x=args.click_x, click_y=args.click_y,
                                   mesh_method=mesh_method)
    elif args.interactive:
        client.save_reference_mesh_interactive(rgb, mesh_path)
    else:
        client.save_reference_mesh(rgb, mesh_path, mesh_method=mesh_method)

    print(f"\n[完了] mesh: {mesh_path}")
    print(f"       サーバ mesh: {client._server_mesh_path}")
    print(f"       テンプレート: {client._template_dir}")
    print(f"\n次のコマンド:")
    print(f"  python test_demo_w_IMU.py --mode online \\")
    print(f"    --rgb {args.rgb} \\")
    print(f"    --depth <depth_path> \\")
    print(f"    --mesh {mesh_path} \\")
    print(f"    --server-mesh-path \"{client._server_mesh_path}\" \\")
    print(f"    --template-dir \"{client._template_dir}\"")


def run_full(args, config):
    """
    RGB + 深度ファイル → SAM マスク取得 → IMU/重力で高さ自動推定
    → SAM-6D pose → 可視化
    """
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from utils.coord_transform import (
        CameraIntrinsics, ObjectPose,
        estimate_scale_from_depth, normalized_to_camera,
    )
    from utils.visualization import project_hands_on_image
    from utils.pointcloud_utils import load_pointcloud_ply

    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]

    # ---- RGB ロード ----
    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[test] RGB ロード: {args.rgb}  shape={rgb.shape}")

    # ---- 深度ロード ----
    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    intrinsics = load_intrinsics(args, w, h)

    # ---- Step 0: 重力ベクトル取得 ----
    if args.object_size > 0:
        # 手動指定が優先
        object_size_mm = args.object_size * 10.0
        print(f"[高さ] 手動指定: {args.object_size} cm = {object_size_mm:.0f} mm (IMU 不使用)")
        gravity_vec = None
    elif args.gravity is not None:
        # --gravity で手動指定
        gravity_vec = np.array(args.gravity, dtype=np.float64)
        gravity_vec = gravity_vec / np.linalg.norm(gravity_vec)
        print(f"[高さ] 重力ベクトル手動指定: {gravity_vec}")
        object_size_mm = 0.0  # マスク取得後に計算
    else:
        # RealSense IMU から自動取得
        gravity_vec = get_gravity_imu(n_samples=args.imu_samples)
        object_size_mm = 0.0  # マスク取得後に計算

    # ---- Step 1: SAM-3D でメッシュ生成（マスクも取得） ----
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mesh_path = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")
    click_x, click_y = args.click_x, args.click_y

    print("\n[Step 1] SAM-3D でメッシュ生成中 (マスク取得)...")
    if args.click_x >= 0 and args.click_y >= 0:
        _, masks, scores = client.save_reference_mesh(
            rgb, mesh_path,
            click_x=click_x, click_y=click_y,
            mesh_method=mesh_method,
            object_size_mm=0.0,  # スケールはここでは設定しない
        )
    elif args.interactive:
        _, click_x, click_y, masks, scores = client.save_reference_mesh_interactive(
            rgb, mesh_path, mesh_method=mesh_method,
        )
    else:
        _, masks, scores = client.save_reference_mesh(
            rgb, mesh_path,
            mesh_method=mesh_method,
            object_size_mm=0.0,
        )

    print(f"[Step 1完了] mesh: {mesh_path}")

    # ---- Step 2: マスク + 深度 → 高さ推定 ----
    if gravity_vec is not None and masks:
        # スコアが最大のマスクを使用
        best_idx = int(np.argmax(scores)) if scores else 0
        best_mask = masks[best_idx]
        print(f"\n[Step 2] マスクから高さ推定中 (mask_idx={best_idx}, score={scores[best_idx] if scores else '?':.3f})...")

        # マスクを深度と同サイズにリサイズ
        if best_mask.shape[:2] != depth.shape[:2]:
            best_mask = cv2.resize(best_mask, (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        height_m = estimate_height_from_depth_mask(
            depth, best_mask,
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            gravity_vec,
        )

        if height_m > 0.005:  # 5mm 以上なら使用
            object_size_mm = height_m * 1000.0
            print(f"[Step 2完了] 推定高さ: {height_m*100:.1f} cm = {object_size_mm:.0f} mm")

            # マスクを保存（デバッグ用）
            os.makedirs("output/test", exist_ok=True)
            mask_vis = cv2.applyColorMap(best_mask, cv2.COLORMAP_JET)
            mask_overlay = cv2.addWeighted(rgb, 0.6, mask_vis, 0.4, 0)
            cv2.imwrite("output/test/sam_mask.png", mask_overlay)
            print("[Step 2] SAM マスク保存: output/test/sam_mask.png")
        else:
            print("[Step 2] 高さ推定失敗。object_size_mm=0 (サーバ深度推定に委譲)。")
            object_size_mm = 0.0
    elif not masks:
        print("[Step 2] SAM マスクが空。高さ推定をスキップ。")

    # ---- Step 3: 6DoF pose 推定 ----
    # object_size_mm を使ってスケールを設定してから推定
    client._object_size_mm = object_size_mm

    print(f"\n[Step 3] SAM-6D で 6DoF pose 推定中 (object_size_mm={object_size_mm:.0f})...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics,
        click_x=click_x, click_y=click_y,
    )
    print(f"[Step 3完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    os.makedirs("output/test", exist_ok=True)
    if img_pose is not None:
        cv2.imwrite("output/test/server_pointcloud.png", img_pose)
        print("[Step 3] 点群投影画像保存: output/test/server_pointcloud.png")
        if not args.no_show:
            cv2.imshow("Pose: pointcloud", img_pose)
    if img_mesh is not None:
        cv2.imwrite("output/test/server_mesh.png", img_mesh)
        print("[Step 3] メッシュ投影画像保存: output/test/server_mesh.png")
        if not args.no_show:
            cv2.imshow("Pose: mesh", img_mesh)
    if not args.no_show:
        print("何かキーを押すと続行...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.skip_grasp:
        print("\n[完了] --skip-grasp が指定されたため把持姿勢生成をスキップします。")
        return

    # ---- Step 4: Shape2Gesture ----
    mesh_pts = load_pointcloud_ply(mesh_path, target_points=2048)
    mask_u = int(intrinsics.fx * t[0] / max(t[2], 0.01) + intrinsics.cx)
    mask_v = int(intrinsics.fy * t[1] / max(t[2], 0.01) + intrinsics.cy)
    scale = estimate_scale_from_depth(depth, mask_u, mask_v, intrinsics, mesh_pts)
    pose = ObjectPose(center_3d=t, scale=scale, R=R)

    print("\n[Step 4] Shape2Gesture で把持姿勢を生成中...")
    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()
    grasp_results = generator.generate(mesh_pts, num_samples=model_cfg["num_samples"])

    print(f"\n[Step 5] {len(grasp_results)} 件の把持姿勢を画像に投影中...")
    for i, (lh_norm, rh_norm) in enumerate(grasp_results):
        result_img = project_hands_on_image(
            rgb, lh_norm, rh_norm,
            object_pose=pose,
            intrinsics=intrinsics,
        )
        save_path = f"output/test/grasp_{i:02d}.png"
        cv2.imwrite(save_path, result_img)
        print(f"  保存: {save_path}")

    if not args.no_show:
        cv2.imshow("Grasp Result", cv2.imread("output/test/grasp_00.png"))
        print("\n何かキーを押すと終了...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_online(args, config):
    """RGB + 深度ファイル → SAM-6D pose → Shape2Gesture → 画像投影"""
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from utils.coord_transform import (
        CameraIntrinsics, ObjectPose,
        estimate_scale_from_depth, normalized_to_camera,
    )
    from utils.visualization import project_hands_on_image, project_pointcloud_on_image, render_mesh_on_image
    from utils.pointcloud_utils import load_pointcloud_ply

    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")

    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    intrinsics = load_intrinsics(args, w, h)

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )
    client.load_reference_mesh(
        args.mesh,
        server_mesh_path=args.server_mesh_path or "",
        template_dir=args.template_dir or "",
    )

    if args.object_size > 0:
        client._object_size_mm = args.object_size * 10.0
        print(f"[online] object_size_mm={client._object_size_mm:.0f} (手動指定)")

    print("\n[Step 1] SAM-6D で 6DoF pose 推定中...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics,
        click_x=args.click_x, click_y=args.click_y,
    )
    print(f"  R=\n{R}")
    print(f"  t={t}")

    mesh_pts = load_pointcloud_ply(args.mesh, target_points=2048)
    os.makedirs("output/test", exist_ok=True)

    if img_pose is not None:
        cv2.imwrite("output/test/server_pointcloud.png", img_pose)
    if img_mesh is not None:
        cv2.imwrite("output/test/server_mesh.png", img_mesh)

    vis_img = render_mesh_on_image(rgb, args.mesh, R, t, intrinsics, mesh_unit="mm")
    cv2.imwrite("output/test/pose_check_mesh.png", vis_img)

    vis_pts_img = project_pointcloud_on_image(rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
    cv2.imwrite("output/test/pose_check_pts.png", vis_pts_img)

    if not args.no_show:
        cv2.imshow("Pose Check: mesh render", vis_img)
        cv2.imshow("Pose Check: pointcloud + bbox", vis_pts_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.skip_grasp:
        print("\n[完了] --skip-grasp が指定されたため把持姿勢生成をスキップします。")
        return

    mask_u = int(intrinsics.fx * t[0] / max(t[2], 0.01) + intrinsics.cx)
    mask_v = int(intrinsics.fy * t[1] / max(t[2], 0.01) + intrinsics.cy)
    scale = estimate_scale_from_depth(depth, mask_u, mask_v, intrinsics, mesh_pts)
    pose = ObjectPose(center_3d=t, scale=scale, R=R)

    print("\n[Step 2] Shape2Gesture で把持姿勢を生成中...")
    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()
    grasp_results = generator.generate(mesh_pts, num_samples=model_cfg["num_samples"])

    print(f"\n[Step 3] {len(grasp_results)} 件の把持姿勢を画像に投影中...")
    for i, (lh_norm, rh_norm) in enumerate(grasp_results):
        result_img = project_hands_on_image(
            rgb, lh_norm, rh_norm,
            object_pose=pose,
            intrinsics=intrinsics,
        )
        cv2.imwrite(f"output/test/grasp_{i:02d}.png", result_img)

    if not args.no_show:
        cv2.imshow("Grasp Result", cv2.imread("output/test/grasp_00.png"))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ===========================================================================
# エントリポイント
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="IMU 自動高さ推定付きパイプラインテスト")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["offline-mesh", "online", "full"], default="full")
    parser.add_argument("--no-show", action="store_true",
                        help="cv2.imshow を使わない (ヘッドレス環境用)")
    parser.add_argument("--skip-grasp", action="store_true",
                        help="Shape2Gesture をスキップして pose 可視化のみ実行")

    # 入力データ
    parser.add_argument("--rgb",   required=True,  help="RGB 画像パス (.png/.jpg)")
    parser.add_argument("--depth", default=None,   help="深度画像パス (.png)")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="深度のスケール係数 (0.001: mm→m, デフォルト)")

    # カメラ内部パラメータ
    parser.add_argument("--cam-json", default=None,
                        help="camera JSON ファイル ({cam_K:[fx,0,cx,0,fy,cy,0,0,1], depth_scale:1.0})")
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.0)
    parser.add_argument("--cx", type=float, default=-1)
    parser.add_argument("--cy", type=float, default=-1)

    # mesh パス
    parser.add_argument("--mesh",     default=None, help="[online] ローカル mesh (.ply)")
    parser.add_argument("--mesh-out", default="meshes/test_object.ply",
                        help="[offline-mesh/full] 保存先")
    parser.add_argument("--server-mesh-path", default=None)
    parser.add_argument("--template-dir",     default=None)

    # クリック・インタラクティブ
    parser.add_argument("--click-x",    type=int, default=-1)
    parser.add_argument("--click-y",    type=int, default=-1)
    parser.add_argument("--interactive", action="store_true", default=True)

    # 高さ指定（3 種類の優先順位: --object-size > --gravity > IMU 自動）
    parser.add_argument("--object-size", type=float, default=0.0,
                        help="物体の高さ [cm]。指定するとIMU・深度推定を上書き。")
    parser.add_argument("--gravity", type=float, nargs=3, default=None,
                        metavar=("GX", "GY", "GZ"),
                        help="重力方向ベクトル (例: 0 -1 0)。IMU 代替。--object-size より低優先。")
    parser.add_argument("--imu-samples", type=int, default=30,
                        help="IMU サンプル数 (デフォルト: 30)")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "offline-mesh":
        run_offline_mesh(args, config)
    elif args.mode == "full":
        if not args.depth:
            print("エラー: --depth を指定してください。")
            sys.exit(1)
        run_full(args, config)
    else:
        if not args.depth:
            print("エラー: --depth を指定してください。")
            sys.exit(1)
        if not args.mesh:
            print("エラー: --mesh を指定してください。")
            sys.exit(1)
        run_online(args, config)


if __name__ == "__main__":
    main()
