"""
デモデータを使ったパイプラインテスト

RealSense なしでファイルから RGB + 深度を送信し、
サーバの SAM-3D + SAM-6D + Shape2Gesture を動作確認する。

使用方法:
    # Step1: reference mesh 生成
    python test_demo.py --mode offline-mesh --rgb test_data/rgb.png

    # Step2: 把持姿勢生成
    python test_demo.py --mode online \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --mesh meshes/test_object.ply \
        --fx 591.0 --fy 590.0 --cx 322.5 --cy 244.0

    # SAM-6D デモデータ (depth は mm PNG → m float32 に変換)
    python test_demo.py --mode online \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --depth-scale 0.001 \
        --mesh meshes/test_object.ply
"""

import argparse
import os
import sys
import yaml
import numpy as np
import cv2

# カレントディレクトリを real_world_demo に設定
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_intrinsics(args, img_w: int, img_h: int):
    """
    カメラ内部パラメータを読み込む。

    --cam-json が指定されている場合はそちらを優先。
    {"cam_K": [fx,0,cx, 0,fy,cy, 0,0,1], "depth_scale": 1.0} 形式を想定。
    """
    import json as _json


    if args.cam_json:
        with open(args.cam_json, "r") as f:
            cam = _json.load(f)
        K = cam["cam_K"]  # 9要素フラット or 3x3
        if len(K) == 9:
            fx, cx = K[0], K[2]
            fy, cy = K[4], K[5]
        else:
            raise ValueError(f"cam_K の形式が不正: {K}")
        # depth_scale も JSON から上書き
        if "depth_scale" in cam and args.depth_scale == 0.001:
            args.depth_scale = cam["depth_scale"]
        print(f"[intrinsics] cam_json から読み込み: fx={fx} fy={fy} cx={cx} cy={cy}  depth_scale={args.depth_scale}")
    else:
        fx = args.fx
        fy = args.fy
        cx = args.cx if args.cx > 0 else img_w / 2
        cy = args.cy if args.cy > 0 else img_h / 2

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=img_w, height=img_h)


def load_depth(depth_path: str, depth_scale: float = 1.0) -> np.ndarray:
    """
    深度画像を float32 [メートル] として読み込む

    depth_scale:
        1.0    → すでにメートル単位 (float32 tif など)
        0.001  → ミリメートル uint16 PNG (SAM-6D デモ形式)
        0.0001 → 0.1mm uint16 PNG (RealSense など)
    """
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"深度画像が読み込めません: {depth_path}")

    depth_m = depth_raw.astype(np.float32) * depth_scale
    print(f"[test] 深度画像ロード: {depth_path}  "
          f"shape={depth_raw.shape}  dtype={depth_raw.dtype}  "
          f"range=[{depth_m.min():.3f}, {depth_m.max():.3f}] m")
    return depth_m


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
    print(f"  python test_demo.py --mode online \\")
    print(f"    --rgb {args.rgb} \\")
    print(f"    --depth <depth_path> \\")
    print(f"    --mesh {mesh_path} \\")
    print(f"    --server-mesh-path \"{client._server_mesh_path}\" \\")
    print(f"    --template-dir \"{client._template_dir}\"")


def run_full(args, config):
    """RGB + 深度ファイル → メッシュ生成 → SAM-6D pose → 可視化 (1コマンド実行)"""
    import subprocess
    import json as json_module
    import shutil
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from utils.coord_transform import (
        CameraIntrinsics, ObjectPose,
        estimate_scale_from_depth, normalized_to_camera,
    )
    from utils.visualization import project_hands_on_image
    from utils.pointcloud_utils import load_pointcloud_ply

    sam_cfg  = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]

    # ---- RGB ロード ----
    rgb = cv2.imread(args.rgb)  # BGR
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[test] RGB ロード: {args.rgb}  shape={rgb.shape}")

    # ---- Step 1: メッシュ生成 ----
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mesh_path = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")
    print("\n[Step 1] SAM-3D でメッシュ生成中...")
    if args.click_x >= 0 and args.click_y >= 0:
        client.save_reference_mesh(rgb, mesh_path,
                                   click_x=args.click_x, click_y=args.click_y,
                                   mesh_method=mesh_method)
    elif args.interactive:
        client.save_reference_mesh_interactive(rgb, mesh_path, mesh_method=mesh_method)
    else:
        client.save_reference_mesh(rgb, mesh_path, mesh_method=mesh_method)

    print(f"[Step 1完了] mesh: {mesh_path}")
    print(f"             サーバ mesh: {client._server_mesh_path}")
    print(f"             テンプレート: {client._template_dir}")

    # ---- 深度ロード ----
    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]

    # ---- カメラ内部パラメータ ----
    intrinsics = load_intrinsics(args, w, h)

    # ---- Step 2: docker exec で run_demo_custom.sh を実行 ----
    print("\n[Step 2] SAM-6D で 6DoF pose 推定中 (run_demo_custom.sh)...")

    TMP_HOST   = "/home/okada/ws/project/tmp"
    TMP_DOCKER = "/workspace/tmp"

    # RGB / Depth を共有ディレクトリに保存
    rgb_host   = os.path.join(TMP_HOST, "rgb.png")
    depth_host = os.path.join(TMP_HOST, "depth.png")
    cam_host   = os.path.join(TMP_HOST, "camera_custom.json")

    cv2.imwrite(rgb_host, rgb)  # BGR のまま保存
    depth_mm = (depth * 1000.0).astype(np.uint16)
    cv2.imwrite(depth_host, depth_mm)

    # カメラ JSON 生成 (depth_scale=1.0: uint16mm → /1000 = m)
    cam_json = {
        "cam_K": [
            intrinsics.fx, 0.0, intrinsics.cx,
            0.0, intrinsics.fy, intrinsics.cy,
            0.0, 0.0, 1.0,
        ],
        "depth_scale": 1.0,
    }
    with open(cam_host, "w") as f:
        json_module.dump(cam_json, f)

    # Docker パス
    rgb_docker   = f"{TMP_DOCKER}/rgb.png"
    depth_docker = f"{TMP_DOCKER}/depth.png"
    cam_docker   = f"{TMP_DOCKER}/camera_custom.json"

    # template_dir (Docker) → output_dir (Docker)
    # _template_dir = ".../object_seed42_mesh_templates/templates"
    # output_dir    = ".../object_seed42_mesh_templates"
    template_dir_docker = client._template_dir.rstrip("/")
    if template_dir_docker.endswith("/templates"):
        output_dir_docker = template_dir_docker[: -len("/templates")]
    else:
        output_dir_docker = template_dir_docker

    mesh_docker    = client._server_mesh_path
    script_docker  = "/workspace/SAM-6D/SAM-6D/run_demo_custom.sh"

    cmd = [
        "docker", "exec", "sam6d_service",
        "bash", script_docker,
        mesh_docker,
        output_dir_docker,
        str(args.click_x),
        str(args.click_y),
        rgb_docker,
        depth_docker,
        cam_docker,
    ]
    print(f"[Step 2] 実行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"run_demo_custom.sh 失敗 (code={result.returncode})")

    # ---- 結果 JSON 読み込み ----
    output_dir_host  = output_dir_docker.replace(TMP_DOCKER, TMP_HOST)
    result_json_path = os.path.join(output_dir_host, "sam6d_results", "detection_pem.json")
    with open(result_json_path, "r") as f:
        detections = json_module.load(f)

    if not detections:
        raise RuntimeError("pose 推定失敗: detection が空です")

    best = max(detections, key=lambda d: d["score"])
    R = np.array(best["R"], dtype=np.float32)        # (3, 3)
    t = np.array(best["t"], dtype=np.float32) / 1000.0  # mm → m
    print(f"[Step 2完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    # ---- 可視化画像を output/test/ にコピー ----
    os.makedirs("output/test", exist_ok=True)
    vis_pem  = os.path.join(output_dir_host, "sam6d_results", "vis_pem.png")
    vis_axes = os.path.join(output_dir_host, "sam6d_results", "vis_pem_axes.png")

    out_pem  = "output/test/pose_check.png"
    out_axes = "output/test/pose_check_axes.png"

    if os.path.exists(vis_pem):
        shutil.copy(vis_pem, out_pem)
        print(f"[Pose確認] 可視化画像を保存: {out_pem}")
    if os.path.exists(vis_axes):
        shutil.copy(vis_axes, out_axes)
        print(f"[Pose確認] 軸付き可視化画像を保存: {out_axes}")

    if not args.no_show:
        for path, title in [(out_pem, "Pose Check"), (out_axes, "Pose Check (axes)")]:
            img = cv2.imread(path)
            if img is not None:
                cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[表示スキップ] --no-show が指定されています。画像はファイルに保存されました。")

    if args.skip_grasp:
        print("\n[完了] --skip-grasp が指定されたため把持姿勢生成をスキップします。")
        return

    # ---- スケール推定 ----
    mesh_pts = load_pointcloud_ply(mesh_path, target_points=2048)
    mask_u = int(intrinsics.fx * t[0] / max(t[2], 0.01) + intrinsics.cx)
    mask_v = int(intrinsics.fy * t[1] / max(t[2], 0.01) + intrinsics.cy)
    scale = estimate_scale_from_depth(depth, mask_u, mask_v, intrinsics, mesh_pts)
    pose = ObjectPose(center_3d=t, scale=scale, R=R)

    # ---- Shape2Gesture ----
    print("\n[Step 3] Shape2Gesture で把持姿勢を生成中...")
    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()

    grasp_results = generator.generate(mesh_pts, num_samples=model_cfg["num_samples"])

    print(f"\n[Step 4] {len(grasp_results)} 件の把持姿勢を画像に投影中...")
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

    sam_cfg  = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]

    # ---- ファイルロード ----
    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")

    depth = load_depth(args.depth, depth_scale=args.depth_scale)

    # 深度と RGB のサイズを合わせる
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        print(f"[test] 深度をリサイズ: {depth.shape}")

    h, w = rgb.shape[:2]

    # ---- カメラ内部パラメータ ----
    intrinsics = load_intrinsics(args, w, h)
    print(f"[test] intrinsics: fx={intrinsics.fx} fy={intrinsics.fy} "
          f"cx={intrinsics.cx} cy={intrinsics.cy}")

    # ---- SAM-6D クライアント ----
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )
    client.load_reference_mesh(
        args.mesh,
        server_mesh_path=args.server_mesh_path or "",
        template_dir=args.template_dir or "",
    )

    # ---- 6DoF pose 推定 ----
    print("\n[Step 1] SAM-6D で 6DoF pose 推定中...")
    R, t = client.estimate_pose(rgb, depth, intrinsics,
                                click_x=args.click_x, click_y=args.click_y)
    print(f"  R=\n{R}")
    print(f"  t={t}")

    # ---- メッシュをRGB画像にレンダリングして確認 ----
    mesh_pts = load_pointcloud_ply(args.mesh, target_points=2048)
    bgr = rgb  # cv2.imread は BGR で読み込むためそのまま使用
    os.makedirs("output/test", exist_ok=True)

    bgr = rgb  # cv2.imread は BGR で読み込むためそのまま使用
    # Open3D OffscreenRenderer でメッシュ面をレンダリング (FoundationPose的手法)
    vis_img = render_mesh_on_image(bgr, args.mesh, R, t, intrinsics, mesh_unit="mm")
    vis_path = "output/test/pose_check_mesh.png"
    cv2.imwrite(vis_path, vis_img)
    print(f"[Pose確認] メッシュレンダリング画像を保存: {vis_path}")

    # 点群+bbox投影も保存 (比較用)
    vis_pts_img = project_pointcloud_on_image(bgr, mesh_pts, R, t, intrinsics, points_unit="mm")
    vis_pts_path = "output/test/pose_check_pts.png"
    cv2.imwrite(vis_pts_path, vis_pts_img)
    print(f"[Pose確認] 点群+bbox投影画像を保存: {vis_pts_path}")

    if not args.no_show:
        cv2.imshow("Pose Check: mesh render", vis_img)
        cv2.imshow("Pose Check: pointcloud + bbox", vis_pts_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[表示スキップ] --no-show が指定されています。画像はファイルに保存されました。")

    if args.skip_grasp:
        print("\n[完了] --skip-grasp が指定されたため把持姿勢生成をスキップします。")
        print(f"  出力: {vis_path}")
        print(f"  出力: {vis_pts_path}")
        return

    # ---- スケール推定 ----
    mask_u = int(intrinsics.fx * t[0] / max(t[2], 0.01) + intrinsics.cx)
    mask_v = int(intrinsics.fy * t[1] / max(t[2], 0.01) + intrinsics.cy)
    scale = estimate_scale_from_depth(depth, mask_u, mask_v, intrinsics, mesh_pts)
    pose = ObjectPose(center_3d=t, scale=scale, R=R)

    # ---- Shape2Gesture ----
    print("\n[Step 2] Shape2Gesture で把持姿勢を生成中...")
    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()

    grasp_results = generator.generate(mesh_pts, num_samples=model_cfg["num_samples"])

    # ---- 画像投影 ----
    print(f"\n[Step 3] {len(grasp_results)} 件の把持姿勢を画像に投影中...")
    os.makedirs("output/test", exist_ok=True)

    for i, (lh_norm, rh_norm) in enumerate(grasp_results):
        result_img = project_hands_on_image(
            rgb, lh_norm, rh_norm,
            object_pose=pose,
            intrinsics=intrinsics,
        )
        save_path = f"output/test/grasp_{i:02d}.png"
        cv2.imwrite(save_path, result_img)
        print(f"  保存: {save_path}")

    # 最初の結果を表示
    if not args.no_show:
        cv2.imshow("Grasp Result", cv2.imread("output/test/grasp_00.png"))
        print("\n何かキーを押すと終了...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="デモデータでパイプラインをテスト")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["offline-mesh", "online", "full"], default="online")
    parser.add_argument("--no-show", action="store_true",
                        help="cv2.imshow を使わない (Docker/ヘッドレス環境用)")
    parser.add_argument("--skip-grasp", action="store_true",
                        help="Shape2Gesture をスキップして pose 可視化のみ実行")

    # 入力データ
    parser.add_argument("--rgb",   required=True,  help="RGB 画像パス (.png/.jpg)")
    parser.add_argument("--depth", default=None,   help="[online] 深度画像パス (.png)")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="深度のスケール係数 (0.001: mm→m, デフォルト)")

    # カメラ内部パラメータ: --cam-json を優先、なければ個別指定
    parser.add_argument("--cam-json", default=None,
                        help="camera JSON ファイル ({cam_K:[fx,0,cx,0,fy,cy,0,0,1], depth_scale:1.0})")
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.0)
    parser.add_argument("--cx", type=float, default=-1)   # -1 で画像中央
    parser.add_argument("--cy", type=float, default=-1)

    # mesh パス
    parser.add_argument("--mesh",     default=None, help="[online] ローカル mesh (.ply)")
    parser.add_argument("--mesh-out", default="meshes/test_object.ply",
                        help="[offline-mesh] 保存先")
    parser.add_argument("--server-mesh-path", default=None)
    parser.add_argument("--template-dir",     default=None)

    # offline-mesh オプション
    parser.add_argument("--click-x",    type=int, default=-1)
    parser.add_argument("--click-y",    type=int, default=-1)
    parser.add_argument("--interactive", action="store_true", default=True)

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
