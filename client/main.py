"""
real_world_demo メインエントリーポイント

パイプライン:
    [オフライン] 物体ごとに1回:
        RealSense RGB → サーバ (SAM-3D) → reference mesh (.ply) 保存
        サーバが SAM-6D でテンプレートをレンダリングし保存

    [オンライン] 毎回:
        RealSense RGBD
            → サーバ (SAM-6D) → 6DoF pose (R, t)
            → reference mesh × (R, t, scale) → カメラ座標系の完全点群
            → Shape2Gesture → 把持姿勢 (正規化座標系)
            → (R, t, scale) でカメラ座標系へ変換
            → 画像に投影して保存 / ロボットへ送信

使用方法:
    # Step1: reference mesh を生成 (物体ごとに1回)
    python main.py --mode offline-mesh --mesh-out meshes/cup.ply

    # Step2: 把持姿勢生成 (毎回)
    python main.py --mesh meshes/cup.ply --no-robot
    python main.py --mesh meshes/cup.ply
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================
# オフラインフェーズ: reference mesh の生成・保存
# ==============================================================

def run_offline_mesh(config: dict, args):
    """
    RealSense RGB → サーバ (SAM-3D + SAM-6D テンプレート) → mesh 保存

    物体ごとに1回だけ実行する。
    生成した mesh は --mesh-out で指定したパスに保存される。
    """
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient

    cam_cfg = config["camera"]
    sam_cfg = config["sam3d"]

    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
    )

    print("\n[操作方法]")
    print("  [c] この画像で reference mesh を生成")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()
            key = camera.show_preview(rgb, depth)

            if key == ord("q"):
                break
            elif key == ord("c"):
                mesh_path = args.mesh_out
                print(f"\n[Step] reference mesh 生成中 → {mesh_path}")
                if sam_cfg.get("interactive", True):
                    client.save_reference_mesh_interactive(rgb, mesh_path)
                else:
                    client.save_reference_mesh(rgb, mesh_path)
                print(f"[完了] {mesh_path} に保存しました。")
                print("       次回: python main.py --mesh", mesh_path)
                break
    finally:
        camera.stop()


# ==============================================================
# オンラインフェーズ: RGBD → 6DoF pose → 把持姿勢 → ロボット
# ==============================================================

def run_online(config: dict, args):
    """
    毎フレーム: RGBD → SAM-6D pose → Shape2Gesture → ロボット
    """
    import cv2 as _cv2
    from datetime import datetime
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from pipeline.robot_interface import RobotInterface, GraspPose
    from utils.visualization import (
        live_visualize_setup, live_visualize_update,
        visualize_multiple_grasps, project_hands_on_image,
    )
    from utils.coord_transform import (
        CameraIntrinsics, ObjectPose,
        estimate_scale_from_depth, project_to_image, normalized_to_camera,
    )
    from utils.pointcloud_utils import load_pointcloud_ply, normalize_pointcloud

    cam_cfg   = config["camera"]
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]
    robot_cfg = config["robot"]
    vis_cfg   = config["visualization"]

    # ---- 初期化 ----
    print("=" * 50)
    print("[Step 1] 初期化中...")
    print("=" * 50)

    # Shape2Gesture モデルのロード
    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()

    # SAM-6D クライアント
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )
    # reference mesh のパスをセット (サーバ側パスは別途指定 or 初回 offline-mesh で取得)
    server_mesh_path = args.server_mesh_path or ""
    template_dir     = args.template_dir or ""
    client.load_reference_mesh(args.mesh, server_mesh_path, template_dir)

    # reference mesh の点群を読み込み (GraspGenerator 入力 + スケール推定)
    mesh_pts = load_pointcloud_ply(args.mesh, target_points=2048)
    mesh_pts_norm = normalize_pointcloud(mesh_pts)  # unit sphere (スケール推定用)

    # ロボット
    mode = robot_cfg["mode"]
    robot_kwargs = robot_cfg.get(mode, {})
    robot = RobotInterface(mode=mode, **robot_kwargs)
    if not args.no_robot:
        robot.connect()

    # カメラ
    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    # カメラ内部パラメータを camera.json として保存
    cam_json = {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "width": cam_cfg["width"],
        "height": cam_cfg["height"],
    }
    os.makedirs("output", exist_ok=True)
    with open("camera.json", "w") as f:
        json.dump(cam_json, f, indent=2)
    print("[Camera] 内部パラメータ保存: camera.json")

    intrinsics = CameraIntrinsics(
        fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy,
        width=cam_cfg["width"], height=cam_cfg["height"],
    )

    fig, ax = live_visualize_setup()

    print("\n[操作方法]")
    print("  [g] 把持姿勢を生成してロボットに送信")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()
            key = camera.show_preview(rgb, depth)

            if key == ord("q"):
                break

            elif key == ord("g"):
                out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(out_dir, exist_ok=True)

                # ---- Step 2: SAM-6D で 6DoF pose 推定 ----
                print("\n" + "=" * 50)
                print("[Step 2] SAM-6D で 6DoF pose 推定中...")
                print("=" * 50)

                R, t, img_pose, img_mesh = client.estimate_pose(
                    rgb, depth, intrinsics,
                    click_x=args.click_x, click_y=args.click_y,
                )

                # サーバから受信した投影画像を保存・表示
                if img_pose is not None:
                    path = os.path.join(out_dir, "server_pointcloud.png")
                    _cv2.imwrite(path, img_pose)
                    _cv2.imshow("Server: Pointcloud Projection", img_pose)
                    _cv2.waitKey(1)
                    print(f"[投影] 点群投影画像: {path}")

                if img_mesh is not None:
                    path = os.path.join(out_dir, "server_mesh.png")
                    _cv2.imwrite(path, img_mesh)
                    _cv2.imshow("Server: Mesh Projection", img_mesh)
                    _cv2.waitKey(1)
                    print(f"[投影] メッシュ投影画像: {path}")

                # mask_u, mask_v: スケール推定用 (tの投影)
                mask_u = int(intrinsics.fx * t[0] / max(t[2], 0.01) + intrinsics.cx)
                mask_v = int(intrinsics.fy * t[1] / max(t[2], 0.01) + intrinsics.cy)

                scale = estimate_scale_from_depth(
                    depth, mask_u, mask_v, intrinsics, mesh_pts_norm
                )

                # 6DoF pose オブジェクト (R, t, scale)
                pose = ObjectPose(center_3d=t, scale=scale, R=R)

                # ---- Step 3: Shape2Gesture で把持姿勢生成 ----
                print("\n" + "=" * 50)
                print("[Step 3] Shape2Gesture で把持姿勢を生成中...")
                print("=" * 50)

                grasp_results = generator.generate(
                    mesh_pts,
                    num_samples=model_cfg["num_samples"],
                )

                norm_pts, seg_labels = generator.get_segmentation(mesh_pts)

                if vis_cfg["show_all_samples"] and len(grasp_results) > 1:
                    visualize_multiple_grasps(
                        norm_pts, grasp_results,
                        labels=seg_labels if vis_cfg["show_segmentation"] else None,
                    )

                left_hand_norm, right_hand_norm = grasp_results[0]
                live_visualize_update(
                    fig, ax, norm_pts, left_hand_norm, right_hand_norm,
                    labels=seg_labels if vis_cfg["show_segmentation"] else None,
                )

                # ---- Step 4: 把持姿勢をカメラ画像に投影して保存・表示 ----
                for i, (lh_norm, rh_norm) in enumerate(grasp_results):
                    result_img = project_hands_on_image(
                        rgb, lh_norm, rh_norm,
                        object_pose=pose,
                        intrinsics=intrinsics,
                    )
                    save_path = os.path.join(out_dir, f"grasp_{i:02d}.png")
                    _cv2.imwrite(save_path, result_img)

                print(f"[投影結果] {len(grasp_results)}枚を {out_dir} に保存しました。")
                _cv2.imshow("Grasp Projection",
                            _cv2.imread(os.path.join(out_dir, "grasp_00.png")))
                _cv2.waitKey(1)

                # ---- Step 5: ロボットへ送信 ----
                print("\n" + "=" * 50)
                print("[Step 5] ロボットに把持姿勢を送信中...")
                print("=" * 50)

                left_hand_cam  = normalized_to_camera(left_hand_norm, pose)
                right_hand_cam = normalized_to_camera(right_hand_norm, pose)

                if not args.no_robot:
                    grasp_pose = GraspPose(
                        left_hand=left_hand_cam,
                        right_hand=right_hand_cam,
                        object_scale=scale,
                        object_center=t,
                    )
                    robot.send_grasp_pose(grasp_pose, execute=robot_cfg["execute"])
                    result = robot.wait_for_result(timeout=robot_cfg["timeout"])
                    print(f"[Robot] 結果: {result}")
                else:
                    print("[main] --no-robot: ロボット送信をスキップ")

    except KeyboardInterrupt:
        print("\n[main] 中断されました。")
    finally:
        camera.stop()
        if not args.no_robot:
            robot.disconnect()


# ==============================================================
# フルパイプライン: mesh生成 → pose推定 → 把持姿勢生成
# ==============================================================

def run_full(config: dict, args):
    """
    フルパイプライン (カメラ起動から把持まで一連で実行)

    操作:
        [c] 物体をクリックして選択 → 3D mesh 生成
        [g] pose 推定 → 把持姿勢生成 → 画像投影 ([c] 済みが必要)
        [q] 終了
    """
    import cv2 as _cv2
    from datetime import datetime
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from pipeline.robot_interface import RobotInterface, GraspPose
    from utils.visualization import (
        live_visualize_setup, live_visualize_update,
        visualize_multiple_grasps, project_hands_on_image,
    )
    from utils.coord_transform import (
        CameraIntrinsics, ObjectPose,
        estimate_scale_from_depth, normalized_to_camera,
    )
    from utils.pointcloud_utils import load_pointcloud_ply, normalize_pointcloud

    cam_cfg   = config["camera"]
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]
    robot_cfg = config["robot"]
    vis_cfg   = config["visualization"]

    mesh_path  = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")

    # ---- 初期化 ----
    print("=" * 50)
    print("[初期化] Shape2Gesture モデルをロード中...")
    print("=" * 50)

    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mode = robot_cfg["mode"]
    robot_kwargs = robot_cfg.get(mode, {})
    robot = RobotInterface(mode=mode, **robot_kwargs)
    if not args.no_robot:
        robot.connect()

    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    # カメラ内部パラメータを camera.json として保存
    cam_json = {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "width": cam_cfg["width"],
        "height": cam_cfg["height"],
    }
    os.makedirs("output", exist_ok=True)
    with open("camera.json", "w") as f:
        json.dump(cam_json, f, indent=2)
    print("[Camera] 内部パラメータ保存: camera.json")

    intrinsics = CameraIntrinsics(
        fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy,
        width=cam_cfg["width"], height=cam_cfg["height"],
    )

    fig, ax = live_visualize_setup()

    # 状態変数
    mesh_pts      = None   # (N, 3) mesh 点群
    mesh_pts_norm = None   # unit sphere 正規化済み
    click_x = click_y = -1

    print("\n[操作方法]")
    print("  [c] 物体をクリック選択 → 3D mesh 生成")
    print("  [g] pose 推定 → 把持姿勢生成 ([c] 実行後に有効)")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()

            # mesh 生成済みかどうかをプレビューに表示
            status = "mesh: OK  [g]で把持" if mesh_pts is not None else "mesh: なし  [c]で物体選択"
            preview = rgb.copy()
            _cv2.putText(preview, status, (10, 30),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            key = camera.show_preview(preview, depth)

            if key == ord("q"):
                break

            elif key == ord("c"):
                # ---- mesh 生成 ----
                print("\n" + "=" * 50)
                print("[mesh生成] 物体をクリックして選択してください...")
                print("=" * 50)
                os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)
                _, click_x, click_y = client.save_reference_mesh_interactive(
                    rgb, mesh_path, mesh_method=mesh_method
                )
                mesh_pts      = load_pointcloud_ply(mesh_path, target_points=2048)
                mesh_pts_norm = normalize_pointcloud(mesh_pts)
                print(f"[mesh生成完了] {mesh_path}  クリック座標: ({click_x}, {click_y})")
                print("[次のステップ] [g] を押して把持姿勢を生成してください。")

            elif key == ord("g"):
                if mesh_pts is None:
                    print("[警告] 先に [c] で mesh を生成してください。")
                    continue

                out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(out_dir, exist_ok=True)

                # ---- pose 推定 ----
                print("\n" + "=" * 50)
                print("[pose推定] SAM-6D で 6DoF pose 推定中...")
                print("=" * 50)

                R, t, img_pose, img_mesh = client.estimate_pose(
                    rgb, depth, intrinsics,
                    click_x=click_x, click_y=click_y,
                )

                if img_pose is not None:
                    path = os.path.join(out_dir, "server_pointcloud.png")
                    _cv2.imwrite(path, img_pose)
                    _cv2.imshow("Server: Pointcloud Projection", img_pose)
                    _cv2.waitKey(1)
                    print(f"[投影] 点群投影画像: {path}")

                if img_mesh is not None:
                    path = os.path.join(out_dir, "server_mesh.png")
                    _cv2.imwrite(path, img_mesh)
                    _cv2.imshow("Server: Mesh Projection", img_mesh)
                    _cv2.waitKey(1)
                    print(f"[投影] メッシュ投影画像: {path}")

                # スケール推定
                mask_u = int(intrinsics.fx * t[0] / max(t[2], 0.01) + intrinsics.cx)
                mask_v = int(intrinsics.fy * t[1] / max(t[2], 0.01) + intrinsics.cy)
                scale = estimate_scale_from_depth(
                    depth, mask_u, mask_v, intrinsics, mesh_pts_norm
                )
                pose = ObjectPose(center_3d=t, scale=scale, R=R)

                # ---- 把持姿勢生成 ----
                print("\n" + "=" * 50)
                print("[把持生成] Shape2Gesture で把持姿勢を生成中...")
                print("=" * 50)

                grasp_results = generator.generate(
                    mesh_pts,
                    num_samples=model_cfg["num_samples"],
                )

                norm_pts, seg_labels = generator.get_segmentation(mesh_pts)

                if vis_cfg["show_all_samples"] and len(grasp_results) > 1:
                    visualize_multiple_grasps(
                        norm_pts, grasp_results,
                        labels=seg_labels if vis_cfg["show_segmentation"] else None,
                    )

                left_hand_norm, right_hand_norm = grasp_results[0]
                live_visualize_update(
                    fig, ax, norm_pts, left_hand_norm, right_hand_norm,
                    labels=seg_labels if vis_cfg["show_segmentation"] else None,
                )

                # ---- 画像投影・保存 ----
                for i, (lh_norm, rh_norm) in enumerate(grasp_results):
                    result_img = project_hands_on_image(
                        rgb, lh_norm, rh_norm,
                        object_pose=pose,
                        intrinsics=intrinsics,
                    )
                    save_path = os.path.join(out_dir, f"grasp_{i:02d}.png")
                    _cv2.imwrite(save_path, result_img)

                print(f"[投影結果] {len(grasp_results)}枚を {out_dir} に保存しました。")
                _cv2.imshow("Grasp Projection",
                            _cv2.imread(os.path.join(out_dir, "grasp_00.png")))
                _cv2.waitKey(1)

                # ---- ロボットへ送信 ----
                left_hand_cam  = normalized_to_camera(left_hand_norm, pose)
                right_hand_cam = normalized_to_camera(right_hand_norm, pose)

                if not args.no_robot:
                    grasp_pose = GraspPose(
                        left_hand=left_hand_cam,
                        right_hand=right_hand_cam,
                        object_scale=scale,
                        object_center=t,
                    )
                    robot.send_grasp_pose(grasp_pose, execute=robot_cfg["execute"])
                    result = robot.wait_for_result(timeout=robot_cfg["timeout"])
                    print(f"[Robot] 結果: {result}")
                else:
                    print("[main] --no-robot: ロボット送信をスキップ")

    except KeyboardInterrupt:
        print("\n[main] 中断されました。")
    finally:
        camera.stop()
        if not args.no_robot:
            robot.disconnect()


# ==============================================================
# エントリーポイント
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description="real_world_demo: RealSense + SAM-3D + SAM-6D + Shape2Gesture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # Step1: reference mesh を生成 (物体ごとに1回)
  python main.py --mode offline-mesh --mesh-out meshes/cup.ply

  # Step2: 把持姿勢生成 (毎回)
  python main.py --mesh meshes/cup.ply --no-robot
  python main.py --mesh meshes/cup.ply
        """
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["full", "online", "offline-mesh"], default="full",
        help="full: 物体選択からすべて一括実行 (デフォルト) / "
             "online: mesh 指定済みで pose 推定のみ / "
             "offline-mesh: mesh 生成のみ",
    )
    parser.add_argument(
        "--mesh", default=None,
        help="[online] ローカルの reference mesh (.ply) パス",
    )
    parser.add_argument(
        "--mesh-out", default="meshes/object.ply",
        help="[full/offline-mesh] mesh 保存先 .ply パス",
    )
    parser.add_argument(
        "--server-mesh-path", default=None,
        help="[online] サーバ側の mesh パス",
    )
    parser.add_argument(
        "--template-dir", default=None,
        help="[online] サーバ側のテンプレートディレクトリ",
    )
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epoch",       type=int, default=None)
    parser.add_argument(
        "--click-x", type=int, default=-1,
        help="[online] 物体クリック座標 X (-1: 画像中央)",
    )
    parser.add_argument(
        "--click-y", type=int, default=-1,
        help="[online] 物体クリック座標 Y (-1: 画像中央)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"エラー: 設定ファイルが見つかりません: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    if args.num_samples is not None:
        config["grasp_model"]["num_samples"] = args.num_samples
    if args.epoch is not None:
        config["grasp_model"]["epoch"] = args.epoch

    print("=" * 50)
    print("  real_world_demo")
    print("  RealSense + SAM-3D + SAM-6D + Shape2Gesture")
    print(f"  モード: {args.mode}")
    print("=" * 50)

    if args.mode == "full":
        run_full(config, args)

    elif args.mode == "offline-mesh":
        run_offline_mesh(config, args)

    else:  # online
        if args.mesh is None:
            print("エラー: --mesh で reference mesh (.ply) を指定してください。")
            sys.exit(1)
        if not os.path.exists(args.mesh):
            print(f"エラー: mesh が見つかりません: {args.mesh}")
            sys.exit(1)
        run_online(config, args)


if __name__ == "__main__":
    main()
