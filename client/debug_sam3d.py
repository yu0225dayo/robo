"""
SAM-3D デバッグスクリプト

RGB画像をサーバに送り、SAM-3D で生成した点群を受け取って可視化する。
SAM-6D や Shape2Gesture は使わず、SAM-3D の動作のみ確認する。

使用方法:
    python debug_sam3d.py --rgb test_data/rgb.png
    python debug_sam3d.py --rgb test_data/rgb.png --click-x 320 --click-y 240
"""

import argparse
import os
import sys
import numpy as np
import cv2
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="SAM-3D デバッグ")
    parser.add_argument("--rgb", required=True, help="RGB画像パス")
    parser.add_argument("--server", default="http://10.40.1.126:8080", help="サーバURL")
    parser.add_argument("--click-x", type=int, default=-1, help="SAMプロンプトX座標 (-1で中央)")
    parser.add_argument("--click-y", type=int, default=-1, help="SAMプロンプトY座標 (-1で中央)")
    parser.add_argument("--points", type=int, default=2048, help="受け取る点数")
    parser.add_argument("--save", default="output/debug_sam3d.png", help="結果画像の保存先")
    args = parser.parse_args()

    # RGB画像読み込み
    bgr = cv2.imread(args.rgb)
    if bgr is None:
        print(f"[ERROR] 画像を読み込めません: {args.rgb}")
        sys.exit(1)
    h, w = bgr.shape[:2]
    click_x = args.click_x if args.click_x >= 0 else w // 2
    click_y = args.click_y if args.click_y >= 0 else h // 2
    print(f"[Info] 画像サイズ: {w}x{h}, プロンプト: ({click_x}, {click_y})")

    # サーバに送信
    print(f"[Step 1] サーバ ({args.server}) に送信中...")
    with open(args.rgb, "rb") as f:
        resp = requests.post(
            f"{args.server}/reconstruct",
            files={"image": ("rgb.png", f, "image/png")},
            data={
                "click_x": click_x,
                "click_y": click_y,
                "target_points": args.points,
            },
            timeout=300,
        )

    if resp.status_code != 200:
        print(f"[ERROR] サーバエラー ({resp.status_code}): {resp.text}")
        sys.exit(1)

    result = resp.json()
    points = np.array(result["points"], dtype=np.float32)  # (N, 3)
    mask_u = result["mask_center_u"]
    mask_v = result["mask_center_v"]
    print(f"[Step 1] 完了: {len(points)} 点を受信")
    print(f"  X: {points[:,0].min():.3f} ~ {points[:,0].max():.3f}")
    print(f"  Y: {points[:,1].min():.3f} ~ {points[:,1].max():.3f}")
    print(f"  Z: {points[:,2].min():.3f} ~ {points[:,2].max():.3f}")
    print(f"  マスク重心: ({mask_u}, {mask_v})")

    # 点群をMatplotlibで3D表示
    print("[Step 2] 点群を3D表示中...")
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 5))

        # 左: 3D点群
        ax3d = fig.add_subplot(121, projection="3d")
        ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="green")
        ax3d.set_title("SAM-3D 点群")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        # 右: RGB + マスク重心
        ax2d = fig.add_subplot(122)
        rgb_disp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax2d.imshow(rgb_disp)
        ax2d.plot(click_x, click_y, "r+", markersize=15, markeredgewidth=2, label="プロンプト")
        ax2d.plot(mask_u, mask_v, "yo", markersize=10, label="マスク重心")
        ax2d.legend()
        ax2d.set_title("入力画像 + SAMプロンプト")

        plt.tight_layout()
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        plt.savefig(args.save, dpi=120)
        print(f"[Step 2] 結果を保存: {args.save}")
        plt.show()

    except ImportError:
        print("[Warning] matplotlib がないため3D表示をスキップ")

    # Open3D で点群表示
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.0, 1.0, 0.0])
        print("[Step 3] Open3D で点群を表示中 (ウィンドウを閉じると終了)...")
        o3d.visualization.draw_geometries([pcd], window_name="SAM-3D 点群確認")
    except ImportError:
        print("[Info] open3d がないためスキップ")


if __name__ == "__main__":
    main()
