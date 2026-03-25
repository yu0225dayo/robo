"""
SAM-3D デバッグスクリプト

RGB画像をクリックして物体を選択し、SAM-3D で生成した点群を受け取って可視化する。

使用方法:
    python debug_sam3d.py --rgb test_data/rgb.png
"""

import argparse
import os
import sys
import numpy as np
import cv2
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def pick_point(bgr: np.ndarray) -> tuple:
    """画像をウィンドウ表示してクリックした座標を返す"""
    clicked = {}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["x"] = x
            clicked["y"] = y

    disp = bgr.copy()
    h, w = disp.shape[:2]
    cv2.putText(disp, "Click object, then press Enter",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.namedWindow("Select Object")
    cv2.setMouseCallback("Select Object", on_click)

    while True:
        show = disp.copy()
        if "x" in clicked:
            cv2.drawMarker(show, (clicked["x"], clicked["y"]),
                           (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow("Select Object", show)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and "x" in clicked:   # Enter で決定
            break
        if key == 27:                       # Esc でキャンセル
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    return clicked["x"], clicked["y"]


def main():
    parser = argparse.ArgumentParser(description="SAM-3D デバッグ")
    parser.add_argument("--rgb", required=True, help="RGB画像パス")
    parser.add_argument("--server", default="http://10.40.1.126:8080", help="サーバURL")
    parser.add_argument("--points", type=int, default=2048, help="受け取る点数")
    parser.add_argument("--save", default="output/debug_sam3d.png", help="結果画像の保存先")
    args = parser.parse_args()

    # RGB画像読み込み
    bgr = cv2.imread(args.rgb)
    if bgr is None:
        print(f"[ERROR] 画像を読み込めません: {args.rgb}")
        sys.exit(1)

    # クリックで物体選択
    print("[Step 0] 画像上でクリックして物体を選択 → Enter で決定 / Esc でキャンセル")
    click_x, click_y = pick_point(bgr)
    print(f"[Step 0] 選択座標: ({click_x}, {click_y})")

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

    # matplotlib で結果を表示
    print("[Step 2] 点群を表示中...")
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 5))

        # 左: 3D点群
        ax3d = fig.add_subplot(121, projection="3d")
        ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="green")
        ax3d.set_title("SAM-3D 点群")
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

        # 右: RGB + クリック位置 + マスク重心
        ax2d = fig.add_subplot(122)
        ax2d.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax2d.plot(click_x, click_y, "r+", markersize=15, markeredgewidth=2, label="クリック")
        ax2d.plot(mask_u, mask_v, "yo", markersize=10, label="マスク重心")
        ax2d.legend()
        ax2d.set_title("入力画像 + SAMプロンプト")

        plt.tight_layout()
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=120)
        print(f"[Step 2] 保存: {args.save}")
        plt.show()

    except ImportError:
        print("[Warning] matplotlib がないためスキップ")

    # Open3D で点群をインタラクティブ表示
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.0, 1.0, 0.0])
        print("[Step 3] Open3D で点群を表示 (ウィンドウを閉じると終了)...")
        o3d.visualization.draw_geometries([pcd], window_name="SAM-3D 点群確認")
    except ImportError:
        print("[Info] open3d がないためスキップ")


if __name__ == "__main__":
    main()
