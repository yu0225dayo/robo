"""
SAM 3D Objects クライアントモジュール (Windows側)

Linux サーバ (A6000) 上で動く server/server.py に
RGB画像を送信し、完全3D点群を受け取る。

依存ライブラリ:
    pip install requests opencv-python numpy
"""

import numpy as np
import cv2
import requests
from typing import Optional, Tuple

from utils.coord_transform import CameraIntrinsics, ObjectPose, estimate_object_pose, add_rotation


class SAM3DClient:
    """
    sam-3d-objects サーバへの HTTP クライアント

    使用例:
        client = SAM3DClient(server_url="http://192.168.1.200:8080")
        client.check_server()
        pointcloud = client.reconstruct_3d_interactive(rgb)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        target_points: int = 2048,
        timeout: float = 120.0,
    ):
        """
        Args:
            server_url:    サーバURL (例: "http://192.168.1.200:8080")
            target_points: 受け取る点群の点数
            timeout:       タイムアウト秒数 (3D生成は数十秒かかる)
        """
        self.server_url = server_url.rstrip("/")
        self.target_points = target_points
        self.timeout = timeout

    def check_server(self) -> bool:
        """サーバの起動確認"""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5.0)
            data = resp.json()
            if data.get("models_loaded"):
                print(f"[SAM3D Client] サーバ接続OK: {self.server_url}")
                return True
            else:
                print(f"[SAM3D Client] サーバは起動中ですがモデル未ロード")
                return False
        except Exception as e:
            print(f"[SAM3D Client] サーバ接続失敗: {e}")
            print(f"  → サーバ起動コマンド (Linux): python server/server.py ...")
            return False

    def reconstruct_3d(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        click_x: int = -1,
        click_y: int = -1,
        seed: int = 42,
    ) -> Tuple[np.ndarray, Optional[ObjectPose]]:
        """
        RGB画像をサーバに送信して3D点群を取得する

        Args:
            rgb:              (H, W, 3) BGR画像
            depth:            (H, W) uint16 深度画像 [mm] (座標変換に使用)
            intrinsics:       カメラ内部パラメータ (座標変換に使用)
            click_x, click_y: SAMプロンプト座標 (-1,-1 で画像中央)
            seed:             sam-3d-objects のシード値

        Returns:
            points:      (N, 3) 正規化座標系の点群
            object_pose: カメラ座標系でのオブジェクト姿勢 (depthがあれば)
        """
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_bytes = buf.tobytes()

        print(f"[SAM3D Client] サーバに画像送信中... "
              f"(プロンプト: ({click_x}, {click_y}), seed={seed})")

        resp = requests.post(
            f"{self.server_url}/reconstruct",
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={
                "click_x": click_x,
                "click_y": click_y,
                "seed": seed,
                "target_points": self.target_points,
            },
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"サーバエラー ({resp.status_code}): {resp.text}"
            )

        data = resp.json()
        points = np.array(data["points"], dtype=np.float32)
        print(f"[SAM3D Client] 点群受信完了: {data['num_points']} points")

        # 深度画像があればカメラ座標系でのオブジェクト姿勢を推定
        object_pose = None
        if depth is not None and intrinsics is not None:
            # サーバから返ってきたマスク重心ピクセルを使って深度を取得
            u_c = data.get("mask_center_u", rgb.shape[1] // 2)
            v_c = data.get("mask_center_v", rgb.shape[0] // 2)

            # マスク重心周辺の小領域からオブジェクト姿勢を推定
            r = 80  # 重心周辺80px
            h, w = depth.shape
            y0, y1 = max(0, v_c - r), min(h, v_c + r)
            x0, x1 = max(0, u_c - r), min(w, u_c + r)
            local_mask = np.zeros_like(depth, dtype=bool)
            local_mask[y0:y1, x0:x1] = depth[y0:y1, x0:x1] > 0

            # depth は camera.py で既にメートル変換済み → depth_scale=1.0
            object_pose = estimate_object_pose(depth, local_mask, intrinsics, depth_scale=1.0)

        return points, object_pose

    def reconstruct_3d_interactive(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, Optional[ObjectPose]]:
        """
        インタラクティブモード: ウィンドウ上でクリックして物体を指定する

        Args:
            rgb:        (H, W, 3) BGR画像
            depth:      (H, W) uint16 深度画像 [mm]
            intrinsics: カメラ内部パラメータ
            seed:       ランダムシード

        Returns:
            points:      (N, 3) 点群
            object_pose: カメラ座標系でのオブジェクト姿勢
        """
        clicked = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked.clear()
                clicked.append((x, y))

        cv2.namedWindow("Select Object (click + Enter)")
        cv2.setMouseCallback("Select Object (click + Enter)", mouse_callback)
        print("[SAM3D Client] 物体をクリックして選択し、Enter で確定してください。")

        while True:
            display = rgb.copy()
            if clicked:
                cv2.circle(display, clicked[0], 8, (0, 255, 0), -1)
                cv2.putText(display, "Press Enter to confirm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Select Object (click + Enter)", display)
            key = cv2.waitKey(1)
            if key == 13 and clicked:   # Enter
                break
            elif key == 27:             # ESC
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("キャンセルされました。")

        cv2.destroyAllWindows()
        cx, cy = clicked[0]
        return self.reconstruct_3d(rgb, depth, intrinsics,
                                   click_x=cx, click_y=cy, seed=seed)
