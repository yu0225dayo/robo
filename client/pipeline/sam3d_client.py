"""
SAM-3D クライアントモジュール (Windows側)

RGB画像をサーバに送り、SAM-3D で生成した完全3D点群 (N,3) を取得する。

依存ライブラリ:
    pip install requests opencv-python numpy
"""

import numpy as np
import cv2
import requests
from typing import Tuple


class SAM3DClient:
    """
    SAM-3D サーバへの HTTP クライアント

    使用例:
        client = SAM3DClient(server_url="http://192.168.1.200:8080")
        points, mu, mv = client.get_point_cloud(rgb)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout: float = 120.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def get_point_cloud(
        self,
        rgb: np.ndarray,
        click_x: int = -1,
        click_y: int = -1,
        seed: int = 42,
        target_points: int = 2048,
    ) -> Tuple[np.ndarray, int, int]:
        """
        RGB画像をサーバに送り、SAM-3D で生成した完全3D点群を取得する

        Args:
            rgb:           (H, W, 3) BGR画像
            click_x, click_y: SAMプロンプト座標 (-1,-1 で画像中央)
            seed:          SAM-3D シード値
            target_points: 受け取る点群の点数

        Returns:
            points:       (N, 3) float32 正規化点群 (unit sphere)
            mask_center_u: int マスク重心U座標
            mask_center_v: int マスク重心V座標
        """
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_bytes = buf.tobytes()

        print(f"[SAM3D] 3D点群生成中 (プロンプト: ({click_x},{click_y}))...")
        resp = requests.post(
            f"{self.server_url}/reconstruct",
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={
                "click_x": click_x,
                "click_y": click_y,
                "seed": seed,
                "target_points": target_points,
            },
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"サーバエラー ({resp.status_code}): {resp.text}")

        data = resp.json()
        points = np.array(data["points"], dtype=np.float32)
        mask_u = data.get("mask_center_u", -1)
        mask_v = data.get("mask_center_v", -1)
        print(f"[SAM3D] 点群取得完了: {len(points)} points, mask_center=({mask_u},{mask_v})")
        return points, mask_u, mask_v

    def get_point_cloud_interactive(
        self,
        rgb: np.ndarray,
        seed: int = 42,
        target_points: int = 2048,
    ) -> Tuple[np.ndarray, int, int]:
        """
        インタラクティブモード: ウィンドウ上でクリックして物体を指定する
        """
        clicked = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked.clear()
                clicked.append((x, y))

        cv2.namedWindow("Select Object (click + Enter)")
        cv2.setMouseCallback("Select Object (click + Enter)", mouse_callback)
        print("[SAM3D] 物体をクリックして選択し、Enter で確定してください。")

        while True:
            display = rgb.copy()
            if clicked:
                cv2.circle(display, clicked[0], 8, (0, 255, 0), -1)
                cv2.putText(display, "Press Enter to confirm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Select Object (click + Enter)", display)
            key = cv2.waitKey(1)
            if key == 13 and clicked:
                break
            elif key == 27:
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("キャンセルされました。")

        cv2.destroyAllWindows()
        cx, cy = clicked[0]
        return self.get_point_cloud(rgb, click_x=cx, click_y=cy,
                                    seed=seed, target_points=target_points)
