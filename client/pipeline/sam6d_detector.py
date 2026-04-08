"""
SAM-6D クライアントモジュール (ローカルPC側)

サーバとの通信:
    オフライン (物体ごとに1回):
        RGB → /reconstruct_mesh → reference mesh (.ply) 保存
        サーバが SAM-3D でメッシュ生成 + SAM-6D でテンプレートレンダリング

    オンライン (毎フレーム):
        RGB + depth → /pose_estimate → 6DoF pose (R, t) 取得
        サーバが SAM-6D でポーズ推定
"""

import os
import numpy as np
import cv2
import requests
from typing import Tuple, Optional

from utils.coord_transform import CameraIntrinsics


class SAM6DClient:
    """
    SAM-6D サーバへの HTTP クライアント

    使用例 (オフライン):
        client = SAM6DClient(server_url="http://10.40.1.126:8080")
        client.save_reference_mesh(rgb, "meshes/cup.ply")

    使用例 (オンライン):
        client.load_reference_mesh("meshes/cup.ply")
        R, t = client.estimate_pose(rgb, depth, intrinsics)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout_mesh: float = 300.0,
        timeout_pose: float = 30.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout_mesh = timeout_mesh
        self.timeout_pose = timeout_pose
        self._mesh_path: Optional[str] = None      # ローカルの .ply パス
        self._server_mesh_path: str = ""            # サーバ側の .ply パス
        self._template_dir: str = ""               # サーバ側テンプレートディレクトリ

    # ------------------------------------------------------------------
    # オフライン: reference mesh の生成・保存
    # ------------------------------------------------------------------

    def save_reference_mesh(
        self,
        rgb: np.ndarray,
        mesh_save_path: str,
        click_x: int = -1,
        click_y: int = -1,
        seed: int = 42,
        mesh_method: str = "bpa",
    ) -> str:
        """
        RGB画像をサーバに送り、SAM-3D で生成した reference mesh を保存する

        サーバは SAM-3D でメッシュ生成後、SAM-6D でテンプレートをレンダリングし
        テンプレートディレクトリのパスをヘッダで返す。

        Args:
            rgb:             (H, W, 3) BGR 画像
            mesh_save_path:  ローカルの保存先 .ply パス
            click_x, click_y: SAM プロンプト座標 (-1,-1 で画像中央)
            seed:            SAM-3D シード値

        Returns:
            保存した .ply のパス
        """
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_bytes = buf.tobytes()

        print(f"[SAM6D] reference mesh 生成中 (プロンプト: ({click_x},{click_y}))...")
        resp = requests.post(
            f"{self.server_url}/reconstruct_mesh",
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={"click_x": click_x, "click_y": click_y, "seed": seed, "mesh_method": mesh_method},
            timeout=self.timeout_mesh,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"サーバエラー ({resp.status_code}): {resp.text}")

        os.makedirs(os.path.dirname(os.path.abspath(mesh_save_path)), exist_ok=True)
        with open(mesh_save_path, "wb") as f:
            f.write(resp.content)

        self._server_mesh_path = resp.headers.get("X-Mesh-Path", "")
        self._template_dir     = resp.headers.get("X-Template-Dir", "")
        mask_u = resp.headers.get("X-Mask-Center-U", "?")
        mask_v = resp.headers.get("X-Mask-Center-V", "?")

        print(f"[SAM6D] mesh 保存完了: {mesh_save_path}  mask_center=({mask_u},{mask_v})")
        if self._server_mesh_path:
            print(f"[SAM6D] サーバ側 mesh: {self._server_mesh_path}")
        if self._template_dir:
            print(f"[SAM6D] テンプレート: {self._template_dir}")

        self._mesh_path = mesh_save_path
        return mesh_save_path

    def save_reference_mesh_interactive(
        self,
        rgb: np.ndarray,
        mesh_save_path: str,
        seed: int = 42,
        mesh_method: str = "bpa",
    ) -> Tuple[str, int, int]:
        """インタラクティブモード: クリックして物体を指定する"""
        clicked = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked.clear()
                clicked.append((x, y))

        cv2.namedWindow("Select Object (click + Enter)")
        cv2.setMouseCallback("Select Object (click + Enter)", mouse_callback)
        print("[SAM6D] 物体をクリックして選択し、Enter で確定してください。")

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
        result = self.save_reference_mesh(rgb, mesh_save_path,
                                          click_x=cx, click_y=cy, seed=seed,
                                          mesh_method=mesh_method)
        return result, cx, cy

    def load_reference_mesh(
        self,
        mesh_path: str,
        server_mesh_path: str = "",
        template_dir: str = "",
    ):
        """
        保存済みの reference mesh をセットする (セッション再開時)

        Args:
            mesh_path:        ローカルの .ply パス
            server_mesh_path: サーバ側の .ply パス
            template_dir:     サーバ側テンプレートディレクトリ
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"reference mesh が見つかりません: {mesh_path}")
        self._mesh_path        = mesh_path
        self._server_mesh_path = server_mesh_path
        self._template_dir     = template_dir
        print(f"[SAM6D] reference mesh ロード: {mesh_path}")

    # ------------------------------------------------------------------
    # オンライン: 6DoF pose 推定
    # ------------------------------------------------------------------

    def estimate_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        click_x: int = -1,
        click_y: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        RGBD + reference mesh からカメラ座標系での物体 6DoF pose を推定する

        サーバは SAM-6D (Docker) で推定し R, t を返す。
        メッシュファイルはサーバ側に保存済みのパスを指定するため再送不要。

        Args:
            rgb:        (H, W, 3) BGR 画像
            depth:      (H, W) float32 深度画像 [m]
            intrinsics: カメラ内部パラメータ

        Returns:
            R: (3, 3) float32 回転行列 (物体座標系 → カメラ座標系)
            t: (3,)   float32 平行移動ベクトル [m]
        """
        if not self._server_mesh_path:
            raise RuntimeError(
                "サーバ側 mesh パスが未設定です。"
                "save_reference_mesh() を実行するか、"
                "load_reference_mesh(server_mesh_path=...) で指定してください。"
            )

        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rgb_bytes   = buf.tobytes()
        depth_bytes = depth.astype(np.float32).tobytes()

        print("[SAM6D] 6DoF pose 推定中...")
        resp = requests.post(
            f"{self.server_url}/pose_estimate",
            files={
                "rgb_image":   ("frame.jpg", rgb_bytes,   "image/jpeg"),
                "depth_image": ("depth.bin", depth_bytes, "application/octet-stream"),
            },
            data={
                "fx":           intrinsics.fx,
                "fy":           intrinsics.fy,
                "cx":           intrinsics.cx,
                "cy":           intrinsics.cy,
                "mesh_path":    self._server_mesh_path,
                "template_dir": self._template_dir,
                "click_x":      click_x,
                "click_y":      click_y,
            },
            timeout=self.timeout_pose,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"サーバエラー ({resp.status_code}): {resp.text}")

        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"pose 推定失敗: {data.get('error')}")

        R = np.array(data["R"], dtype=np.float32)  # (3, 3)
        t = np.array(data["t"], dtype=np.float32)  # (3,)
        print(f"[SAM6D] pose 推定完了: t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")

        # サーバ生成画像をデコードして返す
        import base64, cv2 as _cv2
        img_pose_bgr: Optional[np.ndarray] = None
        img_mesh_bgr: Optional[np.ndarray] = None
        for key, attr in [("img_pose", "pose"), ("img_mesh", "mesh")]:
            b64 = data.get(key, "")
            if b64:
                img = _cv2.imdecode(
                    np.frombuffer(base64.b64decode(b64), np.uint8),
                    _cv2.IMREAD_COLOR,
                )
                if key == "img_pose":
                    img_pose_bgr = img
                else:
                    img_mesh_bgr = img

        return R, t, img_pose_bgr, img_mesh_bgr
