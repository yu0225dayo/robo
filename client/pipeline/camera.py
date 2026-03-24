"""
Intel RealSense RGBDカメラインターフェース

RealSense D400シリーズ (D435, D415 等) からRGBD画像を取得し、
点群データとして出力するモジュール。

依存ライブラリ:
    pip install pyrealsense2 opencv-python numpy open3d
"""

import numpy as np
import cv2


class RealSenseCamera:
    """
    Intel RealSense カメラの制御クラス

    使用例:
        camera = RealSenseCamera(width=640, height=480, fps=30)
        camera.start()
        rgb, depth, pointcloud = camera.capture()
        camera.stop()
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Args:
            width:  カラー/深度フレームの幅 (pixel)
            height: カラー/深度フレームの高さ (pixel)
            fps:    フレームレート
        """
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 が見つかりません。\n"
                "インストール: pip install pyrealsense2"
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.align = None
        self._running = False

    def start(self):
        """カメラストリームを開始する"""
        rs = self.rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # カラーストリーム設定
        self.config.enable_stream(
            rs.stream.color, self.width, self.height,
            rs.format.bgr8, self.fps
        )
        # 深度ストリーム設定
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height,
            rs.format.z16, self.fps
        )

        profile = self.pipeline.start(self.config)

        # 深度スケール取得
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 深度フレームをカラーフレームに合わせるアライン処理
        self.align = rs.align(rs.stream.color)

        # カメラ内部パラメータ取得
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy

        self._running = True
        print(f"[Camera] RealSense 起動完了 ({self.width}x{self.height} @ {self.fps}fps)")
        print(f"[Camera] 内部パラメータ: fx={self.fx:.2f}, fy={self.fy:.2f}, "
              f"cx={self.cx:.2f}, cy={self.cy:.2f}")

    def capture(self):
        """
        1フレーム分のRGBD画像と点群を取得する

        Returns:
            rgb:        (H, W, 3) uint8 numpy array (BGR)
            depth:      (H, W) float32 numpy array [メートル単位]
            pointcloud: (N, 3) float32 numpy array [メートル単位]
        """
        if not self._running:
            raise RuntimeError("カメラが起動していません。start() を先に呼んでください。")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("フレーム取得に失敗しました。")

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth = depth * self.depth_scale  # ミリメートル → メートル

        pointcloud = self._depth_to_pointcloud(depth)
        return rgb, depth, pointcloud

    def _depth_to_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        """
        深度画像から点群を生成する

        Args:
            depth: (H, W) float32 深度画像 [メートル]

        Returns:
            (N, 3) float32 点群 (有効点のみ)
        """
        h, w = depth.shape
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        z = depth
        x = (u_coords - self.cx) * z / self.fx
        y = (v_coords - self.cy) * z / self.fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # 深度が0の無効点を除去
        valid_mask = z.reshape(-1) > 0
        points = points[valid_mask]

        return points.astype(np.float32)

    def show_preview(self, rgb: np.ndarray, depth: np.ndarray, window_name: str = "RealSense Preview"):
        """
        RGB・深度画像をウィンドウ表示する

        Args:
            rgb:    (H, W, 3) カラー画像 (BGR)
            depth:  (H, W) 深度画像 [メートル]
        """
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        combined = np.hstack([rgb, depth_colormap])
        cv2.imshow(window_name, combined)
        return cv2.waitKey(1)

    def stop(self):
        """カメラストリームを停止する"""
        if self._running and self.pipeline is not None:
            self.pipeline.stop()
            self._running = False
            cv2.destroyAllWindows()
            print("[Camera] RealSense 停止")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
