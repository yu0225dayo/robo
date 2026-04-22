"""
RealSense から RGB・深度・IMU（重力ベクトル）・カメラパラメータを保存するスクリプト

引数なしで実行すると saved_data/test_YYYYMMDD_HHMMSS/ に自動保存する。
カウントダウン中に IMU サンプルを収集するため待ち時間は1回分。

保存されるファイル:
    rgb.png       - カラー画像 (BGR, 8bit)
    depth.png     - 深度画像 (uint16, mm 単位)
    cam.json      - カメラ内部パラメータ + depth_scale
    gravity.npy   - 重力方向の単位ベクトル (3,) float64

使用方法:
    python save_data_IMU.py
    python save_data_IMU.py --countdown 5   # カウントダウン秒数を変更
"""

import json
import os
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs


COUNTDOWN_DEFAULT = 3.0
RESOLUTION        = (640, 480)
FPS               = 30


def main():
    parser = argparse.ArgumentParser(description="RealSense RGBD + IMU データ保存")
    parser.add_argument("--countdown", type=float, default=COUNTDOWN_DEFAULT,
                        help=f"撮影までのカウントダウン秒数 (デフォルト: {COUNTDOWN_DEFAULT})")
    args = parser.parse_args()

    # 保存先: saved_data/test_YYYYMMDD_HHMMSS/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("saved_data", f"test_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[保存先] {out_dir}")

    # ---- パイプライン起動 ----
    pipeline = rs.pipeline()
    cfg = rs.config()
    w, h = RESOLUTION
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16,  FPS)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)

    print("[RealSense] パイプライン起動中...")
    profile = pipeline.start(cfg)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"[intrinsics] fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.ppx:.2f} cy={intr.ppy:.2f}  depth_scale={depth_scale}")

    align = rs.align(rs.stream.color)

    try:
        # ---- Step 1 & 2: カウントダウン中に IMU 収集 + プレビュー ----
        print(f"\n[撮影] {args.countdown:.0f} 秒後に自動撮影します。物体を映してください。")
        print(f"[IMU] カウントダウン中に重力ベクトルを収集します。カメラを固定してください。")
        start = time.time()
        accel_samples = []
        color_image = depth_raw = None

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_raw   = np.asanyarray(depth_frame.get_data())

            # IMU サンプル収集
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                m = accel_frame.as_motion_frame().get_motion_data()
                accel_samples.append([m.x, m.y, m.z])

            # プレビュー
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET
            )
            preview = np.hstack([color_image, depth_colormap])
            remain = max(0.0, args.countdown - (time.time() - start))
            cv2.putText(preview, f"Capturing in {remain:.1f}s  IMU={len(accel_samples)}samples",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("RealSense Preview", preview)
            cv2.waitKey(1)

            if time.time() - start >= args.countdown:
                print(f"[撮影] キャプチャしました。IMU サンプル数: {len(accel_samples)}")
                break

        # 重力ベクトル計算
        g = np.mean(accel_samples, axis=0)
        gravity_vec = (g / np.linalg.norm(g)).astype(np.float64)
        print(f"[IMU] 重力ベクトル g = [{gravity_vec[0]:.4f}, {gravity_vec[1]:.4f}, {gravity_vec[2]:.4f}]")

        cv2.destroyAllWindows()

        # ---- Step 3: 保存 ----
        cv2.imwrite(os.path.join(out_dir, "rgb.png"),   color_image)
        cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_raw)

        cam_data = {
            "cam_K": [
                intr.fx, 0.0, intr.ppx,
                0.0, intr.fy, intr.ppy,
                0.0, 0.0, 1.0,
            ],
            "depth_scale": depth_scale,
            "width":  intr.width,
            "height": intr.height,
            "gravity": gravity_vec.tolist(),
        }
        with open(os.path.join(out_dir, "cam.json"), "w") as f:
            json.dump(cam_data, f, indent=2)

        cv2.imwrite(os.path.join(out_dir, "rgb.png"),   color_image)
        cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_raw)

        g = gravity_vec
        print(f"\n[保存完了] {out_dir}/")
        print(f"  rgb.png  depth.png  cam.json")
        print(f"\n[次のコマンド]")
        print(f"  python test_demo_w_IMU.py --mode full \\")
        print(f"    --rgb {out_dir}/rgb.png \\")
        print(f"    --depth {out_dir}/depth.png \\")
        print(f"    --cam-json {out_dir}/cam.json \\")
        print(f"    --no-show --skip-grasp")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
