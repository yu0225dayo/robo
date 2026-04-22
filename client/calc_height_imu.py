import pyrealsense2 as rs
import numpy as np

# ============================================================
# 設定
# ============================================================
DEPTH_SCALE_DEFAULT = 0.001  # mm → m

# ============================================================
# パイプライン初期化
# ============================================================
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16,  30)
config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.accel,  rs.format.motion_xyz32f, 100)

profile = pipeline.start(config)

# depth scale取得
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# depth intrinsics取得
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_stream.get_intrinsics()

print(f"depth_scale: {depth_scale}")
print(f"intrinsics: fx={intrinsics.fx}, fy={intrinsics.fy}, cx={intrinsics.ppx}, cy={intrinsics.ppy}")

# ============================================================
# 重力ベクトルを取得（加速度センサー）
# ============================================================
def get_gravity_vector(pipeline, n_samples=30):
    """
    加速度センサーのフレームをn_samples枚平均して
    重力方向の単位ベクトルを返す（RealSense座標系）
    """
    samples = []
    collected = 0

    print("重力ベクトル取得中...")
    while collected < n_samples:
        frames = pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        if not accel_frame:
            continue
        motion = accel_frame.as_motion_frame().get_motion_data()
        samples.append([motion.x, motion.y, motion.z])
        collected += 1

    g = np.mean(samples, axis=0)
    g = g / np.linalg.norm(g)  # 単位ベクトルに正規化
    print(f"重力ベクトル g = {g}")
    return g

# ============================================================
# 画素 → 3D点 変換
# ============================================================
def pixel_to_3d(u, v, depth_m, intrinsics):
    """
    画素座標(u, v)と深度depth_m[m]から
    3D点(X, Y, Z)[m]を返す（RealSense座標系）
    """
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_m)
    return np.array(point)  # [X, Y, Z]

# ============================================================
# マスク内の点群を3D変換
# ============================================================
def get_points_3d(depth_image, mask, intrinsics, depth_scale):
    """
    マスク内の全画素を3D点群に変換して返す

    Args:
        depth_image : (H, W) uint16のdepth画像
        mask        : (H, W) bool、物体領域がTrue
        intrinsics  : RealSenseのintrinsics
        depth_scale : depth値をメートルに変換するスケール

    Returns:
        points_3d : (N, 3) float64の3D点群
    """
    vs, us = np.where(mask)  # マスク内の画素座標一覧
    points = []

    for v, u in zip(vs, us):
        d = depth_image[v, u] * depth_scale  # メートル換算
        if d <= 0:
            continue  # depth無効値はスキップ
        p = pixel_to_3d(u, v, d, intrinsics)
        points.append(p)

    if len(points) == 0:
        return None

    return np.array(points)  # shape: (N, 3)

# ============================================================
# 高さ計算
# ============================================================
def calc_height(points_3d, gravity_vec):
    """
    3D点群を重力方向に射影して高さを計算

    Args:
        points_3d  : (N, 3) 3D点群
        gravity_vec: (3,)   重力方向の単位ベクトル

    Returns:
        height_m : 高さ [m]
    """
    # 各点を重力方向に射影（内積）→ スカラー値のリスト
    projections = points_3d @ gravity_vec  # shape: (N,)

    height_m = projections.max() - projections.min()
    return height_m

# ============================================================
# メイン処理
# ============================================================
try:
    # Step1: 重力ベクトルを先に取得（カメラを固定した状態で）
    gravity_vec = get_gravity_vector(pipeline, n_samples=30)

    print("RGBDフレーム取得中... Ctrl+Cで終了")

    while True:
        # Step2: RGBDフレーム取得
        frames = pipeline.wait_for_frames()

        # depth/colorを時間的に同期
        align = rs.align(rs.stream.color)
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())  # (H, W) uint16

        # -------------------------------------------------------
        # Step3: マスクを用意（ここを物体検出器の出力に差し替える）
        # 例：画像中央付近をマスクとして仮定
        H, W = depth_image.shape
        mask = np.zeros((H, W), dtype=bool)
        mask[H//4 : H*3//4, W//4 : W*3//4] = True
        # ↑ 実際はSAM・YOLO等のマスクに差し替えてください
        # -------------------------------------------------------

        # Step4: マスク内を3D点群に変換
        points_3d = get_points_3d(depth_image, mask, intrinsics, depth_scale)
        if points_3d is None:
            print("有効なdepth点なし、スキップ")
            continue

        # Step5: 重力方向に射影して高さ計算
        height = calc_height(points_3d, gravity_vec)
        print(f"高さ: {height:.4f} m  ({height*100:.1f} cm)  点数: {len(points_3d)}")

except KeyboardInterrupt:
    print("終了")

finally:
    pipeline.stop()