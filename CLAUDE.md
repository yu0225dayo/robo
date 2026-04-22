# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## システム概要

RealSense カメラで物体を撮影し、GPU サーバで 3D 再構築と姿勢推定を行い、把持姿勢をロボットへ送信するパイプライン。

```
クライアント (ローカルPC)          サーバ (10.40.1.126)
  RealSense RGBD
    → main.py / test_demo.py  ──HTTP:8080──→  server/server.py  (SAM2 + SAM-3D)
                                                    ↓ HTTP:8081
                                               SAM-6D Docker
    ← R, t (6DoF pose) ←─────────────────────────────
    → GraspGenerator (Shape2Gesture)
    → ロボット送信
```

## 起動手順

### サーバ側（GPU 計算機）

```bash
# SAM-6D Docker を起動 (初回のみビルド: docker compose build sam6d)
cd ~/ws/project/server
docker compose up -d sam6d
curl http://localhost:8081/health  # 起動確認

# server.py を起動
python server.py \
    --sam-checkpoint /ws/okada/project/sam_vit_h_4b8939.pth \
    --sam3d-config   ~/ws/sam-3d-objects/checkpoints/hf/pipeline.yaml \
    --sam3d-repo     ~/ws/sam-3d-objects \
    --sam6d-service  http://localhost:8081 \
    --host 0.0.0.0 --port 8080
```

### クライアント側（ローカル PC）

```bash
cd ~/ws/project/client

# フルパイプライン（mesh生成 + 姿勢推定 + 把持）
python main.py

# テスト用（カメラなし・ロボットなし）
python test_demo.py --mode full \
    --rgb test_data/rgb.png --depth test_data/depth.png \
    --click-x 400 --click-y 280 --no-show --skip-grasp

# mesh 生成済みで姿勢推定のみ
python main.py --mode online --mesh meshes/cup.ply --no-robot
```

## アーキテクチャ

### サーバ (`server/server.py`)

- `POST /reconstruct_mesh` — RGB → SAM2 マスク → SAM-3D → PLY 返却 + SAM-6D テンプレートレンダリング
- `POST /pose_estimate` — RGB + depth → SAM2 マスク → SAM-6D (Docker) → R, t 返却
- SAM-3D はオンデマンドロード（推論後に即削除して GPU 解放）
- SAM-6D は Docker コンテナ (sam6d_service) へ HTTP プロキシ

### SAM-6D サービス (`server/sam6d_service.py`)

- Docker コンテナ内 FastAPI サービス (port 8081)
- `POST /render_templates` — メッシュから視点テンプレートを生成
- `POST /run_pose_estimation` — テンプレート + RGB + マスク → R, t

### クライアント (`client/`)

| モジュール | 役割 |
|---|---|
| `main.py` | RealSense カメラ制御 + キーボード操作UI |
| `test_demo.py` | 静止画ファイルでのテスト用エントリポイント |
| `pipeline/sam6d_detector.py` | サーバ HTTP クライアント (`SAM6DClient`) |
| `pipeline/grasp_generator.py` | Shape2Gesture モデルで把持姿勢生成 |
| `pipeline/robot_interface.py` | ロボット送信 (ROS / TCP / Serial / Mock) |
| `utils/coord_transform.py` | カメラ座標系変換・`CameraIntrinsics` クラス |

### 設定ファイル (`client/config.yaml`)

- `sam3d.server_url` — サーバ IP:ポート（ここを変更して接続先を切り替え）
- `sam3d.mesh_method` — メッシュ生成方式: `bpa` / `poisson` / `knn`
- `robot.mode` — `mock`（デバッグ）/ `tcp` / `ros` / `serial`

## 既知の問題

- **スケール不一致**: SAM-3D が生成する PLY のスケールが実寸と合わず SAM-6D の姿勢推定精度が低下する。`server.py` 内に深度から推定サイズでスケール補正するコードがあるが根本解決には至っていない。
- 代替パイプライン (`~/ws/project-fp/`) で FoundationPose + 深度直接 BPA メッシュを検証中。

## 共有ディレクトリ

- `tmp/` — サーバ・Docker 間の中間ファイル共有（Docker 内では `/workspace/tmp/`）
- `server/SAM-6D/` — SAM-6D リポジトリ（submodule）
- `server/sam-3d-objects/` — SAM-3D リポジトリ（submodule）
