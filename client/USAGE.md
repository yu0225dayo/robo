# main.py 使い方

RealSense カメラを使った物体把持デモのメインスクリプト。

## 前提条件

- Intel RealSense D400系カメラが接続済み
- 計算機サーバ (`server/server.py`) が起動済み
- `config.yaml` の `server_url` が正しく設定済み
- Shape2Gesture 学習済みモデルが `save_model/` に配置済み

---

## Step 1: reference mesh を生成（物体ごとに1回）

```bash
python main.py --mode offline-mesh --mesh-out meshes/cup.ply
```

- RealSense プレビューが表示される
- `[c]` キー → クリックで物体を指定 → Enter で確定 → mesh 生成開始
- `[q]` キー → 終了
- 完了すると `meshes/cup.ply` が保存される
- サーバ側の mesh パスとテンプレートディレクトリがターミナルに表示される（Step 2 で使用）

---

## Step 2: 把持姿勢生成（毎回）

### ロボットなし（デバッグ用）

```bash
python main.py --mesh meshes/cup.ply --no-robot
```

### ロボットあり

```bash
python main.py --mesh meshes/cup.ply
```

### サーバ側 mesh パスを指定（精度向上）

Step 1 完了時に表示されるパスを `--server-mesh-path` と `--template-dir` に渡す。

```bash
python main.py \
  --mesh meshes/cup.ply \
  --server-mesh-path "/path/to/server/mesh.ply" \
  --template-dir "/path/to/templates" \
  --no-robot
```

### 物体位置をクリック座標で指定

```bash
python main.py --mesh meshes/cup.ply --click-x 320 --click-y 240 --no-robot
```

---

## 操作方法（Step 2 実行中）

| キー | 動作 |
|------|------|
| `g`  | SAM-6D で pose 推定 → 把持姿勢生成 → 画像保存 |
| `q`  | 終了 |

---

## 出力ファイル

`[g]` キー押下ごとに `output/<YYYYMMDD_HHMMSS>/` が作成される。

| ファイル | 内容 |
|---------|------|
| `server_pointcloud.png` | サーバが生成した点群投影画像 |
| `server_mesh.png`       | サーバが生成した3Dメッシュ投影画像 |
| `grasp_00.png` 〜       | 把持姿勢をRGB画像に投影した結果 |

また、起動時に `camera.json` がカレントディレクトリに保存される。

---

## 引数一覧

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--config` | `config.yaml` | 設定ファイルパス |
| `--mode` | `online` | `online`: 把持生成 / `offline-mesh`: mesh生成 |
| `--mesh` | — | [online] ローカル reference mesh (.ply) パス |
| `--mesh-out` | `meshes/object.ply` | [offline-mesh] mesh 保存先 |
| `--server-mesh-path` | — | サーバ側 mesh パス（offline-mesh 後に表示） |
| `--template-dir` | — | サーバ側テンプレートディレクトリ |
| `--click-x` | `-1` | 物体クリック座標 X（-1: 画像中央） |
| `--click-y` | `-1` | 物体クリック座標 Y（-1: 画像中央） |
| `--no-robot` | `False` | ロボット送信をスキップ |
| `--num-samples` | config参照 | 把持候補生成数 |
| `--epoch` | config参照 | PositionVAE エポック番号 |

---

## config.yaml 主要設定

```yaml
sam3d:
  server_url: "http://10.40.1.126:8080"  # 計算機サーバのIPとポート
  mesh_method: "knn"                      # メッシュ生成方法: bpa / poisson / knn
  timeout: 6000.0                         # mesh生成タイムアウト[秒]

grasp_model:
  model_dir: "save_model"   # Shape2Gesture モデルディレクトリ
  epoch: 69                 # PositionVAE エポック番号
  num_samples: 6            # 把持候補数

robot:
  mode: "mock"              # ros / tcp / serial / mock
```
