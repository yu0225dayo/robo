# main.py 使い方

RealSense カメラを使った物体把持デモのメインスクリプト。

## 前提条件

- Intel RealSense D400系カメラが接続済み
- 計算機サーバ (`server/server.py`) が起動済み
- `config.yaml` の `server_url` が正しく設定済み
- Shape2Gesture 学習済みモデルが `save_model/` に配置済み

---

## 通常の使い方（full モード）

```bash
python main.py
```

起動するとRealSenseのプレビューが表示される。

| キー | 動作 |
|------|------|
| `c`  | 物体をクリックして選択 → 3D mesh 生成（サーバで SAM-3D 実行） |
| `g`  | pose 推定 → 把持姿勢生成 → 画像投影（`c` 実行後に有効） |
| `q`  | 終了 |

### 手順

1. `c` → ウィンドウ上で把持したい物体をクリック → Enter で確定
2. サーバで 3D mesh が生成されるまで待つ（数十秒）
3. `g` → SAM-6D で pose 推定 → 把持姿勢を画像に投影
4. 物体を変える場合は再度 `c` で mesh を再生成

---

## mesh 保存先を指定する場合

```bash
python main.py --mesh-out meshes/cup.ply
```

デフォルトは `meshes/object.ply`。

---

## ロボットなしで動作確認

```bash
python main.py --no-robot
```

---

## 出力ファイル

`[g]` キー押下ごとに `output/<YYYYMMDD_HHMMSS>/` が作成される。

| ファイル | 内容 |
|---------|------|
| `server_pointcloud.png` | サーバが生成した点群投影画像 |
| `server_mesh.png`       | サーバが生成した3Dメッシュ投影画像 |
| `grasp_00.png` 〜       | 把持姿勢をRGB画像に投影した結果 |

起動時に `camera.json` がカレントディレクトリに保存される（カメラ内部パラメータ）。

---

## その他のモード

### online モード（mesh 生成済みの場合）

```bash
python main.py --mode online --mesh meshes/cup.ply --no-robot
```

### offline-mesh モード（mesh 生成のみ）

```bash
python main.py --mode offline-mesh --mesh-out meshes/cup.ply
```

---

## 引数一覧

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--config` | `config.yaml` | 設定ファイルパス |
| `--mode` | `full` | `full` / `online` / `offline-mesh` |
| `--mesh-out` | `meshes/object.ply` | [full/offline-mesh] mesh 保存先 |
| `--mesh` | — | [online] ローカル reference mesh (.ply) パス |
| `--server-mesh-path` | — | [online] サーバ側 mesh パス |
| `--template-dir` | — | [online] サーバ側テンプレートディレクトリ |
| `--click-x` | `-1` | [online] 物体クリック座標 X（-1: 画像中央） |
| `--click-y` | `-1` | [online] 物体クリック座標 Y（-1: 画像中央） |
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
