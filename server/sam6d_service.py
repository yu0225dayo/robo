"""
SAM-6D マイクロサービス (Docker コンテナ内で実行)

Docker コンテナ内で SAM-6D の推論を行い、REST API として提供する。
メインサーバ (server.py) から HTTP で呼び出される。

ボリュームマウント前提:
    /workspace/SAM-6D  → ホストの ~/ws/SAM-6D
    /workspace/tmp     → ホストの ~/ws/tmp  (共有テンポラリ)

起動方法 (docker-compose 経由):
    docker compose up sam6d
"""

import argparse
import os
import sys
import json
import shutil
import tempfile
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="SAM-6D Service")

# ---- グローバル状態 ----
_wrapper = None
_template_cache: dict[str, str] = {}   # cad_path → template_dir
_args_global = None


# ==================== Pydantic スキーマ ====================

class RenderTemplatesRequest(BaseModel):
    cad_path: str                    # メッシュ PLY パス [mm単位]
    output_dir: Optional[str] = None
    num_templates: int = 42
    pcd_path: Optional[str] = None   # 点群 PLY パス (指定時は点群直接投影)


class EstimatePoseRequest(BaseModel):
    rgb_path: str           # 計算機上の RGB PNG パス
    depth_path: str         # 計算機上の depth PNG パス [uint16, mm単位]
    cam_json_path: str      # 計算機上の camera.json パス
    cad_path: str           # 計算機上の PLY パス [mm単位]
    template_dir: str       # render_templates() 出力ディレクトリ
    det_score_thresh: float = 0.2
    click_x: int = -1
    click_y: int = -1


class FullEstimateRequest(BaseModel):
    """RGB+depth の numpy バイナリを受け取り、一括でpose推定"""
    rgb_path: str
    depth_path: str         # mm単位 uint16 PNG
    intrinsics: dict        # {"fx","fy","cx","cy"}
    cad_path: str
    template_dir: Optional[str] = None   # None の場合は自動レンダリング
    det_score_thresh: float = 0.2


# ==================== 起動時モデルロード ====================

def load_sam6d(sam6d_repo: str, device: str = "cuda", segmentor: str = "sam"):
    global _wrapper

    # sam6d_wrapper.py をパスに追加 (ホスト側と共有)
    ws_root = os.path.dirname(sam6d_repo.rstrip("/"))
    if ws_root not in sys.path:
        sys.path.insert(0, ws_root)

    from sam6d_wrapper import SAM6DWrapper
    _wrapper = SAM6DWrapper(sam6d_repo=sam6d_repo, device=device)
    _wrapper.load_models(segmentor=segmentor)
    print(f"[sam6d_service] モデルロード完了 (device={device}, segmentor={segmentor})")


# ==================== エンドポイント ====================

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _wrapper is not None}


@app.post("/render_templates")
def render_templates(req: RenderTemplatesRequest):
    """
    CADモデルから複数視点テンプレートをレンダリングする (物体ごとに一度だけ実行)

    Returns:
        {"template_dir": "/path/to/templates"}
    """
    if _wrapper is None:
        raise HTTPException(503, "モデルがロードされていません")
    if not os.path.exists(req.cad_path):
        raise HTTPException(400, f"CADファイルが見つかりません: {req.cad_path}")

    cache_key = req.cad_path
    if cache_key in _template_cache:
        tdir = _template_cache[cache_key]
        if os.path.isdir(tdir):
            src_mtime = os.path.getmtime(req.cad_path)
            tpl_mtime = os.path.getmtime(tdir)
            if src_mtime <= tpl_mtime:
                print(f"[sam6d_service] テンプレートキャッシュ使用: {tdir}")
                return {"template_dir": tdir}
            print(f"[sam6d_service] ソースが更新されたため再生成: {req.cad_path}")

    tdir = _wrapper.render_templates(
        cad_path_mm=req.cad_path,
        output_dir=req.output_dir,
        num_templates=req.num_templates,
        pcd_path=req.pcd_path,
    )
    _template_cache[cache_key] = tdir
    return {"template_dir": tdir}


@app.post("/estimate_pose")
def estimate_pose(req: EstimatePoseRequest):
    """
    6DoF 姿勢推定

    Returns:
        {
            "R": [[...], [...], [...]],   # (3,3) 回転行列
            "t": [x, y, z],               # (3,) 平行移動 [m]
            "mask_area": int
        }
    """
    if _wrapper is None:
        raise HTTPException(503, "モデルがロードされていません")

    for path, label in [(req.rgb_path, "RGB"), (req.depth_path, "depth"),
                         (req.cam_json_path, "camera.json"), (req.cad_path, "CAD"),
                         (req.template_dir, "テンプレートディレクトリ")]:
        if not os.path.exists(path):
            raise HTTPException(400, f"{label}が見つかりません: {path}")

    # RGB 読み込み
    bgr = cv2.imread(req.rgb_path)
    if bgr is None:
        raise HTTPException(400, f"RGB画像の読み込み失敗: {req.rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # depth 読み込み (uint16 mm → float32 m)
    depth_mm = cv2.imread(req.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_m = depth_mm / 1000.0

    # カメラパラメータ
    with open(req.cam_json_path) as f:
        cam = json.load(f)
    K = cam["cam_K"]
    intrinsics = {"fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5]}

    import torch
    R, t, mask_area = _wrapper.estimate_pose(
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        cad_path_mm=req.cad_path,
        template_dir=req.template_dir,
        det_score_thresh=req.det_score_thresh,
        click_x=req.click_x,
        click_y=req.click_y,
    )
    torch.cuda.empty_cache()

    return JSONResponse({
        "R": R.tolist(),
        "t": t.tolist(),
        "mask_area": mask_area,
    })


@app.post("/full_estimate")
def full_estimate(req: FullEstimateRequest):
    """
    RGB + depth ファイルパスから全工程を実行
    テンプレートが未生成の場合は自動レンダリングも行う

    Returns:
        {
            "R": ..., "t": ..., "mask_area": ...,
            "template_dir": "..."
        }
    """
    if _wrapper is None:
        raise HTTPException(503, "モデルがロードされていません")

    # カメラパラメータを一時ファイルへ書き出し
    tmpdir = tempfile.mkdtemp(dir="/workspace/tmp")
    try:
        cam_path = os.path.join(tmpdir, "camera.json")
        intr = req.intrinsics
        cam_json = {
            "cam_K": [intr["fx"], 0.0, intr["cx"],
                      0.0, intr["fy"], intr["cy"],
                      0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        with open(cam_path, "w") as f:
            json.dump(cam_json, f)

        # テンプレート (キャッシュ or 新規レンダリング)
        tdir = req.template_dir
        if tdir is None or not os.path.isdir(tdir):
            tdir = _wrapper.render_templates(cad_path_mm=req.cad_path)
            _template_cache[req.cad_path] = tdir

        # RGB / depth 読み込み
        bgr = cv2.imread(req.rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_mm = cv2.imread(req.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_mm / 1000.0

        import torch
        intr_dict = {k: float(v) for k, v in req.intrinsics.items()}
        R, t, mask_area = _wrapper.estimate_pose(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intr_dict,
            cad_path_mm=req.cad_path,
            template_dir=tdir,
        )
        torch.cuda.empty_cache()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return JSONResponse({
        "R": R.tolist(),
        "t": t.tolist(),
        "mask_area": mask_area,
        "template_dir": tdir,
    })


# ==================== エントリポイント ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-6D マイクロサービス")
    parser.add_argument("--sam6d-repo", default="/workspace/SAM-6D",
                        help="SAM-6D リポジトリのパス (ボリュームマウント先)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segmentor", default="sam", choices=["sam", "fastsam"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    _args_global = args

    print("=" * 50)
    print("  SAM-6D Microservice")
    print(f"  sam6d_repo: {args.sam6d_repo}")
    print(f"  device:     {args.device}")
    print(f"  host:port:  {args.host}:{args.port}")
    print("=" * 50)

    load_sam6d(args.sam6d_repo, device=args.device, segmentor=args.segmentor)
    uvicorn.run(app, host=args.host, port=args.port)

# ===== Dockerfile CMD 用エントリポイントメモ =====
# /opt/conda/envs/sam6d/bin/python /service/sam6d_service.py --port 8081
