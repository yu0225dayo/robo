"""
sam-3d-objects 推論サーバ (Linux + A6000 で実行)

クライアント (Windows) から RGB画像を受け取り、
SAM + sam-3d-objects で完全3Dモデルを生成して返す。

起動方法 (Linux サーバ上で):
    cd /path/to/real_world_demo
    pip install fastapi uvicorn plyfile segment_anything
    python server/server.py --sam-checkpoint /path/to/sam_vit_h.pth \
                            --sam3d-config /path/to/pipeline.yaml \
                            --sam3d-repo /path/to/sam-3d-objects \
                            --host 0.0.0.0 --port 8080
"""

import argparse
import sys
import os
import numpy as np
import cv2
import requests as _requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn


app = FastAPI(title="SAM 3D Objects Server")

# グローバルモデル (起動時にロード)
sam_predictor = None
sam3d_inference = None
args_global = None
SAM6D_SERVICE_URL = ""   # --sam6d-service で設定


def load_models(sam_checkpoint: str, sam3d_config: str, sam3d_repo: str,
                device: str = "cuda"):
    global sam_predictor, sam3d_inference

    # sam-3d-objects をパスに追加 (notebook/ に inference.py がある)
    notebook_path = os.path.join(sam3d_repo, "notebook")
    for p in [sam3d_repo, notebook_path]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # SAM
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print(f"[Server] SAM ロード完了 ({device})")

    # sam-3d-objects
    from inference import Inference
    sam3d_inference = Inference(sam3d_config, compile=False)
    print("[Server] sam-3d-objects ロード完了")


@app.get("/health")
def health():
    """サーバの死活確認"""
    return {"status": "ok", "models_loaded": sam_predictor is not None}


@app.post("/reconstruct")
async def reconstruct(
    image: UploadFile = File(...),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    seed: int = Form(42),
    target_points: int = Form(2048),
    output_dir: str = Form("tmp/server_reconstructions"),
):
    """
    RGB画像から物体の完全3D点群を生成して返す

    Args:
        image:         RGB画像ファイル (JPEG/PNG)
        click_x, click_y: SAMプロンプト座標 (-1,-1 で画像中央を使用)
        seed:          sam-3d-objects のランダムシード
        target_points: 返す点群の点数

    Returns:
        JSON: {"points": [[x,y,z], ...], "num_points": N,
               "mask_center_u": int, "mask_center_v": int}
    """
    if sam_predictor is None or sam3d_inference is None:
        raise HTTPException(status_code=503, detail="モデルがロードされていません")

    # 画像をデコード
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="画像のデコードに失敗しました")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # SAMプロンプト設定
    if click_x < 0 or click_y < 0:
        prompt_point = np.array([[w // 2, h // 2]])
    else:
        prompt_point = np.array([[click_x, click_y]])

    # Step 1: SAM で2Dマスク生成
    sam_predictor.set_image(rgb)
    masks, scores, _ = sam_predictor.predict(
        point_coords=prompt_point,
        point_labels=np.array([1]),
        multimask_output=True,
    )
    best_mask = masks[np.argmax(scores)]
    print(f"[Server] SAM マスク生成完了 (面積: {best_mask.sum()} px, "
          f"プロンプト: ({prompt_point[0][0]}, {prompt_point[0][1]}))")

    # Step 2: sam-3d-objects で完全3Dモデル生成
    print("[Server] sam-3d-objects 推論中...")
    output = sam3d_inference(rgb, best_mask, seed=seed)

    # Step 3: PLY保存 → XYZ抽出
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"object_seed{seed}.ply")
    output["gs"].save_ply(ply_path)
    print(f"[Server] PLY保存: {ply_path}")

    # Gaussian splat PLY から XYZ 座標を抽出
    from plyfile import PlyData
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    points = np.stack([
        vertex["x"].astype(np.float32),
        vertex["y"].astype(np.float32),
        vertex["z"].astype(np.float32),
    ], axis=-1)
    print(f"[Server] Gaussian数: {len(points)}")

    # リサンプリング
    n = len(points)
    choice = np.random.choice(n, target_points, replace=(n < target_points))
    points = points[choice]

    # マスク重心ピクセル (Windows側でカメラ座標変換に使用)
    ys, xs = np.where(best_mask)
    mask_center_u = int(xs.mean())
    mask_center_v = int(ys.mean())

    print(f"[Server] 点群送信: {len(points)} points, mask_center=({mask_center_u},{mask_center_v})")
    return JSONResponse({
        "points": points.tolist(),
        "num_points": len(points),
        "ply_path": ply_path,
        "mask_center_u": mask_center_u,
        "mask_center_v": mask_center_v,
    })


@app.post("/reconstruct_mesh")
async def reconstruct_mesh(
    image: UploadFile = File(...),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    seed: int = Form(42),
    output_dir: str = Form("tmp/server_reconstructions"),
):
    """
    RGB画像から物体の3Dメッシュ (.ply) を生成してファイルとして返す。
    SAM-6D サービスが有効なら、テンプレートを事前レンダリングする。

    Returns:
        PLY ファイル (バイナリ)
        Headers: X-Mesh-Path, X-Template-Dir, X-Mask-Center-U/V
    """
    if sam_predictor is None or sam3d_inference is None:
        raise HTTPException(status_code=503, detail="モデルがロードされていません")

    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="画像のデコードに失敗しました")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    prompt_point = np.array([[click_x if click_x >= 0 else w // 2,
                               click_y if click_y >= 0 else h // 2]])

    sam_predictor.set_image(rgb)
    masks, scores, _ = sam_predictor.predict(
        point_coords=prompt_point,
        point_labels=np.array([1]),
        multimask_output=True,
    )
    best_mask = masks[np.argmax(scores)]

    print("[Server] sam-3d-objects 推論中 (mesh出力)...")
    output = sam3d_inference(rgb, best_mask, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"object_seed{seed}.ply")
    output["gs"].save_ply(ply_path)
    ply_path = os.path.abspath(ply_path)

    ys, xs = np.where(best_mask)
    mask_center_u = int(xs.mean())
    mask_center_v = int(ys.mean())

    # SAM-6D サービスでテンプレートをレンダリング
    template_dir = ""
    if SAM6D_SERVICE_URL:
        try:
            resp = _requests.post(
                f"{SAM6D_SERVICE_URL}/render_templates",
                data={"mesh_path": ply_path},
                timeout=120,
            )
            if resp.status_code == 200:
                template_dir = resp.json().get("template_dir", "")
                print(f"[Server] テンプレートレンダリング完了: {template_dir}")
            else:
                print(f"[Server] テンプレートレンダリング失敗: {resp.text}")
        except Exception as e:
            print(f"[Server] SAM-6D サービス接続エラー (スキップ): {e}")

    print(f"[Server] mesh送信: {ply_path}")
    return FileResponse(
        ply_path,
        media_type="application/octet-stream",
        filename="reference_mesh.ply",
        headers={
            "X-Mesh-Path":     ply_path,
            "X-Template-Dir":  template_dir,
            "X-Mask-Center-U": str(mask_center_u),
            "X-Mask-Center-V": str(mask_center_v),
        },
    )


@app.post("/pose_estimate")
async def pose_estimate(
    rgb_image:   UploadFile = File(...),
    depth_image: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    mesh_path:    str = Form(...),
    template_dir: str = Form(""),
):
    """
    SAM-6D サービス (Docker) に 6DoF pose 推定をプロキシする。

    Returns:
        JSON: {"R": [[...]], "t": [...], "success": bool}
    """
    if not SAM6D_SERVICE_URL:
        raise HTTPException(status_code=503, detail="SAM-6D サービスが設定されていません")

    rgb_bytes   = await rgb_image.read()
    depth_bytes = await depth_image.read()

    resp = _requests.post(
        f"{SAM6D_SERVICE_URL}/pose_estimate",
        files={
            "rgb_image":   ("frame.jpg",  rgb_bytes,   "image/jpeg"),
            "depth_image": ("depth.bin",  depth_bytes, "application/octet-stream"),
        },
        data={
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "mesh_path":    mesh_path,
            "template_dir": template_dir,
        },
        timeout=60,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JSONResponse(resp.json())


@app.post("/segment_only")
async def segment_only(
    image: UploadFile = File(...),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
):
    """
    SAMのマスクのみ返す (デバッグ用)
    """
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="SAMがロードされていません")

    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    prompt_point = np.array([[click_x if click_x >= 0 else w // 2,
                               click_y if click_y >= 0 else h // 2]])
    sam_predictor.set_image(rgb)
    masks, scores, _ = sam_predictor.predict(
        point_coords=prompt_point,
        point_labels=np.array([1]),
        multimask_output=True,
    )
    best_mask = masks[np.argmax(scores)]
    return JSONResponse({
        "mask_area": int(best_mask.sum()),
        "score": float(scores[np.argmax(scores)]),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-checkpoint", required=True,
                        help="SAM ViT-H モデル重みパス")
    parser.add_argument("--sam3d-config", required=True,
                        help="sam-3d-objects の pipeline.yaml パス")
    parser.add_argument("--sam3d-repo", required=True,
                        help="sam-3d-objects リポジトリのパス")
    parser.add_argument("--sam6d-service", default="",
                        help="SAM-6D Docker サービスの URL (例: http://localhost:8081)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    args_global = args

    global SAM6D_SERVICE_URL
    SAM6D_SERVICE_URL = args.sam6d_service.rstrip("/")
    if SAM6D_SERVICE_URL:
        print(f"  SAM-6D service: {SAM6D_SERVICE_URL}")

    print("=" * 50)
    print(f"  SAM 3D Objects Server")
    print(f"  device: {args.device}")
    print(f"  host:   {args.host}:{args.port}")
    print("=" * 50)

    load_models(args.sam_checkpoint, args.sam3d_config, args.sam3d_repo, args.device)

    uvicorn.run(app, host=args.host, port=args.port)
