"""
SAM 3D Objects + SAM-6D パイプラインサーバ (Linux + A6000 で実行)

クライアント (ローカルPC) から RGB+深度画像を受け取り、以下を行う:
    1. SAM + sam-3d-objects で完全3Dモデル (PLY) を生成
    2. SAM-6D Docker サービス (port 8081) へプロキシして6DoF姿勢推定

起動方法 (Linux サーバ上で):
    python server.py \
        --sam-checkpoint ~/ws/sam_vit_h_4b8939.pth \
        --sam3d-config   ~/ws/sam-3d-objects/checkpoints/hf/pipeline.yaml \
        --sam3d-repo     ~/ws/sam-3d-objects \
        --sam6d-service  http://localhost:8081 \
        --host 0.0.0.0 --port 8080
"""

import argparse
import sys
import os
import json
import numpy as np
import cv2
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI(title="SAM 3D + SAM-6D Pipeline Server")

# グローバルモデル (起動時にロード)
sam_predictor = None
sam3d_inference = None
args_global = None

# SAM-6D サービス URL (Docker コンテナ)
_sam6d_url: str = "http://localhost:8081"

# ホスト↔Dockerコンテナ間の共有tmpディレクトリパスマッピング
_host_tmp: str   = "/home/okada/ws/project/tmp"
_docker_tmp: str = "/workspace/tmp"


def to_docker_path(host_path: str) -> str:
    """ホスト側の絶対パスをDockerコンテナ内のパスに変換する"""
    abs_path = os.path.abspath(host_path)
    if abs_path.startswith(_host_tmp):
        return _docker_tmp + abs_path[len(_host_tmp):]
    return abs_path


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


def _sam6d_post(endpoint: str, payload: dict, timeout: float = 300.0) -> dict:
    """SAM-6D Docker サービスへ JSON POST し、レスポンスを返す"""
    url = f"{_sam6d_url}/{endpoint.lstrip('/')}"
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        raise HTTPException(503, f"SAM-6D サービスに接続できません ({url}). "
                                 "docker compose up sam6d を確認してください。")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code,
                            f"SAM-6D サービスエラー: {e.response.text}")


@app.post("/reconstruct_mesh")
async def reconstruct_mesh(
    image: UploadFile = File(...),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    seed: int = Form(42),
    target_points: int = Form(2048),
    output_dir: str = Form(""),
):
    """
    クライアント互換エンドポイント: SAM-3D でメッシュ生成 + SAM-6D テンプレートレンダリング

    レスポンス: PLY バイナリ
    ヘッダ:
        X-Mesh-Path:      サーバ側の .ply パス
        X-Template-Dir:   SAM-6D テンプレートディレクトリ
        X-Mask-Center-U:  マスク重心 U 座標
        X-Mask-Center-V:  マスク重心 V 座標
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
    print(f"[Server] SAM マスク完了 (面積:{best_mask.sum()}px)")

    print("[Server] SAM-3D 推論中...")
    output = sam3d_inference(rgb, best_mask, seed=seed)

    # 共有tmpに保存してDockerからアクセスできるようにする
    save_dir = output_dir if output_dir else os.path.join(_host_tmp, "server_reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    ply_path = os.path.join(save_dir, f"object_seed{seed}.ply")
    output["gs"].save_ply(ply_path)
    print(f"[Server] PLY 保存: {ply_path}")

    # 点群 → メッシュ変換 (SAM-6D は面付きメッシュを必要とする)
    import open3d as o3d
    print("[Server] 点群をメッシュに変換中 (Poisson reconstruction)...")
    gs_ply = o3d.io.read_point_cloud(ply_path)

    # ダウンサンプリングで点数を削減
    gs_ply = gs_ply.voxel_down_sample(voxel_size=0.005)
    print(f"[Server] ダウンサンプル後: {len(gs_ply.points)} points")

    gs_ply.estimate_normals()
    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(gs_ply, depth=8)

    # メッシュ簡略化 (三角形数を上限10000に)
    mesh_o3d = mesh_o3d.simplify_quadric_decimation(10000)
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_unreferenced_vertices()

    mesh_path = ply_path.replace(".ply", "_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh_o3d)
    print(f"[Server] メッシュ保存: {mesh_path} ({len(mesh_o3d.triangles)} triangles)")

    ys, xs = np.where(best_mask)
    mask_center_u = int(xs.mean())
    mask_center_v = int(ys.mean())

    # SAM-6D テンプレートレンダリング (Dockerコンテナ内パスで渡す)
    print("[Server] SAM-6D テンプレートレンダリング中...")
    tdir_resp = _sam6d_post("render_templates", {
        "cad_path": to_docker_path(mesh_path),
        "output_dir": None,
        "num_templates": 42,
    }, timeout=600.0)
    template_dir = tdir_resp["template_dir"]
    print(f"[Server] テンプレート完了: {template_dir}")

    from fastapi.responses import Response
    with open(mesh_path, "rb") as f:
        ply_bytes = f.read()

    return Response(
        content=ply_bytes,
        media_type="application/octet-stream",
        headers={
            "X-Mesh-Path":     mesh_path,
            "X-Template-Dir":  template_dir,
            "X-Mask-Center-U": str(mask_center_u),
            "X-Mask-Center-V": str(mask_center_v),
        },
    )


@app.post("/pose_estimate")
async def pose_estimate(
    rgb_image: UploadFile = File(...),
    depth_image: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    mesh_path: str = Form(...),
    template_dir: str = Form(...),
    det_score_thresh: float = Form(0.2),
):
    """
    クライアント互換エンドポイント: 6DoF 姿勢推定

    depth_image: float32 生バイト列 (H×W×4 bytes, メートル単位)
    """
    import tempfile, shutil

    rgb_bytes   = await rgb_image.read()
    depth_bytes = await depth_image.read()

    # RGB デコード
    nparr = np.frombuffer(rgb_bytes, np.uint8)
    bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "RGB画像のデコードに失敗しました")
    h, w = bgr.shape[:2]

    # depth: float32 生バイト → uint16 PNG [mm]
    depth_f32 = np.frombuffer(depth_bytes, dtype=np.float32).reshape(h, w)
    depth_mm  = (depth_f32 * 1000.0).astype(np.uint16)

    tmpdir = tempfile.mkdtemp()
    try:
        rgb_path   = os.path.join(tmpdir, "rgb.png")
        depth_path = os.path.join(tmpdir, "depth.png")
        cam_path   = os.path.join(tmpdir, "camera.json")

        cv2.imwrite(rgb_path, bgr)
        cv2.imwrite(depth_path, depth_mm)

        cam_json = {
            "cam_K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        with open(cam_path, "w") as f:
            json.dump(cam_json, f)

        result = _sam6d_post("estimate_pose", {
            "rgb_path":         rgb_path,
            "depth_path":       depth_path,
            "cam_json_path":    cam_path,
            "cad_path":         mesh_path,
            "template_dir":     template_dir,
            "det_score_thresh": det_score_thresh,
        }, timeout=300.0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return JSONResponse({
        "success": True,
        "R": result["R"],
        "t": result["t"],
        "mask_area": result.get("mask_area", 0),
    })


@app.get("/sam6d/health")
def sam6d_health():
    """SAM-6D Docker サービスの死活確認"""
    return _sam6d_post("health", {})


@app.post("/render_templates")
async def render_templates(
    cad_path: str = Form(...),
    output_dir: str = Form(""),
    num_templates: int = Form(42),
):
    """
    CADモデル (.ply) からテンプレートをレンダリングする (物体ごとに一度)

    Args:
        cad_path:      サーバ上の PLY ファイルパス [mm単位]
        output_dir:    テンプレート保存先 (空文字で自動生成)
        num_templates: テンプレート数 (デフォルト 42)

    Returns:
        {"template_dir": "/path/to/templates"}
    """
    payload = {
        "cad_path": cad_path,
        "output_dir": output_dir if output_dir else None,
        "num_templates": num_templates,
    }
    return JSONResponse(_sam6d_post("render_templates", payload))


@app.post("/estimate_pose")
async def estimate_pose(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    intrinsics_json: str = Form(...),   # JSON文字列: {"fx","fy","cx","cy"}
    cad_path: str = Form(...),
    template_dir: str = Form(...),
    det_score_thresh: float = Form(0.2),
):
    """
    RGB + 深度画像から 6DoF 姿勢推定

    Args:
        rgb:              RGB 画像 (PNG/JPEG)
        depth:            深度画像 (uint16 PNG, mm 単位)
        intrinsics_json:  カメラ内部パラメータ JSON
        cad_path:         サーバ上の CAD (.ply) パス [mm]
        template_dir:     render_templates() の出力ディレクトリ
        det_score_thresh: 検出スコア閾値

    Returns:
        {"R": [[...]], "t": [...], "mask_area": int}
    """
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp(dir=os.path.join(
        os.path.dirname(args_global.sam3d_repo), "tmp"))
    try:
        # アップロードファイルを一時保存
        rgb_path = os.path.join(tmpdir, "rgb.png")
        depth_path = os.path.join(tmpdir, "depth.png")
        cam_path = os.path.join(tmpdir, "camera.json")

        rgb_bytes = await rgb.read()
        with open(rgb_path, "wb") as f:
            f.write(rgb_bytes)

        depth_bytes = await depth.read()
        with open(depth_path, "wb") as f:
            f.write(depth_bytes)

        intrinsics = json.loads(intrinsics_json)
        cam_json = {
            "cam_K": [intrinsics["fx"], 0.0, intrinsics["cx"],
                      0.0, intrinsics["fy"], intrinsics["cy"],
                      0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        with open(cam_path, "w") as f:
            json.dump(cam_json, f)

        payload = {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "cam_json_path": cam_path,
            "cad_path": cad_path,
            "template_dir": template_dir,
            "det_score_thresh": det_score_thresh,
        }
        result = _sam6d_post("estimate_pose", payload, timeout=300.0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return JSONResponse(result)


@app.post("/full_pipeline")
async def full_pipeline(
    image: UploadFile = File(...),
    depth: UploadFile = File(...),
    intrinsics_json: str = Form(...),   # {"fx","fy","cx","cy"}
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    seed: int = Form(42),
    target_points: int = Form(2048),
    det_score_thresh: float = Form(0.2),
    output_dir: str = Form("tmp/pipeline"),
):
    """
    フルパイプライン: SAM-3D 再構成 → SAM-6D 姿勢推定 を一括実行

    Args:
        image:           RGB 画像 (PNG/JPEG)
        depth:           深度画像 (uint16 PNG, mm 単位)
        intrinsics_json: カメラ内部パラメータ JSON {"fx","fy","cx","cy"}
        click_x, click_y: SAM プロンプト座標 (-1,-1 で中央)
        seed:            sam-3d-objects シード
        target_points:   点群点数
        det_score_thresh: SAM-6D 検出閾値

    Returns:
        {
            "points":       [[x,y,z], ...],   # SAM-3D 点群 (object 座標系)
            "ply_path":     "/path/to/obj.ply",
            "template_dir": "/path/to/templates",
            "R":            [[...], [...], [...]],
            "t":            [x, y, z],          # カメラ座標系 [m]
            "mask_center_u": int,
            "mask_center_v": int,
        }
    """
    if sam_predictor is None or sam3d_inference is None:
        raise HTTPException(503, "SAM-3D モデルがロードされていません")

    import tempfile, shutil

    # ---- Step 1: SAM-3D で PLY 生成 ----
    image_bytes = await image.read()
    depth_bytes = await depth.read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "RGB画像のデコードに失敗しました")
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
    print(f"[Pipeline] SAM マスク完了 (面積:{best_mask.sum()}px)")

    print("[Pipeline] SAM-3D 推論中...")
    recon_output = sam3d_inference(rgb, best_mask, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"object_seed{seed}.ply")
    recon_output["gs"].save_ply(ply_path)
    print(f"[Pipeline] PLY 保存: {ply_path}")

    from plyfile import PlyData
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    n = len(points)
    choice = np.random.choice(n, target_points, replace=(n < target_points))
    points = points[choice]

    ys, xs = np.where(best_mask)
    mask_center_u = int(xs.mean())
    mask_center_v = int(ys.mean())

    # ---- Step 2: 深度・カメラパラメータを一時保存 ----
    tmpdir = tempfile.mkdtemp(dir=output_dir)
    try:
        rgb_path_tmp   = os.path.join(tmpdir, "rgb.png")
        depth_path_tmp = os.path.join(tmpdir, "depth.png")
        cam_path_tmp   = os.path.join(tmpdir, "camera.json")

        cv2.imwrite(rgb_path_tmp, bgr)

        depth_arr = np.frombuffer(depth_bytes, np.uint8)
        depth_img = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise HTTPException(400, "深度画像のデコードに失敗しました")
        cv2.imwrite(depth_path_tmp, depth_img)

        intrinsics = json.loads(intrinsics_json)
        cam_json = {
            "cam_K": [intrinsics["fx"], 0.0, intrinsics["cx"],
                      0.0, intrinsics["fy"], intrinsics["cy"],
                      0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        with open(cam_path_tmp, "w") as f:
            json.dump(cam_json, f)

        # ---- Step 3: テンプレートレンダリング (キャッシュ済みなら省略) ----
        tdir_resp = _sam6d_post("render_templates", {
            "cad_path": ply_path,
            "output_dir": None,
            "num_templates": 42,
        }, timeout=600.0)
        template_dir = tdir_resp["template_dir"]
        print(f"[Pipeline] テンプレートディレクトリ: {template_dir}")

        # ---- Step 4: SAM-6D 姿勢推定 ----
        pose_resp = _sam6d_post("estimate_pose", {
            "rgb_path": rgb_path_tmp,
            "depth_path": depth_path_tmp,
            "cam_json_path": cam_path_tmp,
            "cad_path": ply_path,
            "template_dir": template_dir,
            "det_score_thresh": det_score_thresh,
        }, timeout=300.0)
        print(f"[Pipeline] 姿勢推定完了")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return JSONResponse({
        "points":        points.tolist(),
        "num_points":    len(points),
        "ply_path":      ply_path,
        "template_dir":  template_dir,
        "R":             pose_resp["R"],
        "t":             pose_resp["t"],
        "mask_area":     pose_resp["mask_area"],
        "mask_center_u": mask_center_u,
        "mask_center_v": mask_center_v,
    })


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
    parser.add_argument("--sam6d-service", default="http://localhost:8081",
                        help="SAM-6D Docker サービスの URL (デフォルト: http://localhost:8081)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host-tmp", default="/home/okada/ws/project/tmp",
                        help="ホスト側の共有tmpディレクトリ (Dockerマウント元)")
    parser.add_argument("--docker-tmp", default="/workspace/tmp",
                        help="Dockerコンテナ内の共有tmpディレクトリ (マウント先)")
    args = parser.parse_args()
    args_global = args
    _sam6d_url  = args.sam6d_service
    _host_tmp   = args.host_tmp
    _docker_tmp = args.docker_tmp

    print("=" * 50)
    print(f"  SAM 3D + SAM-6D Pipeline Server")
    print(f"  device:       {args.device}")
    print(f"  host:port:    {args.host}:{args.port}")
    print(f"  sam6d_service: {_sam6d_url}")
    print("=" * 50)

    load_models(args.sam_checkpoint, args.sam3d_config, args.sam3d_repo, args.device)

    uvicorn.run(app, host=args.host, port=args.port)
