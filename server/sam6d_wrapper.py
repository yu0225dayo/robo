"""
SAM-6D ラッパー (サーバ側)

JiehongLin/SAM-6D の2ステージパイプラインをラップする。
    Stage 1: Instance Segmentation Model (ISM) - SAMベースのセグメンテーション
    Stage 2: Pose Estimation Model (PEM) - 6DoF pose推定

注意:
    - CAD model (.ply) はミリメートル単位
    - 深度画像はミリメートル単位 (カメラからのメートル値を変換して渡すこと)
    - テンプレートは物体ごとに事前レンダリング (render_templates() を一度だけ呼ぶ)
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import numpy as np
import cv2
from typing import Dict, Any, Optional


class SAM6DWrapper:
    """
    SAM-6D の2ステージパイプラインをラップするクラス

    使用例:
        wrapper = SAM6DWrapper(sam6d_repo="/path/to/SAM-6D", device="cuda")
        wrapper.load_models()

        # 物体ごとに一度: テンプレートレンダリング
        template_dir = wrapper.render_templates(cad_path_mm="obj.ply")

        # リアルタイム: pose推定
        R, t, mask_area = wrapper.estimate_pose(rgb, depth_m, intrinsics, cad_path_mm, template_dir)
    """

    def __init__(self, sam6d_repo: str, device: str = "cuda"):
        self.sam6d_repo = sam6d_repo
        self.device = device
        self._ism = None   # Instance Segmentation Model
        self._pem = None   # Pose Estimation Model
        self._ism_cfg = None
        self._pem_cfg = None

    def load_models(self, segmentor: str = "sam"):
        """
        ISM と PEM をロードする

        Args:
            segmentor: "sam" or "fastsam"
        """
        ism_dir = os.path.join(self.sam6d_repo, "SAM-6D", "Instance_Segmentation_Model")
        pem_dir = os.path.join(self.sam6d_repo, "SAM-6D", "Pose_Estimation_Model")

        pem_model_dir    = os.path.join(pem_dir, "model")
        pem_pnet2_dir    = os.path.join(pem_dir, "model", "pointnet2")
        pem_prov_dir     = os.path.join(pem_dir, "provider")
        pem_utils_dir    = os.path.join(pem_dir, "utils")
        for p in [ism_dir, pem_dir, pem_model_dir, pem_pnet2_dir,
                  pem_prov_dir, pem_utils_dir]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # --- ISM (Instance Segmentation Model) ---
        orig_dir = os.getcwd()
        try:
            os.chdir(ism_dir)
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from hydra.utils import instantiate
            import torch

            GlobalHydra.instance().clear()
            cfg_dir = os.path.join(ism_dir, "configs")
            with initialize_config_dir(version_base=None, config_dir=cfg_dir):
                cfg = compose(config_name="run_inference.yaml",
                              overrides=[f"model=ISM_{segmentor}",
                                         "save_dir=/tmp/sam6d_ism_log"])
            self._ism_cfg = cfg.model
            self._ism = instantiate(cfg.model)

            device = torch.device(self.device)
            self._ism.descriptor_model.model = self._ism.descriptor_model.model.to(device)
            self._ism.descriptor_model.model.device = device
            if hasattr(self._ism.segmentor_model, "predictor"):
                self._ism.segmentor_model.predictor.model = (
                    self._ism.segmentor_model.predictor.model.to(device)
                )
            else:
                self._ism.segmentor_model.model.setup_model(device=device, verbose=True)
            print(f"[SAM-6D] ISM ロード完了 ({segmentor})")
        finally:
            os.chdir(orig_dir)

        # --- PEM (Pose Estimation Model) ---
        try:
            os.chdir(pem_dir)
            from pose_estimation_model import Net as PoseEstimationModel
            from omegaconf import OmegaConf
            pem_cfg_path = os.path.join(pem_dir, "config", "base.yaml")
            self._pem_cfg = OmegaConf.load(pem_cfg_path)
            self._pem = PoseEstimationModel(self._pem_cfg.model)

            # 学習済み重みをロード
            import torch
            pem_ckpt = os.path.join(pem_dir, "checkpoints", "sam-6d-pem-base.pth")
            if os.path.exists(pem_ckpt):
                state = torch.load(pem_ckpt, map_location="cpu")
                # checkpoint 形式によってキーが異なる場合があるため両方試す
                sd = state.get("model", state.get("state_dict", state))
                self._pem.load_state_dict(sd, strict=False)
                print(f"[SAM-6D] PEM 重みロード完了: {pem_ckpt}")
            else:
                print(f"[SAM-6D] 警告: PEM 重みファイルが見つかりません: {pem_ckpt}")

            self._pem.to(self.device)
            self._pem.eval()
            print("[SAM-6D] PEM ロード完了")
        finally:
            os.chdir(orig_dir)

    def render_templates(
        self,
        cad_path_mm: str,
        output_dir: Optional[str] = None,
        num_templates: int = 42,
    ) -> str:
        """
        CADモデルから42視点のテンプレートをレンダリングする (物体ごとに一度)

        Args:
            cad_path_mm: CADモデル (.ply) パス [ミリメートル単位]
            output_dir:  テンプレート保存先 (None で自動生成)
            num_templates: テンプレート数 (デフォルト42)

        Returns:
            テンプレートディレクトリのパス
        """
        if output_dir is None:
            base = os.path.splitext(cad_path_mm)[0]
            output_dir = base + "_templates"

        os.makedirs(output_dir, exist_ok=True)
        render_script = os.path.join(
            self.sam6d_repo, "SAM-6D", "Render", "render_custom_templates.py"
        )
        blenderproc = "/opt/conda/envs/sam6d/bin/blenderproc"

        cmd = [
            blenderproc, "run", render_script,
            "--cad_path", cad_path_mm,
            "--output_dir", output_dir,
        ]
        print(f"[SAM-6D] テンプレートレンダリング開始: {cad_path_mm}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"テンプレートレンダリング失敗 (code={result.returncode}):\n{result.stderr[-2000:]}"
            )
        print(f"[SAM-6D] テンプレートレンダリング完了: {output_dir}")
        return output_dir

    def estimate_pose(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: Dict[str, float],
        cad_path_mm: str,
        template_dir: str,
        det_score_thresh: float = 0.2,
    ):
        """
        RGBD + CADモデルから 6DoF pose を推定する

        Args:
            rgb:             (H, W, 3) RGB画像 (uint8)
            depth_m:         (H, W) 深度画像 [メートル] → 内部でmmに変換
            intrinsics:      {"fx": ..., "fy": ..., "cx": ..., "cy": ...}
            cad_path_mm:     CADモデル (.ply) [ミリメートル単位]
            template_dir:    render_templates() の出力ディレクトリ
            det_score_thresh: 検出スコア閾値

        Returns:
            R:         (3, 3) float32 回転行列 (物体→カメラ座標系)
            t:         (3,)   float32 平行移動 [メートル]
            mask_area: int    セグメンテーション面積 [px]
        """
        if self._ism is None or self._pem is None:
            raise RuntimeError("load_models() を先に呼んでください。")

        # 一時ディレクトリにファイルを保存 (SAM-6Dはファイルパス入力)
        tmpdir = tempfile.mkdtemp()
        try:
            rgb_path   = os.path.join(tmpdir, "rgb.png")
            depth_path = os.path.join(tmpdir, "depth.png")
            cam_path   = os.path.join(tmpdir, "camera.json")
            seg_path   = os.path.join(tmpdir, "seg.json")

            # RGB 保存
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # 深度 保存 (メートル → ミリメートル → uint16 PNG)
            depth_mm = (depth_m * 1000.0).astype(np.uint16)
            cv2.imwrite(depth_path, depth_mm)

            # カメラ内部パラメータ保存 (SAM-6D の camera.json 形式)
            fx, fy = intrinsics["fx"], intrinsics["fy"]
            cx, cy = intrinsics["cx"], intrinsics["cy"]
            cam_json = {
                "cam_K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                "depth_scale": 1.0,
            }
            with open(cam_path, "w") as f:
                json.dump(cam_json, f)

            pem_dir = os.path.join(self.sam6d_repo, "SAM-6D", "Pose_Estimation_Model")
            if pem_dir not in sys.path:
                sys.path.insert(0, pem_dir)

            import torch
            from run_inference_custom import get_templates, get_test_data

            # Stage 1: SAM でマスク生成 → COCO RLE 形式で seg.json 保存
            import pycocotools.mask as cocomask

            bgr_np = cv2.imread(rgb_path)
            rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)

            proposals = self._ism.segmentor_model.generate_masks(rgb_np)
            masks = proposals["masks"]
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()  # (N, H, W) bool
            del proposals  # GPU中間テンソルを解放

            if len(masks) == 0:
                raise RuntimeError("物体が検出されませんでした。")

            # 面積の大きい順に上位10個に絞る (GPU OOM 対策: pairwise_distance が O(N^2) のため)
            areas = masks.sum(axis=(1, 2))
            top_idx = np.argsort(areas)[::-1][:10]
            masks = masks[top_idx]

            seg_data = []
            for mask in masks:
                mask_u8 = np.asfortranarray(mask.astype(np.uint8))
                rle = cocomask.encode(mask_u8)
                rle["counts"] = rle["counts"].decode("utf-8")
                seg_data.append({
                    "segmentation": {"size": list(mask.shape), "counts": rle["counts"]},
                    "score": 1.0,
                })
            with open(seg_path, "w") as f:
                json.dump(seg_data, f)
            print(f"[SAM-6D] SAM マスク生成完了: {len(seg_data)} 個")
            torch.cuda.empty_cache()  # ISM後のキャッシュ解放

            # Stage 2: テンプレート特徴量取得
            # render_custom_templates.py は output_dir/templates/ に保存する
            tem_path = os.path.join(template_dir, "templates")
            dataset_cfg = self._pem_cfg.test_dataset
            all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, dataset_cfg)

            with torch.no_grad():
                all_tem_pts, all_tem_feat = self._pem.feature_extraction.get_obj_feats(
                    all_tem, all_tem_pts, all_tem_choose
                )

            # Stage 3: 観測データ取得 → Pose Estimation
            input_data, _, _, _, dets = get_test_data(
                rgb_path=rgb_path,
                depth_path=depth_path,
                cam_path=cam_path,
                cad_path=cad_path_mm,
                seg_path=seg_path,
                det_score_thresh=det_score_thresh,
                cfg=dataset_cfg,
            )

            ninstance = input_data['pts'].size(0)
            with torch.no_grad():
                input_data['dense_po'] = all_tem_pts.repeat(ninstance, 1, 1)
                input_data['dense_fo'] = all_tem_feat.repeat(ninstance, 1, 1)
                out = self._pem(input_data)

            if 'pred_pose_score' in out:
                pose_scores = (out['pred_pose_score'] * out['score']).detach().cpu().numpy()
            else:
                pose_scores = out['score'].detach().cpu().numpy()

            pred_rot   = out['pred_R'].detach().cpu().numpy()
            pred_trans = out['pred_t'].detach().cpu().numpy() * 1000  # → mm

            best_idx = int(pose_scores.argmax())
            R   = pred_rot[best_idx].astype(np.float32)
            t_m = (pred_trans[best_idx] / 1000.0).astype(np.float32)  # mm → m
            mask_area = int(dets[best_idx].get("area", 0)) if best_idx < len(dets) else 0

            # 推論後に中間テンソルを明示的に解放
            del all_tem, all_tem_pts, all_tem_choose, all_tem_feat
            del input_data, out
            torch.cuda.empty_cache()

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return R, t_m, mask_area
