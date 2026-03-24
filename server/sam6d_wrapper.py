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
        pem_dir = os.path.join(self.sam6d_repo, "SAM-6D", "Pose_Estimation_Model")

        orig_dir = os.getcwd()
        try:
            os.chdir(pem_dir)
            if pem_dir not in sys.path:
                sys.path.insert(0, pem_dir)
            from utils.render_utils import render_templates as _render
            _render(
                cad_path=cad_path_mm,
                output_dir=output_dir,
                num_templates=num_templates,
            )
            print(f"[SAM-6D] テンプレートレンダリング完了: {output_dir} ({num_templates}視点)")
        finally:
            os.chdir(orig_dir)

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
            orig_dir = os.getcwd()
            try:
                os.chdir(pem_dir)

                # Stage 1: Instance Segmentation
                from run_inference_custom import batch_input_data
                detections = self._ism.generate_proposals(
                    rgb_path=rgb_path,
                    stability_score_thresh=0.97,
                )
                if detections is None or len(detections) == 0:
                    raise RuntimeError("物体が検出されませんでした。")

                # 検出結果を seg.json として保存
                seg_data = []
                for det in detections:
                    seg_data.append({
                        "segmentation": det["segmentation"],
                        "score": float(det["stability_score"]),
                        "bbox": det["bbox"],
                    })
                with open(seg_path, "w") as f:
                    json.dump(seg_data, f)

                # Stage 2: Pose Estimation
                input_data, _, _ = batch_input_data(
                    depth_path=depth_path,
                    cam_path=cam_path,
                    cad_path=cad_path_mm,
                    seg_path=seg_path,
                    det_score_thresh=det_score_thresh,
                    cfg=self._pem_cfg,
                )

                # テンプレート特徴量取得
                from run_inference_custom import get_templates
                templates = get_templates(template_dir, self._pem_cfg)

                import torch
                with torch.no_grad():
                    obj_feats = self._pem.feature_extraction.get_obj_feats(
                        templates[0], templates[1], templates[2]
                    )
                    R_pred, t_pred, scores = self._pem(input_data, obj_feats)

                # 最高スコアの結果を取得
                best_idx = scores.argmax().item()
                R = R_pred[best_idx].cpu().numpy().astype(np.float32)   # (3, 3)
                t_mm = t_pred[best_idx].cpu().numpy().astype(np.float32)  # (3,) [mm]
                t_m = t_mm / 1000.0  # mm → m

                # マスク面積
                best_det = detections[best_idx] if best_idx < len(detections) else detections[0]
                mask_area = int(best_det.get("area", 0))

            finally:
                os.chdir(orig_dir)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return R, t_m, mask_area
