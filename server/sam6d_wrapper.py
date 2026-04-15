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
        pcd_path: Optional[str] = None,
    ) -> str:
        """
        テンプレートをレンダリングする (物体ごとに一度)

        pcd_path が指定された場合は点群直接投影 (Blenderproc不要)。
        指定がなければ従来の Blenderproc + メッシュ方式を使用。

        Args:
            cad_path_mm:   メッシュ (.ply) パス [mm単位] (Blenderproc方式で使用)
            output_dir:    テンプレート保存先 (None で自動生成)
            num_templates: テンプレート数 (デフォルト42)
            pcd_path:      点群 PLY パス [mm単位] (指定時は点群直接投影を使用)

        Returns:
            テンプレートディレクトリのパス
        """
        if output_dir is None:
            base = os.path.splitext(cad_path_mm)[0]
            output_dir = base + "_templates"

        os.makedirs(output_dir, exist_ok=True)

        if pcd_path is not None:
            # 点群直接投影方式
            render_script = os.path.join(
                self.sam6d_repo, "SAM-6D", "Render", "render_pointcloud_templates.py"
            )
            python = "/opt/conda/envs/sam6d/bin/python"
            cmd = [
                python, render_script,
                "--pcd_path", pcd_path,
                "--output_dir", output_dir,
                "--num_views", str(num_templates),
            ]
            print(f"[SAM-6D] 点群直接投影テンプレート生成: {pcd_path}")
        else:
            # 従来の Blenderproc 方式
            render_script = os.path.join(
                self.sam6d_repo, "SAM-6D", "Render", "render_custom_templates.py"
            )
            blenderproc = "/opt/conda/envs/sam6d/bin/blenderproc"
            cmd = [
                blenderproc, "run", render_script,
                "--cad_path", cad_path_mm,
                "--output_dir", output_dir,
            ]
            print(f"[SAM-6D] Blenderprocテンプレートレンダリング開始: {cad_path_mm}")

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
        click_x: int = -1,
        click_y: int = -1,
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

            # Stage 1: セグメンテーション
            import pycocotools.mask as cocomask
            bgr_np = cv2.imread(rgb_path)
            rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)

            _click_mode = click_x >= 0 and click_y >= 0
            if _click_mode:
                # --- クリック座標あり: SAM 点プロンプトで直接セグメント ---
                print(f"[SAM-6D] SAM 点プロンプト: ({click_x}, {click_y})")
                predictor = self._ism.segmentor_model.predictor
                predictor.set_image(rgb_np)
                sam_masks, sam_scores, _ = predictor.predict(
                    point_coords=np.array([[click_x, click_y]]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                # mask index 2 (最大マスク) を先頭に固定し、その後 index 1, 0
                order = [2, 1, 0]
                seg_data = []
                for i in order:
                    mask = sam_masks[i]
                    mask_u8 = np.asfortranarray(mask.astype(np.uint8))
                    rle = cocomask.encode(mask_u8)
                    rle["counts"] = rle["counts"].decode("utf-8")
                    seg_data.append({
                        "segmentation": {"size": list(mask.shape), "counts": rle["counts"]},
                        "score": float(sam_scores[i]),
                    })
                # mask_1, 2, 3 として sam6d_results ディレクトリに保存
                results_dir = os.path.join(os.path.dirname(template_dir.rstrip("/")), "sam6d_results")
                os.makedirs(results_dir, exist_ok=True)
                for i in range(3):
                    save_path = os.path.join(results_dir, f"mask_{i+1}.png")
                    cv2.imwrite(save_path, (sam_masks[i].astype(np.uint8) * 255))
                    print(f"[SAM-6D] mask_{i+1}.png 保存: score={sam_scores[i]:.3f} → {save_path}")
                print(f"[SAM-6D] 点プロンプト完了: {len(seg_data)} マスク")
            else:
                # --- クリック座標なし: 完全 ISM パイプライン ---
                import glob as _glob
                import trimesh
                from PIL import Image as PILImage
                from utils.bbox_utils import CropResizePad
                from utils.poses.pose_utils import (
                    get_obj_poses_from_template_level,
                    load_index_level_in_level2,
                )
                from model.utils import Detections

                device = torch.device(self.device)

                tem_path_ism = os.path.join(template_dir, "templates")
                n_tem = len(_glob.glob(f"{tem_path_ism}/rgb_*.png"))
                if n_tem == 0:
                    raise RuntimeError(f"テンプレートが見つかりません: {tem_path_ism}")

                boxes_t, masks_t, templates_t = [], [], []
                for idx in range(n_tem):
                    img_t = PILImage.open(os.path.join(tem_path_ism, f"rgb_{idx}.png"))
                    msk_t = PILImage.open(os.path.join(tem_path_ism, f"mask_{idx}.png"))
                    boxes_t.append(msk_t.getbbox())
                    img_arr = torch.from_numpy(np.array(img_t.convert("RGB")) / 255).float()
                    msk_arr = torch.from_numpy(np.array(msk_t.convert("L")) / 255).float()
                    img_arr = img_arr * msk_arr[:, :, None]
                    templates_t.append(img_arr)
                    masks_t.append(msk_arr.unsqueeze(-1))

                templates_t    = torch.stack(templates_t).permute(0, 3, 1, 2)
                masks_t        = torch.stack(masks_t).permute(0, 3, 1, 2)
                boxes_t_tensor = torch.tensor(np.array(boxes_t))

                proposal_processor = CropResizePad(224)
                templates_proc = proposal_processor(images=templates_t, boxes=boxes_t_tensor).to(device)
                masks_cropped  = proposal_processor(images=masks_t,     boxes=boxes_t_tensor).to(device)

                self._ism.ref_data = {}
                self._ism.ref_data["descriptors"] = self._ism.descriptor_model.compute_features(
                    templates_proc, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
                self._ism.ref_data["appe_descriptors"] = self._ism.descriptor_model.compute_masked_patch_feature(
                    templates_proc, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data

                raw_dets = self._ism.segmentor_model.generate_masks(rgb_np)
                detections_ism = Detections(raw_dets)
                if len(detections_ism) == 0:
                    raise RuntimeError("物体が検出されませんでした。")

                query_desc, query_appe_desc = self._ism.descriptor_model.forward(rgb_np, detections_ism)
                (idx_selected, pred_idx_objects, semantic_score, best_template) = \
                    self._ism.compute_semantic_score(query_desc)
                detections_ism.filter(idx_selected)
                query_appe_desc = query_appe_desc[idx_selected, :]

                appe_scores, ref_aux_descriptor = self._ism.compute_appearance_score(
                    best_template, pred_idx_objects, query_appe_desc)

                template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
                template_poses[:, :3, 3] *= 0.4
                poses = torch.tensor(template_poses).to(torch.float32).to(device)
                self._ism.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

                mesh_ism = trimesh.load_mesh(cad_path_mm)
                model_pts_ism = mesh_ism.sample(2048).astype(np.float32) / 1000.0
                self._ism.ref_data["pointcloud"] = torch.tensor(model_pts_ism).unsqueeze(0).data.to(device)

                cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                batch_ism = {
                    "depth":         torch.from_numpy(depth_mm.astype(np.int32)).unsqueeze(0).to(device),
                    "cam_intrinsic": torch.from_numpy(cam_K).unsqueeze(0).to(device),
                    "depth_scale":   torch.tensor(np.array(1.0)).unsqueeze(0).to(device),
                }
                image_uv = self._ism.project_template_to_image(
                    best_template, pred_idx_objects, batch_ism, detections_ism.masks)
                geometric_score, visible_ratio = self._ism.compute_geometric_score(
                    image_uv, detections_ism, query_appe_desc, ref_aux_descriptor,
                    visible_thred=self._ism.visible_thred)

                final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (1 + 1 + visible_ratio)
                detections_ism.add_attribute("scores", final_score)
                detections_ism.add_attribute("object_ids", torch.zeros_like(final_score))
                detections_ism.to_numpy()

                sorted_idx = np.argsort(detections_ism.scores)[::-1][:5]
                masks_ism  = detections_ism.masks[sorted_idx]
                scores_ism = detections_ism.scores[sorted_idx]
                print(f"[SAM-6D] ISM 完了: 上位{len(sorted_idx)}件 scores={scores_ism.round(3)}")

                del detections_ism, query_desc, query_appe_desc
                del semantic_score, appe_scores, geometric_score, visible_ratio, final_score
                del image_uv, ref_aux_descriptor, best_template, pred_idx_objects, idx_selected
                del templates_proc, masks_cropped, batch_ism
                self._ism.ref_data = {}
                torch.cuda.empty_cache()

                seg_data = []
                for i in range(len(masks_ism)):
                    mask  = masks_ism[i]
                    score = float(scores_ism[i])
                    mask_u8 = np.asfortranarray(mask.astype(np.uint8))
                    rle = cocomask.encode(mask_u8)
                    rle["counts"] = rle["counts"].decode("utf-8")
                    seg_data.append({
                        "segmentation": {"size": list(mask.shape), "counts": rle["counts"]},
                        "score": score,
                    })

            with open(seg_path, "w") as f:
                json.dump(seg_data, f)
            torch.cuda.empty_cache()

            # PEM用: オブジェクト領域を白色に置き換えた RGB を生成
            # 白メッシュから生成したテンプレートと色条件を合わせるため
            import pycocotools.mask as _cocomask
            rgb_path_pem = rgb_path
            if seg_data:
                # クリックモードはmask index 2 (最大マスク=seg_data[0]) を使用
                # ISMモードはスコア最高のマスクを使用
                best_seg = seg_data[0] if _click_mode else max(seg_data, key=lambda x: x["score"])
                rle = best_seg["segmentation"]
                rle_decode = {"size": rle["size"],
                              "counts": rle["counts"].encode("utf-8") if isinstance(rle["counts"], str) else rle["counts"]}
                obj_mask = _cocomask.decode(rle_decode).astype(bool)
                bgr_pem = cv2.imread(rgb_path).copy()
                bgr_pem[obj_mask] = (255, 255, 255)
                rgb_path_pem = os.path.join(tmpdir, "rgb_pem.png")
                cv2.imwrite(rgb_path_pem, bgr_pem)
                print(f"[SAM-6D] PEM用白マスクRGB生成: {obj_mask.sum()} px を白色化")

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
                rgb_path=rgb_path_pem,
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
