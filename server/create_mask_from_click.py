"""
クリック座標からSAMマスクを生成し、PEMが読めるdetection_ism.jsonを作成する。
ISMのテンプレートマッチングをバイパスして、指定した物体のポーズ推定を行うために使用。
"""
import argparse
import json
import os
import numpy as np
from PIL import Image
import torch
import pycocotools.mask as cocomask

SAM_CHECKPOINT = os.path.join(os.path.dirname(__file__),
                              "checkpoints/segment-anything/sam_vit_h_4b8939.pth")


def create_detection_from_click(rgb_path, click_x, click_y, output_dir):
    from segment_anything import sam_model_registry, SamPredictor

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    h, w = rgb.shape[:2]

    # クリック座標がなければ画像中央を使う
    if click_x < 0:
        click_x = w // 2
    if click_y < 0:
        click_y = h // 2

    print(f"[SAM] クリック点: ({click_x}, {click_y})")

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[click_x, click_y]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    # mask index 2 (最大マスク) を使用
    best_idx = 2
    mask = masks[best_idx]  # (H, W) bool
    score = float(scores[best_idx])
    print(f"[SAM] マスク面積: {mask.sum()}px  スコア: {score:.3f}")

    # 3枚のマスクをすべて保存 (スコアを画像に重畳)
    import cv2 as _cv2
    os.makedirs(f"{output_dir}/sam6d_results", exist_ok=True)
    for i in range(3):
        mask_save_path = f"{output_dir}/sam6d_results/mask_{i+1}.png"
        # グレースケールマスクをBGR3chに変換してスコアテキストを描画
        mask_bgr = _cv2.cvtColor(masks[i].astype(np.uint8) * 255, _cv2.COLOR_GRAY2BGR)
        label = f"mask_{i+1}  score={scores[i]:.3f}"
        _cv2.putText(mask_bgr, label, (10, 30), _cv2.FONT_HERSHEY_SIMPLEX,
                     0.9, (0, 255, 0), 2, _cv2.LINE_AA)
        _cv2.imwrite(mask_save_path, mask_bgr)
        print(f"[SAM] mask_{i+1}.png 保存: score={scores[i]:.3f} → {mask_save_path}")

    # bbox (x, y, w, h)
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = [x1, y1, x2 - x1, y2 - y1]

    # RLE エンコード
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")

    detection = [{
        "scene_id": 0,
        "image_id": 0,
        "category_id": 0,
        "bbox": bbox,
        "score": score,
        "segmentation": rle,
        "time": 0.0,
    }]

    os.makedirs(f"{output_dir}/sam6d_results", exist_ok=True)
    save_path = f"{output_dir}/sam6d_results/detection_ism.json"
    with open(save_path, "w") as f:
        json.dump(detection, f)
    print(f"[SAM] 保存: {save_path}")

    # マスク可視化
    vis = rgb.copy()
    vis[mask] = (vis[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    # クリック点を赤丸で描画
    import cv2
    cv2.circle(vis, (click_x, click_y), 6, (255, 0, 0), -1)
    # bbox を青枠で描画
    cv2.rectangle(vis, (x1, y1), (x1 + bbox[2], y1 + bbox[3]), (0, 0, 255), 2)
    vis_path = f"{output_dir}/sam6d_results/vis_mask.png"
    Image.fromarray(vis).save(vis_path)
    print(f"[SAM] マスク可視化: {vis_path}")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--click_x", default=-1, type=int)
    parser.add_argument("--click_y", default=-1, type=int)
    args = parser.parse_args()

    create_detection_from_click(args.rgb_path, args.click_x, args.click_y, args.output_dir)
