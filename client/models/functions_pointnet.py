"""
PointNet パーツセグメンテーション関連ユーティリティ
- get_patseg_wo_target: ターゲットなしでパーツセグメンテーション実行
- farthest_point_sampling: FPSによる点群サブサンプリング
"""

import numpy as np
import torch
import random


def farthest_point_sampling(points, target_num=100):
    """
    最遠点サンプリング (FPS)
    Args:
        points: (N, 3) numpy array
        target_num: サンプリング後の点数
    Returns:
        (target_num, 3) numpy array
    """
    sampled = [random.randint(0, len(points) - 1)]
    dists = np.full(len(points), np.inf)
    for _ in range(1, target_num):
        last = points[sampled[-1]]
        dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))
        sampled.append(np.argmax(dists))
    return points[sampled]


def get_patseg_wo_target(model, point, num_classes=3):
    """
    ターゲットラベルなしでパーツセグメンテーションを実行する
    (実世界推論用)

    Args:
        model: PointNetDenseCls インスタンス
        point: (B, N, 3) torch.Tensor の点群
        num_classes: セグメントクラス数 (デフォルト: 3)

    Returns:
        pl:       (B, 3, 256) 左手接触点群 (centering+scaling済み)
        pr:       (B, 3, 256) 右手接触点群 (centering+scaling済み)
        all_feat: (B, 1024) 全体形状特徴ベクトル
        plout:    (B, 3, 256) 左手接触点群 (scalingのみ)
        prout:    (B, 3, 256) 右手接触点群 (scalingのみ)
    """
    B = point.size(0)
    points = point.transpose(2, 1)  # (B, 3, N)

    # パーツセグメンテーション
    pred, trans, trans_feat, all_feat = model(points)
    pred = pred.view(-1, num_classes)
    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()

    print(f"  セグメンテーション結果 - 右手(1): {np.count_nonzero(pred_choice.cpu()==1)}, "
          f"左手(2): {np.count_nonzero(pred_choice.cpu()==2)}, "
          f"非接触(0): {np.count_nonzero(pred_choice.cpu()==0)}")

    # (B, 3, N) → (B, N, 3)
    points = points.transpose(1, 2).cpu().data.numpy()
    pred_np = pred_np.reshape(B, -1, 1)

    pl_out, pr_out = np.empty((0, 3)), np.empty((0, 3))
    plout_raw, prout_raw = np.empty((0, 3)), np.empty((0, 3))

    for batch in range(B):
        parts_l_list = np.array([])
        parts_r_list = np.array([])

        target_l = pred_np
        target_r = pred_np

        n_pts = points.shape[1]
        count_label2 = np.count_nonzero(pred_np[batch] == 2)
        count_label1 = np.count_nonzero(pred_np[batch] == 1)

        if count_label2 <= 10 or count_label1 <= 10:
            print("  警告: パーツ点が極端に少ないです。セグメンテーション結果を確認してください。")

        for j in range(n_pts):
            if target_l[batch][j] == 2:
                parts_l_list = np.append(parts_l_list, points[batch][j])
            if target_r[batch][j] == 1:
                parts_r_list = np.append(parts_r_list, points[batch][j])

        # 点が少なければコピーで拡張
        while len(parts_l_list) <= (3 * 256):
            add_list = parts_l_list * 1.01
            parts_l_list = np.append(parts_l_list, add_list)
        while len(parts_r_list) <= (3 * 256):
            add_list = parts_r_list * 1.01
            parts_r_list = np.append(parts_r_list, add_list)

        # FPS サンプリング
        parts_l_list = parts_l_list.reshape(int(len(parts_l_list) / 3), 3)
        pl = farthest_point_sampling(parts_l_list, target_num=256)
        parts_r_list = parts_r_list.reshape(int(len(parts_r_list) / 3), 3)
        pr = farthest_point_sampling(parts_r_list, target_num=256)

        plout_raw = np.vstack((plout_raw, pl))
        prout_raw = np.vstack((prout_raw, pr))

        # centering + scaling
        pl_center = np.expand_dims(np.mean(pl, axis=0), 0)
        pl_c = pl - pl_center
        dist_l = np.max(np.sqrt(np.sum(pl_c ** 2, axis=1)), 0)

        pr_center = np.expand_dims(np.mean(pr, axis=0), 0)
        pr_c = pr - pr_center
        dist_r = np.max(np.sqrt(np.sum(pr_c ** 2, axis=1)), 0)

        pl_out = np.vstack((pl_out, pl_c / dist_l))
        pr_out = np.vstack((pr_out, pr_c / dist_r))

    # numpy → torch
    pl = torch.from_numpy(pl_out.reshape(B, 256, 3).astype(np.float32)).transpose(2, 1)
    pr = torch.from_numpy(pr_out.reshape(B, 256, 3).astype(np.float32)).transpose(2, 1)
    plout = torch.from_numpy(plout_raw.reshape(B, 256, 3).astype(np.float32)).transpose(2, 1)
    prout = torch.from_numpy(prout_raw.reshape(B, 256, 3).astype(np.float32)).transpose(2, 1)

    return pl, pr, all_feat, plout, prout
