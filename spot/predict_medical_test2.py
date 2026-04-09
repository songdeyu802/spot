import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
import torch
import cv2
from skimage.morphology import binary_erosion, disk
from nets.unet import Unet
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment  # 添加这个导入
import itertools
from copy import deepcopy

# =========================
# 配置区
# =========================
MODEL_PATH = "logs/best_epoch_weights.pth"
#13  261  255 25
IMAGE_PATH = "F:/西物所博士/FPGA信号阈值设置/right/论文/光斑原图/image_68.png"
GT_MASK_PATH = "E:/unet-pytorch-main/Medical_Datasets/Labels/image_68.png"  # 没有GT可设为None

#IMAGE_PATH = "E:/Infrared-Small-Target-Detection-master/dataset/NUDT-SIRST/images/000004.png"
#GT_MASK_PATH = "E:/Infrared-Small-Target-Detection-master/dataset/NUDT-SIRST/masks/000004.png"
#INPUT_SHAPE = [256, 256]
SAVE_DIR = "pred_results"

BACKBONE = "myunet"
NUM_CLASSES = 2
INPUT_SHAPE = [720, 720]
CUDA = True

# 单次预测阈值
THRESHOLD = 0.8

# 圆形度阈值
MIN_CIRCULARITY = 0.6

# 连通域最小面积
MIN_COMPONENT_AREA = 2

# 腐蚀半径，0 表示不腐蚀
EROSION_RADIUS = 1

# 多百分位列表：先收缩到较少几个，降低误判
HIGH_PERCENT_LIST = [95.0, 97.0, 99.0, 99.5]
# 投票阈值：
# 例如4次预测里，至少有2次判为前景才保留
VOTE_THRESHOLD = 2

# 是否使用投票融合；False 则退回 OR 融合
USE_VOTE_FUSION = True

# 添加中心匹配的最大距离阈值
MAX_MATCH_DISTANCE = 8.0


# =========================
# 图像预处理函数
# =========================
def normalize_16bit_image(img, low_percent=1, high_percent=99.5):
    img = img.astype(np.float32)
    low = np.percentile(img, low_percent)
    high = np.percentile(img, high_percent)
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)
    return img


def resize_image_keep_ratio(image, size):
    iw, ih = image.size
    h, w = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("I;16", (w, h), 0)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image.paste(image, (dx, dy))
    return new_image, nw, nh, dx, dy


# =========================
# 圆形度过滤
# =========================
def filter_by_circularity(mask, min_circularity=MIN_CIRCULARITY):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], closed=True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity >= min_circularity:
            filtered[labels == i] = 1
    return filtered


# =========================
# GT 读取与评估
# =========================
def load_binary_mask(mask_path, target_size=None):
    """
    读取GT mask并转成二值图:
    前景=1, 背景=0
    target_size: (W, H)
    """
    gt = Image.open(mask_path)
    if gt.mode != "L":
        gt = gt.convert("L")

    if target_size is not None:
        gt = gt.resize(target_size, Image.NEAREST)

    gt = np.array(gt, dtype=np.uint8)
    gt = (gt > 0).astype(np.uint8)
    return gt


def compute_binary_seg_metrics(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"pred_mask shape {pred_mask.shape} != gt_mask shape {gt_mask.shape}")

    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": float(precision),
        "Recall": float(recall),
        "IoU": float(iou),
        "Dice": float(dice),
        "Accuracy": float(accuracy),
        "ConfusionMatrix": [[int(tn), int(fp)],
                            [int(fn), int(tp)]]
    }


def save_metrics(metrics, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("========== 分割评估结果 ==========\n")
        f.write(f"TP: {metrics['TP']}\n")
        f.write(f"FP: {metrics['FP']}\n")
        f.write(f"FN: {metrics['FN']}\n")
        f.write(f"TN: {metrics['TN']}\n")
        f.write(f"Precision: {metrics['Precision']:.6f}\n")
        f.write(f"Recall:    {metrics['Recall']:.6f}\n")
        f.write(f"IoU:       {metrics['IoU']:.6f}\n")
        f.write(f"Dice:      {metrics['Dice']:.6f}\n")
        f.write(f"Accuracy:  {metrics['Accuracy']:.6f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(np.array(metrics["ConfusionMatrix"])))
        f.write("\n")


def save_error_map(pred_mask, gt_mask, save_path):
    """
    误差图:
    TP = 白色
    FP = 红色
    FN = 蓝色
    """
    err_vis = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

    tp_mask = (pred_mask == 1) & (gt_mask == 1)
    fp_mask = (pred_mask == 1) & (gt_mask == 0)
    fn_mask = (pred_mask == 0) & (gt_mask == 1)

    err_vis[tp_mask] = [255, 255, 255]  # 白
    err_vis[fp_mask] = [255, 0, 0]  # 红
    err_vis[fn_mask] = [0, 0, 255]  # 蓝

    Image.fromarray(err_vis).save(save_path)


def extract_spot_centers(mask, min_area=1):
    """
    从二值mask中提取每个连通域的中心点
    返回:
        centers: np.ndarray, shape [N, 2], 每行为 [x, y]
        areas:   np.ndarray, shape [N]
    """
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    centers = []
    areas = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        centers.append([cx, cy])
        areas.append(area)

    if len(centers) == 0:
        centers = np.zeros((0, 2), dtype=np.float32)
        areas = np.zeros((0,), dtype=np.int32)
    else:
        centers = np.array(centers, dtype=np.float32)
        areas = np.array(areas, dtype=np.int32)

    return centers, areas


def evaluate_spot_centers(pred_mask, gt_mask, min_area=1, max_match_distance=8.0):
    """
    基于光斑中心的一对一匹配评估
    pred_mask, gt_mask: 二值mask, 0/1
    返回:
        metrics: dict
        matches: list of (gt_idx, pred_idx, dist)
        gt_centers: [Ng,2]
        pred_centers: [Np,2]
    """
    gt_centers, gt_areas = extract_spot_centers(gt_mask, min_area=min_area)
    pred_centers, pred_areas = extract_spot_centers(pred_mask, min_area=min_area)

    num_gt = len(gt_centers)
    num_pred = len(pred_centers)

    # 边界情况
    if num_gt == 0 and num_pred == 0:
        metrics = {
            "GT_Count": 0,
            "Pred_Count": 0,
            "Matched_Count": 0,
            "FN_Count": 0,
            "FP_Count": 0,
            "Instance_Precision": 1.0,
            "Instance_Recall": 1.0,
            "Mean_Center_Error": 0.0,
            "Median_Center_Error": 0.0,
            "Max_Center_Error": 0.0
        }
        return metrics, [], gt_centers, pred_centers

    if num_gt == 0:
        metrics = {
            "GT_Count": 0,
            "Pred_Count": num_pred,
            "Matched_Count": 0,
            "FN_Count": 0,
            "FP_Count": num_pred,
            "Instance_Precision": 0.0,
            "Instance_Recall": 1.0,
            "Mean_Center_Error": None,
            "Median_Center_Error": None,
            "Max_Center_Error": None
        }
        return metrics, [], gt_centers, pred_centers

    if num_pred == 0:
        metrics = {
            "GT_Count": num_gt,
            "Pred_Count": 0,
            "Matched_Count": 0,
            "FN_Count": num_gt,
            "FP_Count": 0,
            "Instance_Precision": 1.0,
            "Instance_Recall": 0.0,
            "Mean_Center_Error": None,
            "Median_Center_Error": None,
            "Max_Center_Error": None
        }
        return metrics, [], gt_centers, pred_centers

    # 构造距离矩阵
    diff = gt_centers[:, None, :] - pred_centers[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # [num_gt, num_pred]

    # 匈牙利匹配
    gt_idx, pred_idx = linear_sum_assignment(dist_matrix)

    matches = []
    matched_gt = set()
    matched_pred = set()

    for g, p in zip(gt_idx, pred_idx):
        d = dist_matrix[g, p]
        if d <= max_match_distance:
            matches.append((int(g), int(p), float(d)))
            matched_gt.add(int(g))
            matched_pred.add(int(p))

    matched_count = len(matches)
    fn_count = num_gt - matched_count
    fp_count = num_pred - matched_count

    inst_precision = matched_count / (num_pred + 1e-8)
    inst_recall = matched_count / (num_gt + 1e-8)

    if matched_count > 0:
        dists = np.array([m[2] for m in matches], dtype=np.float32)
        mean_err = float(np.mean(dists))
        median_err = float(np.median(dists))
        max_err = float(np.max(dists))
    else:
        mean_err = None
        median_err = None
        max_err = None

    metrics = {
        "GT_Count": int(num_gt),
        "Pred_Count": int(num_pred),
        "Matched_Count": int(matched_count),
        "FN_Count": int(fn_count),
        "FP_Count": int(fp_count),
        "Instance_Precision": float(inst_precision),
        "Instance_Recall": float(inst_recall),
        "Mean_Center_Error": mean_err,
        "Median_Center_Error": median_err,
        "Max_Center_Error": max_err
    }

    return metrics, matches, gt_centers, pred_centers


def save_center_match_visualization(image_gray, gt_centers, pred_centers, matches, save_path):
    """
    可视化:
    - GT中心: 绿色
    - Pred中心: 红色
    - 匹配线: 黄色
    """
    if image_gray.ndim == 2:
        vis = np.stack([image_gray, image_gray, image_gray], axis=-1)
    else:
        vis = image_gray.copy()

    vis = vis.astype(np.uint8).copy()

    # 画GT中心
    for x, y in gt_centers:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 255, 0), 1)

    # 画Pred中心
    for x, y in pred_centers:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (255, 0, 0), 1)

    # 画匹配线
    for g, p, d in matches:
        gx, gy = gt_centers[g]
        px, py = pred_centers[p]
        cv2.line(
            vis,
            (int(round(gx)), int(round(gy))),
            (int(round(px)), int(round(py))),
            (255, 255, 0),
            1
        )

    Image.fromarray(vis).save(save_path)


# =========================
# 共线性+等间距评分函数
# =========================
def eval_subset_by_line_and_spacing(Q, params):
    """
    对保留点集 Q 进行评分。
    返回 (score, info_dict)
    """
    M = Q.shape[0]
    if M < 4:
        return np.inf, None

    centroid = np.mean(Q, axis=0)
    Q0 = Q - centroid
    U, s, Vt = svd(Q0, full_matrices=False)
    dir_main = Vt[0, :]
    dir_norm = Vt[1, :]

    s_proj = Q0 @ dir_main
    e_proj = Q0 @ dir_norm

    order = np.argsort(s_proj)
    s_sort = s_proj[order]
    Q_sort = Q[order, :]
    e_sort = e_proj[order]

    idx = np.arange(M)
    coeff = np.polyfit(idx, s_sort, 1)
    d = abs(coeff[0])
    if d < 1e-6:
        return np.inf, None
    s_fit = np.polyval(coeff, idx)

    D = np.abs(s_sort[:, None] - s_sort[None, :])

    ID = np.abs(idx[:, None] - idx[None, :])
    D0 = ID * d

    triu_mask = np.triu(np.ones((M, M), dtype=bool), k=1)

    r_line = np.sqrt(np.mean(e_sort ** 2)) / d
    diff_D = D[triu_mask] - D0[triu_mask]
    r_dist = np.sqrt(np.mean(diff_D ** 2)) / d
    g = np.diff(s_sort)
    r_gap = np.sqrt(np.mean((g - d) ** 2)) / d

    if np.mean(g) > 1e-6:
        r_cv = np.std(g) / np.mean(g)
    else:
        r_cv = 1.0

    score = (
            params['w_line'] * r_line +
            params['w_dist'] * r_dist +
            params['w_gap'] * r_gap +
            0.05 * r_cv
    )

    info = {
        'centroid': centroid,
        'dir_main': dir_main,
        'dir_norm': dir_norm,
        's_sort': s_sort,
        's_fit': s_fit,
        'spacing': d,
        'D': D,
        'D0': D0,
        'Qsort': Q_sort,
        'r_line': r_line,
        'r_dist': r_dist,
        'r_gap': r_gap,
        'r_cv': r_cv
    }
    return score, info


# =========================
# 穷举剔除主函数
# =========================
def remove_outliers_by_exhaustive(P, max_remove=3, target_num=None, params=None):
    """
    穷举删除 0~max_remove 个点，返回最优保留点集。
    P: (N,2) 候选点坐标
    max_remove: 最多删除点数
    target_num: 期望保留点数，若指定则只尝试删除 N-target_num 个点
    params: 评分权重字典
    返回 (P_clean, noise_idx, best_info)
    """
    if params is None or not isinstance(params, dict):
        print(f"警告：params 类型为 {type(params)}，已重置为默认字典")
        params = {
            'w_line': 0.35,
            'w_dist': 0.45,
            'w_gap': 0.20,
            'lambda_delete': 0.08
        }

    N = P.shape[0]
    if N < 4:
        raise ValueError("点数少于4，无法进行稳健剔除。")

    if target_num is not None:
        k_need = N - target_num
        if k_need < 0:
            raise ValueError("target_num 大于候选点数。")
        if k_need > max_remove:
            raise ValueError(f"需要删除 {k_need} 个点，超过 max_remove={max_remove}。")
        candidates_k = [k_need]
    else:
        candidates_k = range(0, max_remove + 1)

    best_score = np.inf
    best_keep_mask = None
    best_noise_idx = None
    best_info = None

    for k in candidates_k:
        if k == 0:
            comb_iter = [()]
        else:
            comb_iter = itertools.combinations(range(N), k)

        for remove_idx in comb_iter:
            keep_mask = np.ones(N, dtype=bool)
            keep_mask[list(remove_idx)] = False
            Q = P[keep_mask, :]

            if Q.shape[0] < 4:
                continue

            score, info = eval_subset_by_line_and_spacing(Q, params)
            if score is None:
                continue

            score += params['lambda_delete'] * (k / N)

            if score < best_score:
                best_score = score
                best_keep_mask = keep_mask.copy()
                best_noise_idx = np.array(list(remove_idx))
                best_info = deepcopy(info)
                best_info['score'] = score
                best_info['remove_idx'] = best_noise_idx
                best_info['keep_mask'] = best_keep_mask

    if best_keep_mask is None:
        raise RuntimeError("未找到有效子集。")

    P_clean = P[best_keep_mask, :]
    noise_idx = best_noise_idx

    return P_clean, noise_idx, best_info


# =========================
# 主函数
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

    print("==== 当前配置 ====")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"IMAGE_PATH: {IMAGE_PATH}")
    print(f"GT_MASK_PATH: {GT_MASK_PATH}")
    print(f"HIGH_PERCENT_LIST: {HIGH_PERCENT_LIST}")
    print(f"THRESHOLD: {THRESHOLD}")
    print(f"USE_VOTE_FUSION: {USE_VOTE_FUSION}")
    print(f"VOTE_THRESHOLD: {VOTE_THRESHOLD}")
    print("=================\n")

    # 1. 加载模型
    model = Unet(num_classes=NUM_CLASSES, pretrained=False, backbone=BACKBONE)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 2. 读取原始图像
    image = Image.open(IMAGE_PATH)
    old_image = image.copy()

    # 用于显示的 8-bit 原图
    old_np = np.array(old_image).astype(np.float32)
    if old_np.ndim == 3:
        old_np = old_np[..., 0]
    old_disp = old_np.copy()
    old_disp = old_disp - old_disp.min()
    old_disp = old_disp / (old_disp.max() + 1e-8) * 255.0
    old_disp = old_disp.astype(np.uint8)

    old_rgb = np.stack([old_disp, old_disp, old_disp], axis=-1)
    print(f"Min: {old_np.min()}, Max: {old_np.max()}")
    print(f"Valid values ratio: {(old_np > 0).mean():.2%}")
    # 3. 先把 resize 做一次，后面只改归一化百分位
    resized_image, nw, nh, dx, dy = resize_image_keep_ratio(image, INPUT_SHAPE)
    resized_np_base = np.array(resized_image, dtype=np.uint16)
    if resized_np_base.ndim == 3:
        resized_np_base = resized_np_base[..., 0]

    # 4. 多百分位预测融合：改成 vote map
    vote_map = np.zeros(old_disp.shape, dtype=np.uint16)

    for hp in HIGH_PERCENT_LIST:
        print(f"Processing with high_percent = {hp} ...")

        image_data = normalize_16bit_image(resized_np_base, low_percent=1, high_percent=hp)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = np.expand_dims(image_data, axis=0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).float().to(device)
            outputs, _, _, _ = model(images)
            probs = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()

            # 裁掉padding区域
            probs = probs[dy:dy + nh, dx:dx + nw]

            # resize回原图大小
            probs_img = Image.fromarray((probs * 255).astype(np.uint8))
            probs_img = probs_img.resize(old_image.size, Image.BICUBIC)
            probs = np.array(probs_img, dtype=np.float32) / 255.0

            pred_mask = (probs > THRESHOLD).astype(np.uint8)

        vote_map += pred_mask.astype(np.uint16)

    # 保存投票热图，便于观察
    vote_vis = (vote_map.astype(np.float32) / max(len(HIGH_PERCENT_LIST), 1) * 255.0).astype(np.uint8)
    Image.fromarray(vote_vis).save(os.path.join(SAVE_DIR, "vote_map.png"))

    # 根据投票图生成 final_mask
    if USE_VOTE_FUSION:
        final_mask = (vote_map >= VOTE_THRESHOLD).astype(np.uint8)
    else:
        final_mask = (vote_map >= 1).astype(np.uint8)

    cv2.imwrite(os.path.join(SAVE_DIR, "step0_vote_fused.png"), final_mask * 255)

    # 5. 圆形度过滤
    final_mask = filter_by_circularity(final_mask, min_circularity=MIN_CIRCULARITY)
    cv2.imwrite(os.path.join(SAVE_DIR, "step1_circularity.png"), final_mask * 255)

    # 6. 屏蔽已知的固定盲元
    bad_spots_raw = [
        (6, 111), (58, 512), (75, 138), (257, 352), (268, 342),
        (441, 281), (520, 348), (567, 95), (574, 23), (637, 474)
    ]
    bad_spots = [(x - 1, y - 1) for x, y in bad_spots_raw]

    radius = 1
    mask_to_clear = np.zeros_like(final_mask, dtype=bool)
    for x, y in bad_spots:
        x_min = max(0, x - radius)
        x_max = min(final_mask.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(final_mask.shape[0], y + radius + 1)
        mask_to_clear[y_min:y_max, x_min:x_max] = True

    final_mask[mask_to_clear] = 0
    cv2.imwrite(os.path.join(SAVE_DIR, "step2_blind_spot_clean.png"), final_mask * 255)

    # 7. 提取候选点并做穷举剔除
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        final_mask.astype(np.uint8), connectivity=8
    )

    if num_labels > 1:
        points = []
        point_labels = []

        for i in range(1, num_labels):
            cx, cy = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_COMPONENT_AREA:
                continue
            points.append([cx, cy])
            point_labels.append(i)

        points = np.array(points)

        if len(points) >= 4:
            max_remove = 5
            target_num = None
            params = {
                'w_line': 0.35,
                'w_dist': 0.45,
                'w_gap': 0.20,
                'lambda_delete': 0.08
            }

            print(f"开始穷举剔除，候选点数: {len(points)}")
            try:
                P_clean, noise_idx, best_info = remove_outliers_by_exhaustive(
                    points, max_remove, target_num, params
                )
                print(f"实际删除点数: {len(noise_idx)}")

                new_mask = np.zeros_like(final_mask, dtype=np.uint8)

                # 修复这里：point_labels 是 list，不能高级索引
                keep_labels = [point_labels[i] for i in range(len(point_labels)) if i not in noise_idx]

                for lbl in keep_labels:
                    new_mask[labels == lbl] = 1

                final_mask = new_mask
                cv2.imwrite(os.path.join(SAVE_DIR, "step3_outlier_removed.png"), final_mask * 255)

            except Exception as e:
                print(f"剔除过程出错: {e}")
        else:
            print(f"点数不足4，跳过剔除。当前点数: {len(points)}")
    else:
        print("没有连通域，跳过剔除。")

    # 8. 可选：形态学腐蚀
   # if EROSION_RADIUS > 0:
   #     final_mask = binary_erosion(final_mask, disk(EROSION_RADIUS)).astype(np.uint8)

    #cv2.imwrite(os.path.join(SAVE_DIR, "step4_eroded.png"), final_mask * 255)

    # 9. 评估（如果提供GT）
    if GT_MASK_PATH is not None and os.path.exists(GT_MASK_PATH):
        gt_mask = load_binary_mask(GT_MASK_PATH, target_size=old_image.size)
        metrics = compute_binary_seg_metrics(final_mask, gt_mask)

        print("\n========== 分割评估结果 ==========")
        print(f"TP: {metrics['TP']}")
        print(f"FP: {metrics['FP']}")
        print(f"FN: {metrics['FN']}")
        print(f"TN: {metrics['TN']}")
        print(f"Precision: {metrics['Precision']:.6f}")
        print(f"Recall:    {metrics['Recall']:.6f}")
        print(f"IoU:       {metrics['IoU']:.6f}")
        print(f"Dice:      {metrics['Dice']:.6f}")
        print(f"Accuracy:  {metrics['Accuracy']:.6f}")
        print("Confusion Matrix:")
        print(np.array(metrics["ConfusionMatrix"]))

        save_metrics(metrics, os.path.join(SAVE_DIR, "metrics.txt"))

        # GT 叠加图（绿色）
        gt_overlay = old_rgb.copy()
        gt_overlay[gt_mask == 1] = [0, 255, 0]
        Image.fromarray(gt_overlay).save(os.path.join(SAVE_DIR, "overlay_gt.png"))

        # =========================
        # 光斑中心级评估
        # =========================
        center_metrics, matches, gt_centers, pred_centers = evaluate_spot_centers(
            pred_mask=final_mask,
            gt_mask=gt_mask,
            min_area=MIN_COMPONENT_AREA,
            max_match_distance=MAX_MATCH_DISTANCE
        )

        print("\n========== 光斑中心评估结果 ==========")
        print(f"GT_Count:           {center_metrics['GT_Count']}")
        print(f"Pred_Count:         {center_metrics['Pred_Count']}")
        print(f"Matched_Count:      {center_metrics['Matched_Count']}")
        print(f"FN_Count:           {center_metrics['FN_Count']}")
        print(f"FP_Count:           {center_metrics['FP_Count']}")
        print(f"Instance_Precision: {center_metrics['Instance_Precision']:.6f}")
        print(f"Instance_Recall:    {center_metrics['Instance_Recall']:.6f}")

        if center_metrics["Mean_Center_Error"] is not None:
            print(f"Mean_Center_Error:  {center_metrics['Mean_Center_Error']:.4f} px")
            print(f"Median_Center_Error:{center_metrics['Median_Center_Error']:.4f} px")
            print(f"Max_Center_Error:   {center_metrics['Max_Center_Error']:.4f} px")
        else:
            print("Mean_Center_Error:  None")
            print("Median_Center_Error:None")
            print("Max_Center_Error:   None")

        # 保存到文本
        with open(os.path.join(SAVE_DIR, "center_metrics.txt"), "w", encoding="utf-8") as f:
            f.write("========== 光斑中心评估结果 ==========\n")
            f.write(f"GT_Count:           {center_metrics['GT_Count']}\n")
            f.write(f"Pred_Count:         {center_metrics['Pred_Count']}\n")
            f.write(f"Matched_Count:      {center_metrics['Matched_Count']}\n")
            f.write(f"FN_Count:           {center_metrics['FN_Count']}\n")
            f.write(f"FP_Count:           {center_metrics['FP_Count']}\n")
            f.write(f"Instance_Precision: {center_metrics['Instance_Precision']:.6f}\n")
            f.write(f"Instance_Recall:    {center_metrics['Instance_Recall']:.6f}\n")
            f.write(f"Mean_Center_Error:  {center_metrics['Mean_Center_Error']}\n")
            f.write(f"Median_Center_Error:{center_metrics['Median_Center_Error']}\n")
            f.write(f"Max_Center_Error:   {center_metrics['Max_Center_Error']}\n")

        save_center_match_visualization(
            image_gray=old_disp,
            gt_centers=gt_centers,
            pred_centers=pred_centers,
            matches=matches,
            save_path=os.path.join(SAVE_DIR, "center_match_vis.png")
        )
        # 误差图
        save_error_map(final_mask, gt_mask, os.path.join(SAVE_DIR, "error_map.png"))
    else:
        print("\n未找到 GT_MASK_PATH，跳过评估。")

    # 10. 保存最终结果
    mask_save = final_mask * 255
    Image.fromarray(mask_save).save(os.path.join(SAVE_DIR, "final_mask.png"))
    Image.fromarray(old_disp).save(os.path.join(SAVE_DIR, "original.png"))

    overlay = old_rgb.copy()
    overlay[final_mask == 1] = [255, 0, 0]
    Image.fromarray(overlay).save(os.path.join(SAVE_DIR, "overlay_final.png"))

    print("\n预测完成，结果保存在：", SAVE_DIR)


if __name__ == "__main__":
    main()