import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
import torch
import cv2
from skimage.morphology import binary_erosion, disk
from nets.unet import Unet, DualBranchUnet
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
import itertools
from copy import deepcopy

# =========================
# 配置区
# =========================
MODEL_PATH = "logs/best_epoch_weights.pth"
IMAGE_PATH = "F:/西物所博士/FPGA信号阈值设置/right/论文/光斑原图/image_51.png"
GT_MASK_PATH = "E:/unet-pytorch-main/Medical_Datasets/Labels/image_51.png"

#IMAGE_PATH = "E:/Infrared-Small-Target-Detection-master/dataset/NUDT-SIRST/images/000968.png"
#GT_MASK_PATH = "E:/Infrared-Small-Target-Detection-master/dataset/NUDT-SIRST/masks/000968.png"
#INPUT_SHAPE = [256, 256]

SAVE_DIR = "pred_results"
BACKBONE = "myunet"
NUM_CLASSES = 2
USE_DUAL_BRANCH = True
# 使用正方形输入，减少边缘漏检
INPUT_SHAPE = [720, 720]  # [height, width]
CUDA = True

# 单次预测阈值
THRESHOLD = 0.8
DET_THRESHOLD = 0.7
NOISE_THRESHOLD = 0.5

# 圆形度阈值
MIN_CIRCULARITY = 0.6

# 连通域最小面积
MIN_COMPONENT_AREA = 1

# 腐蚀半径，0 表示不腐蚀
EROSION_RADIUS = 1

# 多百分位列表
HIGH_PERCENT_LIST = [95.0, 97.0, 99.0, 99.5]
# 投票阈值
VOTE_THRESHOLD = 2

# 是否使用投票融合
USE_VOTE_FUSION = True

# 添加中心匹配的最大距离阈值（像素）
MAX_MATCH_DISTANCE = 8.0


# =========================
# 图像预处理函数
# =========================
def normalize_16bit_image(img, low_percent=1, high_percent=99.5):
    """百分位归一化"""
    img = img.astype(np.float32)
    low = np.percentile(img, low_percent)
    high = np.percentile(img, high_percent)
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)
    return img


def normalize_16bit_image_global(img):
    """全局线性归一化，保留所有深度信息（备用）"""
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return np.zeros_like(img)
    normalized = (img - img_min) / (img_max - img_min + 1e-8)
    return normalized


def resize_image_keep_ratio(image, size):
    """
    保持宽高比的resize，并记录变换参数
    返回: (resized_image, nw, nh, dx, dy, scale)
    """
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
    return new_image, nw, nh, dx, dy, scale


def map_coords_to_original(coords, scale, dx, dy, original_size):
    """
    将resize+padding后的坐标映射回原图坐标
    coords: (N, 2) 在resize后图像上的坐标 [x, y]
    scale: 缩放因子
    dx, dy: padding偏移
    original_size: (width, height)
    返回: (N, 2) 原图坐标
    """
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    mapped = []
    for x, y in coords:
        # 移除padding
        x_no_pad = x - dx
        y_no_pad = y - dy

        # 缩放到原图
        orig_x = x_no_pad / scale
        orig_y = y_no_pad / scale

        # 边界裁剪
        orig_x = np.clip(orig_x, 0, original_size[0] - 1)
        orig_y = np.clip(orig_y, 0, original_size[1] - 1)

        mapped.append([orig_x, orig_y])

    return np.array(mapped, dtype=np.float32)


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
    读取GT mask并转成二值图
    重要：确保坐标系统一致
    """
    if not os.path.exists(mask_path):
        print(f"警告: GT文件不存在: {mask_path}")
        return None

    gt = Image.open(mask_path)
    print(f"原始GT信息:")
    print(f"  尺寸: {gt.size}")
    print(f"  模式: {gt.mode}")

    # 检查GT中前景像素的数量
    gt_np = np.array(gt)
    print(f"  GT像素范围: {gt_np.min()} - {gt_np.max()}")
    print(f"  唯一值: {np.unique(gt_np)}")

    # 转换为灰度图
    if gt.mode != "L":
        gt = gt.convert("L")

    # 缩放到目标尺寸
    if target_size is not None:
        print(f"将GT从{gt.size}缩放到: {target_size}")
        gt = gt.resize(target_size, Image.NEAREST)

    gt = np.array(gt, dtype=np.uint8)
    # 二值化：注意GT中前景像素的值可能不是255
    gt_original = gt.copy()
    gt = (gt > 0).astype(np.uint8)

    print(f"GT二值图信息:")
    print(f"  尺寸: {gt.shape}")
    print(f"  前景像素数: {gt.sum()}")
    print(f"  前景像素占比: {gt.sum() / gt.size * 100:.2f}%")

    # 统计连通域数量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt, connectivity=8)
    print(f"  连通域数量: {num_labels - 1}")

    # 打印每个连通域的中心
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        print(f"    GT连通域{i}: 中心=({cx:.1f}, {cy:.1f}), 面积={area}")

    return gt


# 在主函数中添加坐标验证
def verify_coordinate_systems(image, gt_mask, pred_mask, scale, dx, dy):
    """
    验证三个坐标系统是否一致
    """
    print("\n" + "=" * 60)
    print("坐标系统验证")
    print("=" * 60)

    # 1. 原图尺寸
    print(f"1. 原图尺寸: {image.size}")

    # 2. GT中心（已经是原图坐标）
    gt_centers, _ = extract_spot_centers(gt_mask)
    print(f"\n2. GT中心 (原图坐标):")
    for i, (x, y) in enumerate(gt_centers):
        print(f"   {i}: ({x:.1f}, {y:.1f})")

    # 3. 预测mask的中心（未映射，在pred_mask上的坐标）
    pred_centers_resized, _ = extract_spot_centers(pred_mask)
    print(f"\n3. Pred中心 (在pred_mask上的坐标, 尺寸{pred_mask.shape[1]}x{pred_mask.shape[0]}):")
    for i, (x, y) in enumerate(pred_centers_resized):
        print(f"   {i}: ({x:.1f}, {y:.1f})")

    # 4. 映射后的Pred中心
    if len(pred_centers_resized) > 0:
        mapped_centers = map_coords_to_original(
            pred_centers_resized, scale, dx, dy, image.size
        )
        print(f"\n4. Pred中心 (映射到原图后):")
        for i, (x, y) in enumerate(mapped_centers):
            print(f"   {i}: ({x:.1f}, {y:.1f})")

        # 5. 计算最小距离
        if len(gt_centers) > 0:
            from scipy.spatial.distance import cdist
            distances = cdist(gt_centers, mapped_centers)
            min_distances = distances.min(axis=1)
            print(f"\n5. GT到最近Pred的最小距离:")
            for i, d in enumerate(min_distances):
                print(f"   GT {i}: {d:.2f} px")
            print(f"   平均最小距离: {min_distances.mean():.2f} px")
            print(f"   建议匹配阈值: {min_distances.mean() * 1.5:.2f} px")

    print("=" * 60)

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
    误差图: TP=白色, FP=红色, FN=蓝色
    """
    err_vis = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

    tp_mask = (pred_mask == 1) & (gt_mask == 1)
    fp_mask = (pred_mask == 1) & (gt_mask == 0)
    fn_mask = (pred_mask == 0) & (gt_mask == 1)

    err_vis[tp_mask] = [255, 255, 255]
    err_vis[fp_mask] = [255, 0, 0]
    err_vis[fn_mask] = [0, 0, 255]

    Image.fromarray(err_vis).save(save_path)


def extract_spot_centers(mask, min_area=1):
    """
    从二值mask中提取每个连通域的中心点
    注意：OpenCV的centroids返回的是(x, y)，其中x是列坐标，y是行坐标
    """
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    centers = []
    areas = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[i]  # OpenCV返回(x, y)
        # 确保坐标是整数或浮点数
        centers.append([float(cx), float(cy)])
        areas.append(area)
        print(f"  连通域{i}: 中心=({cx:.1f}, {cy:.1f}), 面积={area}")  # 调试输出

    if len(centers) == 0:
        centers = np.zeros((0, 2), dtype=np.float32)
        areas = np.zeros((0,), dtype=np.int32)
    else:
        centers = np.array(centers, dtype=np.float32)
        areas = np.array(areas, dtype=np.int32)

    return centers, areas


def evaluate_spot_centers(pred_mask, gt_mask, min_area=1, max_match_distance=8.0,
                          scale=1.0, dx=0, dy=0, original_size=None):
    """
    基于光斑中心的一对一匹配评估
    注意：pred_mask和gt_mask都应该是原图尺寸

    参数:
        pred_mask: 预测的二值mask (H, W)，已经是原图尺寸
        gt_mask: GT二值mask (H, W)
        min_area: 最小连通域面积
        max_match_distance: 最大匹配距离（像素）
        scale, dx, dy, original_size: 保留参数为了兼容性，但不再使用

    返回:
        metrics: dict 评估指标
        matches: list of (gt_idx, pred_idx, dist)
        gt_centers: (Ng, 2) GT中心坐标
        pred_centers: (Np, 2) 预测中心坐标
    """
    # 1. 提取GT中心（已经是原图坐标）
    gt_centers, gt_areas = extract_spot_centers(gt_mask, min_area=min_area)

    # 2. 提取预测中心（已经是原图坐标，因为pred_mask已经是原图尺寸）
    pred_centers, pred_areas = extract_spot_centers(pred_mask, min_area=min_area)

    num_gt = len(gt_centers)
    num_pred = len(pred_centers)

    # 调试输出
    print(f"\n提取中心信息:")
    print(f"  GT中心数: {num_gt}")
    if num_gt > 0:
        print(f"  GT中心示例: {gt_centers[:3].tolist()}")
    print(f"  Pred中心数: {num_pred}")
    if num_pred > 0:
        print(f"  Pred中心示例: {pred_centers[:3].tolist()}")

    # 边界情况处理
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
            "Max_Center_Error": 0.0,
            "All_Matches": []
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
            "Max_Center_Error": None,
            "All_Matches": []
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
            "Max_Center_Error": None,
            "All_Matches": []
        }
        return metrics, [], gt_centers, pred_centers

    # 3. 构造距离矩阵
    diff = gt_centers[:, None, :] - pred_centers[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # [num_gt, num_pred]

    print(f"\n距离矩阵统计:")
    print(f"  最小距离: {dist_matrix.min():.2f}")
    print(f"  最大距离: {dist_matrix.max():.2f}")
    print(f"  平均距离: {dist_matrix.mean():.2f}")

    # 4. 使用匈牙利算法进行最优匹配
    gt_idx, pred_idx = linear_sum_assignment(dist_matrix)

    # 5. 根据距离阈值筛选有效匹配
    matches = []
    matched_gt = set()
    matched_pred = set()

    for g, p in zip(gt_idx, pred_idx):
        d = dist_matrix[g, p]
        if d <= max_match_distance:
            matches.append((int(g), int(p), float(d)))
            matched_gt.add(int(g))
            matched_pred.add(int(p))
            print(f"  匹配成功: GT[{g}]={gt_centers[g]} <-> Pred[{p}]={pred_centers[p]}, 距离={d:.2f}")
        else:
            print(
                f"  匹配失败(距离过大): GT[{g}]={gt_centers[g]} <-> Pred[{p}]={pred_centers[p]}, 距离={d:.2f} > {max_match_distance}")

    # 6. 计算额外的不匹配点
    unmatched_gt = set(range(num_gt)) - matched_gt
    unmatched_pred = set(range(num_pred)) - matched_pred

    for g in unmatched_gt:
        # 找最近的pred距离（即使没匹配上）
        min_dist = dist_matrix[g].min()
        print(f"  未匹配GT[{g}]={gt_centers[g]}, 最近Pred距离={min_dist:.2f}")

    for p in unmatched_pred:
        # 找最近的gt距离（即使没匹配上）
        min_dist = dist_matrix[:, p].min()
        print(f"  未匹配Pred[{p}]={pred_centers[p]}, 最近GT距离={min_dist:.2f}")

    matched_count = len(matches)
    fn_count = num_gt - matched_count
    fp_count = num_pred - matched_count

    # 7. 计算实例级别的精确率和召回率
    inst_precision = matched_count / (num_pred + 1e-8)
    inst_recall = matched_count / (num_gt + 1e-8)

    # 8. 计算中心误差统计
    if matched_count > 0:
        dists = np.array([m[2] for m in matches], dtype=np.float32)
        mean_err = float(np.mean(dists))
        median_err = float(np.median(dists))
        max_err = float(np.max(dists))
        min_err = float(np.min(dists))
        std_err = float(np.std(dists))

        print(f"\n匹配误差统计:")
        print(f"  平均误差: {mean_err:.2f} px")
        print(f"  中位数误差: {median_err:.2f} px")
        print(f"  最小误差: {min_err:.2f} px")
        print(f"  最大误差: {max_err:.2f} px")
        print(f"  标准差: {std_err:.2f} px")
    else:
        mean_err = None
        median_err = None
        max_err = None
        min_err = None
        std_err = None
        print(f"\n没有成功匹配的点对!")

    # 9. 构建评估指标字典
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
        "Max_Center_Error": max_err,
        "Min_Center_Error": min_err,
        "Std_Center_Error": std_err,
        "All_Matches": matches,
        "Unmatched_GT": list(unmatched_gt),
        "Unmatched_Pred": list(unmatched_pred),
        "Distance_Matrix": dist_matrix.tolist()  # 用于调试
    }

    return metrics, matches, gt_centers, pred_centers


def extract_spot_centers(mask, min_area=1):
    """
    从二值mask中提取每个连通域的中心点
    返回: centers (N,2) [x, y], areas (N,)
    """
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    centers = []
    areas = []

    print(f"  连通域检测: 共{num_labels - 1}个连通域")

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            print(f"    跳过连通域{i}: 面积={area} < {min_area}")
            continue
        cx, cy = centroids[i]  # OpenCV返回(x, y)
        centers.append([float(cx), float(cy)])
        areas.append(area)
        print(f"    连通域{i}: 中心=({cx:.1f}, {cy:.1f}), 面积={area}")

    if len(centers) == 0:
        centers = np.zeros((0, 2), dtype=np.float32)
        areas = np.zeros((0,), dtype=np.int32)
    else:
        centers = np.array(centers, dtype=np.float32)
        areas = np.array(areas, dtype=np.int32)

    return centers, areas


def save_center_match_visualization(image_gray, gt_centers, pred_centers, matches, save_path):
    """
    可视化: GT中心绿色, Pred中心红色, 匹配线黄色
    """
    if image_gray.ndim == 2:
        vis = np.stack([image_gray, image_gray, image_gray], axis=-1)
    else:
        vis = image_gray.copy()

    vis = vis.astype(np.uint8).copy()

    # 画GT中心
    for x, y in gt_centers:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 255, 0), 2)

    # 画Pred中心
    for x, y in pred_centers:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, (255, 0, 0), 2)

    # 画匹配线
    for g, p, d in matches:
        gx, gy = gt_centers[g]
        px, py = pred_centers[p]
        cv2.line(
            vis,
            (int(round(gx)), int(round(gy))),
            (int(round(px)), int(round(py))),
            (255, 255, 0),
            2
        )

    Image.fromarray(vis).save(save_path)


# =========================
# 共线性+等间距评分函数（保留原功能）
# =========================
def eval_subset_by_line_and_spacing(Q, params):
    """对保留点集 Q 进行评分"""
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


def remove_outliers_by_exhaustive(P, max_remove=3, target_num=None, params=None):
    """穷举删除点，返回最优保留点集"""
    if params is None or not isinstance(params, dict):
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
    print(f"INPUT_SHAPE: {INPUT_SHAPE}")
    print(f"HIGH_PERCENT_LIST: {HIGH_PERCENT_LIST}")
    print(f"THRESHOLD: {THRESHOLD}")
    print(f"USE_VOTE_FUSION: {USE_VOTE_FUSION}")
    print(f"VOTE_THRESHOLD: {VOTE_THRESHOLD}")
    print("=================\n")

    # 1. 加载模型
    if USE_DUAL_BRANCH:
        model = DualBranchUnet(
            num_classes=NUM_CLASSES,
            det_backbone=BACKBONE,
            det_pretrained=False,
            use_unet_dna=True,
            fusion_alpha_init=1.0
        )
    else:
        model = Unet(num_classes=NUM_CLASSES, pretrained=False, backbone=BACKBONE)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 2. 读取原始图像
    image = Image.open(IMAGE_PATH)
    old_image = image.copy()
    original_size = old_image.size  # (width, height)

    print(f"\n原图信息:")
    print(f"  尺寸: {original_size}")
    print(f"  模式: {image.mode}")

    # 用于显示的 8-bit 原图
    old_np = np.array(old_image).astype(np.float32)
    if old_np.ndim == 3:
        old_np = old_np[..., 0]
    old_disp = old_np.copy()
    old_disp = old_disp - old_disp.min()
    old_disp = old_disp / (old_disp.max() + 1e-8) * 255.0
    old_disp = old_disp.astype(np.uint8)

    old_rgb = np.stack([old_disp, old_disp, old_disp], axis=-1)
    print(f"深度范围: Min={old_np.min()}, Max={old_np.max()}")
    print(f"有效值比例: {(old_np > 0).mean():.2%}")

    # 3. resize图像，获取变换参数
    resized_image, nw, nh, dx, dy, scale = resize_image_keep_ratio(image, INPUT_SHAPE)
    print(f"\n图像变换参数:")
    print(f"  Resize后尺寸: {resized_image.size}")
    print(f"  缩放因子: {scale:.4f}")
    print(f"  Padding偏移: dx={dx}, dy={dy}")
    print(f"  有效区域尺寸: {nw}x{nh}")

    resized_np_base = np.array(resized_image, dtype=np.uint16)
    if resized_np_base.ndim == 3:
        resized_np_base = resized_np_base[..., 0]

    # 4. 多百分位预测融合
    vote_map = np.zeros(old_disp.shape, dtype=np.uint16)          # final
    vote_map_det = np.zeros(old_disp.shape, dtype=np.uint16)      # detector raw
    noise_prob_sum = np.zeros(old_disp.shape, dtype=np.float32)   # noise prob

    for hp in HIGH_PERCENT_LIST:
        print(f"\nProcessing with high_percent = {hp} ...")

        image_data = normalize_16bit_image(resized_np_base, low_percent=1, high_percent=hp)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = np.expand_dims(image_data, axis=0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).float().to(device)
            if USE_DUAL_BRANCH:
                det_out = model.det_branch(images)
                det_logit = det_out[0] if isinstance(det_out, (list, tuple)) else det_out
                noise_logit = model.noise_branch(images)
                final_logit = model._fuse(det_logit, noise_logit)
            else:
                model_out = model(images)
                final_logit = model_out[0] if isinstance(model_out, (list, tuple)) else model_out
                det_logit = final_logit
                noise_logit = torch.zeros_like(final_logit[:, :1, :, :])

            det_prob = torch.softmax(det_logit, dim=1)[0, 1].cpu().numpy()
            final_prob = torch.softmax(final_logit, dim=1)[0, 1].cpu().numpy()
            noise_prob = torch.sigmoid(noise_logit)[0, 0].cpu().numpy()

            # 裁掉padding区域
            det_prob = det_prob[dy:dy + nh, dx:dx + nw]
            final_prob = final_prob[dy:dy + nh, dx:dx + nw]
            noise_prob = noise_prob[dy:dy + nh, dx:dx + nw]

            # resize回原图大小
            def _resize_to_orig(prob_map):
                prob_img = Image.fromarray((prob_map * 255).astype(np.uint8))
                prob_img = prob_img.resize(old_image.size, Image.BICUBIC)
                return np.array(prob_img, dtype=np.float32) / 255.0

            det_prob = _resize_to_orig(det_prob)
            final_prob = _resize_to_orig(final_prob)
            noise_prob = _resize_to_orig(noise_prob)

            pred_mask_det = (det_prob > DET_THRESHOLD).astype(np.uint8)
            pred_mask_final = (final_prob > THRESHOLD).astype(np.uint8)

        vote_map_det += pred_mask_det.astype(np.uint16)
        vote_map += pred_mask_final.astype(np.uint16)
        noise_prob_sum += noise_prob

    # 保存投票热图
    vote_vis = (vote_map.astype(np.float32) / max(len(HIGH_PERCENT_LIST), 1) * 255.0).astype(np.uint8)
    Image.fromarray(vote_vis).save(os.path.join(SAVE_DIR, "vote_map.png"))
    vote_vis_det = (vote_map_det.astype(np.float32) / max(len(HIGH_PERCENT_LIST), 1) * 255.0).astype(np.uint8)
    Image.fromarray(vote_vis_det).save(os.path.join(SAVE_DIR, "vote_map_det.png"))

    # 根据投票图生成 final_mask
    if USE_VOTE_FUSION:
        final_mask = (vote_map >= VOTE_THRESHOLD).astype(np.uint8)
        det_mask = (vote_map_det >= VOTE_THRESHOLD).astype(np.uint8)
    else:
        final_mask = (vote_map >= 1).astype(np.uint8)
        det_mask = (vote_map_det >= 1).astype(np.uint8)

    noise_prob_avg = noise_prob_sum / max(len(HIGH_PERCENT_LIST), 1)
    noise_mask = (noise_prob_avg >= NOISE_THRESHOLD).astype(np.uint8)

    cv2.imwrite(os.path.join(SAVE_DIR, "step0_vote_fused.png"), final_mask * 255)
    cv2.imwrite(os.path.join(SAVE_DIR, "det_raw_mask.png"), det_mask * 255)
    cv2.imwrite(os.path.join(SAVE_DIR, "noise_mask.png"), noise_mask * 255)
    cv2.imwrite(os.path.join(SAVE_DIR, "final_minus_noise_mask.png"), final_mask * 255)

    # 5. 圆形度过滤（可选）
    # final_mask = filter_by_circularity(final_mask, min_circularity=MIN_CIRCULARITY)
    # cv2.imwrite(os.path.join(SAVE_DIR, "step1_circularity.png"), final_mask * 255)

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
    print(f"\n最终预测结果:")
    print(f"  前景像素数: {final_mask.sum()}")
    print(f"  连通域数量: {cv2.connectedComponents(final_mask.astype(np.uint8))[0] - 1}")

    # 在评估之前添加验证
    if GT_MASK_PATH is not None and os.path.exists(GT_MASK_PATH):
        gt_mask = load_binary_mask(GT_MASK_PATH, target_size=old_image.size)

        if gt_mask is not None:
            # 验证坐标系统
            verify_coordinate_systems(old_image, gt_mask, final_mask, scale, dx, dy)

            # 确保尺寸一致
            if gt_mask.shape != final_mask.shape:
                print(f"\n警告: 尺寸不一致，调整GT尺寸")
                gt_mask = cv2.resize(gt_mask, (final_mask.shape[1], final_mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

            # 使用更大的匹配阈值进行测试
            test_thresholds = [20, 30, 50, 100]
            for test_th in test_thresholds:
                print(f"\n测试匹配阈值: {test_th}")
                center_metrics, matches, gt_centers, pred_centers = evaluate_spot_centers(
                    pred_mask=final_mask,
                    gt_mask=gt_mask,
                    min_area=1,  # 使用1而不是MIN_COMPONENT_AREA
                    max_match_distance=MAX_MATCH_DISTANCE,
                    scale=1.0,  # 不需要映射
                    dx=0,
                    dy=0,
                    original_size=original_size
                )
                print(f"  匹配数: {center_metrics['Matched_Count']}")
                if center_metrics['Matched_Count'] > 0:
                    print(f"  找到了匹配! 使用阈值 {test_th}")
                    # 使用这个阈值继续
                    break
    # 在评估前添加
    print("\n===== 详细调试信息 =====")
    print(f"final_mask shape: {final_mask.shape}")
    print(f"final_mask 前景像素位置:")
    ys, xs = np.where(final_mask > 0)
    if len(ys) > 0:
        print(f"  X范围: {xs.min()} - {xs.max()}")
        print(f"  Y范围: {ys.min()} - {ys.max()}")
        print(f"  中心点: ({xs.mean():.1f}, {ys.mean():.1f})")

    print(f"\nGT mask shape: {gt_mask.shape}")
    ys_gt, xs_gt = np.where(gt_mask > 0)
    if len(ys_gt) > 0:
        print(f"  X范围: {xs_gt.min()} - {xs_gt.max()}")
        print(f"  Y范围: {ys_gt.min()} - {ys_gt.max()}")
        print(f"  中心点: ({xs_gt.mean():.1f}, {ys_gt.mean():.1f})")

    # 保存可视化对比
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(final_mask, cmap='gray')
    axes[0].set_title(f'Pred Mask (中心: {xs.mean():.1f}, {ys.mean():.1f})')
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title(f'GT Mask (中心: {xs_gt.mean():.1f}, {ys_gt.mean():.1f})')
    plt.savefig(os.path.join(SAVE_DIR, "mask_comparison.png"))
    print(f"\n已保存mask对比图: mask_comparison.png")
    # 7. 评估（如果提供GT）
    if GT_MASK_PATH is not None and os.path.exists(GT_MASK_PATH):
        print(f"\n加载GT: {GT_MASK_PATH}")
        gt_mask = load_binary_mask(GT_MASK_PATH, target_size=old_image.size)

        if gt_mask is not None:
            # 确保GT和预测尺寸一致
            if gt_mask.shape != final_mask.shape:
                print(f"警告: GT尺寸{gt_mask.shape}与预测尺寸{final_mask.shape}不一致!")
                gt_mask = cv2.resize(gt_mask, (final_mask.shape[1], final_mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

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

            save_metrics(metrics, os.path.join(SAVE_DIR, "metrics.txt"))

            # GT 叠加图
            gt_overlay = old_rgb.copy()
            gt_overlay[gt_mask == 1] = [0, 255, 0]
            Image.fromarray(gt_overlay).save(os.path.join(SAVE_DIR, "overlay_gt.png"))

            # =========================
            # 光斑中心级评估（带坐标映射）
            # =========================
            # 动态调整匹配距离：原图上的距离阈值需要除以缩放因子
            adjusted_match_distance = MAX_MATCH_DISTANCE / scale
            print(f"\n匹配距离阈值: 原图={MAX_MATCH_DISTANCE}px, 调整后={adjusted_match_distance:.2f}px")

            center_metrics, matches, gt_centers, pred_centers = evaluate_spot_centers(
                pred_mask=final_mask,
                gt_mask=gt_mask,
                min_area=MIN_COMPONENT_AREA,
                max_match_distance=adjusted_match_distance,
                scale=scale,
                dx=dx,
                dy=dy,
                original_size=original_size
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

            # 保存中心评估结果
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
                f.write(f"\n变换参数:\n")
                f.write(f"  scale: {scale:.6f}\n")
                f.write(f"  dx: {dx}\n")
                f.write(f"  dy: {dy}\n")
                f.write(f"  original_size: {original_size}\n")

            # 可视化中心匹配
            save_center_match_visualization(
                image_gray=old_disp,
                gt_centers=gt_centers,
                pred_centers=pred_centers,
                matches=matches,
                save_path=os.path.join(SAVE_DIR, "center_match_vis.png")
            )

            # 误差图
            save_error_map(final_mask, gt_mask, os.path.join(SAVE_DIR, "error_map.png"))

            # 打印坐标信息用于调试
            print("\n===== 坐标调试信息 =====")
            if len(gt_centers) > 0:
                print(f"GT中心示例 (前3个):")
                for i, center in enumerate(gt_centers[:3]):
                    print(f"  {i}: ({center[0]:.1f}, {center[1]:.1f})")
            if len(pred_centers) > 0:
                print(f"Pred中心示例 (前3个):")
                for i, center in enumerate(pred_centers[:3]):
                    print(f"  {i}: ({center[0]:.1f}, {center[1]:.1f})")
            if len(matches) > 0:
                print(f"匹配对示例 (前3个):")
                for i, (g, p, d) in enumerate(matches[:3]):
                    print(f"  {i}: GT[{g}]={gt_centers[g]} <-> Pred[{p}]={pred_centers[p]}, 距离={d:.2f}")
        else:
            print("GT加载失败，跳过评估")
    else:
        print(f"\n未找到GT文件: {GT_MASK_PATH}，跳过评估。")

    # 8. 保存最终结果
    mask_save = final_mask * 255
    Image.fromarray(mask_save).save(os.path.join(SAVE_DIR, "final_mask.png"))
    Image.fromarray(old_disp).save(os.path.join(SAVE_DIR, "original.png"))

    overlay_det = old_rgb.copy()
    overlay_det[det_mask == 1] = [255, 255, 0]
    Image.fromarray(overlay_det).save(os.path.join(SAVE_DIR, "overlay_det_raw.png"))

    overlay = old_rgb.copy()
    overlay[final_mask == 1] = [255, 0, 0]
    Image.fromarray(overlay).save(os.path.join(SAVE_DIR, "overlay_final.png"))

    if GT_MASK_PATH is not None and os.path.exists(GT_MASK_PATH):
        gt_mask_for_vis = load_binary_mask(GT_MASK_PATH, target_size=old_image.size)
        if gt_mask_for_vis is not None:
            gt_mask_for_vis = (gt_mask_for_vis > 0).astype(np.uint8)
            compare_rgb = old_rgb.copy()
            compare_rgb[(det_mask == 1) & (gt_mask_for_vis == 1)] = [0, 255, 0]      # TP
            compare_rgb[(det_mask == 1) & (gt_mask_for_vis == 0)] = [255, 0, 0]      # FP
            compare_rgb[(det_mask == 0) & (gt_mask_for_vis == 1)] = [0, 0, 255]      # FN
            Image.fromarray(compare_rgb).save(os.path.join(SAVE_DIR, "compare_det_gt.png"))

    print(f"\n预测完成，结果保存在：{SAVE_DIR}")
    print("生成的文件:")
    for f in os.listdir(SAVE_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
