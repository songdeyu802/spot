import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
import torch
import cv2
from skimage.morphology import binary_erosion, disk
from nets.unet import Unet
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
import itertools
from copy import deepcopy
import glob
from tqdm import tqdm

# =========================
# 配置区
# =========================
MODEL_PATH = "logs/best_epoch_weights.pth"

# 数据集路径配置
DATASET_ROOT = "E:/unet-pytorch-main/Medical_Datasets/"  # 数据集根目录
IMAGES_DIR = os.path.join(DATASET_ROOT, "Images")  # 原图文件夹
LABELS_DIR = os.path.join(DATASET_ROOT, "Labels")  # 标签文件夹
SPLIT_TXT_PATH = os.path.join(DATASET_ROOT, "ImageSets/Segmentation/test.txt")  # test.txt路径

# 输出目录
OUTPUT_DIR = "test_results"

BACKBONE = "myunet"
NUM_CLASSES = 2
INPUT_SHAPE = [720, 720]
CUDA = True

# 单次预测阈值
THRESHOLD = 0.8

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

# 中心匹配的最大距离阈值（像素）
MAX_MATCH_DISTANCE = 8.0

# 支持的图像格式
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']


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
    """将resize+padding后的坐标映射回原图坐标"""
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    mapped = []
    for x, y in coords:
        x_no_pad = x - dx
        y_no_pad = y - dy
        orig_x = x_no_pad / scale
        orig_y = y_no_pad / scale
        orig_x = np.clip(orig_x, 0, original_size[0] - 1)
        orig_y = np.clip(orig_y, 0, original_size[1] - 1)
        mapped.append([orig_x, orig_y])

    return np.array(mapped, dtype=np.float32)


def filter_by_circularity(mask, min_circularity=MIN_CIRCULARITY):
    """圆形度过滤"""
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


def load_binary_mask(mask_path, target_size=None):
    """读取GT mask并转成二值图"""
    if not os.path.exists(mask_path):
        return None

    gt = Image.open(mask_path)
    if gt.mode != "L":
        gt = gt.convert("L")

    if target_size is not None:
        gt = gt.resize(target_size, Image.NEAREST)

    gt = np.array(gt, dtype=np.uint8)
    gt = (gt > 0).astype(np.uint8)

    return gt


def compute_binary_seg_metrics(pred_mask, gt_mask):
    """计算二值分割指标"""
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
    }


def extract_spot_centers(mask, min_area=1):
    """从二值mask中提取每个连通域的中心点"""
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    centers = []
    areas = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        centers.append([float(cx), float(cy)])
        areas.append(area)

    if len(centers) == 0:
        centers = np.zeros((0, 2), dtype=np.float32)
        areas = np.zeros((0,), dtype=np.int32)
    else:
        centers = np.array(centers, dtype=np.float32)
        areas = np.array(areas, dtype=np.int32)

    return centers, areas


def evaluate_spot_centers(pred_mask, gt_mask, min_area=1, max_match_distance=8.0):
    """基于光斑中心的一对一匹配评估"""
    gt_centers, gt_areas = extract_spot_centers(gt_mask, min_area=min_area)
    pred_centers, pred_areas = extract_spot_centers(pred_mask, min_area=min_area)

    num_gt = len(gt_centers)
    num_pred = len(pred_centers)

    if num_gt == 0 and num_pred == 0:
        metrics = {
            "GT_Count": 0, "Pred_Count": 0, "Matched_Count": 0,
            "FN_Count": 0, "FP_Count": 0, "Instance_Precision": 1.0,
            "Instance_Recall": 1.0, "Mean_Center_Error": 0.0,
        }
        return metrics, [], gt_centers, pred_centers

    if num_gt == 0:
        metrics = {
            "GT_Count": 0, "Pred_Count": num_pred, "Matched_Count": 0,
            "FN_Count": 0, "FP_Count": num_pred, "Instance_Precision": 0.0,
            "Instance_Recall": 1.0, "Mean_Center_Error": None,
        }
        return metrics, [], gt_centers, pred_centers

    if num_pred == 0:
        metrics = {
            "GT_Count": num_gt, "Pred_Count": 0, "Matched_Count": 0,
            "FN_Count": num_gt, "FP_Count": 0, "Instance_Precision": 1.0,
            "Instance_Recall": 0.0, "Mean_Center_Error": None,
        }
        return metrics, [], gt_centers, pred_centers

    diff = gt_centers[:, None, :] - pred_centers[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

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
    else:
        mean_err = None

    metrics = {
        "GT_Count": int(num_gt),
        "Pred_Count": int(num_pred),
        "Matched_Count": int(matched_count),
        "FN_Count": int(fn_count),
        "FP_Count": int(fp_count),
        "Instance_Precision": float(inst_precision),
        "Instance_Recall": float(inst_recall),
        "Mean_Center_Error": mean_err,
    }

    return metrics, matches, gt_centers, pred_centers


def remove_outliers_by_exhaustive(P, max_remove=3, target_num=None, params=None):
    """穷举删除点，返回最优保留点集（简化版）"""
    if params is None:
        params = {'w_line': 0.35, 'w_dist': 0.45, 'w_gap': 0.20, 'lambda_delete': 0.08}

    N = P.shape[0]
    if N < 4:
        return P, np.array([]), None

    # 简化：如果点数大于4，返回所有点（实际应用中可保留完整算法）
    return P, np.array([]), None


def get_test_images_from_txt(txt_path, images_dir, labels_dir, image_extensions):
    """
    从test.txt文件中读取图像列表，返回(image_path, label_path)的列表

    test.txt每行格式通常是：图像文件名（不带扩展名）或相对路径
    例如：
    image_001
    image_002
    或
    001
    002
    """
    if not os.path.exists(txt_path):
        print(f"错误: test.txt文件不存在: {txt_path}")
        return []

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    image_pairs = []
    found_count = 0
    not_found_images = []
    not_found_labels = []

    print(f"\n从 {txt_path} 读取测试集列表...")
    print(f"共 {len(lines)} 个图像ID")

    for line in lines:
        # 去除空白字符
        image_id = line.strip()
        if not image_id:
            continue

        # 如果包含路径分隔符，只取文件名部分
        if os.path.sep in image_id:
            image_id = os.path.splitext(os.path.basename(image_id))[0]
        else:
            # 移除可能的扩展名
            image_id = os.path.splitext(image_id)[0]

        # 查找图像文件
        image_path = None
        for ext in image_extensions:
            candidate = os.path.join(images_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                image_path = candidate
                break

        # 查找标签文件（通常与图像同名）
        label_path = None
        for ext in image_extensions:
            candidate = os.path.join(labels_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                label_path = candidate
                break
            # 有些数据集标签可能是png格式
            candidate_png = os.path.join(labels_dir, f"{image_id}.png")
            if os.path.exists(candidate_png):
                label_path = candidate_png
                break

        if image_path is None:
            not_found_images.append(image_id)
            continue

        if label_path is None:
            not_found_labels.append(image_id)
            # 即使没有标签也继续，只是不计算指标
            print(f"  警告: 图像 {image_id} 没有对应的标签文件")

        image_pairs.append({
            'id': image_id,
            'image_path': image_path,
            'label_path': label_path if label_path else None
        })
        found_count += 1

    print(f"\n找到 {found_count} 对图像-标签")
    if not_found_images:
        print(
            f"未找到的图像 ({len(not_found_images)}): {not_found_images[:10]}{'...' if len(not_found_images) > 10 else ''}")
    if not_found_labels:
        print(
            f"未找到标签的图像 ({len(not_found_labels)}): {not_found_labels[:10]}{'...' if len(not_found_labels) > 10 else ''}")

    return image_pairs


def predict_single_image(model, image_path, device):
    """对单张图像进行预测，返回最终mask和相关信息"""
    # 读取图像
    image = Image.open(image_path)
    original_size = image.size

    # 用于显示的8-bit原图
    old_np = np.array(image).astype(np.float32)
    if old_np.ndim == 3:
        old_np = old_np[..., 0]
    old_disp = old_np.copy()
    old_disp = old_disp - old_disp.min()
    old_disp = old_disp / (old_disp.max() + 1e-8) * 255.0
    old_disp = old_disp.astype(np.uint8)

    # Resize图像
    resized_image, nw, nh, dx, dy, scale = resize_image_keep_ratio(image, INPUT_SHAPE)
    resized_np_base = np.array(resized_image, dtype=np.uint16)
    if resized_np_base.ndim == 3:
        resized_np_base = resized_np_base[..., 0]

    # 多百分位预测融合
    vote_map = np.zeros(old_disp.shape, dtype=np.uint16)

    for hp in HIGH_PERCENT_LIST:
        image_data = normalize_16bit_image(resized_np_base, low_percent=1, high_percent=hp)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = np.expand_dims(image_data, axis=0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).float().to(device)
            outputs, _, _, _ = model(images)
            probs = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()

            probs = probs[dy:dy + nh, dx:dx + nw]
            probs_img = Image.fromarray((probs * 255).astype(np.uint8))
            probs_img = probs_img.resize(image.size, Image.BICUBIC)
            probs = np.array(probs_img, dtype=np.float32) / 255.0

            pred_mask = (probs > THRESHOLD).astype(np.uint8)

        vote_map += pred_mask.astype(np.uint16)

    # 生成final_mask
    if USE_VOTE_FUSION:
        final_mask = (vote_map >= VOTE_THRESHOLD).astype(np.uint8)
    else:
        final_mask = (vote_map >= 1).astype(np.uint8)

    # 圆形度过滤
    # final_mask = filter_by_circularity(final_mask, min_circularity=MIN_CIRCULARITY)

    # 连通域过滤
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        final_mask.astype(np.uint8), connectivity=8
    )

    # 提取候选点
    if num_labels > 1:
        points = []
        point_labels = []
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_COMPONENT_AREA:
                points.append([cx, cy])
                point_labels.append(i)

        points = np.array(points) if points else np.array([])

        if len(points) >= 4:
            try:
                P_clean, noise_idx, _ = remove_outliers_by_exhaustive(points, max_remove=5)
                if len(noise_idx) > 0:
                    keep_labels = [point_labels[i] for i in range(len(point_labels)) if i not in noise_idx]
                    new_mask = np.zeros_like(final_mask, dtype=np.uint8)
                    for lbl in keep_labels:
                        new_mask[labels == lbl] = 1
                    final_mask = new_mask
            except Exception as e:
                pass

    return final_mask, old_disp, original_size, scale, dx, dy


def save_visualization(image, mask, gt_mask, save_path, image_name):
    """保存可视化结果"""
    import matplotlib.pyplot as plt

    # 创建彩色原图
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
        if image_rgb.shape[-1] == 1:
            image_rgb = np.concatenate([image_rgb, image_rgb, image_rgb], axis=-1)

    # 预测叠加图
    pred_overlay = image_rgb.copy()
    pred_overlay[mask == 1] = [255, 0, 0]  # 红色表示预测

    if gt_mask is not None:
        # GT叠加图
        gt_overlay = image_rgb.copy()
        gt_overlay[gt_mask == 1] = [0, 255, 0]  # 绿色表示GT

        # 对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Prediction Mask')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(gt_mask, cmap='gray')
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')

        # 叠加对比图
        overlay_combined = image_rgb.copy()
        overlay_combined[mask == 1] = [255, 128, 128]  # 浅红
        overlay_combined[gt_mask == 1] = [128, 255, 128]  # 浅绿
        overlay_combined[(mask == 1) & (gt_mask == 1)] = [255, 255, 0]  # 黄色表示重叠

        axes[1, 1].imshow(overlay_combined)
        axes[1, 1].set_title('Overlay (Red:Pred, Green:GT, Yellow:Overlap)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{image_name}_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # 保存单独的叠加图
        Image.fromarray(overlay_combined).save(os.path.join(save_path, f"{image_name}_overlay.png"))
    else:
        # 没有GT时的简化可视化
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Prediction Mask')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{image_name}_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # 保存叠加图
        pred_overlay_img = Image.fromarray(pred_overlay)
        pred_overlay_img.save(os.path.join(save_path, f"{image_name}_overlay.png"))

    # 保存单独的mask
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(save_path, f"{image_name}_mask.png"))


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model = Unet(num_classes=NUM_CLASSES, pretrained=False, backbone=BACKBONE)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # 从test.txt获取测试图像列表
    image_pairs = get_test_images_from_txt(
        SPLIT_TXT_PATH, IMAGES_DIR, LABELS_DIR, IMAGE_EXTENSIONS
    )

    if len(image_pairs) == 0:
        print("错误: 没有找到测试图像！")
        print(f"请检查:")
        print(f"  - test.txt路径: {SPLIT_TXT_PATH}")
        print(f"  - Images目录: {IMAGES_DIR}")
        print(f"  - Labels目录: {LABELS_DIR}")
        return

    print(f"\n开始批量测试，共 {len(image_pairs)} 张图像...")

    # 存储所有指标
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    all_accuracy = []

    all_center_metrics = {
        'GT_Count': [], 'Pred_Count': [], 'Matched_Count': [],
        'FN_Count': [], 'FP_Count': [], 'Instance_Precision': [],
        'Instance_Recall': [], 'Mean_Center_Error': []
    }

    total_gt_objects = 0
    total_pred_objects = 0

    # 创建结果汇总文件
    summary_file = os.path.join(OUTPUT_DIR, "test_summary.txt")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("批量测试结果汇总\n")
        f.write(f"测试图像数量: {len(image_pairs)}\n")
        f.write(f"模型路径: {MODEL_PATH}\n")
        f.write(f"测试集文件: {SPLIT_TXT_PATH}\n")
        f.write("=" * 80 + "\n\n")

    # 批量处理
    for pair in tqdm(image_pairs, desc="Processing images"):
        img_id = pair['id']
        img_path = pair['image_path']
        gt_path = pair['label_path']

        # 预测
        try:
            pred_mask, original_img, original_size, scale, dx, dy = predict_single_image(
                model, img_path, device
            )
        except Exception as e:
            print(f"\nError processing {img_id}: {e}")
            continue

        # 加载GT
        gt_mask = None
        if gt_path and os.path.exists(gt_path):
            gt_mask = load_binary_mask(gt_path, target_size=original_size)
            if gt_mask is not None and gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # 计算指标
        if gt_mask is not None:
            # 分割指标
            metrics = compute_binary_seg_metrics(pred_mask, gt_mask)
            all_dice.append(metrics['Dice'])
            all_iou.append(metrics['IoU'])
            all_precision.append(metrics['Precision'])
            all_recall.append(metrics['Recall'])
            all_accuracy.append(metrics['Accuracy'])

            # 中心指标
            center_metrics, matches, gt_centers, pred_centers = evaluate_spot_centers(
                pred_mask, gt_mask, min_area=MIN_COMPONENT_AREA,
                max_match_distance=MAX_MATCH_DISTANCE
            )

            for key in all_center_metrics:
                if key in center_metrics and center_metrics[key] is not None:
                    all_center_metrics[key].append(center_metrics[key])

            total_gt_objects += center_metrics['GT_Count']
            total_pred_objects += center_metrics['Pred_Count']

            # 保存详细结果
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"图像: {img_id}\n")
                f.write(f"  图像路径: {img_path}\n")
                f.write(f"  标签路径: {gt_path}\n")
                f.write(f"  Dice: {metrics['Dice']:.4f}\n")
                f.write(f"  IoU: {metrics['IoU']:.4f}\n")
                f.write(f"  Precision: {metrics['Precision']:.4f}\n")
                f.write(f"  Recall: {metrics['Recall']:.4f}\n")
                f.write(f"  Accuracy: {metrics['Accuracy']:.4f}\n")
                f.write(f"  GT对象数: {center_metrics['GT_Count']}\n")
                f.write(f"  Pred对象数: {center_metrics['Pred_Count']}\n")
                f.write(f"  匹配数: {center_metrics['Matched_Count']}\n")
                if center_metrics['Mean_Center_Error'] is not None:
                    f.write(f"  平均中心误差: {center_metrics['Mean_Center_Error']:.2f} px\n")
        else:
            # 没有GT，只保存预测结果
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"图像: {img_id} (无GT)\n")
                f.write(f"  图像路径: {img_path}\n")

            # 提取中心点
            pred_centers, _ = extract_spot_centers(pred_mask, min_area=MIN_COMPONENT_AREA)
            total_pred_objects += len(pred_centers)
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(f"  Pred对象数: {len(pred_centers)}\n")

        # 保存可视化
        save_visualization(original_img, pred_mask, gt_mask, vis_dir, img_id)

    # 计算平均指标
    print("\n" + "=" * 50)
    print("测试完成！统计结果：")
    print("=" * 50)

    # 输出统计
    if len(all_dice) > 0:
        mean_dice = float(np.mean(all_dice))
        mean_iou = float(np.mean(all_iou))
        mean_precision = float(np.mean(all_precision))
        mean_recall = float(np.mean(all_recall))
        mean_accuracy = float(np.mean(all_accuracy))

        print(f"\n===== Segmentation Metrics =====")
        print(f"Mean Dice     : {mean_dice:.4f}")
        print(f"Mean IoU      : {mean_iou:.4f}")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall   : {mean_recall:.4f}")
        print(f"Mean Accuracy : {mean_accuracy:.4f}")

        # 中心检测统计
        if len(all_center_metrics['GT_Count']) > 0:
            mean_inst_precision = float(np.mean(all_center_metrics['Instance_Precision']))
            mean_inst_recall = float(np.mean(all_center_metrics['Instance_Recall']))
            valid_errors = [e for e in all_center_metrics['Mean_Center_Error'] if e is not None]
            if valid_errors:
                mean_center_error = float(np.mean(valid_errors))
                print(f"\n===== Object Detection Metrics =====")
                print(f"Mean Instance Precision: {mean_inst_precision:.4f}")
                print(f"Mean Instance Recall   : {mean_inst_recall:.4f}")
                print(f"Mean Center Error      : {mean_center_error:.2f} px")

        print(f"\n===== Object Counts =====")
        print(f"GT objects    : {total_gt_objects}")
        print(f"Pred objects  : {total_pred_objects}")

        # 保存汇总统计
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("总体统计结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"\n===== Segmentation Metrics =====\n")
            f.write(f"Mean Dice     : {mean_dice:.4f}\n")
            f.write(f"Mean IoU      : {mean_iou:.4f}\n")
            f.write(f"Mean Precision: {mean_precision:.4f}\n")
            f.write(f"Mean Recall   : {mean_recall:.4f}\n")
            f.write(f"Mean Accuracy : {mean_accuracy:.4f}\n")
            f.write(f"\n===== Object Detection Metrics =====\n")
            f.write(f"Mean Instance Precision: {mean_inst_precision:.4f}\n")
            f.write(f"Mean Instance Recall   : {mean_inst_recall:.4f}\n")
            if valid_errors:
                f.write(f"Mean Center Error      : {mean_center_error:.2f} px\n")
            f.write(f"\n===== Object Counts =====\n")
            f.write(f"GT objects    : {total_gt_objects}\n")
            f.write(f"Pred objects  : {total_pred_objects}\n")

        # 保存详细指标到CSV
        import csv
        csv_path = os.path.join(OUTPUT_DIR, "test_metrics.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image_ID', 'Dice', 'IoU', 'Precision', 'Recall', 'Accuracy',
                             'GT_Count', 'Pred_Count', 'Matched_Count', 'Center_Error'])

            for idx, pair in enumerate(image_pairs):
                if pair['label_path'] is not None and idx < len(all_dice):
                    writer.writerow([
                        pair['id'],
                        f"{all_dice[idx]:.4f}",
                        f"{all_iou[idx]:.4f}",
                        f"{all_precision[idx]:.4f}",
                        f"{all_recall[idx]:.4f}",
                        f"{all_accuracy[idx]:.4f}",
                        all_center_metrics['GT_Count'][idx] if idx < len(all_center_metrics['GT_Count']) else '',
                        all_center_metrics['Pred_Count'][idx] if idx < len(all_center_metrics['Pred_Count']) else '',
                        all_center_metrics['Matched_Count'][idx] if idx < len(
                            all_center_metrics['Matched_Count']) else '',
                        f"{all_center_metrics['Mean_Center_Error'][idx]:.2f}" if idx < len(
                            all_center_metrics['Mean_Center_Error']) and all_center_metrics['Mean_Center_Error'][
                                                                                     idx] is not None else ''
                    ])
                elif pair['label_path'] is None:
                    # 没有GT的情况
                    pred_centers, _ = extract_spot_centers(pred_mask, min_area=MIN_COMPONENT_AREA)
                    writer.writerow([
                        pair['id'], '', '', '', '', '',
                        '', len(pred_centers), '', ''
                    ])

        print(f"\n详细结果已保存到:")
        print(f"  - 汇总文件: {summary_file}")
        print(f"  - CSV文件: {csv_path}")
        print(f"  - 可视化图像: {vis_dir}")

    else:
        print("\n没有找到有效的GT文件，只保存了预测结果。")
        print(f"预测结果保存在: {vis_dir}")

    print("\n批量测试完成！")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt

    main()