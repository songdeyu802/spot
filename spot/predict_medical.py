import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
import torch
import cv2
from skimage.morphology import binary_erosion, disk
from nets.unet import Unet
from scipy.linalg import svd
# =========================
# 配置区
# =========================
MODEL_PATH = "logs/best_epoch_weights.pth"
#IMAGE_PATH = "F:/西物所博士/FPGA信号阈值设置/right/论文/光斑原图/image_216.png"
IMAGE_PATH = "E:/Infrared-Small-Target-Detection-master/dataset/NUDT-SIRST/images/000001.png"
#IMAGE_PATH = "F:/1.06um_20251111am8.30/yang/120A/06-32-54.png"
SAVE_DIR = "pred_results"

BACKBONE = "myunet"
NUM_CLASSES = 2
INPUT_SHAPE = [640, 640]
CUDA = True
THRESHOLD = 0.8
MIN_CIRCULARITY = 0.6           # 提高圆形度阈值
MIN_LINE_POINTS = 3     # 新参数：构成一条直线所需的最少光斑数

# 要尝试的百分位列表
#HIGH_PERCENT_LIST = [99.5, 70.0, 1.0]
HIGH_PERCENT_LIST = [0.5, 1.0, 5.0, 10.0, 30.0, 50.0, 70.0, 90.0, 95.0, 99.0, 99.5, 99.9]
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

def filter_by_circularity(mask, min_circularity=MIN_CIRCULARITY):
    """
    保留圆度大于 min_circularity 且面积大于 min_area 的连通区域。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # 提取当前连通区域的二值图像
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

def filter_by_line(mask, min_points, dist_thresh=5, max_y_std=10):
    """
    保留至少包含 min_points 个点、且这些点到某条直线的距离小于 dist_thresh，
    并且这些点的 y 坐标标准差小于 max_y_std（近似水平）的连通区域。
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels - 1 < min_points:
        return np.zeros_like(mask, dtype=np.uint8)

    points = [centroids[i] for i in range(1, num_labels)]
    n = len(points)

    # 记录所有满足距离条件的组及其 y 标准差
    all_groups = []   # 每个元素为 (indices, y_std)

    for i in range(n):
        for j in range(i+1, n):
            p1 = np.array(points[i])
            p2 = np.array(points[j])
            if np.linalg.norm(p2 - p1) < 1e-3:
                continue
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            unit_vec = line_vec / line_len
            indices = []
            for k, pt in enumerate(points):
                pt = np.array(pt)
                t = np.dot(pt - p1, unit_vec)
                proj = p1 + t * unit_vec
                dist = np.linalg.norm(pt - proj)
                if dist <= dist_thresh:
                    indices.append(k)
            if len(indices) >= min_points:
                # 计算这些点的 y 坐标标准差
                y_vals = [points[idx][1] for idx in indices]
                y_std = np.std(y_vals)
                all_groups.append((indices, y_std))

    if not all_groups:
        return np.zeros_like(mask, dtype=np.uint8)

    # 筛选出 y 标准差小于阈值的组（即水平组）
    valid_groups = [(indices, y_std) for indices, y_std in all_groups if y_std <= max_y_std]
    if not valid_groups:
        return np.zeros_like(mask, dtype=np.uint8)

    # 在符合条件的组中，选择点数最多的一个
    best_group = max(valid_groups, key=lambda x: len(x[0]))[0]

    # 构建最终掩码
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for idx in best_group:
        new_mask[labels == idx+1] = 1
    return new_mask

import numpy as np
import cv2
from scipy.linalg import svd

def ordered_dist_matrix(points):
    """
    输入: points, shape (N, 2)
    输出:
        D: 距离矩阵 (N, N)
        sorted_points: 按主方向排序后的点 (N, 2)
    """
    # 中心化
    centroid = np.mean(points, axis=0)
    P0 = points - centroid

    # SVD 获取主方向
    U, s, Vt = svd(P0, full_matrices=False)
    dir1 = Vt[0, :]          # 主方向（第一主成分）

    # 投影到主方向
    proj = P0 @ dir1
    order = np.argsort(proj)
    sorted_points = points[order, :]

    # 计算距离矩阵（所有点对距离）
    N = len(points)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(sorted_points[i] - sorted_points[j])
            D[i, j] = d
            D[j, i] = d
    return D, sorted_points

def compute_anomaly_score(points):
    """
    评估每个点作为异常点时的“规则性”评分。
    返回: scores (N,) 越小越可能是异常点
    """
    N = len(points)
    if N < 4:
        return np.zeros(N)  # 点数太少，不做剔除

    scores = np.full(N, np.inf)

    # 留一法
    for i in range(N):
        keep = np.ones(N, dtype=bool)
        keep[i] = False
        Q = points[keep, :]

        # 1. 主方向排序并计算距离矩阵
        try:
            D, Qs = ordered_dist_matrix(Q)
        except:
            continue

        # 相邻间距（排序后点与下一个点的距离）
        d_adj = np.diag(D, 1)   # 对角线偏移1的元素
        if len(d_adj) < 2:
            continue

        # 基值：相邻间距的中位数（避免异常值影响）
        base = max(np.median(d_adj), 1e-6)

        # (a) 相邻间距的一致性：标准差
        spacing_score = np.std(d_adj) / base

        # (b) 间距变化的平滑度：相邻间距差值的均方根
        if len(d_adj) > 1:
            smooth_score = np.sqrt(np.mean(np.diff(d_adj)**2)) / base
        else:
            smooth_score = 0

        # (c) 点偏离直线的程度：垂直主方向上的投影误差
        # 计算 Q 中各点到主方向的垂直距离
        centroid = np.mean(Q, axis=0)
        Q0 = Q - centroid
        # 主方向（第一主成分）
        U, s, Vt = svd(Q0, full_matrices=False)
        dir1 = Vt[0, :]          # 主方向
        dir2 = Vt[1, :]          # 垂直方向
        # 投影到垂直方向
        err = Q0 @ dir2
        line_score = np.sqrt(np.mean(err**2)) / base

        # 综合评分（权重可调）
        scores[i] = spacing_score + 0.5 * smooth_score + 0.2 * line_score

    return scores

def remove_spacing_outlier(mask, max_iter=1, min_points=4):
    """
    迭代删除异常点，直到剩余点数量小于 min_points 或不再有异常点。
    输入:
        mask: 二值掩码 (numpy array, dtype=uint8)
        max_iter: 最大迭代次数（通常为1次即可）
        min_points: 最少保留点数
    输出:
        clean_mask: 清洗后的掩码
        removed_indices: 所有被删除点的原始索引（相对于第一次提取的点列表）
    """
    # 提取连通区域中心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 2:
        return mask, []

    # 收集所有有效点（排除背景）
    points = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        area = stats[i, cv2.CC_STAT_AREA]
        # 可选：根据面积过滤小噪点（可根据需要调整）
        if area < 2:   # 极小噪点直接忽略
            continue
        points.append((cx, cy, i))   # 存储坐标和对应的 label
    if len(points) < min_points:
        return mask, []

    # 转换为 numpy 数组（坐标）
    coords = np.array([[p[0], p[1]] for p in points])
    removed = []   # 记录被删点的原始索引（在 points 中的位置）

    for _ in range(max_iter):
        N = len(coords)
        if N < min_points:
            break

        scores = compute_anomaly_score(coords)
        outlier_idx = np.argmin(scores)   # 评分最小的点最异常

        # 可选：如果最小评分仍很大（比如大于阈值），则不再剔除
        # if scores[outlier_idx] > some_threshold: break

        # 记录该点在原始 points 中的索引
        removed.append(outlier_idx)

        # 删除该点
        keep = np.ones(N, dtype=bool)
        keep[outlier_idx] = False
        coords = coords[keep, :]

    # 构建最终掩码
    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    # 保留所有未被删除的点（注意：索引需要与原始的 points 对应）
    keep_indices = [i for i in range(len(points)) if i not in removed]
    for idx in keep_indices:
        label = points[idx][2]
        clean_mask[labels == label] = 1

    # 返回删除点在原始点列表中的索引（如果多次迭代，可能需要映射，这里返回直接删掉的所有原始索引）
    # 注意：若迭代删除多次，removed 中的索引是基于当前 coords 的，需要映射回原始点列表，这里简化处理。
    # 如果需要准确的原始索引，可以在每次删除时记录映射关系。但通常我们只关心最终掩码，不需要索引。
    # 所以返回空列表即可，或返回 removed 的原始索引（此处省略复杂映射）。

    return clean_mask, removed

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

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

    # 3. 初始化最终掩码
    final_mask = np.zeros(old_disp.shape, dtype=np.uint8)

    # 4. 多百分位预测融合
    for hp in HIGH_PERCENT_LIST:
        print(f"Processing with high_percent = {hp} ...")
        image_data, nw, nh, dx, dy = resize_image_keep_ratio(image, INPUT_SHAPE)
        image_data = np.array(image_data, dtype=np.uint16)
        if image_data.ndim == 3:
            image_data = image_data[..., 0]
        image_data = normalize_16bit_image(image_data, low_percent=1, high_percent=hp)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = np.expand_dims(image_data, axis=0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).float().to(device)
            outputs, _, _, _ = model(images)
            probs = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()
            probs = probs[dy:dy+nh, dx:dx+nw]
            probs_img = Image.fromarray((probs * 255).astype(np.uint8))
            probs_img = probs_img.resize(old_image.size, Image.BICUBIC)
            probs = np.array(probs_img, dtype=np.float32) / 255.0
            pred_mask = (probs > THRESHOLD).astype(np.uint8)

        final_mask = np.logical_or(final_mask, pred_mask).astype(np.uint8)

    # 5. 后处理：面积过滤
    final_mask = filter_by_circularity(final_mask, min_circularity=MIN_CIRCULARITY)
    cv2.imwrite(os.path.join(SAVE_DIR, "step1_area.png"), final_mask * 255)
    # 6. 屏蔽已知的固定盲元（坐标邻域）
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
    cv2.imwrite(os.path.join(SAVE_DIR, "step2_clean.png"), final_mask * 255)
    # ... 前面的过滤步骤
    final_mask = filter_by_line(final_mask, min_points=MIN_LINE_POINTS)
    cv2.imwrite(os.path.join(SAVE_DIR, "step3_line.png"), final_mask * 255)

    # 新增：基于间距一致性的异常点剔除（最多删除1个点）
    final_mask, _ = remove_spacing_outlier(final_mask, max_iter=1, min_points=4)
    cv2.imwrite(os.path.join(SAVE_DIR, "step4_spacing_outlier.png"), final_mask * 255)

    # 保存结果
    cv2.imwrite(os.path.join(SAVE_DIR, "step4_spacing_outlier.png"), final_mask * 255)

    # 可选：形态学腐蚀
    final_mask = binary_erosion(final_mask, disk(1)).astype(np.uint8)
    # 9. 保存结果
    mask_save = final_mask * 255
    Image.fromarray(mask_save).save(os.path.join(SAVE_DIR, "final_mask.png"))
    Image.fromarray(old_disp).save(os.path.join(SAVE_DIR, "original.png"))

    old_rgb = np.stack([old_disp, old_disp, old_disp], axis=-1)
    overlay = old_rgb.copy()
    overlay[final_mask == 1] = [255, 0, 0]
    Image.fromarray(overlay).save(os.path.join(SAVE_DIR, "overlay_final.png"))

    print("预测完成，结果保存在：", SAVE_DIR)

if __name__ == "__main__":
    main()