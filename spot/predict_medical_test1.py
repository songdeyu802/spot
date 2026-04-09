import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
import torch
import cv2
from skimage.morphology import binary_erosion, disk
from nets.unet import Unet
from scipy.linalg import svd
import itertools
from copy import deepcopy

# =========================
# 配置区
# =========================
MODEL_PATH = "logs/best_epoch_weights.pth"
IMAGE_PATH = "F:/西物所博士/FPGA信号阈值设置/right/论文/光斑原图/image_33.png"
SAVE_DIR = "pred_results"

BACKBONE = "myunet"
NUM_CLASSES = 2
INPUT_SHAPE = [640, 640]
CUDA = True
THRESHOLD = 0.8
MIN_CIRCULARITY = 0.6           # 圆形度阈值

# 多百分位列表（可根据需要精简）
HIGH_PERCENT_LIST = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                     15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                     55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0,
                     91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
                     99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]

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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
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

    # 1) 中心化，SVD 获取主方向
    centroid = np.mean(Q, axis=0)
    Q0 = Q - centroid
    U, s, Vt = svd(Q0, full_matrices=False)  # Vt 的行是主成分方向
    dir_main = Vt[0, :]           # 第一主成分（主方向）
    dir_norm = Vt[1, :]           # 垂直方向

    # 投影到主方向
    s_proj = Q0 @ dir_main         # 一维坐标
    e_proj = Q0 @ dir_norm         # 法向距离

    # 2) 按主方向排序
    order = np.argsort(s_proj)
    s_sort = s_proj[order]
    Q_sort = Q[order, :]
    e_sort = e_proj[order]

    # 3) 用等差序列拟合投影坐标
    idx = np.arange(M)
    coeff = np.polyfit(idx, s_sort, 1)
    d = abs(coeff[0])               # 估计间距
    if d < 1e-6:
        return np.inf, None
    s_fit = np.polyval(coeff, idx)

    # 4) 构造实际距离矩阵（使用投影坐标计算距离，更快）
    D = np.abs(s_sort[:, None] - s_sort[None, :])

    # 5) 构造理想距离矩阵（等差）
    ID = np.abs(idx[:, None] - idx[None, :])
    D0 = ID * d

    # 只取上三角（不包括对角线）
    triu_mask = np.triu(np.ones((M, M), dtype=bool), k=1)

    # 6) 计算三种残差
    r_line = np.sqrt(np.mean(e_sort**2)) / d          # 共线性
    diff_D = D[triu_mask] - D0[triu_mask]
    r_dist = np.sqrt(np.mean(diff_D**2)) / d          # 距离矩阵一致性
    g = np.diff(s_sort)
    r_gap = np.sqrt(np.mean((g - d)**2)) / d           # 相邻间距一致性

    # 附加：间距变异系数
    if np.mean(g) > 1e-6:
        r_cv = np.std(g) / np.mean(g)
    else:
        r_cv = 1.0

    # 7) 综合评分
    score = (params['w_line'] * r_line +
             params['w_dist'] * r_dist +
             params['w_gap']  * r_gap +
             0.05 * r_cv)

    # 保存信息用于调试
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
    # 确保 params 是字典
    if params is None or not isinstance(params, dict):
        print(f"警告：params 类型为 {type(params)}，已重置为默认字典")
        params = {
            'w_line': 0.35,
            'w_dist': 0.45,
            'w_gap':  0.20,
            'lambda_delete': 0.08
        }

    N = P.shape[0]
    if N < 4:
        raise ValueError("点数少于4，无法进行稳健剔除。")

    # 确定尝试的删除个数列表
    if target_num is not None:
        k_need = N - target_num
        if k_need < 0:
            raise ValueError("target_num 大于候选点数。")
        if k_need > max_remove:
            raise ValueError(f"需要删除 {k_need} 个点，超过 max_remove={max_remove}。")
        candidates_k = [k_need]
    else:
        candidates_k = range(0, max_remove+1)

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

            # 加入删除点数惩罚，防止过度删除
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

    # 5. 圆形度过滤
    final_mask = filter_by_circularity(final_mask, min_circularity=MIN_CIRCULARITY)
    cv2.imwrite(os.path.join(SAVE_DIR, "step1_area.png"), final_mask * 255)

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
    cv2.imwrite(os.path.join(SAVE_DIR, "step2_clean.png"), final_mask * 255)

    # 7. 直接提取所有候选点（不再使用 filter_by_line，避免误判）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        points = []
        point_labels = []
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 2:
                continue
            points.append([cx, cy])
            point_labels.append(i)
        points = np.array(points)

        if len(points) >= 4:
            # 设置穷举剔除参数
            max_remove = 5               # 最多删除5个点
            target_num = None            # 若已知真实光斑数，可设置具体数值，例如 6
            params = {
                'w_line': 0.35,
                'w_dist': 0.45,
                'w_gap': 0.20,
                'lambda_delete': 0.08
            }
            print(f"开始穷举剔除，候选点数: {len(points)}")
            try:
                P_clean, noise_idx, best_info = remove_outliers_by_exhaustive(points, max_remove, target_num, params)
                print(f"实际删除点数: {len(noise_idx)}")
                # 重建掩码
                new_mask = np.zeros_like(final_mask, dtype=np.uint8)
                keep_labels = point_labels[[i for i in range(len(point_labels)) if i not in noise_idx]]
                for lbl in keep_labels:
                    new_mask[labels == lbl] = 1
                final_mask = new_mask
                cv2.imwrite(os.path.join(SAVE_DIR, "step4_combined.png"), final_mask * 255)
            except Exception as e:
                print(f"剔除过程出错: {e}")
        else:
            print(f"点数不足4，跳过剔除。当前点数: {len(points)}")
    else:
        print("没有连通域，跳过剔除。")

    # 8. 可选：形态学腐蚀
    final_mask = binary_erosion(final_mask, disk(1)).astype(np.uint8)

    # 9. 保存最终结果
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