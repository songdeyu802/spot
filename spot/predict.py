import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from nets.unet import Unet
from utils.utils import cvtColor, preprocess_input


# =========================
# 配置区
# =========================
MODEL_PATH = "logs/best_epoch_weights.pth"
IMAGE_PATH = "Medical_Datasets/Images/image_104.png"   # 改成你要测试的图
SAVE_DIR = "pred_results"

BACKBONE = "myunet"
NUM_CLASSES = 2
INPUT_SHAPE = [512, 512]   # 训练时用的尺寸
CUDA = True


def resize_image_keep_ratio(image, size):
    iw, ih = image.size
    h, w = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", (w, h), (128, 128, 128))
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image.paste(image, (dx, dy))
    return new_image, nw, nh, dx, dy


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

    # 1. 建模并加载权重
    model = Unet(num_classes=NUM_CLASSES, pretrained=False, backbone=BACKBONE)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 2. 读取图像
    image = Image.open(IMAGE_PATH)
    image = cvtColor(image)   # 转成 RGB 兼容原项目流程
    old_image = image.copy()

    # 3. resize（和训练验证同风格）
    image_data, nw, nh, dx, dy = resize_image_keep_ratio(image, INPUT_SHAPE)

    # 4. 预处理
    image_data = np.array(image_data, np.float32)
    image_data = np.transpose(preprocess_input(image_data), (2, 0, 1))
    image_data = np.expand_dims(image_data, 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data).float().to(device)
        outputs = model(images)                         # [1, 2, H, W]
        probs = torch.softmax(outputs, dim=1)[0, 1]    # 取前景概率图

        # 去掉 padding 区域
        probs = probs.cpu().numpy()
        probs = probs[dy:dy + nh, dx:dx + nw]

        # resize 回原图尺寸
        probs_img = Image.fromarray((probs * 255).astype(np.uint8))
        probs_img = probs_img.resize(old_image.size, Image.BICUBIC)
        probs = np.array(probs_img, dtype=np.float32) / 255.0

        # 阈值，可后面调
        pred_mask = (probs > 0.5).astype(np.uint8) * 255

    # 5. 保存概率图
    prob_save = (probs * 255).astype(np.uint8)
    Image.fromarray(prob_save).save(os.path.join(SAVE_DIR, "probability.png"))

    # 6. 保存二值 mask
    Image.fromarray(pred_mask).save(os.path.join(SAVE_DIR, "pred_mask.png"))

    # 7. 保存原图
    old_image.save(os.path.join(SAVE_DIR, "original.png"))

    # 8. 保存叠加图
    old_np = np.array(old_image).astype(np.uint8)
    overlay = old_np.copy()

    # 用红色高亮前景
    overlay[pred_mask > 0] = [255, 0, 0]

    overlay = (0.6 * old_np + 0.4 * overlay).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(SAVE_DIR, "overlay.png"))

    print("预测完成，结果保存在：", SAVE_DIR)


if __name__ == "__main__":
    main()