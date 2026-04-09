import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input


def normalize_16bit_image(img, low_percent=1, high_percent=99.5):
    img = img.astype(np.float32)

    low = np.percentile(img, low_percent)
    high = np.percentile(img, high_percent)

    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)

    return img


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, use_patch=False,
                 patch_size=128, noise_label_dir="NoiseLabels"):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.use_patch = use_patch
        self.patch_size = patch_size
        self.noise_label_dir = noise_label_dir

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#
        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), name + ".png"))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"), name + ".png"))
        noise_png_path = os.path.join(self.dataset_path, self.noise_label_dir, name + ".png")
        if os.path.exists(noise_png_path):
            noise_png = Image.open(noise_png_path)
        else:
            noise_png = Image.new("L", png.size, 0)

        jpg, png, noise_png = self.get_random_data(jpg, png, noise_png, self.input_shape, random=self.train)

        # 转成 numpy
        jpg = np.array(jpg, dtype=np.uint16)
        png = np.array(png)

        # 如果意外读成了三通道，只取一个通道
        if jpg.ndim == 3:
            jpg = jpg[..., 0]

        jpg = normalize_16bit_image(jpg)
        jpg = np.expand_dims(jpg, axis=0).astype(np.float32)

        # 标签二值化
        modify_png = (png > 0).astype(np.uint8)
        noise_gt = (np.array(noise_png) > 0).astype(np.uint8)

        # patch 裁剪
        if self.use_patch:
            coords = np.argwhere(modify_png > 0)

            # 70% 围绕前景裁，30% 随机裁背景
            if len(coords) > 0 and np.random.rand() < 0.7:
                cy, cx = coords[np.random.randint(len(coords))]
            else:
                cy = np.random.randint(0, modify_png.shape[0])
                cx = np.random.randint(0, modify_png.shape[1])

            ps = self.patch_size
            h, w = modify_png.shape

            y1 = max(0, cy - ps // 2)
            x1 = max(0, cx - ps // 2)
            y2 = min(h, y1 + ps)
            x2 = min(w, x1 + ps)

            y1 = max(0, y2 - ps)
            x1 = max(0, x2 - ps)

            jpg = jpg[:, y1:y2, x1:x2]
            modify_png = modify_png[y1:y2, x1:x2]

        # one-hot 标签，注意尺寸不能再写死 input_shape
        h, w = modify_png.shape
        # 将2D标签展平，转为one-hot，再reshape
        seg_labels = np.eye(self.num_classes, dtype=np.float32)[modify_png.reshape([-1])]
        seg_labels = seg_labels.reshape((h, w, self.num_classes))
        return jpg, modify_png, noise_gt, seg_labels


    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, noise_label, input_shape, jitter=0.05, random=True):
        # """
        # 专为彩色星点图像设计的数据增强：
        # - 微幅缩放（±5%）
        # - 90° 旋转
        # - 随机水平翻转
        # - 随机平移填充
        # - 亮度 / 对比度 微调
        # - 高斯噪声
        # """
        # 保留原始步骤
        image = Image.fromarray(np.array(image, dtype=np.uint16))
        label = Image.fromarray(np.array(label))
        noise_label = Image.fromarray(np.array(noise_label))

        iw, ih = image.size
        h, w = input_shape

        if not random:
            # 等比例缩放 + 居中填充
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            label = label.resize((nw, nh), Image.NEAREST)

            new_img = Image.new('I;16', (w, h), 0)
            new_lbl = Image.new('L', (w, h), 0)
            new_noise_lbl = Image.new('L', (w, h), 0)
            new_img.paste(image, ((w - nw) // 2, (h - nh) // 2))
            new_lbl.paste(label, ((w - nw) // 2, (h - nh) // 2))
            new_noise_lbl.paste(noise_label, ((w - nw) // 2, (h - nh) // 2))
            return new_img, new_lbl, new_noise_lbl

        # —— 微幅缩放 —— #
        scale = self.rand(1 - jitter, 1 + jitter)  # ±5%
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # —— 90° 旋转 —— #
        angle = np.random.choice([0, 90, 180, 270])
        image = image.rotate(angle)
        label = label.rotate(angle)
        noise_label = noise_label.rotate(angle)

        # —— 随机水平翻转 —— #
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            noise_label = noise_label.transpose(Image.FLIP_LEFT_RIGHT)

        # —— 随机平移并填充背景 —— #
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_img = Image.new('I;16', (w, h), 0)
        new_lbl = Image.new('L', (w, h), 0)
        new_noise_lbl = Image.new('L', (w, h), 0)
        new_img.paste(image, (dx, dy))
        new_lbl.paste(label, (dx, dy))
        new_noise_lbl.paste(noise_label, (dx, dy))

        img_np = np.array(new_img, dtype=np.float32)

        alpha = self.rand(0.95, 1.05)
        beta = self.rand(-500, 500)
        img_np = np.clip(img_np * alpha + beta, 0, 65535)

        if self.rand() < 0.3:
            noise = np.random.normal(0, 50, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 65535)

        final_img = Image.fromarray(img_np.astype(np.uint16))
        return final_img, new_lbl, new_noise_lbl

    # def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
    #     image   = cvtColor(image)
    #     label   = Image.fromarray(np.array(label))
    #     #------------------------------#
    #     #   获得图像的高宽与目标高宽
    #     #------------------------------#
    #     iw, ih  = image.size
    #     h, w    = input_shape

    #     if not random:
    #         iw, ih  = image.size
    #         scale   = min(w/iw, h/ih)
    #         nw      = int(iw*scale)
    #         nh      = int(ih*scale)

    #         image       = image.resize((nw,nh), Image.BICUBIC)
    #         new_image   = Image.new('RGB', [w, h], (128,128,128))
    #         new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    #         label       = label.resize((nw,nh), Image.NEAREST)
    #         new_label   = Image.new('L', [w, h], (0))
    #         new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    #         return new_image, new_label

    #     #------------------------------------------#
    #     #   对图像进行缩放并且进行长和宽的扭曲
    #     #------------------------------------------#
    #     new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
    #     scale = self.rand(0.9, 1.1)
    #     if new_ar < 1:
    #         nh = int(scale*h)
    #         nw = int(nh*new_ar)
    #     else:
    #         nw = int(scale*w)
    #         nh = int(nw/new_ar)
    #     image = image.resize((nw,nh), Image.BICUBIC)
    #     label = label.resize((nw,nh), Image.NEAREST)

    #     #------------------------------------------#
    #     #   翻转图像
    #     #------------------------------------------#
    #     flip = self.rand()<.5
    #     if flip:
    #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #         label = label.transpose(Image.FLIP_LEFT_RIGHT)

    #     #------------------------------------------#
    #     #   将图像多余的部分加上灰条
    #     #------------------------------------------#
    #     dx = int(self.rand(0, w-nw))
    #     dy = int(self.rand(0, h-nh))
    #     new_image = Image.new('RGB', (w,h), (128,128,128))
    #     new_label = Image.new('L', (w,h), (0))
    #     new_image.paste(image, (dx, dy))
    #     new_label.paste(label, (dx, dy))
    #     image = new_image
    #     label = new_label

    #     image_data      = np.array(image, np.uint8)
    #     #---------------------------------#
    #     #   对图像进行色域变换
    #     #   计算色域变换的参数
    #     #---------------------------------#
    #     r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #     #---------------------------------#
    #     #   将图像转到HSV上
    #     #---------------------------------#
    #     hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    #     dtype           = image_data.dtype
    #     #---------------------------------#
    #     #   应用变换
    #     #---------------------------------#
    #     x       = np.arange(0, 256, dtype=r.dtype)
    #     lut_hue = ((x * r[0]) % 180).astype(dtype)
    #     lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    #     lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    #     image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    #     image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    #     return image_data, label


def unet_dataset_collate(batch):
    images = []
    pngs = []
    noise_gts = []
    seg_labels = []
    for img, png, noise_gt, labels in batch:
        images.append(img)
        pngs.append(png)
        noise_gts.append(noise_gt)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    noise_gts = torch.from_numpy(np.array(noise_gts)).float()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, noise_gts, seg_labels
