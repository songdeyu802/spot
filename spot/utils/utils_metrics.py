import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


import torch
import torch.nn.functional as F


def f_score(inputs, target, beta=2.0, smooth=1e-5, threshold=0.5, foreground_index=1):
    """
    更适合 TinyObjectLoss / 小目标分割 的前景 F_beta 指标

    参数
    ----------
    inputs : torch.Tensor
        模型输出 logits
        支持:
        - [N, 1, H, W]   单通道二分类
        - [N, 2, H, W]   双通道(背景/前景)
        - [N, C, H, W]   多分类（默认取 foreground_index 作为前景类）

    target : torch.Tensor
        标签，支持多种格式:
        - [N, H, W]                 0/1 mask 或 类别索引图
        - [N, 1, H, W]              0/1 mask
        - [N, H, W, Ct]             one-hot，可能最后一维含 ignore 通道
        - [N, Ct, H, W]             one-hot，可能第一维通道含 ignore 通道

    beta : float
        F_beta 的 beta，默认 2.0，更偏向召回，适合小目标

    smooth : float
        平滑项，避免分母为 0

    threshold : float
        前景概率二值化阈值

    foreground_index : int
        当前景是多分类中的某一类时，对应的类别下标
        对双通道背景/前景输出，默认前景是 1
    """

    n, c, h, w = inputs.shape

    # -----------------------------
    # 1) 解析 target，提取：
    #    - target_fg: 前景真值 [N,1,Ht,Wt]
    #    - valid_mask: 有效像素掩码 [N,1,Ht,Wt]
    # -----------------------------
    valid_mask = None

    if target.dim() == 3:
        # [N,H,W]，可能是 0/1 mask，也可能是类别索引图
        ht, wt = target.shape[1], target.shape[2]
        if c == 1:
            target_fg = target.unsqueeze(1).float()
        else:
            target_fg = (target == foreground_index).float().unsqueeze(1)

    elif target.dim() == 4:
        # 情况 A: [N,1,H,W]
        if target.shape[1] == 1:
            ht, wt = target.shape[2], target.shape[3]
            target_fg = target.float()

        # 情况 B: [N,Ct,H,W] one-hot / one-hot+ignore
        elif target.shape[2] == h and target.shape[3] == w or target.shape[1] <= 10:
            # 更偏向按 NCHW one-hot 处理
            _, ct, ht, wt = target.shape
            target = target.float()

            has_ignore = (ct == c + 1) or (c == 1 and ct >= 2)
            valid_ct = ct - 1 if has_ignore else ct

            if has_ignore:
                valid_mask = 1.0 - target[:, -1:, :, :]

            if c == 1:
                # 单通道输出时，优先取第 0 个有效类别为前景
                fg_idx = 0 if valid_ct == 1 else min(foreground_index, valid_ct - 1)
            else:
                fg_idx = min(foreground_index, valid_ct - 1)

            target_fg = target[:, fg_idx:fg_idx + 1, :, :]

        # 情况 C: [N,H,W,Ct] one-hot / one-hot+ignore
        else:
            nt, ht, wt, ct = target.shape
            target = target.float()

            has_ignore = (ct == c + 1) or (c == 1 and ct >= 2)
            valid_ct = ct - 1 if has_ignore else ct

            if has_ignore:
                valid_mask = 1.0 - target[..., -1].unsqueeze(1)

            if c == 1:
                fg_idx = 0 if valid_ct == 1 else min(foreground_index, valid_ct - 1)
            else:
                fg_idx = min(foreground_index, valid_ct - 1)

            target_fg = target[..., fg_idx].unsqueeze(1)

    else:
        raise ValueError(f"Unsupported target shape: {target.shape}")

    target_fg = target_fg.float()

    # 如果没有显式 ignore 通道，则所有像素都有效
    if valid_mask is None:
        valid_mask = torch.ones_like(target_fg)

    # -----------------------------
    # 2) 尺寸对齐
    # -----------------------------
    if (h != ht) or (w != wt):
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=False)

    # -----------------------------
    # 3) 从 logits 得到前景概率
    # -----------------------------
    if inputs.shape[1] == 1:
        # 单通道二分类
        prob_fg = torch.sigmoid(inputs)
    else:
        # 双通道/多分类
        prob = torch.softmax(inputs, dim=1)
        fg_idx = min(foreground_index, inputs.shape[1] - 1)
        prob_fg = prob[:, fg_idx:fg_idx + 1, :, :]

    # -----------------------------
    # 4) 二值化预测
    # -----------------------------
    pred_fg = (prob_fg > threshold).float()

    # 只在有效区域内统计
    pred_fg = pred_fg * valid_mask
    target_fg = target_fg * valid_mask

    # -----------------------------
    # 5) 计算前景 F_beta
    # -----------------------------
    tp = torch.sum(pred_fg * target_fg)
    fp = torch.sum(pred_fg * (1.0 - target_fg))
    fn = torch.sum((1.0 - pred_fg) * target_fg)

    beta2 = beta ** 2
    score = ((1.0 + beta2) * tp + smooth) / ((1.0 + beta2) * tp + beta2 * fn + fp + smooth)

    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
    return np.array(hist, np.int), IoUs, PA_Recall, Precision

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            