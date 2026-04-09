import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss, FG_BCE_Loss

# tqdm 用来显示进度条
from tqdm import tqdm

# 获取当前学习率的函数
from utils.utils import get_lr

# 计算分割任务评价指标 f_score 的函数
from utils.utils_metrics import f_score

# ==================== 新增损失函数定义 ====================

class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - 适合极小目标，减少漏检"""

    def __init__(self, alpha=0.45, beta=0.55, gamma=1.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # FP 权重
        self.beta = beta  # FN 权重，beta > alpha 更重视召回
        self.gamma = gamma  # focal 参数
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: [B,1,H,W] 或 [B,2,H,W]
        # target: [B,1,H,W] 或 [B,H,W]
        if logits.shape[1] > 1:
            logits = logits[:, 1:2, :, :]  # 取前景通道

        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        prob = torch.sigmoid(logits)

        dims = (0, 2, 3)
        tp = (prob * target).sum(dims)
        fp = (prob * (1 - target)).sum(dims)
        fn = ((1 - prob) * target).sum(dims)

        tversky = (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = (1 - tversky) ** self.gamma
        return loss.mean()


class OHEMBCEWithLogitsLoss(nn.Module):
    """OHEM BCE - 专门压 hardest negatives，减少假阳性"""

    def __init__(self, pos_weight=1.0, neg_pos_ratio=20, empty_neg_ratio=0.01):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.empty_neg_ratio = empty_neg_ratio

        if torch.is_tensor(pos_weight):
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits, target):
        # logits: [B,1,H,W] 或 [B,2,H,W]
        # target: [B,1,H,W] 或 [B,H,W]
        if logits.shape[1] > 1:
            logits = logits[:, 1:2, :, :]

        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        bce_map = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none', pos_weight=self.pos_weight
        )

        pos_mask = target > 0.5
        neg_mask = ~pos_mask

        pos_loss = bce_map[pos_mask]
        neg_loss = bce_map[neg_mask]

        num_pos = int(pos_mask.sum().item())
        num_neg = int(neg_mask.sum().item())

        if num_neg == 0:
            neg_keep = logits.new_tensor(0.0)
        else:
            if num_pos > 0:
                k = min(num_neg, self.neg_pos_ratio * num_pos)
            else:
                # 整张图没有目标，只保留最难的那 1% 负样本
                k = max(1, int(num_neg * self.empty_neg_ratio))
            neg_keep = torch.topk(neg_loss, k=k, largest=True).values.mean()

        if pos_loss.numel() > 0:
            pos_keep = pos_loss.mean()
        else:
            pos_keep = logits.new_tensor(0.0)

        return pos_keep + neg_keep


class TinyObjectLoss(nn.Module):
    """组合损失：OHEM BCE + Focal Tversky，专为极小目标设计"""

    def __init__(self, pos_weight=8.0, neg_pos_ratio=20,
                 lambda_bce=0.65, lambda_ft=0.35):
        super().__init__()
        self.ohem_bce = OHEMBCEWithLogitsLoss(
            pos_weight=pos_weight,
            neg_pos_ratio=neg_pos_ratio,
            empty_neg_ratio=0.01
        )
        self.focal_tversky = FocalTverskyLoss(
            alpha=0.45, beta=0.55, gamma=1.5
        )
        self.lambda_bce = lambda_bce
        self.lambda_ft = lambda_ft

    def forward(self, pred, target):
        # pred: [B,1,H,W] 或 [B,2,H,W]
        # target: [B,1,H,W] 或 [B,H,W]
        return (
                self.lambda_bce * self.ohem_bce(pred, target) +
                self.lambda_ft * self.focal_tversky(pred, target)
        )


def dilate_mask(mask, kernel_size):
    """
    对 mask 进行膨胀
    mask: [B,1,H,W] 或 [B,H,W]
    kernel_size: 膨胀核大小 (3, 5, 等)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()

    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    padding = kernel_size // 2
    out = F.conv2d(mask, kernel, padding=padding)
    return (out > 0).float()


# ==================== 修改后的 fit_one_epoch ====================

def fit_one_epoch(
        model_train,  # 用来训练的模型（可能是包装后的模型）
        model,  # 真正保存权重用的模型
        loss_history,  # 记录 loss 历史的对象
        eval_callback,  # 每个 epoch 结束时调用的评估回调
        optimizer,  # 优化器
        epoch,  # 当前是第几个 epoch（从 0 开始）
        epoch_step,  # 训练集一轮要跑多少个 batch
        epoch_step_val,  # 验证集一轮要跑多少个 batch
        gen,  # 训练集数据迭代器
        gen_val,  # 验证集数据迭代器
        Epoch,  # 总 epoch 数
        cuda,  # 是否使用 GPU
        dice_loss,  # 是否额外加 Dice Loss (已废弃，保留参数兼容性)
        focal_loss,  # 是否使用 Focal Loss (已废弃，保留参数兼容性)
        cls_weights,  # 类别权重 (已废弃，保留参数兼容性)
        num_classes,  # 类别数
        fp16,  # 是否使用混合精度训练
        scaler,  # 混合精度训练用的梯度缩放器
        save_period,  # 每隔多少轮保存一次模型
        save_dir,  # 模型保存目录
        local_rank=0,  # 分布式训练时当前进程序号，默认 0
        pos_weight=None  # 正样本权重，默认 8.0（用于 TinyObjectLoss）
):
    # -----------------------------
    # 初始化 TinyObjectLoss
    # -----------------------------
    # 设置正样本权重，极小目标建议 8~15
    if pos_weight is None:
        pos_weight = 8.0

    tiny_loss_fn = TinyObjectLoss(
        pos_weight=pos_weight,
        neg_pos_ratio=20,
        lambda_bce=0.65,
        lambda_ft=0.35
    )

    # -----------------------------
    # 一、初始化训练和验证阶段的累计变量
    # -----------------------------
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(
            total=epoch_step,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            postfix=dict,
            mininterval=0.3
        )

    # -----------------------------
    # 二、训练阶段
    # -----------------------------
    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs, pngs, labels = batch

        with torch.no_grad():
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
            # 前向传播
            outputs, aux1, aux2, aux3 = model_train(imgs)

            # 准备不同层级的 GT（膨胀策略）
            # final & ds1: 原始 GT（精确监督）
            # ds2: 3x3 膨胀（感知附近有目标）
            # ds3: 5x5 膨胀（粗定位）
            gt_main = pngs.float()
            gt_ds1 = gt_main
            gt_ds2 = dilate_mask(gt_main, 3)
            gt_ds3 = dilate_mask(gt_main, 5)

            # 计算各级损失（使用 TinyObjectLoss）
            main_loss = tiny_loss_fn(outputs, gt_main)
            aux1_loss = tiny_loss_fn(aux1, gt_ds1)
            aux2_loss = tiny_loss_fn(aux2, gt_ds2)
            aux3_loss = tiny_loss_fn(aux3, gt_ds3)

            # 深监督权重分配：高分辨率权重越大
            # main: 1.0, aux1: 0.75, aux2: 0.20, aux3: 0.05
            loss = main_loss + 0.75 * aux1_loss + 0.20 * aux2_loss + 0.05 * aux3_loss

            # 计算 f_score 指标
            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

        else:
            # 混合精度训练
            from torch.cuda.amp import autocast

            with autocast():
                outputs, aux1, aux2, aux3 = model_train(imgs)

                # 准备不同层级的 GT（膨胀策略）
                gt_main = pngs.float()
                gt_ds1 = gt_main
                gt_ds2 = dilate_mask(gt_main, 3)
                gt_ds3 = dilate_mask(gt_main, 5)

                # 计算各级损失（使用 TinyObjectLoss）
                main_loss = tiny_loss_fn(outputs, gt_main)
                aux1_loss = tiny_loss_fn(aux1, gt_ds1)
                aux2_loss = tiny_loss_fn(aux2, gt_ds2)
                aux3_loss = tiny_loss_fn(aux3, gt_ds3)

                # 深监督权重分配
                loss = main_loss + 0.75 * aux1_loss + 0.20 * aux2_loss + 0.05 * aux3_loss

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    # 训练阶段结束
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(
            total=epoch_step_val,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            postfix=dict,
            mininterval=0.3
        )

    # -----------------------------
    # 三、验证阶段
    # -----------------------------
    model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        imgs, pngs, labels = batch

        with torch.no_grad():
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)

            outputs, aux1, aux2, aux3 = model_train(imgs)

            # 验证时也使用膨胀策略，保持一致性
            gt_main = pngs.float()
            gt_ds1 = gt_main
            gt_ds2 = dilate_mask(gt_main, 3)
            gt_ds3 = dilate_mask(gt_main, 5)

            main_loss = tiny_loss_fn(outputs, gt_main)
            aux1_loss = tiny_loss_fn(aux1, gt_ds1)
            aux2_loss = tiny_loss_fn(aux2, gt_ds2)
            aux3_loss = tiny_loss_fn(aux3, gt_ds3)

            loss = main_loss + 0.75 * aux1_loss + 0.20 * aux2_loss + 0.05 * aux3_loss

            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
                'f_score': val_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    # -----------------------------
    # 四、验证结束后，记录结果并保存模型
    # -----------------------------
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        loss_history.append_loss(
            epoch + 1,
            total_loss / epoch_step,
            val_loss / epoch_step_val
        )

        if eval_callback is not None:
            eval_callback.on_epoch_end(epoch + 1, model_train)

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (
            total_loss / epoch_step,
            val_loss / epoch_step_val
        ))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir,
                    'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                        (epoch + 1),
                        total_loss / epoch_step,
                        val_loss / epoch_step_val
                    )
                )
            )

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_epoch_weights.pth")
            )

        torch.save(
            model.state_dict(),
            os.path.join(save_dir, "last_epoch_weights.pth")
        )