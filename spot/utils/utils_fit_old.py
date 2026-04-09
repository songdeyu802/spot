# 导入 Python 标准库 os
# 因为后面保存模型时要用 os.path.join 拼接路径
import os

# 导入 PyTorch 深度学习框架
import torch

# 从你项目中的 unet_training 模块里导入几个损失函数
# CE_Loss：交叉熵损失（这段代码里其实没用到）
# Dice_loss：Dice 损失，分割任务常用
# Focal_Loss：Focal Loss，适合处理类别不平衡
# FG_BCE_Loss：前景二分类 BCE 损失
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss, FG_BCE_Loss

# tqdm 用来显示进度条
from tqdm import tqdm

# 获取当前学习率的函数
from utils.utils import get_lr

# 计算分割任务评价指标 f_score 的函数
from utils.utils_metrics import f_score


# 遗传算法优化得到的最优正样本权重
# 这个值通常用于 BCE 类损失里，解决正负样本不平衡问题
BEST_POS_WEIGHT = 107.1416

def fit_one_epoch(
    model_train,        # 用来训练的模型（可能是包装后的模型）
    model,              # 真正保存权重用的模型
    loss_history,       # 记录 loss 历史的对象
    eval_callback,      # 每个 epoch 结束时调用的评估回调
    optimizer,          # 优化器
    epoch,              # 当前是第几个 epoch（从 0 开始）
    epoch_step,         # 训练集一轮要跑多少个 batch
    epoch_step_val,     # 验证集一轮要跑多少个 batch
    gen,                # 训练集数据迭代器
    gen_val,            # 验证集数据迭代器
    Epoch,              # 总 epoch 数
    cuda,               # 是否使用 GPU
    dice_loss,          # 是否额外加 Dice Loss
    focal_loss,         # 是否使用 Focal Loss；否则使用 FG_BCE_Loss
    cls_weights,        # 类别权重
    num_classes,        # 类别数
    fp16,               # 是否使用混合精度训练
    scaler,             # 混合精度训练用的梯度缩放器
    save_period,        # 每隔多少轮保存一次模型
    save_dir,           # 模型保存目录
    local_rank=0,       # 分布式训练时当前进程序号，默认 0
    pos_weight=None     # BCE 中正样本权重，默认 None，None 时用 BEST_POS_WEIGHT
):
    # -----------------------------
    # 一、初始化训练和验证阶段的累计变量
    # -----------------------------

    # 训练总损失
    total_loss = 0

    # 训练总 f_score
    total_f_score = 0

    # 验证总损失
    val_loss = 0

    # 验证总 f_score
    val_f_score = 0

    # 如果当前是主进程（通常 local_rank == 0 表示主进程）
    # 才打印日志和显示进度条，避免多进程时重复输出
    if local_rank == 0:
        print('Start Train')

        # 创建训练进度条
        # total=epoch_step 表示这一轮训练一共多少步
        # desc 显示当前 epoch 信息
        pbar = tqdm(
            total=epoch_step,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            postfix=dict,
            mininterval=0.3
        )

    # -----------------------------
    # 二、训练阶段
    # -----------------------------

    # 把模型切换到训练模式
    # 训练模式下，Dropout、BatchNorm 等会按训练方式工作
    model_train.train()

    # 遍历训练集
    for iteration, batch in enumerate(gen):
        # 如果已经达到这一轮规定的训练步数，就停止
        if iteration >= epoch_step:
            break

        # 从一个 batch 中取出：
        # imgs   : 输入图片
        # pngs   : 标签图（通常是类别索引图）
        # labels : 另一种标签形式（通常给 Dice Loss / 指标使用）
        imgs, pngs, labels = batch

        # 下面这些操作不需要梯度，因此放在 no_grad 里
        with torch.no_grad():
            # 把 numpy 格式的类别权重转成 PyTorch Tensor
            weights = torch.from_numpy(cls_weights)

            # 如果使用 GPU，就把数据和权重都搬到 GPU 上
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        # 每个 batch 开始前先把优化器中的梯度清零
        # 因为 PyTorch 默认梯度会累加，不清零会出问题
        optimizer.zero_grad()

        # ===========================================
        # 情况 1：不用混合精度，走普通精度训练
        # ===========================================
        if not fp16:
            # 前向传播
            # 模型输出主输出 + 3 个辅助输出
            # 说明这是一个带深监督结构的网络
            #这里的model_train(imgs)，本质上就是在调用模型的 forward(imgs)
            outputs, aux1, aux2, aux3 = model_train(imgs)

            # -----------------------------
            # 计算主损失
            # -----------------------------
            if focal_loss:
                # 如果 focal_loss=True，则主损失使用 Focal Loss
                main_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                aux1_loss = Focal_Loss(aux1, pngs, weights, num_classes=num_classes)
                aux2_loss = Focal_Loss(aux2, pngs, weights, num_classes=num_classes)
                aux3_loss = Focal_Loss(aux3, pngs, weights, num_classes=num_classes)
            else:
                # 否则使用前景 BCE 损失 FG_BCE_Loss

                # 如果没有手动传入 pos_weight，就使用预先设好的 BEST_POS_WEIGHT
                if pos_weight is None:
                    pos_w = torch.tensor(BEST_POS_WEIGHT).to(outputs.device)
                else:
                    # 如果传入的 pos_weight 本身已经是 Tensor
                    if torch.is_tensor(pos_weight):
                        pos_w = pos_weight.to(outputs.device)
                    else:
                        # 如果传入的是普通数字，就先转成 Tensor
                        pos_w = torch.tensor(pos_weight).to(outputs.device)

                # outputs[:, 1:2, :, :] 表示只取第 2 个通道（索引 1）
                # 并且保留通道维度
                # 这里通常表示：通道 0 是背景，通道 1 是前景
                main_loss = FG_BCE_Loss(outputs[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux1_loss = FG_BCE_Loss(aux1[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux2_loss = FG_BCE_Loss(aux2[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux3_loss = FG_BCE_Loss(aux3[:, 1:2, :, :], pngs, pos_weight=pos_w)

            # 把主输出和辅助输出的损失加权求和
            # 主输出权重大，辅助输出权重小
            loss = main_loss + 0.5 * aux1_loss + 0.25 * aux2_loss + 0.25 * aux3_loss

            # 如果开启了 Dice Loss，就继续把 Dice Loss 加进去
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                aux1_dice = Dice_loss(aux1, labels)
                aux2_dice = Dice_loss(aux2, labels)
                aux3_dice = Dice_loss(aux3, labels)

                loss = loss + main_dice + 0.5 * aux1_dice + 0.25 * aux2_dice + 0.25 * aux3_dice

            # 计算 f_score 指标
            # 这里只是评估，不需要参与梯度计算
            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            # 反向传播：根据 loss 计算梯度
            loss.backward()

            # 优化器更新参数
            optimizer.step()

        # ===========================================
        # 情况 2：使用混合精度训练
        # ===========================================
        else:
            # autocast 是自动混合精度上下文
            # 会自动选择合适精度进行计算，减少显存占用并提速
            from torch.cuda.amp import autocast

            with autocast():
                # 前向传播
                outputs, aux1, aux2, aux3 = model_train(imgs)

                # -----------------------------
                # 计算主损失
                # -----------------------------
                if focal_loss:
                    # 使用 Focal Loss
                    main_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    aux1_loss = Focal_Loss(aux1, pngs, weights, num_classes=num_classes)
                    aux2_loss = Focal_Loss(aux2, pngs, weights, num_classes=num_classes)
                    aux3_loss = Focal_Loss(aux3, pngs, weights, num_classes=num_classes)
                else:
                    # 使用 FG_BCE_Loss

                    # 动态处理 pos_weight
                    if pos_weight is None:
                        pos_w = torch.tensor(BEST_POS_WEIGHT).to(outputs.device)
                    else:
                        if torch.is_tensor(pos_weight):
                            pos_w = pos_weight.to(outputs.device)
                        else:
                            pos_w = torch.tensor(pos_weight).to(outputs.device)

                    main_loss = FG_BCE_Loss(outputs[:, 1:2, :, :], pngs, pos_weight=pos_w)
                    aux1_loss = FG_BCE_Loss(aux1[:, 1:2, :, :], pngs, pos_weight=pos_w)
                    aux2_loss = FG_BCE_Loss(aux2[:, 1:2, :, :], pngs, pos_weight=pos_w)
                    aux3_loss = FG_BCE_Loss(aux3[:, 1:2, :, :], pngs, pos_weight=pos_w)

                # 把主损失和辅助损失加权相加
                loss = main_loss + 0.5 * aux1_loss + 0.25 * aux2_loss + 0.25 * aux3_loss

                # 如果启用 Dice Loss，再继续加上 Dice 部分
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    aux1_dice = Dice_loss(aux1, labels)
                    aux2_dice = Dice_loss(aux2, labels)
                    aux3_dice = Dice_loss(aux3, labels)

                    loss = loss + main_dice + 0.5 * aux1_dice + 0.25 * aux2_dice + 0.25 * aux3_dice

                # 计算 f_score 指标
                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            # 混合精度训练时不能直接 loss.backward()
            # 需要先用 scaler.scale(loss) 放大 loss，再反向传播
            scaler.scale(loss).backward()

            # 使用 scaler.step 代替 optimizer.step()
            scaler.step(optimizer)

            # 更新 scaler 的缩放因子
            scaler.update()

        # -----------------------------
        # 累加训练阶段 loss 和 f_score
        # -----------------------------

        # loss.item()：把单值 Tensor 取成 Python 数字
        total_loss += loss.item()
        total_f_score += _f_score.item()

        # 如果是主进程，就更新进度条显示
        if local_rank == 0:
            pbar.set_postfix(**{
                # 当前平均训练损失
                'total_loss': total_loss / (iteration + 1),

                # 当前平均 f_score
                'f_score': total_f_score / (iteration + 1),

                # 当前学习率
                'lr': get_lr(optimizer)
            })

            # 进度条前进一步
            pbar.update(1)

    # 训练阶段结束
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')

        # 创建验证阶段进度条
        pbar = tqdm(
            total=epoch_step_val,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            postfix=dict,
            mininterval=0.3
        )

    # -----------------------------
    # 三、验证阶段
    # -----------------------------

    # 切换为验证模式
    # BatchNorm、Dropout 等将按验证/推理方式工作
    model_train.eval()

    # 遍历验证集
    for iteration, batch in enumerate(gen_val):
        # 如果达到验证步数上限，就停止
        if iteration >= epoch_step_val:
            break

        # 取出一个 batch
        imgs, pngs, labels = batch

        # 验证阶段一般不需要梯度
        with torch.no_grad():
            # 把 numpy 格式类别权重转成 Tensor
            weights = torch.from_numpy(cls_weights)

            # 如果使用 GPU，就把数据搬到 GPU
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # 前向传播
            outputs, aux1, aux2, aux3 = model_train(imgs)

            # -----------------------------
            # 计算验证损失
            # -----------------------------
            if focal_loss:
                # 使用 Focal Loss
                main_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                aux1_loss = Focal_Loss(aux1, pngs, weights, num_classes=num_classes)
                aux2_loss = Focal_Loss(aux2, pngs, weights, num_classes=num_classes)
                aux3_loss = Focal_Loss(aux3, pngs, weights, num_classes=num_classes)
            else:
                # 使用 FG_BCE_Loss
                if pos_weight is None:
                    pos_w = torch.tensor(BEST_POS_WEIGHT).to(outputs.device)
                else:
                    if torch.is_tensor(pos_weight):
                        pos_w = pos_weight.to(outputs.device)
                    else:
                        pos_w = torch.tensor(pos_weight).to(outputs.device)

                main_loss = FG_BCE_Loss(outputs[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux1_loss = FG_BCE_Loss(aux1[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux2_loss = FG_BCE_Loss(aux2[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux3_loss = FG_BCE_Loss(aux3[:, 1:2, :, :], pngs, pos_weight=pos_w)

            # 把主输出和辅助输出损失加权求和
            loss = main_loss + 0.5 * aux1_loss + 0.25 * aux2_loss + 0.25 * aux3_loss

            # 如果开启 Dice Loss，就加上 Dice Loss
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                aux1_dice = Dice_loss(aux1, labels)
                aux2_dice = Dice_loss(aux2, labels)
                aux3_dice = Dice_loss(aux3, labels)

                loss = loss + main_dice + 0.5 * aux1_dice + 0.25 * aux2_dice + 0.25 * aux3_dice

            # 计算验证集上的 f_score
            _f_score = f_score(outputs, labels)

            # 累加验证损失和指标
            val_loss += loss.item()
            val_f_score += _f_score.item()

        # 如果是主进程，更新验证进度条
        if local_rank == 0:
            pbar.set_postfix(**{
                # 当前平均验证损失
                'val_loss': val_loss / (iteration + 1),

                # 当前平均验证集 f_score
                'f_score': val_f_score / (iteration + 1),

                # 当前学习率
                'lr': get_lr(optimizer)
            })

            pbar.update(1)

    # -----------------------------
    # 四、验证结束后，记录结果并保存模型
    # -----------------------------
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        # 把这一轮训练损失和验证损失记录到 loss_history 中
        loss_history.append_loss(
            epoch + 1,
            total_loss / epoch_step,
            val_loss / epoch_step_val
        )

        # 如果设置了评估回调，就在 epoch 结束后执行
        if eval_callback is not None:
            eval_callback.on_epoch_end(epoch + 1, model_train)

        # 打印当前 epoch 的结果
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (
            total_loss / epoch_step,
            val_loss / epoch_step_val
        ))

        # 如果达到保存周期，或者已经是最后一轮，则保存一次模型
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

        # 如果当前验证损失是历史最好，则保存为最佳模型
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_epoch_weights.pth")
            )

        # 无论如何，都保存一份“最后一轮权重”
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, "last_epoch_weights.pth")
        )

def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0, pos_weight=None):
    total_loss    = 0
    total_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
            outputs, aux1, aux2, aux3 = model_train(imgs)

            if focal_loss:
                main_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                aux1_loss = Focal_Loss(aux1, pngs, weights, num_classes=num_classes)
                aux2_loss = Focal_Loss(aux2, pngs, weights, num_classes=num_classes)
                aux3_loss = Focal_Loss(aux3, pngs, weights, num_classes=num_classes)
            else:
                # 动态处理 pos_weight
                if pos_weight is None:
                    pos_w = torch.tensor(BEST_POS_WEIGHT).to(outputs.device)
                else:
                    if torch.is_tensor(pos_weight):
                        pos_w = pos_weight.to(outputs.device)
                    else:
                        pos_w = torch.tensor(pos_weight).to(outputs.device)

                main_loss = FG_BCE_Loss(outputs[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux1_loss = FG_BCE_Loss(aux1[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux2_loss = FG_BCE_Loss(aux2[:, 1:2, :, :], pngs, pos_weight=pos_w)
                aux3_loss = FG_BCE_Loss(aux3[:, 1:2, :, :], pngs, pos_weight=pos_w)

            loss = main_loss + 0.5 * aux1_loss + 0.25 * aux2_loss + 0.25 * aux3_loss

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                aux1_dice = Dice_loss(aux1, labels)
                aux2_dice = Dice_loss(aux2, labels)
                aux3_dice = Dice_loss(aux3, labels)

                loss = loss + main_dice + 0.5 * aux1_dice + 0.25 * aux2_dice + 0.25 * aux3_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs, aux1, aux2, aux3 = model_train(imgs)

                if focal_loss:
                    main_loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    aux1_loss = Focal_Loss(aux1, pngs, weights, num_classes=num_classes)
                    aux2_loss = Focal_Loss(aux2, pngs, weights, num_classes=num_classes)
                    aux3_loss = Focal_Loss(aux3, pngs, weights, num_classes=num_classes)
                else:
                    # 动态处理 pos_weight
                    if pos_weight is None:
                        pos_w = torch.tensor(BEST_POS_WEIGHT).to(outputs.device)
                    else:
                        if torch.is_tensor(pos_weight):
                            pos_w = pos_weight.to(outputs.device)
                        else:
                            pos_w = torch.tensor(pos_weight).to(outputs.device)

                    main_loss = FG_BCE_Loss(outputs[:, 1:2, :, :], pngs, pos_weight=pos_w)
                    aux1_loss = FG_BCE_Loss(aux1[:, 1:2, :, :], pngs, pos_weight=pos_w)
                    aux2_loss = FG_BCE_Loss(aux2[:, 1:2, :, :], pngs, pos_weight=pos_w)
                    aux3_loss = FG_BCE_Loss(aux3[:, 1:2, :, :], pngs, pos_weight=pos_w)

                loss = main_loss + 0.5 * aux1_loss + 0.25 * aux2_loss + 0.25 * aux3_loss

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    aux1_dice = Dice_loss(aux1, labels)
                    aux2_dice = Dice_loss(aux2, labels)
                    aux3_dice = Dice_loss(aux3, labels)

                    loss = loss + main_dice + 0.5 * aux1_dice + 0.25 * aux2_dice + 0.25 * aux3_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss    += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))