import os
import random
import datetime
import pickle
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from deap import base, creator, tools, algorithms

# 你的项目模块
from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
from utils.dataloader_medical import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, worker_init_fn
from utils.utils_fit_old import fit_one_epoch   # 注意：该函数需已添加 pos_weight 参数

# ---------- 全局配置（与 train_medical.py 保持一致） ----------
Cuda = True
seed = 11
num_classes = 2
backbone = "myunet"
pretrained = False
model_path = ""   # GA阶段不使用预训练权重，从0开始或使用主干预训练（可选）
input_shape = [512, 512]
VOCdevkit_path = 'Medical_Datasets'
dice_loss = True
focal_loss = False
cls_weights = np.array([1, 10000], np.float32)   # 正样本权重很高，与训练时一致
num_workers = 0
save_dir = 'temp_ga'          # 临时保存模型，训练结束后可删除
fp16 = False
sync_bn = False
distributed = False

# 遗传算法参数
#GA_EPOCHS = 10                # 每个个体训练的 epoch 数（快速评估）
#POPULATION_SIZE = 10          # 种群大小
#GENERATIONS = 20              # 进化代数
#BATCH_SIZE = 2                # 与 train_medical.py 中冻结阶段 batch_size 一致
GA_EPOCHS = 1                # 每个个体训练的 epoch 数（快速评估）
POPULATION_SIZE = 1         # 种群大小
GENERATIONS = 2              # 进化代数
BATCH_SIZE = 2                # 与 train_medical.py 中冻结阶段 batch_s
# 学习率及优化器参数（与 train_medical.py 一致）
Init_lr = 1e-5
Min_lr = Init_lr * 0.01
optimizer_type = "adam"
momentum = 0.9
weight_decay = 0
lr_decay_type = 'cos'

# 搜索范围（根据任务调整）
POS_WEIGHT_MIN = 100
POS_WEIGHT_MAX = 1000

def save_population(population, logbook, filename='ga_checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({'population': population, 'logbook': logbook}, f)

def load_population(filename='ga_checkpoint.pkl'):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data['population'], data['logbook']
    except FileNotFoundError:
        return None, None

# ---------- 辅助函数 ----------
def get_train_val_loader(batch_size, use_patch=True):
    """返回训练和验证 DataLoader（与 train_medical.py 逻辑一致）"""
    # 读取文件列表
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = [line.strip() for line in f.readlines()]
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = [line.strip() for line in f.readlines()]

    # 创建数据集
    train_dataset = UnetDataset(
        train_lines, input_shape, num_classes, True, VOCdevkit_path,
        use_patch=use_patch, patch_size=128
    )
    val_dataset = UnetDataset(
        val_lines, input_shape, num_classes, False, VOCdevkit_path,
        use_patch=False
    )

    # DataLoader 参数（不使用分布式，单卡）
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=unet_dataset_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed)
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=unet_dataset_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed)
    )
    return train_loader, val_loader

def train_model(pos_weight, epochs=GA_EPOCHS, batch_size=BATCH_SIZE):
    """
    使用给定 pos_weight 训练模型，返回验证集上的平均 f_score
    """
    seed_everything(seed)   # 固定随机种子，确保可重复性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 初始化模型
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    # 2. 数据加载器
    train_loader, val_loader = get_train_val_loader(batch_size, use_patch=True)
    epoch_step = len(train_loader)
    epoch_step_val = len(val_loader)

    # 3. 学习率与优化器
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epochs)

    # 4. 训练循环
    # 确保临时保存目录存在，并创建虚拟 LossHistory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    class DummyLossHistory:
        def __init__(self):
            self.val_loss = []  # 模拟真实 LossHistory 的 val_loss 列表
            self.losses = []  # 模拟 losses 列表

        def append_loss(self, *args):
            pass
    loss_history = DummyLossHistory()

    for epoch in range(epochs):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(
            model_train, model, loss_history, None, optimizer, epoch,
            epoch_step, epoch_step_val, train_loader, val_loader, epochs,
            Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, None,
            save_period=99999, save_dir=save_dir, local_rank=0,
            pos_weight=pos_weight
        )

    # 5. 评估模型
    model.eval()
    total_f_score = 0.0
    with torch.no_grad():
        for imgs, pngs, labels in val_loader:
            if Cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            outputs, _, _, _ = model(imgs)
            from utils.utils_metrics import f_score
            f = f_score(outputs, labels)
            total_f_score += f.item()
    avg_f_score = total_f_score / epoch_step_val

    return avg_f_score

# ---------- 遗传算法设置 ----------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    """生成个体（pos_weight）"""
    return [random.uniform(POS_WEIGHT_MIN, POS_WEIGHT_MAX)]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)        # 混合交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.2)  # 高斯变异
toolbox.register("select", tools.selTournament, tournsize=3)               # 锦标赛选择

def evaluate(individual):
    pos_weight = individual[0]
    fitness = train_model(pos_weight, epochs=GA_EPOCHS)
    return fitness,

toolbox.register("evaluate", evaluate)

# ---------- 主函数 ----------
# ---------- 主函数 ----------
if __name__ == "__main__":
    seed_everything(seed)

    # 尝试加载已有种群
    population, logbook = load_population()
    if population is None:
        population = toolbox.population(n=POPULATION_SIZE)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "avg", "std", "min", "max"
        start_gen = 0
    else:
        start_gen = len(logbook)
        print(f"Resuming from generation {start_gen} with population size {len(population)}")

    # 注册统计
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 检查是否有未评分的个体（如果从检查点加载，所有个体应该都有 fitness，但以防万一）
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if invalid_ind:
        print(f"Evaluating {len(invalid_ind)} invalid individuals...")
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

    # 开始进化
    for gen in range(start_gen, GENERATIONS):
        # 选择下一代
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估未评分的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 替换种群
        population[:] = offspring

        # 记录统计
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(population), **record)
        print(logbook.stream)

        # 保存检查点
        save_population(population, logbook)

    # 输出最终最优个体
    best_individual = tools.selBest(population, k=1)[0]
    best_pos_weight = best_individual[0]
    print(f"\n最优 pos_weight: {best_pos_weight:.4f}")

    # 可选：用最优参数进行一次最终训练（全量 epoch）
    # final_score = train_model(best_pos_weight, epochs=50, batch_size=BATCH_SIZE)
    # print(f"最终模型 f_score: {final_score:.4f}")