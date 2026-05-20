##train.py的修改，用于wandb自动调参@chy_25_04_06

import os
from time import time
from tqdm import tqdm
import argparse
#import wandb  # 直接导入wandb库而不是自己的模块

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from getdata import MyDataset
from network import RCNet
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='RCNet模型训练')
    
    # 训练参数
    parser.add_argument('--batch-size', default=256, type=int, 
                        help='批大小')
    parser.add_argument('--lr', default=0.08, type=float, 
                        help='学习率')
    parser.add_argument('--lr-last', default=0.0007687936451848682, type=float, 
                        help='最终学习率')
    parser.add_argument('--epochs', default=500, type=int, 
                        help='训练轮数')
    parser.add_argument('--patience', default=10, type=int, 
                        help='早停耐心值')
    parser.add_argument('--budget', default=0.7296224819760568, type=float, 
                        help='置信度损失预算')
    parser.add_argument('--lmbda', default=0.13175275840292658, type=float, 
                        help='置信度损失初始系数')
    parser.add_argument('--trainset_rate', default=0.9, type=float, 
                        help='训练集比例')
    parser.add_argument('--validset_rate', default=0.1, type=float, 
                        help='验证集比例')
    parser.add_argument('--testset_rate', default=0.0, type=float, 
                        help='测试集比例')
    #parser.add_argument('--project', default='mp_cut18_p123_cluster_60', type=str,
    #                    help='wandb项目名称')

    # 添加优化器参数
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['Adam', 'SGD', 'AdamW'],
                      help='优化器选择')

    # 网络参数

    parser.add_argument('--n-classes', default=18, type=int,
                        help='分类数量')

    parser.add_argument('--n-block', default=16, type=int,
                        help='BasicBlock的数量')
    parser.add_argument('--base-filters', default=12, type=int,
                        help='基础滤波器数量')
    
    # 添加学习率调度器参数
    parser.add_argument('--lr-scheduler', default='linear_with_warmup', type=str, 
                        choices=['linear', 'cosine', 'step', 'plateau', 'constant_with_warmup', 'linear_with_warmup'],
                        help='学习率调度策略')
    parser.add_argument('--step-size', default=30, type=int,
                        help='step scheduler的步长')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='step scheduler的衰减率')
    parser.add_argument('--decay-epochs', default=100, type=int,
                        help='constant_with_warmup模式下的衰减轮数')

    # 添加warm-up参数
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        help='学习率预热的epoch数量')

    return parser.parse_args()

# 首先解析命令行参数
args = parse_args()

# 注释wandb初始化
#wandb.init(
#    project=args.project,
#    config=vars(args)  # 使用命令行参数作为wandb配置
#)

class Model():

    def __init__(self, net, resume, checkpoint_model_path=None, args=None):
        # 直接使用已经解析好并用于wandb初始化的参数
        self.args = args if args is not None else parse_args()
        
        # 不需要更新wandb配置，因为它已经在前面初始化时使用了命令行参数
        # wandb.config.update(vars(self.args), allow_val_change=True)

        # set save path
        self.root_path = './model_results/'
        self.dataset_name = 'dataset_0_partial'#改为相应数据集的文件名称
        self.model_save_path = self.root_path
        self.figure_save_path = self.root_path
        self.dataset_path = f"../data_123/{self.dataset_name}.npy" #该数据集文件存储的路径
        self.board_writer_path = f'{self.root_path}tensorboard/'
        self.checkpoint_path = f'{self.root_path}checkpoints/'

        # create empty filefolds
        if not os.path.isdir(self.root_path):
            os.mkdir(self.root_path)
        if not os.path.isdir(self.model_save_path):
            os.mkdir(self.model_save_path)
        if not os.path.isdir(self.figure_save_path):
            os.mkdir(self.figure_save_path)
        if not os.path.isdir(self.board_writer_path):
            os.mkdir(self.board_writer_path)
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        # initialize tensorboard
        self.writer = SummaryWriter(self.board_writer_path)

        # checkpoint setting
        self.RESUME = resume                                     # resume training from checkpoint or not
        self.checkpoint_model_path = checkpoint_model_path       # checkpoint file path

        # create logger
        self.logger_path = f'{self.root_path}RCNet.log'
        self.logger = creat_log("RCNet", f"{self.logger_path}")
        self.logger.info("Initializing RCNet...")

        # 从wandb.config获取参数，而不是args
        self.num_works = 8  # 可以选择将这个也加入wandb配置
        self.batch_size = self.args.batch_size  #wandb.config.batch_size
        self.trainset_rate = self.args.trainset_rate  #wandb.config.trainset_rate
        self.validset_rate = self.args.validset_rate  #wandb.config.validset_rate
        self.testset_rate = self.args.testset_rate  #wandb.config.testset_rate
        self.seed = 42  # 可以选择将这个也加入wandb配置
        self.lr = self.args.lr
        self.lr_last = self.args.lr_last
        self.start_epoch = -1  # 可以选择将这个也加入wandb配置
        self.EPOCH = self.args.epochs
        self.warm_up = self.args.warmup_epochs
        self.patience = self.args.patience
        self.budget = self.args.budget
        self.lmbda = self.args.lmbda
        self.decay_epochs = self.args.decay_epochs

        self.logger.info(f'''
                        dataset: {self.dataset_name},
                        num_works: {self.num_works},
                        batch_size: {self.batch_size},
                        trainset_rate: {self.trainset_rate},
                        validset_rate: {self.validset_rate},
                        testset_rate: {self.testset_rate},
                        seed: {self.seed},
                        lr: {self.lr},
                        lr_last: {self.lr_last},
                        start_epoch: {self.start_epoch},
                        EPOCH: {self.EPOCH},
                        warm_up: {self.warm_up},
                        patience: {self.patience},
                        budget: {self.budget},
                        lmbda: {self.lmbda},
                        net: {net}
                        ''')
        
        # initialize the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.net = torch.nn.DataParallel(self.net)
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        elif self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # recover model parameters from checkpoint
        if self.RESUME:

            try:
                checkpoint = torch.load(self.checkpoint_model_path)
            except:
                raise ValueError("Cannot find checkpoint file!")

            self.logger.info("Resume from checkpoint...")
            self.net.load_state_dict(checkpoint['net'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']

        # 记录warm-up参数
        self.warmup_epochs = self.args.warmup_epochs
        
        # 创建学习率调度器
        if self.args.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.EPOCH, eta_min=self.lr_last)
        elif self.args.lr_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        elif self.args.lr_scheduler == 'step_with_warmup':
            # step with warmup将在训练循环中手动实现
            self.scheduler = None
        elif self.args.lr_scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=5)
        elif self.args.lr_scheduler == 'plateau_with_warmup':
            # plateau with warmup将在训练循环中手动实现
            self.scheduler = None
        elif self.args.lr_scheduler == 'constant_with_warmup':
            # 自定义学习率调度 - 经过decay_epochs轮后降至lr_last并保持不变
            self.decay_epochs = self.args.decay_epochs
            self.scheduler = None  # 我们将在训练循环中手动调整学习率
        elif self.args.lr_scheduler == 'linear_with_warmup':
            # 线性衰减带warmup
            self.scheduler = None
        else:  # 线性
            self.scheduler = None
        
        # 记录学习率信息
        self.logger.info(f"学习率调度器: {self.args.lr_scheduler}, Warm-up: {self.warmup_epochs} epochs")

        # 添加metrics记录文件
        self.metrics_path = f'{self.root_path}metrics.txt'

    def loss_fn(self, out, label, conf):

        exp = torch.exp(out)
        tmp1 = exp.gather(1, label.unsqueeze(-1)).squeeze()
        tmp2 = exp.sum(1)
        softmax = tmp1 / tmp2
        xentropy_loss = torch.mean(- torch.log(softmax))

        confidence_loss = torch.mean(- torch.log(conf.squeeze()))
        total_loss = xentropy_loss + self.lmbda * confidence_loss

        if self.budget > confidence_loss:
            self.lmbda = self.lmbda / 1.03
        elif self.budget <= confidence_loss:
            self.lmbda = self.lmbda / 0.8

        return total_loss


    def fit(self):
        # 创建或清空metrics文件
        with open(self.metrics_path, 'w') as f:
            f.write("epoch,valid_loss,valid_acc,valid_conf\n")

        # initialize
        min_loss = np.inf
        list_of_train_loss, list_of_train_acc = [], []
        list_of_valid_loss, list_of_valid_acc = [], []

        # load dataset
        self.logger.info("Loading dataset...")
        dataset = np.load(self.dataset_path)

        seed_everything(self.seed)
        np.random.shuffle(dataset)

        x_train = dataset[:int(self.trainset_rate*dataset.shape[0])][:, :-1]
        y_train = dataset[:int(self.trainset_rate*dataset.shape[0])][:, -1]
        x_valid = dataset[int(self.trainset_rate*dataset.shape[0]): int((self.trainset_rate+self.validset_rate)*dataset.shape[0])][:, :-1]
        y_valid = dataset[int(self.trainset_rate*dataset.shape[0]): int((self.trainset_rate+self.validset_rate)*dataset.shape[0])][:, -1]

        train_set = MyDataset(x_train, y_train)
        valid_set = MyDataset(x_valid, y_valid)

        train_loader = DataLoader(train_set,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_works,
                                pin_memory=True)
        valid_loader = DataLoader(valid_set,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_works,
                                pin_memory=True)

        # train model
        self.logger.info("Training model...")
        for epoch in range(self.start_epoch+1, self.EPOCH):

            t1 = time()

            train_loss, train_acc, train_conf = self.train(train_loader)
            valid_loss, valid_acc, valid_conf = self.valid(valid_loader)

            t2 = time()

            # 打印训练信息
            print(f"Epoch [{epoch}/{self.EPOCH}] "
                  f"Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} "
                  f"Train Acc: {train_acc:.4f} Valid Acc: {valid_acc:.4f} "
                  f"Valid Conf: {valid_conf:.4f} "
                  f"Time: {(t2-t1):.1f}s")

            # 记录验证集指标
            with open(self.metrics_path, 'a') as f:
                f.write(f"{epoch},{valid_loss:.4f},{valid_acc:.4f},{valid_conf:.4f}\n")

            # save the output of each epoch
            list_of_train_loss.append(train_loss)
            list_of_valid_loss.append(valid_loss)
            list_of_train_acc.append(train_acc)
            list_of_valid_acc.append(valid_acc)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Valid", valid_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Valid", valid_acc, epoch)

            # save the best model
            if (valid_loss < min_loss) and (epoch > self.warm_up):
                min_loss = valid_loss
                checkpoint = {
                    "net": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }

                self.logger.info(f"best score: {min_loss:.4f} @ {epoch}")
                torch.save(checkpoint, f'{self.checkpoint_path}ckpt_best.pth')

            # 学习率更新
            if epoch < self.warmup_epochs:
                # 线性增加学习率从0到初始学习率
                warmup_factor = epoch / self.warmup_epochs
                current_lr = self.lr * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                if self.args.lr_scheduler == 'constant_with_warmup':
                    # 在decay_epochs轮后降至lr_last并保持不变
                    if epoch < self.decay_epochs:
                        # 线性衰减
                        current_lr = self.lr - (epoch / self.decay_epochs) * (self.lr - self.lr_last)
                    else:
                        current_lr = self.lr_last
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr
                elif self.args.lr_scheduler == 'step_with_warmup':
                    # Step衰减带warmup
                    effective_epoch = epoch - self.warmup_epochs
                    if effective_epoch % self.args.step_size == 0 and effective_epoch > 0:
                        current_lr = self.optimizer.param_groups[0]['lr'] * self.args.gamma
                    else:
                        current_lr = self.optimizer.param_groups[0]['lr']
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr
                elif self.args.lr_scheduler == 'plateau_with_warmup':
                    # 在warmup后使用ReduceLROnPlateau
                    # 这里我们记录并传递valid_loss作为参考值，但不在每个epoch执行scheduler.step()
                    if epoch % 5 == 0 and epoch > self.warmup_epochs + 5:  # 每5个epoch检查一次
                        if min(list_of_valid_loss[-5:]) > min(list_of_valid_loss[-10:-5]):
                            current_lr = self.optimizer.param_groups[0]['lr'] * 0.1
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = current_lr
                elif self.args.lr_scheduler == 'linear_with_warmup':
                    # 线性衰减带warmup
                    remaining_epochs = self.EPOCH - self.warmup_epochs
                    if remaining_epochs > 0:
                        decay_factor = (epoch - self.warmup_epochs) / remaining_epochs
                        current_lr = self.lr - decay_factor * (self.lr - self.lr_last)
                    else:
                        current_lr = self.lr_last
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr
                elif self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(valid_loss)
                    else:
                        self.scheduler.step()
                    
                    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        current_lr = self.scheduler.get_last_lr()[0]
                    else:
                        current_lr = self.optimizer.param_groups[0]['lr']
                else:
                    # 线性衰减（考虑warm-up）
                    remaining_epochs = self.EPOCH - self.warmup_epochs
                    if remaining_epochs > 0:
                        current_lr = np.linspace(self.lr, self.lr_last, remaining_epochs)[epoch - self.warmup_epochs]
                    else:
                        current_lr = self.lr_last
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr
            
            # 记录当前学习率
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)

        # save the final model
        torch.save(self.net.state_dict(), f"{self.model_save_path}model.pth")

        # plot
        self.logger.info("Plotting figure...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

        ax1.plot(list_of_train_loss[1:], label='train loss')
        ax1.plot(list_of_valid_loss[1:], label='valid loss')
        ax1.legend()
        ax1.set_title("Loss Curve")

        ax2.plot(list_of_train_acc[1:], label='train acc')
        ax2.plot(list_of_valid_acc[1:], label='valid acc')
        ax2.legend()
        ax2.set_title("Acc Curve")

        fig.savefig(f"{self.figure_save_path}model_results.png", bbox_inches='tight')

        return 0


    def train(self, train_loader):

        self.net.train()

        epoch_loss = 0
        epoch_acc = 0
        epoch_conf = 0

        for xrd, label in tqdm(train_loader, desc='Training'):

            xrd = xrd.to(self.device)
            label = label.to(self.device)

            out, conf = self.net(xrd)
            label_one_hot = torch.zeros(out.shape).to(self.device).scatter_(1, label.unsqueeze(dim=1), 1)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            b =  Variable(torch.bernoulli(torch.Tensor(conf.size()).uniform_(0, 1))).to(self.device)
            conff = conf * b + (1 - b)

            out_c = out * conff.expand_as(out) + label_one_hot * (1 - conff.expand_as(label_one_hot))
            out_c = torch.log(out_c)

            pred_y = out.detach().argmax(dim=1)
            acc = (pred_y == label).float().mean()

            self.optimizer.zero_grad()
            loss = self.loss_fn(out_c, label, conf)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_conf += conf.detach().cpu().numpy().mean()

        #wandb.log({"train_loss": epoch_loss / len(train_loader), 
        #           "train_acc": epoch_acc / len(train_loader), 
        #           "train_conf": epoch_conf / len(train_loader)})
        
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader), epoch_conf / len(train_loader)


    def valid(self, valid_loader):

        self.net.eval()

        epoch_loss = 0
        epoch_acc = 0
        epoch_conf = 0

        for xrd, label in tqdm(valid_loader, desc='Validing'):

            xrd = xrd.to(self.device)
            label = label.to(self.device)

            out, conf = self.net(xrd)
            out_c = conf * out
            out_c = torch.log(out_c)

            pred_y = out.detach().argmax(dim=1)
            acc = (pred_y == label).float().mean()
            loss = self.loss_fn(out_c, label, conf)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_conf += conf.detach().cpu().numpy().mean()

        #wandb.log({"valid_loss": epoch_loss / len(valid_loader), 
        #           "valid_acc": epoch_acc / len(valid_loader), 
        #           "valid_conf": epoch_conf / len(valid_loader)})
        
        return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader), epoch_conf / len(valid_loader)


# 直接调用训练函数，而不是通过agent
if __name__ == "__main__":
    # n_classes = xxx       这个要在命令行设置好
    net = RCNet(
        n_classes = args.n_classes,
        base_filters = args.base_filters,
        n_block = args.n_block,
    )
    model = Model(net, resume=False)
    model.fit()


