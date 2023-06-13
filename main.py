## Author: Jia-Xuan Jiang
## Year: 2023

from datasets.datasets import Glaucoma_Datasets
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from configs.hparam import hparams
from tensorboardX import SummaryWriter
import shutil
import random
import time

def save_checkpoint(state,is_best,filename='checkpoint.pth'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'model_best_'+filename)


def train(train_loader, model, criterion, optimizer, epoch,device, writer):
    batch_time=statistics()
    data_time = statistics()
    losses = statistics()
    model.train()
    start = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # 计算数据加载时间
        data_time.update(time.time()- start)

        input = input.to(device)
        target = target.to(device)

        # 计算输出
        output = model(input)
        loss = criterion(output,target)
        losses.update(loss.item(),input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-start)
        start = time.time()

        if idx % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.value:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.value:.5f} ({loss.avg:.5f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    torch.save(model.state_dict(), "./checkpoints/unet.pth")
    writer.add_scalar('loss/train_loss', losses.value, global_step=epoch)


def validate(val_loader, model, criterion, epoch,device, writer, phase="VAL"):
    batch_time = statistics()
    losses = statistics()

    model.eval()

    with torch.no_grad():
        start = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))

            batch_time.update(time.time() - start)
            start = time.time()

            if idx % 1 == 0:
                print('Valid-{0}: [{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
                      'Valid Loss {loss.value:.5f} ({loss.avg:.5f})\t'.format(
                              phase, idx, len(val_loader),
                              batch_time=batch_time,
                              loss=losses))
    writer.add_scalar('loss/valid_loss', losses.value, global_step=epoch)

#     _image = image[0]
#     _label = label[0]*(255/2)
#     _out_image = output[0]*(255/2)
#     img=torch.stack([_image,_label,_out_image],dim=0)
#     save_image(img,"./image_save/unet_img_{}.png".format(epoch+1))


class statistics(object):
    """
    统计并储存数据
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,value,n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = False
    ## 设置随机数种子，用于复现，宇宙的答案-->42
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 超参数初始化
    hparams = hparams()
    input_channal = hparams.input_channel
    num_classes = hparams.num_classes
    batch_size = hparams.batch_size
    epochs = hparams.epochs
    num_workers = hparams.num_workers
    train_data_path= hparams.trian_data_path
    val_data_path = hparams.val_data_path
    model = hparams.model_type
    device = hparams.device
    loss_type=hparams.loss_list[0]

    # 加载数据
    train_dataset = Glaucoma_Datasets(train_data_path)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,pin_memory=True,shuffle=False)
    val_dataset = Glaucoma_Datasets(val_data_path)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=batch_size,pin_memory=True,shuffle=False)
    print("训练集数量: %d" % len(train_dataset))
    print("测试集数量: %d" % len(val_dataset))

    # 定义网络
    model = model(n_channels=input_channal,n_classes=num_classes).to(device)

    # 定义损失函数和优化器
    lr_init = 0.0001
    lr_stepsize = 20
    weight_decay = 0.001
    criterion = loss_type(class_num=3,alpha=torch.Tensor([0.25]),gamma=2,size_average=True).to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
    writer = SummaryWriter('logs/unet')

    # 训练
    print("Start training ...")
    for epoch in range(epochs):
        train(train_dataloader,model,criterion,optimizer,epoch,device,writer)
        scheduler.step()
        validate(val_dataloader,model,criterion,epoch,device,writer)
        print("\n")
    writer.close()







