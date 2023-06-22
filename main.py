## Author: Jia-Xuan Jiang
## Year: 2023

from datasets.datasets import Glaucoma_Datasets
import torch
import numpy as np
from torch.utils.data import DataLoader
from configs.hparam import hparams
from tensorboardX import SummaryWriter
import os
import random
import time
from metircs import diceCoeffv2


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
        output = torch.sigmoid(output)
        loss = criterion(output,target)
        losses.update(loss.item(),input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-start)
        start = time.time()

        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.value:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.value:.5f} ({loss.avg:.5f})'.format(
                epoch+1, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    if epoch % 100 ==0:
        torch.save(model.state_dict(), "./checkpoints/"+job_name+"/chekpoints_epoch_{}.pth".format(epoch+1))
    writer.add_scalar('loss/train_loss', losses.value, global_step=epoch)


def validate(val_loader, model, criterion, epoch,device, writer, phase="VAL"):
    batch_time = statistics()
    losses = statistics()
    oc_dice_log = statistics()
    od_dice_log = statistics()
    mean_dice_log = statistics()

    model.eval()

    with torch.no_grad():
        start = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            output = torch.sigmoid(output)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            cur_oc_dice = diceCoeffv2(output[:,1:2,:],target[:,1:2,:])
            oc_dice_log.update(cur_oc_dice,input.size(0))
            mean_dice_log.update(cur_oc_dice,input.size(0))

            cur_od_dice = diceCoeffv2(output[:,2:3,:],target[:,2:3,:])
            od_dice_log.update(cur_od_dice, input.size(0))
            mean_dice_log.update(cur_od_dice, input.size(0))

            batch_time.update(time.time() - start)
            start = time.time()

            if idx % 100 == 0:
                print('Valid-{0}: [{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
                      'Valid Loss {loss.value:.4f} ({loss.avg:.4f})\t'
                      'OC Dice {oc_dice_log.value:.4f} ({oc_dice_log.avg:.4f})\t'
                    'OD Dice {od_dice_log.value:.4f} ({od_dice_log.avg:.4f})\t'
                    'Mean Dice {mean_dice_log.value:.4f} ({mean_dice_log.avg:.4f})\t'.format(
                              phase, idx, len(val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              oc_dice_log=oc_dice_log,
                              od_dice_log=od_dice_log,
                              mean_dice_log=mean_dice_log))
    writer.add_scalar('loss/val_loss', losses.value, global_step=epoch)
    writer.add_scalar('loss/oc_dice', oc_dice_log.avg, global_step=epoch)
    writer.add_scalar('loss/od_dice', od_dice_log.avg, global_step=epoch)
    writer.add_scalar('loss/mean_dice', mean_dice_log.avg, global_step=epoch)
    return oc_dice_log.avg,od_dice_log.avg
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
    device = hparams.device
    job_name = hparams.job_name


    # 加载数据
    train_dataset = Glaucoma_Datasets(train_data_path)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,pin_memory=True,shuffle=True)
    val_dataset = Glaucoma_Datasets(val_data_path)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=batch_size,pin_memory=True,shuffle=False)
    print("训练集数量: %d" % len(train_dataset))
    print("测试集数量: %d" % len(val_dataset))

    # 定义网络
    model = hparams.model_type
    model = model(n_channels=input_channal,n_classes=num_classes).to(device)

    # 定义损失函数和优化器
    lr_init = 0.0001
    lr_stepsize = 20
    weight_decay = 0.001

    DiceLoss = hparams.loss_dict["DiceLoss"]
    criterion = DiceLoss(num_classes=3).to(device)
    # criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
    writer = SummaryWriter('logs/'+job_name)

    # 训练
    print("Start training ...")
    if not os.path.exists(os.path.dirname("./checkpoints/{}/".format(job_name))):
        os.makedirs(os.path.dirname("./checkpoints/{}/".format(job_name)))
    best_oc_dice=0
    best_od_dice=0
    for epoch in range(epochs):
        train(train_dataloader,model,criterion,optimizer,epoch,device,writer)
        scheduler.step()

        oc_dice,od_dice=validate(val_dataloader,model,criterion,epoch,device,writer)
        oc_dice_is_best= oc_dice>best_oc_dice
        best_oc_dice = max(oc_dice,best_oc_dice)
        od_dice_is_best = od_dice > best_od_dice
        best_od_dice = max(od_dice,best_od_dice)
        torch.save(model.state_dict(), "./checkpoints/"+job_name+"/checkpoints_final.pth")
        if oc_dice_is_best:
            torch.save(model.state_dict(), "./checkpoints/"+job_name+"/best_oc_dice.pth")
        if od_dice_is_best:
            torch.save(model.state_dict(), "./checkpoints/"+job_name+"/best_od_dice.pth")

        print("\n")
    print("Everything is done! Good Luck^_^")
    writer.close()







