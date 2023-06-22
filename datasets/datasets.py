## Author: Jia-Xuan Jiang
## Year: 2023

import os
import cv2
import torchvision
from torch.utils.data import Dataset
from torchvision.utils import save_image
import numpy as np
from utils.helper import mask_to_onehot

'''
255:背景
0:视杯
128:视盘
'''
palette = [[255],[0],[128]]


class Glaucoma_Datasets(Dataset):
    def __init__(self,path):
        self.palette = palette
        self.path = path
        self.images=os.listdir(os.path.join(path,"images"))
        self.labels=os.listdir(os.path.join(path,"labels"))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        label_name = self.labels[index]
        images_path=os.path.join(self.path,"images")
        labels_path =os.path.join(self.path, "labels")
        image = cv2.imread(os.path.join(images_path,image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(labels_path,label_name),0)
        label = np.expand_dims(label,axis=2)
        label = mask_to_onehot(label,self.palette)
        # # shape from (H, W, C) to (C, H, W)
        # image = image.transpose([2, 0, 1])
        # label = label.transpose([2, 0, 1])
        return self.trans(image),self.trans(label)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # just for test
    # i = 1
    # dataset = Glaucoma_Datasets(r"../data/train")
    # for a,b in dataset:
    #     print(i)
    #     save_image(a,f"./{i}.jpg",nrow=1)
    #     i += 1
    #     if i>5:
    #         break
    train_dataset = Glaucoma_Datasets(r"../data/build_toy/train")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, pin_memory=True, shuffle=False)
    for idx, (input, target) in enumerate(train_dataloader):
        a = input
        b = target
        if idx >1:
            break
    print(target)

