## Author: Jia-Xuan Jiang
## Year: 2023

import os
import cv2
import torchvision
from torch.utils.data import Dataset
from torchvision.utils import save_image

class Glaucoma_Datasets(Dataset):
    def __init__(self,path):
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
        label = cv2.imread(os.path.join(labels_path,label_name))
        return self.trans(image),self.trans(label)


if __name__ == '__main__':
    ## just for test
    i = 1
    dataset = Glaucoma_Datasets(r"../data/train")
    for a,b in dataset:
        print(i)
        save_image(a,f"./{i}.jpg",nrow=1)
        i += 1
        if i>5:
            break
