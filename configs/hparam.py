import torch
from models.unet import UNet
from loss import *

class hparams:
    def __init__(self):
        self.input_channel =3
        self.num_classes = 3
        self.batch_size = 1
        self.lr = 0.001
        self.epochs = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_save_path = "../image_save"
        self.trian_data_path = "./data/train"
        self.val_data_path= "./data/train"
        self.num_workers = 2
        self.model_type = UNet
        self.loss_list = [FocalLoss]