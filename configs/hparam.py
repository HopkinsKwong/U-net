import torch
from models.unet import UNet
from loss import *

class hparams:
    def __init__(self):
        self.job_name="unet_build_origin_clahe_gamma_Traditional_crop_renumber"
        self.input_channel =3
        self.num_classes = 3
        self.batch_size = 4
        self.lr = 0.001
        self.epochs = 500
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_save_path = "../image_save"
        self.trian_data_path = "./data/build_origin_clahe_gamma_Traditional_crop_renumber/train"
        self.val_data_path= "./data/build_origin_clahe_gamma_Traditional_crop_renumber/val"
        self.num_workers = 16
        self.model_type = UNet
        self.loss_dict = {"DiceLoss":SoftDiceLoss}