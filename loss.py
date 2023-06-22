from torch.nn.modules.loss import _Loss
from metircs import diceCoeffv2

class SoftDiceLoss(_Loss):

    def __init__(self, num_classes):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice
